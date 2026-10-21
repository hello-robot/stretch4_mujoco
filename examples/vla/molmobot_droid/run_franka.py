"""
MolmoBot-DROID driving the Franka DROID in any MolmoSpaces house. Type instructions at the
prompt and watch it in MuJoCo's passive viewer and in Rerun (the policy's cameras, the scene
camera and the whole house in 3D). Ctrl+C stops a running instruction.

Between instructions the sim keeps running, so the start pose can be changed first:
- in the viewer, drag the arm's Control sliders (the arm drives there), or pause (Space) and
  drag its Joint sliders (the arm jumps there);
- at the prompt, `jog` moves joints with the keys, `pose` prints the arm's joint positions,
  `set q1 ... q7` (rad) jumps the arm there, and `home` jumps it to the home pose.

`object` says what the target is and what can replace it; `object <name> [index]` puts that in
its place (a THOR category or prefab, or tennis_ball; see `custom_scene.find_asset()`). The scene
is rebuilt, with the arm and its pedestal where they were.

With `--exo_camera left|right` the exo view is Stretch 4's head fisheye on the Franka, cropped
to DROID's aspect, or with `--rectify` redrawn as the DROID exo camera would see it from there
(`FisheyeRectifier`).

`--use_stretch4_gripper sg4|pg4` and `--use_stretch4_gripper_camera` put Stretch 4's gripper and
gripper camera on the Franka in place of the Robotiq and DROID's wrist camera, mounted on its
flange as they are on Stretch 4's wrist (see `franka_retarget/stretch4_hardware_on_franka.py`):
to see how the policy copes with them on the robot it was trained on. Rerun then also shows what
DROID's wrist camera would see on the Franka as it comes, by its Robotiq, without Stretch 4's
hardware (`franka_wrist`) and with it (`franka_wrist_with_stretch4`), as `run_stretch4_sim.py`
does for the ghost Franka over Stretch.

Rerun is updated between instructions too, whenever the Franka or the target moves, and its 3D
scene starts looking at the Franka.

Usage:
    python -m examples.vla.molmobot_droid.run_franka --scene-id procthor-10k/val/0 --object-type mug
    python -m examples.vla.molmobot_droid.run_franka --robot-pose 5.85,7.57,-1.571 --exo_camera center \\
        --execute-horizon 8 --execute-horizon-do-only-first-n-steps 8
    python -m examples.vla.molmobot_droid.run_franka --use_stretch4_gripper pg4 --use_stretch4_gripper_camera
    python -m examples.vla.molmobot_droid.run_franka --use_stretch4_gripper pg4 --use_stretch4_gripper_camera \\
        --apply_franka_gripper_image_mismatch_fix
    python -m examples.vla.molmobot_droid.run_franka --object-type tennis_ball --in-place-of mug
    python -m examples.vla.molmobot_droid.run_franka --exo_camera right --rectify
"""

from __future__ import annotations

import contextlib
import math
import os
import select
import sys
import threading
import time
from typing import Callable

import click
import mujoco
import numpy as np

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import (
    FRANKA_HOME_QPOS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    POLICY_DT,
    build_instruction,
    load_policy,
    unload_policy,
)
from examples.vla.molmobot_droid.droid import HEAD_CAMERAS, FrankaDroidEnv, Observation, RobotPose, body_transform
from examples.vla.molmobot_droid.franka_retarget.gripper_image_fix import FIX_FLAG, gripper_image_fix
from examples.vla.molmobot_droid.franka_retarget.stretch4_hardware_on_franka import (
    ROBOTIQ_WRIST_CAMERAS,
    STRETCH4_GRIPPER_CAMERA,
    STRETCH4_GRIPPERS,
    RobotiqGhost,
)
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    execution_options,
    exo_camera_option,
    STRETCH_TCP,
    TCP_ALIGN,
    HEAD_SIDES,
    gripper_image_fix_option,
    head_crop_option,
    head_rectifier,
    rectify_option,
    to_droid_frame,
    wrist_view,
)
from examples.vla.molmobot_droid.molmospaces.custom_scene import (
    INSERTABLE_OBJECTS,
    OBJECT_COMMANDS,
    SwapTarget,
    object_command,
    run_with_swaps,
    load_custom_scene,
    free_bodies,
    load_custom_scene_franka_droid,
    object_category,
    scene_camera_pose,
    suggest_robot_pose,
)
from examples.vla.molmobot_droid.rollout import StepInfo, interactive_session


def scene_options(function):
    """Options shared by the interactive simulation runners."""
    options = [
        click.option("--scene-id", default="procthor-10k/val/0", show_default=True,
                     help="<dataset>/<split>/<house index>"),
        click.option("--object-type", default="boiler", show_default=True,
                     help="Category of the target object (mug, apple, ...) or its exact body name. With "
                     f"--in-place-of, an object to put in that one's place instead: {', '.join(INSERTABLE_OBJECTS)}, "
                     "a THOR category (apple, alarm clock, ...) or prefab (Apple_3). 'object' at the prompt lists them."),
        click.option("--object-index", type=int, default=0,
                     help="Which one, if there are several (of --in-place-of's, with it)."),
        click.option("--in-place-of", default=None,
                     help="The house's object (a category or body name) to swap --object-type in for, resting "
                     "where that one did."),
        click.option("--robot-pose", default=None,
                     help="x,y,yaw of the robot's footprint (yaw in rad). Default: a clear spot facing the object."),
        click.option("--max-steps", type=int, default=300, show_default=True, help="Steps per instruction (15/s)."),
        click.option("--checkpoint", default=None, help="Checkpoint directory. Default: download from Hugging Face."),
        click.option("--rerun/--no-rerun", default=True, show_default=True),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def resolve_pose(scene, robot_pose: str | None) -> RobotPose:
    pose = RobotPose.parse(robot_pose) if robot_pose else suggest_robot_pose(scene)
    click.secho(f"Robot pose: --robot-pose {pose}", fg="green")
    return pose


IDLE_LOG_TOLERANCE = 1e-4
"""Between instructions, Rerun is logged to when the Franka's or the target's qpos (m, rad) moved
by more than this. Not the rest of the house's: objects across it keep creeping."""

def robot_and_target_qpos(env: FrankaDroidEnv, target: str) -> np.ndarray:
    """Where in `qpos` the Franka's joints (its tool's too) and the `target` body's are."""
    model = env.model
    target_id = model.body(target).id
    sizes = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}
    indices = []
    for joint in range(model.njnt):
        if model.joint(joint).name.startswith(env.spawn.prefix) or model.jnt_bodyid[joint] == target_id:
            start = model.jnt_qposadr[joint]
            indices.extend(range(start, start + sizes.get(model.jnt_type[joint], 1)))
    return np.array(indices, dtype=int)


REACH = 1.0
"""Objects closer than this to `fr3_link0` (m) are listed as pickable: the FR3 reaches ~0.85 m."""


def pickable_objects_text(env: FrankaDroidEnv) -> str:
    """The free objects in reach, nearest first, by the category to name them by."""
    link0 = env.spawn.world_from_link0[:3, 3]
    distances = {name: np.linalg.norm(env.data.body(name).xpos - link0) for name in free_bodies(env.model)}
    in_reach = sorted((d, name) for name, d in distances.items() if d <= REACH)
    listed = ", ".join(f"{object_category(name)} ({d:.2f} m)" for d, name in in_reach) or "none"
    return f"Pickable objects in reach: {listed} ({len(distances) - len(in_reach)} more out of reach)"


class StartPoseEditor:
    """
    Keeps the sim running in real time between instructions, so the passive viewer stays live,
    and moves the arm from the keyboard. Its `idle()` and `handle_command()` plug into
    `interactive_session()`.

    `jog` and `height <m>` also raise or lower the pedestal under the arm.

    Viewer: while running, the arm follows its Control sliders; while paused (Space), its Joint
    sliders move it directly and its targets follow, so it stays put when unpaused.

    `on_frame` is called by `frame()`, at the policy's rate while jogging, on the main thread
    (where the renderers live), to log the cameras.
    """

    COMMANDS = (
        "'jog' to move the arm and its pedestal with the arrow keys, 'home' to send it home, "
        "'pose', 'set q1 ... q7' (rad), 'height <m>' (pedestal)"
    )
    HZ = 60
    PEDESTAL_RANGE = (0.1, 1.5)

    def __init__(self, env: FrankaDroidEnv, on_frame: Callable[[], None] | None = None):
        self.env = env
        self.on_frame = on_frame
        self.arm = env.view.get_move_group("arm")
        self.gripper = env.view.get_move_group("gripper")
        self.limits = np.array(self.arm.joint_pos_limits)
        self.n_substeps = max(1, round(1 / self.HZ / env.model.opt.timestep))
        self._lock = threading.RLock()
        self._running = threading.Event()

    # -- sim ---------------------------------------------------------------

    @contextlib.contextmanager
    def _editing(self):
        """Exclusive access to `env.data`, synced to the viewer afterwards."""
        with self._lock:
            viewer = self.env.viewer
            with viewer.lock() if viewer is not None else contextlib.nullcontext():
                yield
            self.env.sync_viewer()

    def _viewer_paused(self) -> bool:
        sim = self.env.viewer._get_sim() if self.env.viewer is not None else None
        return sim is not None and not sim.run

    def _tick(self) -> None:
        with self._editing():
            if self._viewer_paused():
                # Joint sliders edit qpos while paused: hold the arm wherever they put it.
                self.arm.ctrl = self.arm.joint_pos
                mujoco.mj_forward(self.env.model, self.env.data)
            else:
                mujoco.mj_step(self.env.model, self.env.data, nstep=self.n_substeps)

    def _loop(self) -> None:
        period = self.n_substeps * self.env.model.opt.timestep
        next_tick = time.perf_counter()
        while self._running.is_set():
            if self.env.viewer is not None and not self.env.viewer.is_running():
                return
            self._tick()
            next_tick = max(next_tick + period, time.perf_counter())
            time.sleep(max(0.0, next_tick - time.perf_counter()))

    @contextlib.contextmanager
    def idle(self):
        self._running.set()
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()
        try:
            yield
        finally:
            self._running.clear()
            thread.join()

    def reset(self) -> None:
        with self._lock:
            self.env.reset()

    def frame(self) -> None:
        """`on_frame()`, with the sim held still."""
        if self.on_frame is not None:
            with self._lock:
                self.on_frame()

    # -- arm ---------------------------------------------------------------

    def set_arm(self, q7, jump: bool) -> None:
        """Set the arm's targets to `q7`, and its joints too if `jump` or the viewer is paused."""
        q7 = np.clip(np.asarray(q7, dtype=float), self.limits[:, 0], self.limits[:, 1])
        with self._editing():
            self.arm.ctrl = q7
            if jump or self._viewer_paused():
                self.arm.joint_pos = q7
                self.arm.joint_vel = np.zeros(7)
                mujoco.mj_forward(self.env.model, self.env.data)

    def set_pedestal_height(self, height: float) -> float:
        height = float(np.clip(height, *self.PEDESTAL_RANGE))
        with self._lock:
            self.env.set_pedestal_height(height)
        return height

    def pose_text(self) -> str:
        return (
            "set " + " ".join(f"{q:.4f}" for q in self.arm.joint_pos)
            + f"\nheight {self.env.spawn.pedestal_height:.3f}"
        )

    def handle_command(self, text: str) -> bool:
        words = text.replace(",", " ").split()
        if not words:
            return False
        if words[0] == "pose" and len(words) == 1:
            print(self.pose_text())
        elif words[0] == "home" and len(words) == 1:
            self.set_arm(FRANKA_HOME_QPOS, jump=True)
        elif words[0] == "set" and len(words) == 8:
            try:
                q7 = [float(w) for w in words[1:]]
            except ValueError:
                return False
            self.set_arm(q7, jump=True)
            print(self.pose_text())
        elif words[0] == "height" and len(words) == 2:
            try:
                height = float(words[1])
            except ValueError:
                return False
            print(f"height {self.set_pedestal_height(height):.3f}")
        elif words[0] == "jog" and len(words) == 1:
            self.jog()
        else:
            return False
        return True

    # -- keyboard jogging --------------------------------------------------

    def jog(self) -> None:
        """Move one joint, or the pedestal, at a time from the terminal until Enter, q or Esc."""
        import termios
        import tty

        print(
            "1-7 or left/right: joint (8: pedestal)   up/down or +/-: move   "
            "[ ]: step size (deg, or cm for the pedestal)   g: gripper   h: home   Enter/q/Esc: done"
        )
        pedestal = 7  # the "joint" after the arm's 7
        joint, step = 0, 5.0
        target = self.arm.ctrl.copy()
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        next_frame = time.perf_counter()
        try:
            tty.setcbreak(fd)
            while True:
                q_deg = "  ".join(
                    (f"[{math.degrees(q):7.1f}]" if i == joint else f" {math.degrees(q):7.1f} ")
                    for i, q in enumerate(target)
                )
                height = f"{self.env.spawn.pedestal_height:.2f} m"
                height = f"[{height}]" if joint == pedestal else f" {height} "
                name = "pedestal" if joint == pedestal else f"joint {joint + 1}"
                sys.stdout.write(f"\r\x1b[K{name}  step {step:g}  target {q_deg}  pedestal {height}")
                sys.stdout.flush()

                if self.on_frame is not None:
                    # Log frames until a key comes.
                    while not select.select([fd], [], [], max(0.0, next_frame - time.perf_counter()))[0]:
                        self.frame()
                        next_frame = max(next_frame + POLICY_DT, time.perf_counter())
                key = os.read(fd, 8).decode(errors="ignore")
                if key in ("", "\n", "\r", "q", "\x1b"):
                    break
                if len(key) == 1 and key in "12345678":
                    joint = int(key) - 1
                elif key == "\x1b[C":
                    joint = (joint + 1) % 8
                elif key == "\x1b[D":
                    joint = (joint - 1) % 8
                elif key in ("\x1b[A", "+", "=", "\x1b[B", "-", "_"):
                    sign = 1 if key in ("\x1b[A", "+", "=") else -1
                    if joint == pedestal:
                        self.set_pedestal_height(self.env.spawn.pedestal_height + sign * step / 100)
                        continue
                    target[joint] += sign * math.radians(step)
                elif key == "]":
                    step = min(step * 2, 45.0)
                elif key == "[":
                    step = max(step / 2, 0.25)
                elif key == "h":
                    target = np.array(FRANKA_HOME_QPOS, dtype=float)
                elif key == "g":
                    with self._editing():
                        closed = self.gripper.ctrl[0] > (GRIPPER_OPEN + GRIPPER_CLOSED) / 2
                        self.gripper.ctrl = [GRIPPER_OPEN if closed else GRIPPER_CLOSED]
                    continue
                else:
                    continue
                target = np.clip(target, self.limits[:, 0], self.limits[:, 1])
                self.set_arm(target, jump=False)
        except KeyboardInterrupt:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            print()
        print(self.pose_text())


@click.command()
@scene_options
@exo_camera_option("droid")
@head_crop_option
@execution_options
@click.option(
    "--use_stretch4_gripper",
    "--use-stretch4-gripper",
    "use_stretch4_gripper",
    type=click.Choice(tuple(STRETCH4_GRIPPERS)),
    default=None,
    help="Put Stretch 4's gripper (SG4) or parallel gripper (PG4) on the Franka in place of the Robotiq, "
    "its quick connect on the Franka's flange. Its grasp center ends up 235 mm (SG4) or 167 mm (PG4) from "
    "the flange, against the Robotiq's 155 mm. Its fingers are driven and reported to the policy in the "
    "Robotiq's units.",
)
@click.option(
    "--use_stretch4_gripper_camera",
    "--use-stretch4-gripper-camera",
    "use_stretch4_gripper_camera",
    is_flag=True,
    help="Use Stretch 4's left gripper camera as the wrist camera instead of DROID's, mounted where it is on "
    "Stretch 4 relative to the tool's quick connect, with the quick connect's place on the flange (with the "
    "Robotiq too), and DROID-framed as retargeting frames it.",
)
@gripper_image_fix_option
@rectify_option
def main(scene_id, object_type, object_index, in_place_of, robot_pose, max_steps, checkpoint, rerun, exo_camera,
         head_crop, execute_horizon, execute_first_n, use_stretch4_gripper, use_stretch4_gripper_camera,
         fix_gripper_image_mismatch, rectify):
    if not 1 <= execute_first_n <= execute_horizon:
        raise click.BadParameter("need 1 <= --execute-horizon-do-only-first-n-steps <= --execute-horizon")
    if fix_gripper_image_mismatch and not (use_stretch4_gripper and use_stretch4_gripper_camera):
        # It takes Stretch's fingers out of the image, and the Robotiq's would be left in.
        raise click.BadParameter("needs --use_stretch4_gripper and --use_stretch4_gripper_camera", param_hint=FIX_FLAG)
    if rectify and (exo_camera not in HEAD_SIDES or head_crop != "droid"):
        raise click.BadParameter("needs --exo_camera left or right, and --head-crop droid", param_hint="--rectify")
    scene = load_custom_scene(scene_id, object_type, object_index, in_place_of=in_place_of)
    pose = resolve_pose(scene, robot_pose)

    def say(message: str, color: str | None = None) -> None:
        click.secho(message, fg=color)

    def run_scene(scene, carry: dict) -> None:
        """The Franka in `scene` until `quit`, or until `object` swaps the target (SwapTarget, its
        `carry` the arm's joints and the pedestal's height, which the next scene starts from)."""
        click.secho(f"Target: {scene.object_name} at {scene.object_pos.round(3)}", fg="green")
        env = load_custom_scene_franka_droid(
            scene,
            pose,
            exo_camera=exo_camera,
            stretch4_gripper=use_stretch4_gripper,
            stretch4_gripper_camera=use_stretch4_gripper_camera,
        )
        if exo_camera in HEAD_CAMERAS and head_crop == "droid":
            env.prepare_exo = head_rectifier(exo_camera) if rectify else to_droid_frame
        if fix_gripper_image_mismatch:
            fix = gripper_image_fix(STRETCH4_GRIPPERS[use_stretch4_gripper], STRETCH4_GRIPPER_CAMERA)
            # Where the tool's grasp center is from the Robotiq's grasp_site, as a grasp offset is.
            tool = np.linalg.inv(env.tcp_world() @ TCP_ALIGN) @ body_transform(
                env.data, f"{env.spawn.prefix}gripper/{STRETCH_TCP}"
            )
            env.prepare_wrist = lambda image: fix(image, env.state8[7], tool)
        elif use_stretch4_gripper_camera:
            env.prepare_wrist = wrist_view
        env.launch_viewer()

        logger = None
        robotiq = None
        if rerun:
            robotiq = RobotiqGhost(env) if env.ghost is not None else None
            eye = scene_camera_pose(pose, scene.object_pos)
            if carry:  # a swapped-in target: a recording of its own, with its own scene geometry
                rerun_scene.new_recording(f"{scene.scene_id}: {scene.object_category}", eye=eye)
            else:
                cameras = ["exo", "wrist", "scene"] + (list(ROBOTIQ_WRIST_CAMERAS) if robotiq else [])
                rerun_scene.init_rerun("MolmoBot-DROID Franka", cameras, eye=eye)
            rerun_scene.set_step(0)
            moving = [env.spawn.base_name, scene.object_name] + ([env.ghost.base_name] if robotiq else [])
            logger = rerun_scene.RerunScene(env.model, robotiq.pose() if robotiq else env.data, moving)
        step_count = 0
        sim_time = last_data_time = 0.0

        def log_frame(observation: Observation, metrics: dict[str, float]):
            """One Rerun step: rollout steps and idle frames share the `step` timeline, and
            `sim_time`, which keeps counting across `reset()` putting `data.time` back to 0."""
            nonlocal step_count, sim_time, last_data_time
            step_count += 1
            sim_time += max(0.0, env.data.time - last_data_time)
            last_data_time = env.data.time
            if logger is not None:
                rerun_scene.set_step(step_count, sim_time)
                images = {"exo": observation.exo_rgb, "wrist": observation.wrist_rgb, "scene": env.render_scene()}
                if robotiq is not None:
                    logger.log(robotiq.pose())
                    images.update(robotiq.render())
                else:
                    logger.log(env.data)
                rerun_scene.log_cameras(images)
                rerun_scene.log_tool_poses(None, env.tcp_world(), actual_label="Franka tool")
                rerun_scene.log_metrics(metrics)

        last_logged_state = None
        watched_qpos = robot_and_target_qpos(env, scene.object_name)

        def log_if_moved():
            """A Rerun step outside a rollout, when the robot or the target moved since the last one."""
            nonlocal last_logged_state
            state = np.append(env.data.qpos[watched_qpos], env.spawn.pedestal_height)
            if last_logged_state is not None and np.abs(state - last_logged_state).max() < IDLE_LOG_TOLERANCE:
                return
            last_logged_state = state
            log_frame(env.observe(), {})

        def on_step(info: StepInfo):
            log_frame(info.observation, {"gripper_command": info.action[7], "query": info.query})
            if env.viewer is not None and not env.viewer.is_running():
                raise SystemExit("Viewer closed")

        editor = StartPoseEditor(env, on_frame=log_if_moved if rerun else None)
        if carry:  # the arm and the pedestal where they were before the swap
            editor.set_pedestal_height(carry["pedestal"])
            editor.set_arm(carry["q7"], jump=True)
        editor.frame()  # the start pose, while the policy loads
        try:
            policy = load_policy(checkpoint)
            try:
                interactive_session(
                    env,
                    policy,
                    build_instruction("pick", scene.object_category),
                    execute_horizon,
                    execute_first_n,
                    max_steps,
                    reset=editor.reset,
                    on_step=on_step,
                    handle_command=lambda text: object_command(text, scene, say) or editor.handle_command(text),
                    idle=editor.idle,
                    extra_help=lambda: f"Start pose: {editor.COMMANDS}.\nTarget: {OBJECT_COMMANDS}.\n{pickable_objects_text(env)}",
                    while_waiting=editor.frame if rerun else None,
                )
            except SwapTarget as swap:
                swap.carry.update(q7=editor.arm.joint_pos.copy(), pedestal=env.spawn.pedestal_height)
                raise
        finally:
            if robotiq is not None:
                robotiq.close()
            env.close()

    try:
        run_with_swaps(scene, run_scene, say)
    finally:
        unload_policy()

if __name__ == "__main__":
    main()
