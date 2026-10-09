"""
MolmoBot-DROID driving the Franka DROID in any MolmoSpaces house. Type instructions at the
prompt and watch it in MuJoCo's passive viewer and in Rerun (the policy's cameras, the scene
camera and the whole house in 3D). Ctrl+C stops a running instruction.

Between instructions the sim keeps running, so the start pose can be changed first:
- in the viewer, drag the arm's Control sliders (the arm drives there), or pause (Space) and
  drag its Joint sliders (the arm jumps there);
- at the prompt, `jog` moves joints with the keys, `pose` prints the arm's joint positions,
  `set q1 ... q7` (rad) jumps the arm there, and `home` jumps it to the home pose.

Usage:
    python -m examples.vla.molmobot_droid.run_franka --scene-id procthor-10k/val/0 --object-type mug
    python -m examples.vla.molmobot_droid.run_franka --robot-pose 5.85,7.57,-1.571 --exo_camera center \\
        --execute-horizon 8 --execute-horizon-do-only-first-n-steps 2
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
from examples.vla.molmobot_droid.droid import HEAD_CAMERAS, FrankaDroidEnv, Observation, RobotPose
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    execution_options,
    exo_camera_option,
    head_crop_option,
    to_droid_frame,
)
from examples.vla.molmobot_droid.molmospaces.custom_scene import (
    load_custom_scene,
    free_bodies,
    load_custom_scene_franka_droid,
    object_category,
    suggest_robot_pose,
)
from examples.vla.molmobot_droid.rollout import StepInfo, interactive_session


def scene_options(function):
    """Options shared by the interactive simulation runners."""
    options = [
        click.option("--scene-id", default="procthor-10k/val/0", show_default=True,
                     help="<dataset>/<split>/<house index>"),
        click.option("--object-type", default="boiler", show_default=True,
                     help="Category of the target object (mug, apple, ...) or its exact body name."),
        click.option("--object-index", type=int, default=0, help="Which one, if there are several."),
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

    Viewer: while running, the arm follows its Control sliders; while paused (Space), its Joint
    sliders move it directly and its targets follow, so it stays put when unpaused.

    `on_frame` is called at the policy's rate while jogging, on the main thread (where the
    renderers live), to log the cameras.
    """

    COMMANDS = (
        "'jog' to move the arm with the arrow keys, 'home' to send it home, "
        "'pose', 'set q1 ... q7' (rad)"
    )
    HZ = 60

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

    def pose_text(self) -> str:
        return "set " + " ".join(f"{q:.4f}" for q in self.arm.joint_pos)

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
        elif words[0] == "jog" and len(words) == 1:
            self.jog()
        else:
            return False
        return True

    # -- keyboard jogging --------------------------------------------------

    def jog(self) -> None:
        """Move one joint at a time from the terminal until Enter, q or Esc."""
        import termios
        import tty

        print(
            "1-7 or left/right: joint   up/down or +/-: move   [ ]: step size   "
            "g: gripper   h: home   Enter/q/Esc: done"
        )
        joint, step_deg = 0, 5.0
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
                sys.stdout.write(f"\r\x1b[Kjoint {joint + 1}  step {step_deg:g} deg  target {q_deg}")
                sys.stdout.flush()

                if self.on_frame is not None:
                    # Log frames until a key comes.
                    while not select.select([fd], [], [], max(0.0, next_frame - time.perf_counter()))[0]:
                        with self._lock:
                            self.on_frame()
                        next_frame = max(next_frame + POLICY_DT, time.perf_counter())
                key = os.read(fd, 8).decode(errors="ignore")
                if key in ("", "\n", "\r", "q", "\x1b"):
                    break
                if len(key) == 1 and key in "1234567":
                    joint = int(key) - 1
                elif key == "\x1b[C":
                    joint = (joint + 1) % 7
                elif key == "\x1b[D":
                    joint = (joint - 1) % 7
                elif key in ("\x1b[A", "+", "="):
                    target[joint] += math.radians(step_deg)
                elif key in ("\x1b[B", "-", "_"):
                    target[joint] -= math.radians(step_deg)
                elif key == "]":
                    step_deg = min(step_deg * 2, 45.0)
                elif key == "[":
                    step_deg = max(step_deg / 2, 0.25)
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
def main(scene_id, object_type, object_index, robot_pose, max_steps, checkpoint, rerun, exo_camera, head_crop,
         execute_horizon, execute_first_n):
    if not 1 <= execute_first_n <= execute_horizon:
        raise click.BadParameter("need 1 <= --execute-horizon-do-only-first-n-steps <= --execute-horizon")
    scene = load_custom_scene(scene_id, object_type, object_index)
    click.secho(f"Target: {scene.object_name} at {scene.object_pos.round(3)}", fg="green")
    pose = resolve_pose(scene, robot_pose)
    env = load_custom_scene_franka_droid(scene, pose, exo_camera=exo_camera)
    if exo_camera in HEAD_CAMERAS and head_crop == "droid":
        env.prepare_exo = to_droid_frame
    env.launch_viewer()

    logger = None
    if rerun:
        rerun_scene.init_rerun("MolmoBot-DROID Franka", ["exo", "wrist", "scene"])
        rerun_scene.set_step(0)
        logger = rerun_scene.RerunScene(env.model, env.data, [env.spawn.base_name, scene.object_name])
    step_count = 0

    def log_frame(observation: Observation, metrics: dict[str, float]):
        """One Rerun step: rollout steps and jogging frames share the `step` timeline."""
        nonlocal step_count
        step_count += 1
        if logger is not None:
            rerun_scene.set_step(step_count)
            logger.log(env.data)
            rerun_scene.log_cameras(
                {"exo": observation.exo_rgb, "wrist": observation.wrist_rgb, "scene": env.render_scene()}
            )
            rerun_scene.log_tool_poses(None, env.tcp_world())
            rerun_scene.log_metrics(metrics)

    def on_step(info: StepInfo):
        log_frame(info.observation, {"gripper_command": info.action[7], "query": info.query})
        if env.viewer is not None and not env.viewer.is_running():
            raise SystemExit("Viewer closed")

    editor = StartPoseEditor(env, on_frame=(lambda: log_frame(env.observe(), {})) if rerun else None)
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
            handle_command=editor.handle_command,
            idle=editor.idle,
            extra_help=lambda: f"Start pose: {editor.COMMANDS}.\n{pickable_objects_text(env)}",
        )
    finally:
        unload_policy()
        env.close()


if __name__ == "__main__":
    main()
