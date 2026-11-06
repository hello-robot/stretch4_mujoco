"""
MolmoBot-DROID driving Stretch 4 in simulation, through Franka retargeting
(`franka_retarget/stretch4_retarget.py`), in any MolmoSpaces house. Type an instruction and
press Enter, and Stretch works on it until told otherwise, as `run_stretch4_real.py` does (see
`franka_retarget/console.py`'s KEYS_HELP): Enter stops it, `home` or `reset` goes back to the
start pose. Watch it in stretch4_mujoco's passive viewer and in Rerun: the policy's views,
Stretch's raw cameras, the scene camera and the house in 3D, with the ghost Franka the policy
thinks it is driving overlaid, and its wrist camera's view with and without Stretch in it, when
--include_franka is given. With --use_franka_gripper_camera the policy's wrist view is that
camera without Stretch in it, instead of Stretch's gripper camera.

With --exo_camera left or right (the default), the policy's exo view is the head fisheye on the
side the tool is toward, as on the real robot (see `head_camera_for_tool()`), unless
--no_head_follows_tool keeps it on --exo_camera's; with --rectify,
redrawn as the DROID exo camera would see it (`FisheyeRectifier`) instead of only cropped to its
aspect.

The start pose can be changed between instructions (see `StretchStartPoseEditor`): `jog` moves
the virtual Franka's joints and pedestal with the keys and Stretch follows, `pose` prints them,
`set q1 ... q7` (rad) moves it there, `height <m>` sets the pedestal, and `offset x y z [roll
yaw pitch]` changes the grasp offset (mm, deg; `offset` alone prints it).

`object` says what the target is and what can replace it; `object <name> [index]` puts that in
its place (a THOR category or prefab, or tennis_ball; see `custom_scene.find_asset()`). The
scene is rebuilt and the simulator restarted, with Stretch back where it was.

Usage:
    python -m examples.vla.molmobot_droid.run_stretch4_sim --scene-id procthor-10k/val/0 --object-type boiler
    python -m examples.vla.molmobot_droid.run_stretch4_sim --include_franka --exo_camera center \\
        --head-crop droid --gripper_camera right --slow --grasp-offset-mm 0,0,10
    python -m examples.vla.molmobot_droid.run_stretch4_sim --include_franka --custom_franka_start_pose
    python -m examples.vla.molmobot_droid.run_stretch4_sim --overlay_franka_gripper
    python -m examples.vla.molmobot_droid.run_stretch4_sim --use_franka_gripper_camera
    python -m examples.vla.molmobot_droid.run_stretch4_sim --rectify
    python -m examples.vla.molmobot_droid.run_stretch4_sim --object-type tennis_ball --in-place-of mug
"""

from __future__ import annotations

import click
import numpy as np

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS, build_instruction, load_policy, unload_policy
from examples.vla.molmobot_droid.franka_retarget.console import KEYS_HELP, Console, continuous_session
from examples.vla.molmobot_droid.franka_retarget.gripper_image_fix import reject_with_fix
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    CUSTOM_START_QPOS,
    Stretch4SimEnv,
    franka_yaw_for_start,
    apply_overlay_grasp_offset,
    custom_start_link0_height,
    custom_start_option,
    overlay_gripper_option,
    params_from_kwargs,
    retarget_options,
    spawn_stretch4,
)
from examples.vla.molmobot_droid.franka_retarget.start_pose_editor import StretchStartPoseEditor
from examples.vla.molmobot_droid.molmospaces.benchmark import allow_ctrl_c
from examples.vla.molmobot_droid.molmospaces.custom_scene import (
    STRETCH_ROOT_BODY,
    SceneMirror,
    SwapTarget,
    free_bodies,
    object_command,
    run_with_swaps,
    load_custom_scene,
    load_custom_scene_stretch4,
    scene_camera_pose,
    virtual_franka_link0_height,
)
from examples.vla.molmobot_droid.rollout import StepInfo
from examples.vla.molmobot_droid.run_franka import resolve_pose, scene_options

RERUN_CAMERAS = ["exo", "wrist", "head_raw", "gripper_raw", "scene"]
FRANKA_WRIST_CAMERAS = {"franka_wrist": "franka_with_stretch", "franka_wrist_no_stretch": "franka_alone"}
"""With --include_franka: what the ghost Franka's wrist camera sees, the Franka in its own look,
with Stretch in view and with Stretch hidden (the `OverlayKind` for
`SceneMirror.render_with_franka()`), by Rerun camera name."""


def include_franka_option(function):
    return click.option(
        "--include_franka",
        "--include-franka",
        "include_franka",
        is_flag=True,
        help="Overlay the Franka the policy is driving: see-through and non-colliding, static at "
        "its home pose in the MuJoCo viewer, moving in Rerun and the scene camera.",
    )(function)


def franka_gripper_camera_option(function):
    return click.option(
        "--use_franka_gripper_camera",
        "--use-franka-gripper-camera",
        "use_franka_gripper_camera",
        is_flag=True,
        help="Feed the policy the ghost Franka's wrist camera, with Stretch hidden, for its wrist "
        "view instead of Stretch's gripper camera. Implies --include_franka.",
    )(function)


class StretchRerunLogger:
    """Logs a `Stretch4SimEnv` step: its mirror scene, cameras, and target vs. actual tool."""

    def __init__(self, env: Stretch4SimEnv):
        self.env = env
        stretch_scene = env.stretch_scene
        moving = [STRETCH_ROOT_BODY, stretch_scene.scene.object_name]
        if stretch_scene.include_franka:
            moving.append(stretch_scene.franka.base_name)
        self.scene = rerun_scene.RerunScene(env.mirror.model, env.mirror.data, moving)

    def log_pose(self, step: int) -> None:
        """Outside a rollout (the start pose, jogging): where the tool should be for the Franka,
        and where it got to (if it could get there at all)."""
        self._log_scene(step, self.env.observe())

    def _log_scene(self, step: int, observation) -> np.ndarray:
        """
        The mirror scene, the cameras, and the target vs. actual tool, as of `observation`
        (Stretch keeps moving after it, so its actual tool is from the state it read, not
        now); returns the actual.
        """
        env = self.env
        sim_time, footprint, joints = env.observed_state
        rerun_scene.set_step(step, sim_time)
        self.scene.log(env.mirror.data)
        images = {"exo": observation.exo_rgb, "wrist": observation.wrist_rgb, **observation.extra_cameras,
                  "scene": env.render_scene()}
        if env.stretch_scene.include_franka:
            wrist_camera = env.stretch_scene.franka.wrist_camera_name
            for name, kind in FRANKA_WRIST_CAMERAS.items():
                images[name] = env.mirror.render_with_franka(wrist_camera, observation.state8[7], kind)
        rerun_scene.log_cameras(images)
        actual = env.retargeter.stretch_tool_world(footprint, joints)
        rerun_scene.log_tool_poses(env.retargeter.last_target_tool_world, actual)
        return actual

    def log(self, step: int, info: StepInfo) -> None:
        env = self.env
        actual = self._log_scene(step, info.observation)
        targets = info.step_result
        metrics = {
            "gripper_command": info.action[7],
            "query": info.query,
            "ik_failures": env.retargeter.ik_failures,
            "ik_clamped": env.retargeter.ik_clamped,
        }
        if env.retargeter.last_target_tool_world is not None:
            metrics["tool_error_m"] = float(np.linalg.norm(actual[:3, 3] - env.retargeter.last_target_tool_world[:3, 3]))
        if targets is not None:
            metrics["base_rotate_by_rad"] = targets.base_rotate_by
        rerun_scene.log_metrics(metrics)


OVERLAY_WATCH_RADIUS = 2.0
"""m. With --overlay_franka_gripper the wrist view is rendered in `SceneMirror`, which moves only
the bodies the simulator reports: so it reports the free objects this close to the robot too."""


def nearby_free_bodies(stretch_scene, radius: float = OVERLAY_WATCH_RADIUS) -> list[str]:
    """The scene's free bodies (the objects that can move) within `radius` of the robot."""
    import mujoco

    model = stretch_scene.model
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    robot = np.array([stretch_scene.robot_pose.x, stretch_scene.robot_pose.y])
    return [name for name in free_bodies(model) if np.linalg.norm(data.body(name).xpos[:2] - robot) <= radius]


def with_nearby_free_bodies(stretch_scene, watched: list[str]) -> list[str]:
    """`watched`, and the free bodies near the robot, for --overlay_franka_gripper."""
    return watched + [name for name in nearby_free_bodies(stretch_scene) if name not in watched]


@click.command()
@scene_options
@retarget_options
@include_franka_option
@custom_start_option
@overlay_gripper_option
@franka_gripper_camera_option
def main(scene_id, object_type, object_index, in_place_of, robot_pose, max_steps, checkpoint, rerun, include_franka,
         custom_franka_start_pose, overlay_franka_gripper, use_franka_gripper_camera, **kwargs):
    params = params_from_kwargs(kwargs)
    reject_with_fix(
        params.fix_gripper_image_mismatch,
        overlay_franka_gripper=overlay_franka_gripper,
        use_franka_gripper_camera=use_franka_gripper_camera,
    )
    # The Robotiq and the Franka's wrist camera are the ghost's.
    include_franka = include_franka or overlay_franka_gripper or use_franka_gripper_camera
    if overlay_franka_gripper:
        apply_overlay_grasp_offset(params)
        click.secho(f"Grasp offset {params.effective_grasp_offset_mm} mm", fg="green")
    run_sim(params, scene_id, object_type, object_index, in_place_of, robot_pose, max_steps, checkpoint, rerun,
            include_franka, custom_franka_start_pose, overlay_franka_gripper, use_franka_gripper_camera)


def run_sim(params, scene_id, object_type, object_index, in_place_of, robot_pose, max_steps, checkpoint, rerun,
            include_franka=False, custom_franka_start_pose=False, overlay_franka_gripper=False,
            use_franka_gripper_camera=False, env_class=Stretch4SimEnv, title="MolmoBot-DROID Stretch 4"):
    """
    The session `main()` runs, with the policy at `checkpoint` driving an `env_class` (a
    `Stretch4SimEnv`, or `finetune.stretch4_env.FinetunedStretch4SimEnv` for a fine-tuned
    policy, whose start pose commands go to its `franka_controls`), and Rerun's recording called `title`.
    """
    scene = load_custom_scene(scene_id, object_type, object_index, in_place_of=in_place_of)
    pose = resolve_pose(scene, robot_pose)
    console = None

    def say(message: str, color: str | None = None) -> None:
        if console is not None:
            console.event(message, color)
        else:
            click.secho(message, fg=color)

    def run_scene(scene, carry: dict) -> None:
        """Stretch in `scene` until `quit`, or until `object` swaps the target (SwapTarget, its
        `carry` the virtual Franka's joints and pedestal, which the next scene starts from)."""
        nonlocal console
        click.secho(f"Target: {scene.object_name} at {scene.object_pos.round(3)}", fg="green")
        link0_height = virtual_franka_link0_height(scene)
        start_q7 = np.array(FRANKA_HOME_QPOS)
        if custom_franka_start_pose:
            link0_height = custom_start_link0_height(params, scene.floor_z)
            start_q7 = np.array(CUSTOM_START_QPOS)
            click.secho(f"Franka fr3_link0 at z={link0_height:.3f} (molmospaces would put it at "
                        f"{virtual_franka_link0_height(scene):.3f})", fg="green")
        if "pedestal" in carry:
            link0_height = scene.floor_z + carry["pedestal"]
        # Stretch spawns turned the way the start pose would turn its base, so it does not have to.
        franka_yaw = franka_yaw_for_start(params, link0_height, start_q7, scene.floor_z)
        stretch_scene = load_custom_scene_stretch4(
            scene, pose, include_franka=include_franka, tool_name=params.tool_name, link0_height=link0_height,
            stretch_yaw=-franka_yaw,
        )

        if stretch_scene.removed_bodies:
            say(f"Removed furniture Stretch 4 would spawn inside: {stretch_scene.removed_bodies}", "yellow")
        if stretch_scene.backed_off:
            say(f"Moved {stretch_scene.backed_off:.2f} m back from the target so Stretch 4 clears what it "
                f"rests on: --robot-pose {stretch_scene.robot_pose}", "yellow")
        # As on the real robot, the exo view follows the tool between the side head fisheyes (--head_follows_tool).
        sim = spawn_stretch4(stretch_scene, params)
        if overlay_franka_gripper:
            sim.watch_bodies(with_nearby_free_bodies(stretch_scene, stretch_scene.watched_bodies))
        env = None
        try:
            sim.start(viewer_look_at_body=STRETCH_ROOT_BODY)
            allow_ctrl_c()  # start() takes Ctrl+C over to stop only the simulator, leaving this waiting on the console
            env = env_class(sim, stretch_scene, params, SceneMirror(stretch_scene))
            env.overlay_franka_gripper = overlay_franka_gripper
            env.use_franka_gripper_camera = use_franka_gripper_camera
            env.on_exo_side_change = lambda side, facing, tool_side: say(
                f"◐ exo view from the {side} head camera (tool facing {facing}, on the {tool_side})"
            )
            logger = None
            if rerun:
                eye = scene_camera_pose(stretch_scene.robot_pose, scene.object_pos)
                if carry:  # a swapped-in target: a recording of its own, with its own scene geometry
                    rerun_scene.new_recording(f"{scene.scene_id}: {scene.object_category}", eye=eye)
                else:
                    franka_cameras = list(FRANKA_WRIST_CAMERAS) if include_franka else []
                    rerun_scene.init_rerun(
                        title, RERUN_CAMERAS + franka_cameras,
                        main_cameras=["exo", "wrist", *franka_cameras], eye=eye,
                    )
                rerun_scene.set_step(0)
                env.observe()  # poses the mirror
                logger = StretchRerunLogger(env)
            step_count = 0

            env.start_q7 = start_q7

            def log_pose() -> None:
                """A Rerun step outside a rollout; rollout steps and these share the `step` timeline."""
                nonlocal step_count
                if logger is not None:
                    step_count += 1
                    logger.log_pose(step_count)

            def go_to_start() -> None:
                """To the start pose, logging its tool target in Rerun even when Stretch cannot reach it."""
                try:
                    env.reset()
                finally:
                    log_pose()

            say("Moving Stretch 4 to the Franka's start pose...", "yellow")
            go_to_start()
            go_to_carried_pose(env, carry, say)
            log_pose()

            def on_step(info: StepInfo):
                nonlocal step_count
                step_count += 1
                if not sim.is_running():
                    raise SystemExit("The simulator stopped")
                if logger is not None:
                    logger.log(step_count, info)

            policy = load_policy(checkpoint)
            if console is None:
                click.echo(f"\n{KEYS_HELP}\n  'object [name [index]]' + Enter  swap another object in for the target\n")
                console = Console()
            say(f"The target's instruction: {build_instruction('pick', scene.object_category)}", "green")
            editor = StretchStartPoseEditor(
                getattr(env, "franka_controls", env), go_to=env.move_to_franka_pose,
                set_pedestal_height=stretch_scene.set_franka_pedestal_height, keys=console.keys, say=console.event, on_move=log_pose,
                extra_commands=lambda text: object_command(text, scene, console.event),
            )
            try:
                continuous_session(env, policy, params, max_steps, on_step, console, editor, reset=env.reset)
            except SwapTarget as swap:
                swap.carry.update(q7=editor.franka_pose(), pedestal=stretch_scene.franka.pedestal_height)
                raise
        finally:
            if env is not None:
                env.close()
            sim.stop()

    try:
        run_with_swaps(scene, run_scene, say)
    except KeyboardInterrupt:
        pass
    finally:
        if console is not None:
            console.close()
        unload_policy()


def go_to_carried_pose(env, carry: dict, say) -> None:
    """After an `object` swap, Stretch back where it was: at the Franka's `carry["q7"]`, from
    the start pose. Stays at the start pose if it cannot get there."""
    if "q7" not in carry:
        return
    try:
        env.move_to_franka_pose(carry["q7"])
    except RuntimeError:
        say("Stretch 4 cannot get back to where it was: staying at the start pose", "yellow")
        env.reset()

if __name__ == "__main__":
    main()
