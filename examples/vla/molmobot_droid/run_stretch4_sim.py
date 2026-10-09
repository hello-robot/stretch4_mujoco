"""
MolmoBot-DROID driving Stretch 4 in simulation, through Franka retargeting
(`franka_retarget/stretch4_retarget.py`), in any MolmoSpaces house. Type instructions at the
prompt and watch it in stretch4_mujoco's passive viewer and in Rerun: the policy's views,
Stretch's raw cameras, the scene camera and the house in 3D, with the ghost Franka the policy
thinks it is driving overlaid when --include_franka is given.

Between instructions, the start pose can be changed at the prompt (see `StretchStartPoseEditor`):
`jog` moves the virtual Franka's joints and pedestal with the keys and Stretch follows, `pose`
prints them, `set q1 ... q7` (rad) moves it there, `height <m>` sets the pedestal, and `home` or
`reset` goes back to the start pose, and `offset x y z [roll yaw pitch]` changes the grasp offset
(mm, deg; `offset` alone prints it).

Usage:
    python -m examples.vla.molmobot_droid.run_stretch4_sim --scene-id procthor-10k/val/0 --object-type boiler
    python -m examples.vla.molmobot_droid.run_stretch4_sim --include_franka --exo_camera center \\
        --head-crop droid --gripper_camera right --slow --grasp-offset-mm 0,0,10
    python -m examples.vla.molmobot_droid.run_stretch4_sim --include_franka --custom_franka_start_pose
    python -m examples.vla.molmobot_droid.run_stretch4_sim --overlay_franka_gripper
"""

from __future__ import annotations

import click
import numpy as np

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS, build_instruction, load_policy, unload_policy
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
from examples.vla.molmobot_droid.molmospaces.custom_scene import (
    STRETCH_ROOT_BODY,
    SceneMirror,
    free_bodies,
    load_custom_scene,
    load_custom_scene_stretch4,
    virtual_franka_link0_height,
)
from examples.vla.molmobot_droid.rollout import StepInfo, interactive_session
from examples.vla.molmobot_droid.run_franka import resolve_pose, scene_options

RERUN_CAMERAS = ["exo", "wrist", "head_raw", "gripper_raw", "scene"]


def include_franka_option(function):
    return click.option(
        "--include_franka",
        "--include-franka",
        "include_franka",
        is_flag=True,
        help="Overlay the Franka the policy is driving: see-through and non-colliding, static at "
        "its home pose in the MuJoCo viewer, moving in Rerun and the scene camera.",
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
        rerun_scene.set_step(step)
        self._log_scene(self.env.observe())

    def _log_scene(self, observation) -> np.ndarray:
        """The mirror scene, the cameras, and the target vs. actual tool; returns the actual."""
        env = self.env
        self.scene.log(env.mirror.data)
        rerun_scene.log_cameras(
            {"exo": observation.exo_rgb, "wrist": observation.wrist_rgb, **observation.extra_cameras,
             "scene": env.render_scene()}
        )
        actual = env.retargeter.stretch_tool_world(env.world_from_footprint(), env.joints())
        rerun_scene.log_tool_poses(env.retargeter.last_target_tool_world, actual)
        return actual

    def log(self, step: int, info: StepInfo) -> None:
        env = self.env
        rerun_scene.set_step(step)
        actual = self._log_scene(info.observation)
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
def main(scene_id, object_type, object_index, robot_pose, max_steps, checkpoint, rerun, include_franka,
         custom_franka_start_pose, overlay_franka_gripper, **kwargs):
    params = params_from_kwargs(kwargs)
    include_franka = include_franka or overlay_franka_gripper  # the Robotiq is the ghost's
    if overlay_franka_gripper:
        apply_overlay_grasp_offset(params)
        click.secho(f"Grasp offset {params.effective_grasp_offset_mm} mm", fg="green")
    scene = load_custom_scene(scene_id, object_type, object_index)
    click.secho(f"Target: {scene.object_name} at {scene.object_pos.round(3)}", fg="green")
    pose = resolve_pose(scene, robot_pose)
    link0_height = virtual_franka_link0_height(scene)
    start_q7 = np.array(FRANKA_HOME_QPOS)
    if custom_franka_start_pose:
        link0_height = custom_start_link0_height(params, scene.floor_z)
        start_q7 = np.array(CUSTOM_START_QPOS)
        click.secho(f"Franka fr3_link0 at z={link0_height:.3f} (molmospaces would put it at "
                    f"{virtual_franka_link0_height(scene):.3f})", fg="green")
    # Stretch spawns turned the way the start pose would turn its base, so it does not have to.
    franka_yaw = franka_yaw_for_start(params, link0_height, start_q7, scene.floor_z)
    stretch_scene = load_custom_scene_stretch4(
        scene, pose, include_franka=include_franka, tool_name=params.tool_name, link0_height=link0_height,
        stretch_yaw=-franka_yaw,
    )

    if stretch_scene.removed_bodies:
        click.secho(f"Removed furniture Stretch 4 would spawn inside: {stretch_scene.removed_bodies}", fg="yellow")
    sim = spawn_stretch4(stretch_scene, params)
    if overlay_franka_gripper:
        sim.watch_bodies(with_nearby_free_bodies(stretch_scene, stretch_scene.watched_bodies))
    sim.start(viewer_look_at_body=STRETCH_ROOT_BODY)
    env = Stretch4SimEnv(sim, stretch_scene, params, SceneMirror(stretch_scene))
    env.overlay_franka_gripper = overlay_franka_gripper
    try:
        logger = None
        if rerun:
            rerun_scene.init_rerun("MolmoBot-DROID Stretch 4", RERUN_CAMERAS)
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

        click.secho("Moving Stretch 4 to the Franka's start pose...", fg="yellow")
        go_to_start()
        editor = StretchStartPoseEditor(
            env, go_to=env.move_to_franka_pose, set_pedestal_height=stretch_scene.set_franka_pedestal_height,
            on_move=log_pose,
        )

        def on_step(info: StepInfo):
            nonlocal step_count
            step_count += 1
            if not sim.is_running():
                raise SystemExit("The simulator stopped")
            if logger is not None:
                logger.log(step_count, info)

        policy = load_policy(checkpoint)
        interactive_session(
            env,
            policy,
            build_instruction("pick", scene.object_category),
            params.execute_horizon,
            params.execute_first_n,
            max_steps,
            reset=go_to_start,
            on_step=on_step,
            handle_command=editor.handle_command,
            extra_help=lambda: f"Start pose: {editor.COMMANDS}.",
        )
    finally:
        unload_policy()
        env.close()
        sim.stop()


if __name__ == "__main__":
    main()
