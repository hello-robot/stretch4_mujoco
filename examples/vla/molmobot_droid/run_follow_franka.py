"""
MolmoBot-DROID driving a Franka DROID in any MolmoSpaces house while Stretch 4 follows it blind
(`franka_retarget/follow_franka.py`): the policy sees only the Franka's cameras, Stretch 4 is
invisible to them, and only Stretch 4 touches anything. For seeing how well Stretch 4 does what
the Franka does: whether it picks the object up (molmospaces' pick criterion, judged on
Stretch), how far its tool is from where the Franka's action put it, how often it cannot follow.

Type an instruction at the prompt (empty for the target's pick instruction) and it runs until
Stretch has picked the object up (unless --no-stop-on-success), --max-steps, or Ctrl+C. `reset`
puts both robots back at the start pose (the objects stay where they are); `quit` exits.

Watch it in stretch4_mujoco's viewer (the Franka is in geom group 5: press 5 to see it, static at
home) and in Rerun: the policy's views (the Franka's exo and wrist cameras), Stretch's head
fisheyes and gripper cameras as a policy would be shown them, the scene camera with the Franka
see-through over Stretch, and the house in 3D with the Franka's tool target and Stretch's tool.

Usage:
    python -m examples.vla.molmobot_droid.run_follow_franka --scene-id procthor-10k/val/0 --object-type mug
    python -m examples.vla.molmobot_droid.run_follow_franka --use_parallel_gripper --no-wait-for-arrival
    python -m examples.vla.molmobot_droid.run_follow_franka --custom_franka_start_pose
"""

from __future__ import annotations

import click
import numpy as np

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS, build_instruction, load_policy, unload_policy
from examples.vla.molmobot_droid.franka_retarget.follow_franka import STRETCH_CAMERAS, FollowFrankaEnv
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    CUSTOM_START_QPOS,
    custom_start_link0_height,
    custom_start_option,
    franka_yaw_for_start,
    params_from_kwargs,
    retarget_options,
    spawn_stretch4,
)
from examples.vla.molmobot_droid.molmospaces.benchmark import PickSuccess, allow_ctrl_c
from examples.vla.molmobot_droid.molmospaces.custom_scene import (
    STRETCH_ROOT_BODY,
    SceneMirror,
    load_custom_scene,
    load_custom_scene_stretch4,
    scene_camera_pose,
    virtual_franka_link0_height,
)
from examples.vla.molmobot_droid.rollout import StepInfo, interactive_session
from examples.vla.molmobot_droid.run_franka import resolve_pose, scene_options
from examples.vla.molmobot_droid.run_stretch4_sim import with_nearby_free_bodies

RERUN_CAMERAS = ["exo", "wrist", "fisheye_left", "fisheye_right", "gripper_left", "gripper_right", "scene"]
"""The policy's (the Franka's) views, Stretch's as a policy would be shown them, the scene camera."""


def spawn_follow_sim(stretch_scene, params, judged_bodies: list[str]):
    """
    Stretch 4's simulator for following the Franka (not started): rendering every camera that is
    recorded, publishing the poses of the bodies success is judged on and of the free objects
    near the robot, which the Franka's views are rendered with (`SceneMirror` moves only those).
    """
    sim = spawn_stretch4(stretch_scene, params, cameras=list(STRETCH_CAMERAS.values()))
    sim.watch_bodies(with_nearby_free_bodies(stretch_scene, stretch_scene.watched_bodies + judged_bodies))
    return sim


class FollowRerunLogger:
    """Logs a `FollowFrankaEnv` step: the mirror scene, the cameras, the Franka's tool target vs. Stretch's tool."""

    def __init__(self, env: FollowFrankaEnv):
        self.env = env
        stretch_scene = env.stretch_scene
        moving = [STRETCH_ROOT_BODY, stretch_scene.scene.object_name, stretch_scene.franka.base_name]
        self.scene = rerun_scene.RerunScene(env.mirror.model, env.mirror.data, moving)

    def log(self, step: int, observation, metrics: dict[str, float] | None = None) -> None:
        env = self.env
        rerun_scene.set_step(step)
        self.scene.log(env.mirror.data)
        images = {"exo": observation.exo_rgb, "wrist": observation.wrist_rgb, "scene": env.render_scene()}
        images.update({name: observation.extra_cameras[name] for name in RERUN_CAMERAS if name in observation.extra_cameras})
        rerun_scene.log_cameras(images)
        actual = env.retargeter.stretch_tool_world(env.world_from_footprint(), env.joints())
        rerun_scene.log_tool_poses(env.retargeter.last_target_tool_world, actual)
        if metrics:
            rerun_scene.log_metrics(metrics)


def stats_text(env: FollowFrankaEnv) -> str:
    stats = env.stats()
    tool = (
        f"Stretch's tool {stats['tool_error_mean_m'] * 1000:.0f} mm from the Franka's on average, "
        f"{stats['tool_error_max_m'] * 1000:.0f} mm at most; "
        if "tool_error_mean_m" in stats
        else ""
    )
    return (
        f"Since the start pose: {tool}{stats['ik_clamped']:.0f} actions clamped "
        f"({stats['ik_twisted']:.0f} twisted), {stats['ik_failures']:.0f} Stretch could not follow."
    )


@click.command()
@scene_options
@retarget_options(camera_options=False)
@custom_start_option
@click.option("--stop-on-success/--no-stop-on-success", default=True, show_default=True,
              help="End an instruction once Stretch 4 has picked the object up.")
def main(scene_id, object_type, object_index, in_place_of, robot_pose, max_steps, checkpoint, rerun,
         custom_franka_start_pose, stop_on_success, **kwargs):
    params = params_from_kwargs(kwargs)
    scene = load_custom_scene(scene_id, object_type, object_index, in_place_of=in_place_of)
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
        scene, pose, include_franka=True, tool_name=params.tool_name, link0_height=link0_height,
        stretch_yaw=-franka_yaw,
    )
    if stretch_scene.removed_bodies:
        click.secho(f"Removed furniture Stretch 4 would spawn inside: {stretch_scene.removed_bodies}", fg="yellow")
    if stretch_scene.backed_off:
        click.secho(f"Moved {stretch_scene.backed_off:.2f} m back from the target so Stretch 4 clears what it "
                    f"rests on: --robot-pose {stretch_scene.robot_pose}", fg="yellow")

    sim = spawn_follow_sim(stretch_scene, params, [scene.object_name])
    sim.start(viewer_look_at_body=STRETCH_ROOT_BODY)
    allow_ctrl_c()  # start() takes Ctrl+C over to stop only the simulator
    env = FollowFrankaEnv(sim, stretch_scene, params, SceneMirror(stretch_scene))
    env.start_q7 = start_q7
    try:
        logger = None
        step_count = 0
        if rerun:
            rerun_scene.init_rerun(
                "MolmoBot-DROID Stretch 4 following the Franka", RERUN_CAMERAS, main_cameras=["exo", "wrist", "gripper_left"],
                eye=scene_camera_pose(stretch_scene.robot_pose, scene.object_pos),
            )
            rerun_scene.set_step(0)
            env.observe()  # poses the mirror
            logger = FollowRerunLogger(env)

        def reset() -> None:
            nonlocal step_count
            click.secho("Moving Stretch 4 to the Franka's start pose...", fg="yellow")
            env.reset()
            if logger is not None:
                step_count += 1
                logger.log(step_count, env.observe())

        reset()
        policy = load_policy(checkpoint)
        judge: PickSuccess | None = None
        succeeded = False

        def on_step(info: StepInfo) -> bool:
            nonlocal step_count, judge, succeeded
            if not sim.is_running():
                raise SystemExit("The simulator stopped")
            if info.step == 1:  # a new instruction: lifted from where the object is now
                judge = PickSuccess(scene.object_name, float(env.body_pose(scene.object_name)[0][2]))
                succeeded = False
            success, metrics = judge(env)
            if success and not succeeded:
                click.secho(f"✓ Stretch 4 picked up the {scene.object_category} at step {info.step} "
                            f"(lifted {metrics['lift_m'] * 100:.1f} cm)", fg="green")
            succeeded = succeeded or success
            step_count += 1
            if logger is not None:
                error = env.tool_error_m()
                logger.log(step_count, info.observation, {
                    "gripper_command": info.action[7], "query": info.query, "lift_m": metrics["lift_m"],
                    "ik_failures": env.retargeter.ik_failures, "ik_clamped": env.retargeter.ik_clamped,
                    **({"tool_error_m": error} if error is not None else {}),
                })
            return stop_on_success and success

        interactive_session(
            env,
            policy,
            build_instruction("pick", scene.object_category),
            params.execute_horizon,
            params.execute_first_n,
            max_steps,
            reset=reset,
            on_step=on_step,
            extra_help=lambda: stats_text(env),
        )
    finally:
        unload_policy()
        env.close()
        sim.stop()


if __name__ == "__main__":
    main()
