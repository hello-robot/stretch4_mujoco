"""
MolmoBot-DROID driving the Franka DROID in any MolmoSpaces house. Type instructions at the
prompt and watch it in MuJoCo's passive viewer and in Rerun (the policy's cameras, the scene
camera and the whole house in 3D).

Usage:
    python -m examples.vla.molmobot_droid.run_franka --scene-id procthor-10k/val/0 --object-type mug
    python -m examples.vla.molmobot_droid.run_franka --robot-pose 5.85,7.57,-1.571 --exo_camera center \\
        --execute-horizon 8 --execute-horizon-do-only-first-n-steps 2
"""

from __future__ import annotations

import click

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import build_instruction, load_policy, unload_policy
from examples.vla.molmobot_droid.droid import RobotPose
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import execution_options, exo_camera_option
from examples.vla.molmobot_droid.molmospaces.custom_scene import (
    load_custom_scene,
    load_custom_scene_franka_droid,
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


@click.command()
@scene_options
@exo_camera_option("droid")
@execution_options
def main(scene_id, object_type, object_index, robot_pose, max_steps, checkpoint, rerun, exo_camera,
         execute_horizon, execute_first_n):
    if not 1 <= execute_first_n <= execute_horizon:
        raise click.BadParameter("need 1 <= --execute-horizon-do-only-first-n-steps <= --execute-horizon")
    scene = load_custom_scene(scene_id, object_type, object_index)
    click.secho(f"Target: {scene.object_name} at {scene.object_pos.round(3)}", fg="green")
    pose = resolve_pose(scene, robot_pose)
    env = load_custom_scene_franka_droid(scene, pose, exo_camera=exo_camera)
    env.launch_viewer()

    logger = None
    if rerun:
        rerun_scene.init_rerun("MolmoBot-DROID Franka", ["exo", "wrist", "scene"])
        rerun_scene.set_step(0)
        logger = rerun_scene.RerunScene(env.model, env.data, [env.spawn.base_name, scene.object_name])
    step_count = 0

    def on_step(info: StepInfo):
        nonlocal step_count
        step_count += 1
        if logger is not None:
            rerun_scene.set_step(step_count)
            logger.log(env.data)
            rerun_scene.log_cameras(
                {"exo": info.observation.exo_rgb, "wrist": info.observation.wrist_rgb, "scene": env.render_scene()}
            )
            rerun_scene.log_tool_poses(None, env.tcp_world())
            rerun_scene.log_metrics({"gripper_command": info.action[7], "query": info.query})
        if env.viewer is not None and not env.viewer.is_running():
            raise SystemExit("Viewer closed")

    policy = load_policy(checkpoint)
    try:
        interactive_session(
            env,
            policy,
            build_instruction("pick", scene.object_category),
            execute_horizon,
            execute_first_n,
            max_steps,
            reset=env.reset,
            on_step=on_step,
        )
    finally:
        unload_policy()
        env.close()


if __name__ == "__main__":
    main()
