"""
MolmoBot-DROID driving Stretch 4 in simulation, through Franka retargeting
(`franka_retarget/stretch4_retarget.py`), in any MolmoSpaces house. Type instructions at the
prompt and watch it in stretch4_mujoco's passive viewer and in Rerun: the policy's views,
Stretch's raw cameras, the scene camera and the house in 3D, with the ghost Franka the policy
thinks it is driving overlaid when --include_franka is given.

Usage:
    python -m examples.vla.molmobot_droid.run_stretch4_sim --scene-id procthor-10k/val/0 --object-type boiler
    python -m examples.vla.molmobot_droid.run_stretch4_sim --include_franka --exo_camera center \\
        --head-crop droid --gripper_camera right --slow --grasp-offset-mm 0,0,10
"""

from __future__ import annotations

import click
import numpy as np

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import build_instruction, load_policy, unload_policy
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    Stretch4SimEnv,
    params_from_kwargs,
    retarget_options,
    spawn_stretch4,
)
from examples.vla.molmobot_droid.molmospaces.custom_scene import (
    STRETCH_ROOT_BODY,
    SceneMirror,
    load_custom_scene,
    load_custom_scene_stretch4,
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

    def log(self, step: int, info: StepInfo) -> None:
        env = self.env
        rerun_scene.set_step(step)
        self.scene.log(env.mirror.data)
        observation = info.observation
        rerun_scene.log_cameras(
            {"exo": observation.exo_rgb, "wrist": observation.wrist_rgb, **observation.extra_cameras,
             "scene": env.render_scene()}
        )
        actual = env.retargeter.stretch_tool_world(env.world_from_footprint(), env.joints())
        rerun_scene.log_tool_poses(env.retargeter.last_target_tool_world, actual)
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


@click.command()
@scene_options
@retarget_options
@include_franka_option
def main(scene_id, object_type, object_index, robot_pose, max_steps, checkpoint, rerun, include_franka, **kwargs):
    params = params_from_kwargs(kwargs)
    scene = load_custom_scene(scene_id, object_type, object_index)
    click.secho(f"Target: {scene.object_name} at {scene.object_pos.round(3)}", fg="green")
    pose = resolve_pose(scene, robot_pose)
    stretch_scene = load_custom_scene_stretch4(scene, pose, include_franka=include_franka, tool_name=params.tool_name)

    if stretch_scene.removed_bodies:
        click.secho(f"Removed furniture Stretch 4 would spawn inside: {stretch_scene.removed_bodies}", fg="yellow")
    sim = spawn_stretch4(stretch_scene, params)
    sim.start(viewer_look_at_body=STRETCH_ROOT_BODY)
    env = Stretch4SimEnv(sim, stretch_scene, params, SceneMirror(stretch_scene))
    try:
        click.secho("Moving Stretch 4 to the Franka's home pose...", fg="yellow")
        env.move_to_franka_pose()

        logger = None
        if rerun:
            rerun_scene.init_rerun("MolmoBot-DROID Stretch 4", RERUN_CAMERAS)
            rerun_scene.set_step(0)
            env.observe()  # poses the mirror
            logger = StretchRerunLogger(env)
        step_count = 0

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
            reset=env.reset,
            on_step=on_step,
        )
    finally:
        unload_policy()
        env.close()
        sim.stop()


if __name__ == "__main__":
    main()
