"""
Run a MolmoSpaces benchmark with MolmoBot-DROID driving a Franka DROID while Stretch 4 follows it
blind (`franka_retarget/follow_franka.py`), and keep the episodes Stretch 4 succeeds in as Stretch
4 demonstrations for `run_finetune.py`.

The Franka stands where the episode puts it and starts from its pose (or, with
--custom_franka_start_pose, from the one that puts Stretch's tool at the top of its reach), and
Stretch 4's footprint goes under it, as in `run_benchmark_stretch4.py`. The policy sees only the
Franka's cameras; success is molmospaces' criterion judged on Stretch 4, the only robot that
touches anything.

Writes to <out>/<run name>/, as the other benchmark runners do: per-episode videos of every
camera, a grid of the policy's (Franka's) views, Stretch's cropped head and gripper views, the
tool-side fisheye cropped and rectified, and the scene camera with the instruction in `grid/` (each episode prints a link to its grid),
`report.md` and `results.json`. The cameras are the Franka's `exo` and `wrist` and every one of
Stretch's: `fisheye_left`/`fisheye_right` (DROID-framed, `_rectified` as --rectify redraws them
as the DROID exo camera, and `_raw`), `center`, and
`gripper_left`/`gripper_right` (as the wrist view, and `_raw`); and `fisheye_tool` (and its
`_raw` and `_rectified`), the fisheye on the tool's side each step as `run_stretch4_sim.py` picks
its exo view, one continuous video (`run_finetune.py --use_fisheye tool`). For each successful episode it
also writes `<run name>_ep<#>.h5`, Stretch's trajectory pointing at those videos
(`finetune/dataset.py`): pass the run directory to `run_finetune.py --data`.

Usage:
    python -m examples.vla.molmobot_droid.run_benchmark_follow_franka --use_parallel_gripper
    python -m examples.vla.molmobot_droid.run_benchmark_follow_franka --use_parallel_gripper \\
        --custom_franka_start_pose --no-wait-for-arrival --max-episodes 200 --no-rerun
"""

from __future__ import annotations

import traceback
from pathlib import Path

import click

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS, load_policy, unload_policy
from examples.vla.molmobot_droid.franka_retarget.follow_franka import (
    GRID_EXCLUDED,
    FollowFrankaEnv,
    follow_flags,
    save_trajectory,
)
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    CUSTOM_START_QPOS,
    custom_start_link0_height,
    custom_start_option,
    franka_yaw_for_start,
    params_from_kwargs,
    retarget_options,
)
from examples.vla.molmobot_droid.molmospaces import benchmark as bench
from examples.vla.molmobot_droid.molmospaces.custom_scene import (
    STRETCH_ROOT_BODY,
    SceneMirror,
    load_custom_scene_stretch4,
    scene_camera_pose,
)
from examples.vla.molmobot_droid.run_benchmark_franka import (
    benchmark_options,
    failed_result,
    parse_episodes,
    print_result,
)
from examples.vla.molmobot_droid.run_follow_franka import RERUN_CAMERAS, FollowRerunLogger, spawn_follow_sim

ROBOT = "follow_franka"


@click.command()
@benchmark_options
@retarget_options(camera_options=False)
@custom_start_option
@click.option("--rerun/--no-rerun", default=True, show_default=True)
@click.option("--viewer", is_flag=True, help="Show stretch4_mujoco's viewer (default: headless).")
def main(benchmark, episodes, max_episodes, out, checkpoint, run_to_horizon, resume, list_only,
         custom_franka_start_pose, rerun, viewer, **kwargs):
    if list_only:
        click.echo("\n".join(bench.list_benchmarks()))
        return
    params = params_from_kwargs(kwargs)
    benchmark_dir, selected = bench.load_episodes(benchmark, parse_episodes(episodes), max_episodes)
    flags = follow_flags(params, custom_franka_start_pose)
    name = bench.run_name(ROBOT, flags)
    out_dir = Path(out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    run = bench.BenchmarkRun(out_dir, ROBOT, name, flags, benchmark_dir, selected, resume)
    click.secho(f"{len(run.selected)} episodes from {benchmark_dir} -> {out_dir}", fg="green")

    if rerun:
        rerun_scene.init_rerun("MolmoBot-DROID Stretch 4 following the Franka, benchmark", RERUN_CAMERAS)
    step_count = 0
    policy = load_policy(checkpoint)
    try:
        with run:
            for index, episode in run.episodes():
                setup = bench.episode_setup(episode)
                click.secho(f"[{run.current}] {setup.scene.scene_id}: {setup.instruction}", fg="cyan")
                if custom_franka_start_pose:
                    link0_height = custom_start_link0_height(params, setup.scene.floor_z)
                    start_q7 = CUSTOM_START_QPOS
                else:
                    link0_height = setup.link0_height
                    start_q7 = setup.franka_init_qpos or FRANKA_HOME_QPOS
                # Stretch spawns turned the way the start pose would turn its base, so it does not have to.
                franka_yaw = franka_yaw_for_start(params, link0_height, start_q7, setup.scene.floor_z)
                try:
                    stretch_scene = load_custom_scene_stretch4(
                        setup.scene,
                        setup.robot_pose,
                        include_franka=True,  # its cameras and its look are the ghost's
                        tool_name=params.tool_name,
                        link0_height=link0_height,
                        stretch_yaw=-franka_yaw,
                        keep_removed_for_franka=True,
                    )
                except Exception as error:  # e.g. Stretch 4 cannot stand clear of the scene here
                    result = failed_result(index, setup, error)
                    print_result(run.current, result, out_dir)
                    run.record(result)
                    continue
                if stretch_scene.removed_bodies:
                    click.secho(f"Removed furniture Stretch 4 would spawn inside (the Franka's views still show it): "
                                f"{stretch_scene.removed_bodies}", fg="yellow")
                if stretch_scene.backed_off:
                    click.secho(f"Moved {stretch_scene.backed_off:.2f} m back from the target so Stretch 4 clears what it "
                                f"rests on: --robot-pose {stretch_scene.robot_pose}", fg="yellow")
                sim = spawn_follow_sim(stretch_scene, params, bench.make_judge(setup).bodies)
                try:
                    sim.start(headless=not viewer, viewer_look_at_body=STRETCH_ROOT_BODY)
                    bench.allow_ctrl_c()  # start() takes Ctrl+C over to stop only the simulator
                    env = FollowFrankaEnv(sim, stretch_scene, params, SceneMirror(stretch_scene))
                    env.start_q7 = start_q7
                    try:
                        env.reset()
                        logger = None
                        if rerun:
                            rerun_scene.new_recording(
                                f"{run.current} {setup.scene.scene_id}: {setup.instruction}",
                                eye=scene_camera_pose(stretch_scene.robot_pose, setup.scene.object_pos),
                            )
                            env.observe()  # poses the mirror
                            logger = FollowRerunLogger(env)

                        def on_step(info, _env):
                            nonlocal step_count
                            step_count += 1
                            if logger is not None:
                                logger.log(step_count, info.observation, {"gripper_command": info.action[7], "query": info.query})

                        result = bench.run_episode(
                            index, setup, env, policy, params.execute_horizon, params.execute_first_n, out_dir,
                            name, end_on_success=not run_to_horizon, on_step=on_step,
                            extra_metrics=lambda e: e.stats(), grid_excluded=GRID_EXCLUDED,
                        )
                        result.metrics["saved"] = False
                        if result.success and result.error is None:
                            prefix = f"{name}_{bench.episode_name(index)}"
                            try:
                                saved = save_trajectory(env, out_dir, prefix, setup.instruction, setup.task_type, {
                                    "run_name": name,
                                    "flags": flags,
                                    "benchmark": str(benchmark_dir),
                                    "episode": index,
                                    "scene_id": setup.scene.scene_id,
                                    "object_name": setup.scene.object_name,
                                })
                                result.metrics["saved"] = True
                                click.echo(f"    trajectory: {saved}")
                            except Exception as error:  # the episode still succeeded
                                traceback.print_exc()
                                result.error = f"Saving the trajectory: {type(error).__name__}: {error}"
                    finally:
                        env.close()
                except Exception as error:  # e.g. Stretch cannot reach the start pose
                    result = failed_result(index, setup, error)
                finally:
                    sim.stop()
                print_result(run.current, result, out_dir)
                run.record(result)
            saved = sum(bool(r.metrics.get("saved")) for r in run.results)
            click.secho(f"{'; '.join(bench.score_lines(run.results))}; {saved} trajectories saved. "
                        f"Report: {out_dir / 'report.md'}", fg="green")
            click.secho(f"Grid videos: {(out_dir / bench.GRID_DIR).resolve().as_uri()}", fg="green")
            click.secho(f"Fine-tune on them with: python -m examples.vla.molmobot_droid.run_finetune --data {out_dir}"
                        + (" --use_parallel_gripper" if params.use_parallel_gripper else ""), fg="green")
    finally:
        unload_policy()


if __name__ == "__main__":
    main()
