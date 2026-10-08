"""
Run a MolmoSpaces benchmark with MolmoBot-DROID on Stretch 4 in simulation, retargeted from the
Franka (`franka_retarget/stretch4_retarget.py`). Stretch's footprint goes where the episode puts
the Franka's base, and the virtual Franka is at the episode's height, so the policy sees the
same geometry it would on the Franka. Compare against `run_benchmark_franka.py` with
`compare_benchmarks.py`.

Writes to <out>/<run name>/: per-episode videos of every camera (the policy's two views,
Stretch's raw head and gripper cameras, the scene camera) and a grid, `report.md`, and
`results.json`. Retargeting counts (IK failures, clamped targets) are in the report.

Usage:
    python -m examples.vla.molmobot_droid.run_benchmark_stretch4
    python -m examples.vla.molmobot_droid.run_benchmark_stretch4 --include_franka --exo_camera center \\
        --head-crop droid --slow --grasp-offset-mm 0,0,10 --no-rerun
"""

from __future__ import annotations

from pathlib import Path

import click

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS, load_policy, unload_policy
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    Stretch4SimEnv,
    params_from_kwargs,
    retarget_options,
    spawn_stretch4,
)
from examples.vla.molmobot_droid.molmospaces import benchmark as bench
from examples.vla.molmobot_droid.molmospaces.custom_scene import SceneMirror, load_custom_scene_stretch4
from examples.vla.molmobot_droid.run_benchmark_franka import benchmark_options, parse_episodes, print_result
from examples.vla.molmobot_droid.run_stretch4_sim import RERUN_CAMERAS, StretchRerunLogger, include_franka_option


@click.command()
@benchmark_options
@retarget_options
@include_franka_option
@click.option("--rerun/--no-rerun", default=True, show_default=True)
@click.option("--viewer", is_flag=True, help="Show stretch4_mujoco's viewer (default: headless).")
def main(benchmark, episodes, max_episodes, out, checkpoint, run_to_horizon, resume, list_only, include_franka,
         rerun, viewer, **kwargs):
    if list_only:
        click.echo("\n".join(bench.list_benchmarks()))
        return
    params = params_from_kwargs(kwargs)
    benchmark_dir, selected = bench.load_episodes(benchmark, parse_episodes(episodes), max_episodes)
    flags = {**params.flags(), "include_franka": include_franka}
    name = bench.run_name("stretch4", flags)
    out_dir = Path(out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    run = bench.BenchmarkRun(out_dir, "stretch4", name, flags, benchmark_dir, selected, resume)
    click.secho(f"{len(run.selected)} episodes from {benchmark_dir} -> {out_dir}", fg="green")

    if rerun:
        rerun_scene.init_rerun("MolmoBot-DROID Stretch 4 benchmark", RERUN_CAMERAS)
    step_count = 0
    policy = load_policy(checkpoint)
    try:
        with run:
            for index, episode in run.episodes():
                setup = bench.episode_setup(episode)
                click.secho(f"[{run.current}] {setup.scene.scene_id}: {setup.instruction}", fg="cyan")
                stretch_scene = load_custom_scene_stretch4(
                    setup.scene,
                    setup.robot_pose,
                    include_franka=include_franka,
                    tool_name=params.tool_name,
                    link0_height=setup.link0_height,
                )
                if stretch_scene.removed_bodies:
                    click.secho(f"Removed furniture Stretch 4 would spawn inside: {stretch_scene.removed_bodies}", fg="yellow")
                sim = spawn_stretch4(stretch_scene, params)
                sim.watch_bodies(stretch_scene.watched_bodies + bench.make_judge(setup).bodies)
                try:
                    sim.start(headless=not viewer, viewer_look_at_body="stretch4")
                    bench.allow_ctrl_c()  # start() takes Ctrl+C over to stop only the simulator
                    env = Stretch4SimEnv(sim, stretch_scene, params, SceneMirror(stretch_scene))
                    try:
                        env.move_to_franka_pose(setup.franka_init_qpos or FRANKA_HOME_QPOS)
                        logger = None
                        if rerun:
                            rerun_scene.clear_scene()
                            env.observe()
                            logger = StretchRerunLogger(env)

                        def on_step(info, _env):
                            nonlocal step_count
                            step_count += 1
                            if logger is not None:
                                logger.log(step_count, info)

                        result = bench.run_episode(
                            index, setup, env, policy, params.execute_horizon, params.execute_first_n, out_dir,
                            name, end_on_success=not run_to_horizon, on_step=on_step,
                            extra_metrics=lambda e: e.stats(),
                        )
                    finally:
                        env.close()
                except Exception as error:  # e.g. Stretch cannot reach the start pose
                    result = bench.EpisodeResult(
                        index, setup.scene.scene_id, setup.scene.object_name, setup.instruction,
                        error=f"{type(error).__name__}: {error}",
                    )
                finally:
                    sim.stop()
                print_result(run.current, result)
                run.record(result)
            successes = sum(r.success for r in run.results)
            click.secho(f"{successes}/{len(run.results)} succeeded. Report: {out_dir / 'report.md'}", fg="green")
    finally:
        unload_policy()


if __name__ == "__main__":
    main()
