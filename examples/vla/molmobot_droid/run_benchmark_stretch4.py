"""
Run a MolmoSpaces benchmark with MolmoBot-DROID on Stretch 4 in simulation, retargeted from the
Franka (`franka_retarget/stretch4_retarget.py`). Stretch's footprint goes where the episode puts
the Franka's base, and the virtual Franka is at the episode's height, so the policy sees the
same geometry it would on the Franka. Compare against `run_benchmark_franka.py` with
`compare_benchmarks.py`. With --custom_franka_start_pose, the virtual Franka instead stands at
the height and starts from the pose that puts Stretch's tool at the top of its reach. With
--overlay_franka_gripper, the policy's views show the Franka instead of Stretch: the Robotiq's
fingers for Stretch's gripper in the wrist view, the whole Franka for Stretch's arm in the exo
view (as in `run_stretch4_sim.py`).

Writes to <out>/<run name>/: per-episode videos of every camera (the policy's two views,
Stretch's raw head and gripper cameras, the scene camera), a grid of them with the instruction
in `grid/` (each episode prints a link to its grid), `report.md`, and `results.json`.
Retargeting counts (IK failures, clamped targets) are in the report.

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
    CUSTOM_START_QPOS,
    Stretch4SimEnv,
    apply_overlay_grasp_offset,
    custom_start_link0_height,
    custom_start_option,
    franka_yaw_for_start,
    overlay_gripper_option,
    params_from_kwargs,
    retarget_options,
    spawn_stretch4,
)
from examples.vla.molmobot_droid.molmospaces import benchmark as bench
from examples.vla.molmobot_droid.molmospaces.custom_scene import SceneMirror, load_custom_scene_stretch4
from examples.vla.molmobot_droid.run_benchmark_franka import benchmark_options, parse_episodes, print_result
from examples.vla.molmobot_droid.run_stretch4_sim import (
    RERUN_CAMERAS,
    StretchRerunLogger,
    include_franka_option,
    with_nearby_free_bodies,
)


@click.command()
@benchmark_options
@retarget_options
@include_franka_option
@custom_start_option
@overlay_gripper_option
@click.option("--rerun/--no-rerun", default=True, show_default=True)
@click.option("--viewer", is_flag=True, help="Show stretch4_mujoco's viewer (default: headless).")
def main(benchmark, episodes, max_episodes, out, checkpoint, run_to_horizon, resume, list_only, include_franka,
         custom_franka_start_pose, overlay_franka_gripper, rerun, viewer, **kwargs):
    if list_only:
        click.echo("\n".join(bench.list_benchmarks()))
        return
    params = params_from_kwargs(kwargs)
    include_franka = include_franka or overlay_franka_gripper  # the Robotiq is the ghost's
    if overlay_franka_gripper:
        apply_overlay_grasp_offset(params)  # before the run name, which has the offset
    benchmark_dir, selected = bench.load_episodes(benchmark, parse_episodes(episodes), max_episodes)
    flags = {**params.flags(), "include_franka": include_franka}
    # Only when given, so earlier runs keep their names for --resume.
    if custom_franka_start_pose:
        flags["custom_franka_start_pose"] = True
    if overlay_franka_gripper:
        flags["overlay_franka_gripper"] = True
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
                if custom_franka_start_pose:
                    link0_height = custom_start_link0_height(params, setup.scene.floor_z)
                    start_q7 = CUSTOM_START_QPOS
                else:
                    link0_height = setup.link0_height
                    start_q7 = setup.franka_init_qpos or FRANKA_HOME_QPOS
                # Stretch spawns turned the way the start pose would turn its base, so it does not have to.
                franka_yaw = franka_yaw_for_start(params, link0_height, start_q7, setup.scene.floor_z)
                stretch_scene = load_custom_scene_stretch4(
                    setup.scene,
                    setup.robot_pose,
                    include_franka=include_franka,
                    tool_name=params.tool_name,
                    link0_height=link0_height,
                    stretch_yaw=-franka_yaw,
                )
                if stretch_scene.removed_bodies:
                    click.secho(f"Removed furniture Stretch 4 would spawn inside: {stretch_scene.removed_bodies}", fg="yellow")
                sim = spawn_stretch4(stretch_scene, params)
                watched = stretch_scene.watched_bodies + bench.make_judge(setup).bodies
                sim.watch_bodies(with_nearby_free_bodies(stretch_scene, watched) if overlay_franka_gripper else watched)
                try:
                    sim.start(headless=not viewer, viewer_look_at_body="stretch4")
                    bench.allow_ctrl_c()  # start() takes Ctrl+C over to stop only the simulator
                    env = Stretch4SimEnv(sim, stretch_scene, params, SceneMirror(stretch_scene))
                    env.overlay_franka_gripper = overlay_franka_gripper
                    try:
                        env.move_to_franka_pose(start_q7)
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
                print_result(run.current, result, out_dir)
                run.record(result)
            successes = sum(r.success for r in run.results)
            click.secho(f"{successes}/{len(run.results)} succeeded. Report: {out_dir / 'report.md'}", fg="green")
            click.secho(f"Grid videos: {(out_dir / bench.GRID_DIR).resolve().as_uri()}", fg="green")
    finally:
        unload_policy()


if __name__ == "__main__":
    main()
