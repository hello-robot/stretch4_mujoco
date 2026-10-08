"""
Run a MolmoSpaces benchmark with MolmoBot-DROID on the Franka DROID it was trained on: the
baseline `run_benchmark_stretch4.py` is compared against (see `compare_benchmarks.py`).

Writes to <out>/<run name>/: per-episode videos of every camera, the scene camera and a grid,
`report.md` and `results.json`.

Usage:
    python -m examples.vla.molmobot_droid.run_benchmark_franka
    python -m examples.vla.molmobot_droid.run_benchmark_franka --exo_camera center \\
        --execute-horizon 8 --execute-horizon-do-only-first-n-steps 2
    python -m examples.vla.molmobot_droid.run_benchmark_franka --list
"""

from __future__ import annotations

from pathlib import Path

import click

from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS, load_policy, unload_policy
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    execution_flags,
    execution_options,
    exo_camera_option,
)
from examples.vla.molmobot_droid.molmospaces import benchmark as bench
from examples.vla.molmobot_droid.molmospaces.custom_scene import load_custom_scene_franka_droid


def benchmark_options(function):
    """Options shared by both benchmark runners."""
    options = [
        click.option("--benchmark", default=bench.DEFAULT_BENCHMARK, show_default=True,
                     help="<suite>/<dataset>/<Benchmark>, or a directory holding benchmark.json."),
        click.option("--episodes", default=None, help="Comma-separated episode indices, e.g. 0,3,7."),
        click.option("--max-episodes", type=int, default=5, show_default=True),
        click.option("--out", default="outputs/molmobot_droid", show_default=True, type=click.Path()),
        click.option("--checkpoint", default=None, help="Checkpoint directory. Default: download from Hugging Face."),
        click.option("--run-to-horizon", is_flag=True, help="Keep going after success, to the episode's time limit."),
        click.option("--resume", default=None, metavar="EPISODE",
                     help="Start from this episode (e.g. ep0003, as printed), keeping the results before it "
                     "from the run directory. Ctrl+C stops a run and prints the command to resume it."),
        click.option("--list", "list_only", is_flag=True, help="List installed benchmarks and exit."),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def parse_episodes(text: str | None) -> list[int] | None:
    return [int(v) for v in text.split(",")] if text else None


@click.command()
@benchmark_options
@exo_camera_option("droid")
@execution_options
def main(benchmark, episodes, max_episodes, out, checkpoint, run_to_horizon, resume, list_only, exo_camera,
         execute_horizon, execute_first_n):
    if list_only:
        click.echo("\n".join(bench.list_benchmarks()))
        return
    if not 1 <= execute_first_n <= execute_horizon:
        raise click.BadParameter("need 1 <= --execute-horizon-do-only-first-n-steps <= --execute-horizon")

    benchmark_dir, selected = bench.load_episodes(benchmark, parse_episodes(episodes), max_episodes)
    flags = {"exo_camera": exo_camera, **execution_flags(execute_horizon, execute_first_n)}
    name = bench.run_name("franka", flags)
    out_dir = Path(out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    run = bench.BenchmarkRun(out_dir, "franka", name, flags, benchmark_dir, selected, resume)
    click.secho(f"{len(run.selected)} episodes from {benchmark_dir} -> {out_dir}", fg="green")

    policy = load_policy(checkpoint)
    try:
        with run:
            for index, episode in run.episodes():
                setup = bench.episode_setup(episode)
                click.secho(f"[{run.current}] {setup.scene.scene_id}: {setup.instruction}", fg="cyan")
                env = load_custom_scene_franka_droid(
                    setup.scene,
                    setup.robot_pose,
                    exo_camera=exo_camera,
                    link0_height=setup.link0_height,
                    standing_on_floor=False,  # the benchmark's floating 0.58 m pedestal
                )
                try:
                    env.reset(setup.franka_init_qpos or FRANKA_HOME_QPOS)
                    result = bench.run_episode(
                        index, setup, env, policy, execute_horizon, execute_first_n, out_dir, name,
                        end_on_success=not run_to_horizon,
                    )
                finally:
                    env.close()
                print_result(run.current, result)
                run.record(result)
            successes = sum(r.success for r in run.results)
            click.secho(f"{successes}/{len(run.results)} succeeded. Report: {out_dir / 'report.md'}", fg="green")
    finally:
        unload_policy()


def print_result(name: str, result) -> None:
    click.secho(
        f"    {name}: {'SUCCESS' if result.success else 'fail'} after {result.steps} steps"
        + (f" ({result.error})" if result.error else ""),
        fg="green" if result.success else "red",
    )


if __name__ == "__main__":
    main()
