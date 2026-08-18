"""
Collect behaviour-cloning demonstrations by running the scripted expert.

Data generation and evaluation are the same pipeline here: MolmoSpaces' rollout
runner records every episode's observations, commanded actions and success flag
to HDF5 + MP4 regardless of which policy produced them. So collecting
demonstrations is running `run_benchmarks`-style evaluation with the scripted
expert and keeping the episodes it got right.

Usage:
    # one benchmark
    python -m examples.machine_learning.molmospaces.training.collect \\
        --benchmark pick --episodes 200 --output-dir data/stretch_pick

    # all the manipulation families, pooled into one dataset
    python -m examples.machine_learning.molmospaces.training.collect \\
        --benchmark pick --benchmark pnp --benchmark open --benchmark close \\
        --episodes 200 --output-dir data/stretch_manipulation

Episodes come from the front of the benchmark's own episode list, so by default
the training data and the evaluation share scenes: a policy trained this way is
being measured on its ability to imitate, not to generalise to unseen houses. For
a held-out split, collect on one benchmark's `--alternate` release, or point
`run_benchmarks.py` at a slice `collect.py` did not see.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from examples.machine_learning.molmospaces.benchmarks import (
    ALL_BENCHMARK_KEYS,
    BENCHMARKS,
    NAVIGATION_BENCHMARK_KEYS,
    resolve_benchmark_dir,
)
from examples.machine_learning.molmospaces.configs import (
    DEFAULT_BASELINE_CONFIGS,
    qualified_config_name,
)
from examples.machine_learning.molmospaces.stretch.config import HEAD_CAMERA, WRIST_CAMERA
from examples.machine_learning.molmospaces.training.dataset import build_dataset

log = logging.getLogger(__name__)


def collect_demonstrations(
    benchmark_keys: list[str],
    output_dir: Path,
    episodes: int,
    num_workers: int = 1,
    successful_only: bool = True,
    rollout_dir: Path | None = None,
) -> Path:
    """Roll out the expert on each benchmark and build a dataset from the result.

    Args:
        benchmark_keys: which of the eight benchmarks to demonstrate.
        output_dir: where the built dataset shards go.
        episodes: episodes to attempt per benchmark.
        num_workers: parallel rollout worker processes.
        successful_only: keep only episodes the task judged successful.
        rollout_dir: where to put the raw rollouts. Defaults to
            `<output_dir>/rollouts`. Kept rather than deleted: the videos are the
            only record of *how* the expert failed on the episodes it dropped.

    Returns:
        The dataset directory.
    """
    from molmo_spaces.evaluation import run_evaluation

    output_dir = Path(output_dir)
    rollout_dir = Path(rollout_dir) if rollout_dir is not None else output_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = []
    for key in benchmark_keys:
        config_name = DEFAULT_BASELINE_CONFIGS[key]
        log.info(
            f"[collect] {BENCHMARKS[key].display_name} with {config_name}, {episodes} episodes"
        )
        results = run_evaluation(
            eval_config_cls=qualified_config_name(config_name),
            benchmark_dir=resolve_benchmark_dir(key),
            output_dir=rollout_dir / key,
            max_episodes=episodes,
            num_workers=num_workers,
            use_wandb=False,
        )
        log.info(
            f"[collect] {BENCHMARKS[key].display_name}: "
            f"{results.success_count}/{results.total_count} successful demonstrations"
        )
        run_dirs.append(results.output_dir)

    build_dataset(
        run_dirs=run_dirs,
        output_dir=output_dir,
        camera_names=[HEAD_CAMERA, WRIST_CAMERA],
        successful_only=successful_only,
    )
    return output_dir


@click.command()
@click.option(
    "--benchmark",
    "benchmarks",
    multiple=True,
    type=click.Choice(ALL_BENCHMARK_KEYS),
    default=("pick",),
    help="Benchmark to demonstrate. Repeatable; several are pooled into one dataset.",
)
@click.option("--episodes", type=int, default=100, help="Episodes to attempt per benchmark.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Where to write the dataset shards.",
)
@click.option("--num-workers", type=int, default=1, help="Parallel rollout workers.")
@click.option(
    "--keep-failures/--successful-only",
    default=False,
    help="Include episodes the task judged unsuccessful. Off by default: a partial "
    "expert's failures are counter-examples, not demonstrations.",
)
def main(
    benchmarks: tuple[str, ...],
    episodes: int,
    output_dir: Path,
    num_workers: int,
    keep_failures: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    navigation = [key for key in benchmarks if key in NAVIGATION_BENCHMARK_KEYS]
    if navigation:
        click.secho(
            f"Note: {', '.join(navigation)} will be demonstrated by the A* planner rather "
            "than the scripted manipulator, per DEFAULT_BASELINE_CONFIGS.",
            fg="yellow",
        )

    dataset_dir = collect_demonstrations(
        benchmark_keys=list(benchmarks),
        output_dir=output_dir,
        episodes=episodes,
        num_workers=num_workers,
        successful_only=not keep_failures,
    )
    click.secho(f"Dataset written to {dataset_dir}", fg="green")


if __name__ == "__main__":
    main()
