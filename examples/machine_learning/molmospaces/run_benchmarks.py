"""
Run Stretch 4 on the eight MolmoSpaces benchmark evaluations.

    # baseline policies, a few episodes each, all eight benchmarks
    python -m examples.machine_learning.molmospaces.run_benchmarks --episodes 5

    # one benchmark, more episodes, in parallel
    python -m examples.machine_learning.molmospaces.run_benchmarks \
        --benchmark pick --episodes 200 --num-workers 8

    # a trained behaviour-cloning checkpoint
    python -m examples.machine_learning.molmospaces.run_benchmarks \
        --policy bc --checkpoint checkpoints/stretch_pick.pt --benchmark pick

    # just list what is registered and whether it is installed
    python -m examples.machine_learning.molmospaces.run_benchmarks --list

Results are written as `results.csv` alongside the per-benchmark evaluation
output, in the same shape MolmoSpaces' own `scripts/benchmarks/eval_to_csv.py`
produces, so runs from here and from `eval_main.py` can be pooled.
"""

from __future__ import annotations

import csv
import datetime
import logging
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import click

from examples.machine_learning.molmospaces.benchmarks import (
    ALL_BENCHMARK_KEYS,
    BENCHMARKS,
    SUPPORTED_BENCHMARK_KEYS,
    resolve_benchmark_dir,
)
from examples.machine_learning.molmospaces.configs import (
    DEFAULT_BASELINE_CONFIGS,
    qualified_config_name,
)

log = logging.getLogger(__name__)

# Policy selector -> how to pick an eval config for a given benchmark.
POLICY_CHOICES = ("baseline", "scripted", "scripted_top_down", "bc", "dummy")


@dataclass
class BenchmarkResult:
    """One benchmark's outcome, one row of `results.csv`."""

    benchmark: str
    display_name: str
    policy: str
    eval_config: str
    episodes: int
    successes: int
    success_rate: float
    output_dir: str
    error: str = ""


def eval_config_for(policy: str, benchmark_key: str) -> str:
    """The eval config class name to run `policy` on `benchmark_key`.

    'baseline' is the only selector that varies by benchmark: navigation needs a
    path planner and everything else needs the scripted manipulator.
    """
    if policy == "baseline":
        return DEFAULT_BASELINE_CONFIGS[benchmark_key]
    return {
        "scripted": "StretchScriptedEvalConfig",
        "scripted_top_down": "StretchScriptedTopDownEvalConfig",
        "bc": "StretchBCEvalConfig",
        "dummy": "StretchDummyEvalConfig",
    }[policy]


def run_benchmark(
    benchmark_key: str,
    policy: str,
    episodes: int | None,
    output_root: Path,
    num_workers: int = 1,
    checkpoint: str | None = None,
    task_horizon_steps: int | None = None,
    alternate: str | None = None,
) -> BenchmarkResult:
    """Evaluate one benchmark and summarise it.

    Failures are captured rather than raised: a sweep over eight benchmarks
    should report which one broke and keep going, not lose the seven that worked.
    """
    from molmo_spaces.evaluation import run_evaluation

    benchmark = BENCHMARKS[benchmark_key]
    config_name = eval_config_for(policy, benchmark_key)
    result = BenchmarkResult(
        benchmark=benchmark_key,
        display_name=benchmark.display_name,
        policy=policy,
        eval_config=config_name,
        episodes=0,
        successes=0,
        success_rate=0.0,
        output_dir="",
    )

    try:
        benchmark_dir = resolve_benchmark_dir(benchmark_key, alternate=alternate)
        log.info(
            f"[run] {benchmark.display_name} | {config_name} | "
            f"{episodes if episodes is not None else 'all'} episodes | {benchmark_dir}"
        )
        evaluation = run_evaluation(
            eval_config_cls=qualified_config_name(config_name),
            benchmark_dir=benchmark_dir,
            checkpoint_path=checkpoint,
            output_dir=output_root / benchmark_key,
            max_episodes=episodes,
            num_workers=num_workers,
            task_horizon_steps=task_horizon_steps,
            use_wandb=False,
        )
        result.episodes = evaluation.total_count
        result.successes = evaluation.success_count
        result.success_rate = evaluation.success_rate
        result.output_dir = str(evaluation.output_dir)
    except Exception as error:  # noqa: BLE001 - one broken benchmark must not sink the sweep
        result.error = f"{type(error).__name__}: {error}"
        log.error(f"[run] {benchmark.display_name} failed: {result.error}")
        log.debug(traceback.format_exc())

    return result


def write_results_csv(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def format_results_table(results: list[BenchmarkResult]) -> str:
    header = f"{'benchmark':14s} {'display name':34s} {'episodes':>8s} {'success':>8s} {'rate':>7s}"
    lines = [header, "-" * len(header)]
    for result in results:
        if result.error:
            lines.append(f"{result.benchmark:14s} {result.display_name:34s} {'ERROR':>25s}")
            lines.append(f"{'':14s} {result.error}")
            continue
        lines.append(
            f"{result.benchmark:14s} {result.display_name:34s} "
            f"{result.episodes:8d} {result.successes:8d} {result.success_rate:6.1%}"
        )

    scored = [result for result in results if not result.error and result.episodes]
    if len(scored) > 1:
        total_episodes = sum(result.episodes for result in scored)
        total_successes = sum(result.successes for result in scored)
        lines.append("-" * len(header))
        lines.append(
            f"{'TOTAL':14s} {'':34s} {total_episodes:8d} {total_successes:8d} "
            f"{total_successes / total_episodes:6.1%}"
        )
    return "\n".join(lines)


@click.command()
@click.option(
    "--benchmark",
    "benchmark_keys",
    multiple=True,
    type=click.Choice(ALL_BENCHMARK_KEYS),
    help="Benchmark to run. Repeatable. Defaults to every benchmark Stretch can "
    "currently be evaluated on; name one explicitly to run it anyway.",
)
@click.option(
    "--policy",
    type=click.Choice(POLICY_CHOICES),
    default="baseline",
    help="'baseline' picks the scripted expert for manipulation and the A* planner "
    "for navigation; the others force one policy everywhere.",
)
@click.option(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint for --policy bc. Overrides the path on the policy config.",
)
@click.option(
    "--episodes",
    type=int,
    default=None,
    help="Episodes per benchmark. Defaults to the whole benchmark, which is 1000-2000 "
    "episodes and hours of wall clock.",
)
@click.option("--num-workers", type=int, default=1, help="Parallel rollout worker processes.")
@click.option(
    "--task-horizon-steps",
    type=int,
    default=None,
    help="Override the per-episode step budget. By default each benchmark's own "
    "task_horizon_sec is converted using the eval config's policy_dt_ms.",
)
@click.option(
    "--alternate",
    type=str,
    default=None,
    help="Use a benchmark's alternate release (e.g. 'ms' for the easier MolmoSpaces "
    "suite on pick/pnp). Only valid with a single --benchmark.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write results. Defaults to eval_output/stretch4/<timestamp>.",
)
@click.option("--list", "list_only", is_flag=True, help="List the benchmarks and exit.")
def main(
    benchmark_keys: tuple[str, ...],
    policy: str,
    checkpoint: str | None,
    episodes: int | None,
    num_workers: int,
    task_horizon_steps: int | None,
    alternate: str | None,
    output_dir: Path | None,
    list_only: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # An unsupported benchmark is skipped when sweeping but honoured when named,
    # so the error it produces stays reachable rather than being hidden.
    keys = list(benchmark_keys) or list(SUPPORTED_BENCHMARK_KEYS)

    if list_only:
        _print_benchmark_listing(list(benchmark_keys) or list(ALL_BENCHMARK_KEYS))
        return

    skipped = [key for key in ALL_BENCHMARK_KEYS if key not in keys]
    if not benchmark_keys and skipped:
        click.secho(
            f"Skipping {', '.join(skipped)}: not evaluable with Stretch "
            "(see --list). Pass --benchmark <key> to run one anyway.",
            fg="yellow",
        )

    if alternate is not None and len(keys) != 1:
        raise click.UsageError("--alternate applies to a single --benchmark.")
    if policy == "bc" and checkpoint is None:
        raise click.UsageError(
            "--policy bc needs --checkpoint. Train one with "
            "`python -m examples.machine_learning.molmospaces.training.train_bc`."
        )

    # MolmoSpaces renders off-screen through EGL; without this a headless run
    # fails inside the camera manager rather than at startup.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(output_dir) if output_dir else Path("eval_output") / "stretch4" / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    results = [
        run_benchmark(
            benchmark_key=key,
            policy=policy,
            episodes=episodes,
            output_root=output_root,
            num_workers=num_workers,
            checkpoint=checkpoint,
            task_horizon_steps=task_horizon_steps,
            alternate=alternate,
        )
        for key in keys
    ]

    results_path = output_root / "results.csv"
    write_results_csv(results, results_path)
    click.echo("\n" + format_results_table(results) + "\n")
    click.secho(f"Wrote {results_path}", fg="green")


def _print_benchmark_listing(keys: list[str]) -> None:
    for key in keys:
        benchmark = BENCHMARKS[key]
        try:
            directory = resolve_benchmark_dir(key)
            status = click.style("installed", fg="green")
            location = str(directory)
        except FileNotFoundError:
            status = click.style("NOT INSTALLED", fg="red")
            location = benchmark.relative_dir
        if not benchmark.supported:
            status += click.style("  (not evaluable with Stretch)", fg="yellow")
        click.echo(f"{key:14s} {benchmark.display_name:34s} {status}")
        click.echo(f"{'':14s} {benchmark.description}")
        click.echo(
            f"{'':14s} {benchmark.num_episodes} episodes, authored with "
            f"{benchmark.authoring_robot}, task {benchmark.task_cls.rsplit('.', 1)[-1]}"
        )
        click.echo(f"{'':14s} {location}")
        if benchmark.alternates:
            click.echo(f"{'':14s} alternates: {', '.join(benchmark.alternates)}")
        click.echo()


if __name__ == "__main__":
    main()
