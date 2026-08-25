"""
The eight MolmoSpaces benchmark evaluations, as a registry.

MolmoSpaces ships its benchmarks as `benchmark.json` files under
`$MLSPACES_ASSETS_DIR/benchmarks/`, one directory per (suite, scene dataset,
generator config). Between `molmospaces-bench-v1` and `-v2` there are ~90 such
directories, but they cover exactly eight distinct task families -- the seven
`TaskSpec` subclasses in `molmo_spaces/evaluation/benchmark_schema.py`, with
`OpeningTask` split into its "open" and "close" suites:

    pick             MS-Pick / MB-Pick        molmo_spaces.tasks.pick_task.PickTask
    pnp              MS-PnP  / MB-PnP         ...pick_and_place_task.PickAndPlaceTask
    pnp_next_to      MB-PnP-next-to           ...pick_and_place_next_to_task.PickAndPlaceNextToTask
    pnp_color        MB-PnP-color             ...pick_and_place_color_task.PickAndPlaceColorTask
    open             MS-Open                  ...opening_tasks.OpeningTask
    close            MS-Close                 ...opening_tasks.OpeningTask
    door_opening     MB-Door                  ...opening_tasks.DoorOpeningTask
    nav_to_obj       MS-Nav                   ...nav_task.NavToObjTask

This module pins one benchmark directory per family, so "run the eight benchmark
evaluations" is a well-defined thing to do. Where both a MolmoSpaces ("MS-",
easier, procthor-10k/iTHOR) and a MolmoBot ("MB-", harder, procthor-objaverse)
suite exist for a family, the harder one is the default and the easier one is
available as an alternate -- see `Benchmark.alternates`.

The episodes themselves were authored for a Franka Droid or an RBY1;
`franka_remapping/episode_overrides.py` retargets each one onto Stretch 4 at load time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Benchmark:
    """One of the eight benchmark evaluations."""

    key: str
    """Short name used on the command line, e.g. `--benchmark pick`."""

    display_name: str
    """Name used in reports and on the MolmoSpaces leaderboard, e.g. "MB-Pick"."""

    task_cls: str
    """Fully-qualified MolmoSpaces task class every episode in this benchmark uses."""

    relative_dir: str
    """Directory under `<assets>/benchmarks/` holding this benchmark's `benchmark.json`."""

    authoring_robot: str
    """Robot the episodes were generated with, before Stretch retargeting."""

    num_episodes: int
    """Episode count, for sanity-checking an install and for progress reporting."""

    description: str = ""

    supported: bool = True
    """
    Whether this benchmark can be evaluated with Stretch at all. False marks a
    benchmark whose *task class* is tied to another robot upstream, which no
    amount of retargeting on this side can fix; `run_benchmarks.py` skips those
    unless asked for by name.
    """

    alternates: dict[str, str] = field(default_factory=dict)
    """Other released benchmark directories for the same task family, by label."""

    def directory(self, benchmarks_root: Path | None = None, alternate: str | None = None) -> Path:
        """Absolute path to this benchmark's directory.

        Args:
            benchmarks_root: `<assets>/benchmarks`. Defaults to the MolmoSpaces
                resource manager's location.
            alternate: pick one of `alternates` instead of the default directory.
        """
        root = benchmarks_root if benchmarks_root is not None else default_benchmarks_root()
        relative = self.relative_dir if alternate is None else self.alternates[alternate]
        return (root / relative).resolve()


def default_benchmarks_root() -> Path:
    """`<assets>/benchmarks`, wherever MolmoSpaces put it for this installation.

    MolmoSpaces hashes the install path into its asset directory, so this is not
    a fixed location; `MLSPACES_ASSETS_DIR` overrides it.
    """
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

    return Path(ASSETS_DIR) / "benchmarks"


_V1 = "molmospaces-bench-v1"
_V2 = "molmospaces-bench-v2"

BENCHMARKS: dict[str, Benchmark] = {
    b.key: b
    for b in (
        Benchmark(
            key="pick",
            display_name="MB-Pick (Pick-v2)",
            task_cls="molmo_spaces.tasks.pick_task.PickTask",
            relative_dir=f"{_V2}/procthor-objaverse/FrankaPickHardBench"
            "/FrankaPickHardBench_20260206_json_benchmark",
            authoring_robot="franka_droid",
            num_episodes=1000,
            description="Pick a named object up off a surface and hold it clear of its support.",
            alternates={
                # MS-Pick (Pick-v1.1): the easier procthor-10k mini-benchmark the
                # MolmoSpaces evaluation README walks through.
                "ms": f"{_V1}/procthor-10k/FrankaPickDroidMiniBench"
                "/FrankaPickDroidMiniBench_json_benchmark_20251231",
            },
        ),
        Benchmark(
            key="pnp",
            display_name="MB-PnP (PnP-v2)",
            task_cls="molmo_spaces.tasks.pick_and_place_task.PickAndPlaceTask",
            relative_dir=f"{_V2}/procthor-objaverse/FrankaPickandPlaceHardBench"
            "/FrankaPickandPlaceHardBench_20260206_json_benchmark",
            authoring_robot="franka_droid",
            num_episodes=1000,
            description="Pick a named object and place it in or on a named receptacle.",
            alternates={
                "ms": f"{_V1}/procthor-10k/FrankaPickandPlaceDroidMiniBench"
                "/FrankaPickandPlaceDroidMiniBench_20260111_json_benchmark",
                # A single-episode cut, useful as a smoke test.
                "smoke": f"{_V2}/procthor-objaverse/FrankaPickandPlaceHardBench"
                "/FrankaPickandPlaceHardBench_20260206_json_1ep_benchmark",
            },
        ),
        Benchmark(
            key="pnp_next_to",
            display_name="MB-PnP-next-to (PnP-next-to-v2)",
            task_cls="molmo_spaces.tasks.pick_and_place_next_to_task.PickAndPlaceNextToTask",
            relative_dir=f"{_V2}/procthor-objaverse/FrankaPickandPlaceNextToHardBench"
            "/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark",
            authoring_robot="franka_droid",
            num_episodes=1000,
            description="Place the picked object beside a named object rather than on it, "
            "within a surface-to-surface gap tolerance.",
        ),
        Benchmark(
            key="pnp_color",
            display_name="MB-PnP-color (PnP-color-v2)",
            task_cls="molmo_spaces.tasks.pick_and_place_color_task.PickAndPlaceColorTask",
            relative_dir=f"{_V2}/procthor-objaverse/FrankaPickandPlaceColorHardBench"
            "/FrankaPickandPlaceColorHardBench_20260304_json_benchmark",
            authoring_robot="franka_droid",
            num_episodes=1000,
            description="Pick-and-place where the receptacle is disambiguated only by colour, "
            "against recoloured distractor receptacles.",
        ),
        Benchmark(
            key="open",
            display_name="MB-Open (Open-v2)",
            task_cls="molmo_spaces.tasks.opening_tasks.OpeningTask",
            relative_dir=f"{_V2}/procthor-objaverse/FrankaOpenHardBench"
            "/FrankaOpenHardBench_20260206_json_benchmark",
            authoring_robot="franka_droid",
            num_episodes=1000,
            description="Open an articulated drawer or door past a fraction of its joint range.",
            alternates={
                # MS-Open (Open-v1), the iTHOR suite the MolmoSpaces ms-bench doc
                # lists. It does not currently load: `JsonEvalTaskSampler.
                # set_joint_values()` requires a per-joint grasp file for the
                # articulated object, and the released `droid` grasp library has
                # none for the iTHOR drawers and cabinets these episodes use, so
                # every episode dies with "No joints with grasp file found". That
                # is an asset gap rather than anything robot-specific -- it
                # reproduces with MolmoSpaces' own Franka `DummyBenchmarkEvalConfig`
                # -- so the procthor-objaverse release is the default instead.
                "ms": f"{_V1}/ithor/FrankaOpenDataGenConfig"
                "/FrankaOpenDataGenConfig_20260123_json_benchmark",
            },
        ),
        Benchmark(
            key="close",
            display_name="MB-Close (Close-v2)",
            task_cls="molmo_spaces.tasks.opening_tasks.OpeningTask",
            relative_dir=f"{_V2}/procthor-objaverse/FrankaCloseHardBench"
            "/FrankaCloseHardBench_20260206_json_benchmark",
            authoring_robot="franka_droid",
            num_episodes=1000,
            description="Close an articulated drawer or door that starts open.",
            alternates={
                # MS-Close (Close-v1). Same missing iTHOR joint grasps as 'open'.
                "ms": f"{_V1}/ithor/FrankaCloseDataGenConfig"
                "/FrankaCloseDataGenConfig_20260123_json_benchmark",
            },
        ),
        Benchmark(
            key="door_opening",
            display_name="MB-Door (Door-v2)",
            task_cls="molmo_spaces.tasks.opening_tasks.DoorOpeningTask",
            relative_dir=f"{_V2}/procthor-10k/rby1_benchmarks/door_opening_benchmark",
            authoring_robot="rby1m",
            num_episodes=2000,
            description="Pull a full-size room door open past two thirds of its swing, "
            "which needs the base and the arm to move together. RBY1-only upstream: "
            "DoorOpeningTask hard-codes a dual-arm RBY1 sensor suite, so it cannot "
            "currently be evaluated with Stretch -- see README 'Known limitations'.",
            supported=False,
        ),
        Benchmark(
            key="nav_to_obj",
            display_name="MS-Nav (NavToObj-v1)",
            task_cls="molmo_spaces.tasks.nav_task.NavToObjTask",
            relative_dir=f"{_V2}/procthor-10k/NavToObjDataGenConfig"
            "/NavToObjProcthor10kBench_20260112_json_benchmark",
            authoring_robot="rby1",
            num_episodes=2000,
            description="Drive to within 1.5m of a named object elsewhere in the house.",
            alternates={
                "holodeck": f"{_V2}/holodeck-objaverse/NavToObjDataGenConfig"
                "/NavToObjHolodeckBench_20260115_json_benchmark",
            },
        ),
    )
}

ALL_BENCHMARK_KEYS = tuple(BENCHMARKS)

SUPPORTED_BENCHMARK_KEYS = tuple(key for key, b in BENCHMARKS.items() if b.supported)
"""The benchmarks a Stretch run sweeps by default. See `Benchmark.supported`."""

# Task families whose success criterion is reached by driving rather than by
# manipulating. `run_benchmarks.py` and the scripted policy branch on this.
NAVIGATION_BENCHMARK_KEYS = ("nav_to_obj",)


def resolve_benchmark_dir(
    key: str,
    benchmarks_root: Path | None = None,
    alternate: str | None = None,
) -> Path:
    """Look a benchmark up by key and check it is actually installed.

    Raises:
        KeyError: unknown benchmark key.
        FileNotFoundError: the benchmark is not installed, with the command that
            installs it.
    """
    if key not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{key}'. Known benchmarks: {', '.join(BENCHMARKS)}")

    benchmark = BENCHMARKS[key]
    directory = benchmark.directory(benchmarks_root, alternate=alternate)
    if not (directory / "benchmark.json").exists():
        suite = benchmark.relative_dir.split("/", 1)[0]
        raise FileNotFoundError(
            f"No benchmark.json under {directory}.\n"
            f"The '{key}' benchmark lives in the '{suite}' asset package; install it with:\n"
            f'  python -c "from molmo_spaces.molmo_spaces_constants import get_resource_manager; '
            f"get_resource_manager().install_all_for_source('benchmarks', '{suite}')\""
        )
    return directory
