"""
Search for the camera and gripper settings that let MolmoBot-DROID work on Stretch 4.

The question this answers is the one `demo_droid_on_stretch.py` raises and cannot
settle: the released DROID checkpoint drives Stretch through a retargeting layer
and a camera it was never trained on, and when a rollout fails there is no way to
tell which of the two is at fault. So this runs the same four grasps through
seven setups that walk from the Franka and camera the policy knows to the Stretch
and fisheye it does not, one change at a time, and then searches the settings
that are actually free.

    # the seven setups at their defaults -- 28 rollouts, and the place to start
    python -m examples.machine_learning.molmospaces.retargetting.params_search

    # one setup, a grid over camera pitch and field of view
    python -m examples.machine_learning.molmospaces.retargetting.params_search \\
        --setup stretch_stretchcam \\
        --search grid --dim pitch_deg=15:50:4 --dim fovy=50:100:3

    # the gripper parameters on the fisheye setup, by CMA-ES
    python -m examples.machine_learning.molmospaces.retargetting.params_search \\
        --setup stretch_fisheye --search cmaes \\
        --dim grasp_offset_m=-0.05:0.15 --dim wrist_tilt_deg=-20:60 \\
        --population 6 --generations 5

Every rollout is written to its own MP4 and every trial to a row of
`trials.csv`; `report.md` ranks the trials and lists what happened to each
object. See `scoring.py` for what `score` means, `setups.py` for the seven
setups and `mini_benchmark.py` for the four grasps.

A note on cost. One trial is four rollouts of ~300 steps with a VLA in the loop,
which is minutes, not seconds -- so the defaults here are deliberately small.
`--search none` over all seven setups is the baseline table; a grid is for one
setup and one or two dimensions at a time; CMA-ES is for the continuous
parameters once you know which setup is worth tuning.

What this cannot tell you
-------------------------
One house, one robot pose, four objects, one episode each. A setting that wins
here has beaten the others on this kitchen, not in general -- the point is to
rank settings against each other cheaply, and then to confirm the winner with
`run_benchmarks.py --policy molmobot_droid` over a real benchmark.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import click

# MuJoCo binds the backend named by MUJOCO_GL when it is first imported, which
# the imports below trigger -- so this has to come before them. Everything here
# renders off-screen through EGL, which needs no display.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

from examples.machine_learning.molmospaces.retargetting import mini_benchmark  # noqa: E402
from examples.machine_learning.molmospaces.retargetting.cameras import (  # noqa: E402
    ExoCameraParams,
    RetargetParams,
)
from examples.machine_learning.molmospaces.retargetting.scoring import (  # noqa: E402
    EpisodeScore,
    GraspProbe,
    TrialResult,
    format_trial_table,
    install_probe,
    write_episode_csv,
    write_report,
    write_trial_csv,
)
from examples.machine_learning.molmospaces.retargetting.setups import (  # noqa: E402
    SETUP_KEYS,
    SETUPS,
    params_to_json,
    publish_params,
    qualified_config_name,
)

log = logging.getLogger(__name__)


# =============================================================================
# The search space
# =============================================================================


@dataclass(frozen=True)
class Dimension:
    """One searchable number: how to read it off a `RetargetParams` and put it back.

    A named handle rather than a path into the dataclass, because the
    interesting parameters are not all fields -- the camera's pitch is one
    number of a euler triple, and its height one of a position -- and because
    the names are what a command line says.
    """

    name: str
    bounds: tuple[float, float]
    read: Callable[[RetargetParams], float]
    write: Callable[[RetargetParams, float], RetargetParams]
    description: str

    sweep: tuple[float, ...]
    """
    The values `--search sweep` tries for this dimension.

    Chosen to span what is physically sensible rather than to bracket the
    current default, so the sweep can find that the shipped value was wrong --
    which, for `grasp_offset_m`, it was. Wider than a search would pick on its
    own, because the point of the default run is coverage.
    """

    robots: tuple[str, ...] = ("franka", "stretch")
    """
    Which setups this dimension does anything to.

    The gripper parameters only exist on the Stretch path -- there is nothing to
    retarget on the Franka the policy was trained on -- so sweeping them on a
    Franka setup would run identical trials and report them as if they differed.
    `--dim` warns rather than silently doing that, and `--search sweep` skips
    them.
    """


def _with_exo(params: RetargetParams, **changes: Any) -> RetargetParams:
    return dataclasses.replace(params, exo=dataclasses.replace(params.exo, **changes))


DIMENSIONS: dict[str, Dimension] = {
    dimension.name: dimension
    for dimension in (
        Dimension(
            name="pitch_deg",
            bounds=(5.0, 60.0),
            read=lambda p: p.exo.pitch_deg,
            write=lambda p, v: _with_exo(p, pitch_deg=float(v)),
            description="Camera tilt, measured up from straight down: 0 looks at "
            "the floor and 90 at the horizon, so SMALLER points further DOWN. "
            "The default 43 is 47 degrees below horizontal.",
            sweep=(23.0, 33.0, 43.0, 53.0),
        ),
        Dimension(
            name="fovy",
            bounds=(30.0, 140.0),
            read=lambda p: p.exo.fovy,
            write=lambda p, v: _with_exo(p, fovy=float(v)),
            description="Vertical field of view. 71 is DROID's, 123 is Stretch's fisheye.",
            sweep=(55.0, 71.0, 95.0, 123.0),
        ),
        Dimension(
            name="grasp_offset_m",
            bounds=(-0.10, 0.20),
            read=lambda p: p.grasp_offset_m,
            write=lambda p, v: dataclasses.replace(p, grasp_offset_m=float(v)),
            description="How far to push the commanded grasp centre along Stretch's "
            "approach axis, to account for its much longer gripper. Stretch setups only.",
            sweep=(0.0, 0.06, 0.09, 0.12),
            robots=("stretch",),
        ),
        Dimension(
            name="wrist_tilt_deg",
            bounds=(-60.0, 60.0),
            read=lambda p: p.wrist_tilt_deg,
            write=lambda p, v: dataclasses.replace(p, wrist_tilt_deg=float(v)),
            description="Extra pitch between the Franka's tool frame and Stretch's. "
            "Stretch setups only.",
            sweep=(-45.0, 0.0, 45.0),
            robots=("stretch",),
        ),
        Dimension(
            name="z_offset_fraction",
            bounds=(0.0, 1.0),
            read=lambda p: p.z_offset_fraction,
            write=lambda p, v: dataclasses.replace(p, z_offset_fraction=float(v)),
            description="How much of the measured lift shortfall to add to every "
            "target. Stretch setups only.",
            sweep=(0.0, 0.5),
            robots=("stretch",),
        ),
    )
}


SWEEP_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("camera", ("pitch_deg", "fovy")),
    ("gripper", ("grasp_offset_m", "z_offset_fraction", "wrist_tilt_deg")),
)
"""
What `--search sweep` does, in order: a full grid per stage, carrying the winner.

Staged rather than one grid over all five dimensions, because the full cross
product is 384 points per Stretch setup -- days of rollouts, most of them spent
re-measuring a camera that a previous point already showed was bad. Staging it
costs 40 points per Stretch setup instead, and each stage answers one question:
where should the camera look, and then, from that view, how should the gripper
be corrected.

The stages are grouped so that parameters which *interact* stay in the same
grid. That grouping is not cosmetic. `grasp_offset_m` and `z_offset_fraction`
turned out to be inseparable -- at `grasp_offset_m = 0` the height offset made
no measurable difference, because the depth error was already losing every
grasp, and only once the depth was right did the height become the limiting
error. Swept one at a time, neither would have looked like the answer.

What staging still cannot see is an interaction *across* stages -- a camera that
is only good with a particular gripper correction. That is the price of not
running the full grid; `--search grid` over a hand-picked pair is how you check
one if you suspect it.
"""


@dataclass(frozen=True)
class Axis:
    """One dimension as a command line asked for it."""

    dimension: Dimension
    low: float
    high: float
    steps: int | None

    values: tuple[float, ...] | None = None
    """
    Explicit values, when the caller has them.

    `--dim name=lo:hi:steps` leaves this unset and gets an even spacing. The
    sweep sets it, because its values are chosen rather than spaced -- a
    `grasp_offset_m` sweep of (0, 0.06, 0.09, 0.12) is four points that mean
    something, and `linspace` over the same range would quietly replace them
    with (0, 0.04, 0.08, 0.12).
    """

    def grid_values(self) -> list[float]:
        if self.values is not None:
            return [float(v) for v in self.values]
        return [float(v) for v in np.linspace(self.low, self.high, self.steps or 3)]


def parse_axis(text: str) -> Axis:
    """`name=lo:hi[:steps]` -> an `Axis`."""
    name, _, spec = text.partition("=")
    if name not in DIMENSIONS:
        raise click.UsageError(
            f"Unknown dimension {name!r}. Available: {', '.join(sorted(DIMENSIONS))}."
        )
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise click.UsageError(f"--dim {text!r} should be name=lo:hi or name=lo:hi:steps.")
    low, high = float(parts[0]), float(parts[1])
    if high < low:
        raise click.UsageError(f"--dim {text!r} has its bounds the wrong way round.")
    return Axis(DIMENSIONS[name], low, high, int(parts[2]) if len(parts) == 3 else None)


def apply_point(base: RetargetParams, axes: list[Axis], values: np.ndarray) -> RetargetParams:
    """`base` with each axis set to its value. The only way a point becomes parameters."""
    params = base
    for axis, value in zip(axes, values):
        params = axis.dimension.write(params, float(value))
    return params


# =============================================================================
# Running one trial
# =============================================================================


@dataclass
class TrialRunner:
    """Everything a trial needs that does not change between trials."""

    benchmark_dir: Path
    output_root: Path
    probe: GraspProbe
    episode_steps: int | None
    checkpoint: str | None

    def run(self, setup_key: str, params: RetargetParams, label: str) -> TrialResult:
        """Evaluate one setup at one point, and score every episode of it."""
        from molmo_spaces.evaluation import run_evaluation

        setup = SETUPS[setup_key]
        result = TrialResult(
            setup=setup_key,
            params_description=params.describe(),
            params_json=params_to_json(setup_key, params),
        )

        # Rebuilt if it has gone missing, rather than assumed. A search is an
        # hour of GPU time and the benchmark is a directory in the run's own
        # output tree -- one that has, in practice, been moved out from under a
        # running search. Regenerating it costs a house compile and is
        # deterministic, so the alternative (every remaining trial failing with
        # FileNotFoundError, and the run reporting nine errors) is strictly
        # worse. `build` is a no-op when the file is where it was left.
        mini_benchmark.build(self.benchmark_dir)

        # Before the config class is resolved: `run_evaluation` builds the
        # experiment config from the "module:Class" string, and that
        # construction is where the trial is read back out of the environment.
        publish_params(setup_key, params)
        self.probe.episodes.clear()

        trial_dir = self.output_root / "trials" / f"{setup_key}__{label}"
        log.info(f"[trial] {setup_key} {label}: {params.describe()}")
        try:
            evaluation = run_evaluation(
                eval_config_cls=qualified_config_name(setup.eval_config),
                benchmark_dir=self.benchmark_dir,
                checkpoint_path=self.checkpoint,
                output_dir=trial_dir,
                max_episodes=len(mini_benchmark.TARGETS),
                # One process, because the probe and the MP4 recorder are hooks
                # in *this* one: a worker would render and score into its own
                # memory and hand back nothing but a success count.
                num_workers=1,
                task_horizon_steps=self.episode_steps,
                use_wandb=False,
            )
            result.output_dir = str(evaluation.output_dir)
        except Exception as error:  # noqa: BLE001 - one bad setting must not sink the search
            result.error = f"{type(error).__name__}: {error}"
            log.error(f"[trial] {setup_key} {label} failed: {result.error}")
            log.debug("", exc_info=True)
            return result

        result.episodes = _label_episodes(list(self.probe.episodes), setup_key)
        _attach_videos(result, Path(result.output_dir))

        if result.incomplete:
            # Worth shouting about rather than logging quietly: the run looks
            # like it worked, the MP4s are there, and the only visible symptom
            # is a trial that scored badly for a reason that has nothing to do
            # with its parameters. See `TrialResult.score`.
            log.warning(
                f"[trial] {setup_key} {label}: {result.incomplete} of "
                f"{len(result.episodes)} episodes CRASHED and are excluded from the "
                "score. Their videos are the ones named '..._incomplete.mp4'. The "
                "usual cause is the policy running out of GPU memory because "
                "something else is holding the card -- check `nvidia-smi`, and see "
                f"{Path(result.output_dir) / 'running_log.log'} for the traceback."
            )
        if not result.usable:
            log.error(
                f"[trial] {setup_key} {label}: every episode crashed, so this trial "
                "measured nothing. It will not be ranked."
            )
            return result

        log.info(
            f"[trial] {setup_key} {label}: score {result.score:.3f}, "
            f"{result.successes}/{len(result.scored_episodes)} picked up"
        )
        return result


def _label_episodes(episodes: list[EpisodeScore], setup_key: str) -> list[EpisodeScore]:
    """Say which object each episode was about, and which setup it belongs to.

    Matched on the instruction rather than on order: the rollout order is the
    benchmark's and has been stable in practice, but an episode that errors out
    before its first step still produces a record, and matching on text degrades
    into "unknown" instead of silently attributing one object's result to
    another.
    """
    by_instruction = {target.instruction: target.key for target in mini_benchmark.TARGETS}
    for index, episode in enumerate(episodes):
        episode.setup = setup_key
        episode.target = by_instruction.get(
            episode.instruction,
            mini_benchmark.episode_target(index).key if index < len(mini_benchmark.TARGETS) else "?",
        )
    return episodes


def _attach_videos(result: TrialResult, output_dir: Path) -> None:
    """Point each episode at the MP4 the recorder wrote for it.

    Matched by order within the run, which is the order `EpisodeVideoRecorder`
    numbers its files in -- it names them per house and per episode index, and
    there is one house here.
    """
    videos = sorted((output_dir / "videos").glob("*.mp4"))
    for episode, video in zip(result.episodes, videos):
        episode.video = str(video)


# =============================================================================
# The searches
# =============================================================================


def grid_points(axes: list[Axis]) -> list[np.ndarray]:
    """Every combination of every axis's values, in odometer order."""
    import itertools

    return [np.array(point, dtype=float) for point in itertools.product(*(a.grid_values() for a in axes))]


class SimpleCMAES:
    """A small (mu, lambda)-CMA-ES, maximising, with box bounds.

    Written out rather than depending on `cma`, which is not in this
    repository's environment and would be a new dependency for one script. It is
    the textbook algorithm with the rank-mu covariance update and no restarts:
    the budget here is tens of evaluations, not thousands, so the parts that
    earn their keep over long runs would never come into play.

    Bounds are handled by clipping what is proposed, which biases the search
    towards a boundary it is pressed against -- acceptable because every bound
    here is a physical limit (a camera cannot have a negative field of view) and
    a search that wants to sit on one is telling you something.
    """

    def __init__(
        self,
        x0: np.ndarray,
        sigma0: float,
        bounds: np.ndarray,
        population: int,
        seed: int = 0,
    ) -> None:
        self.dim = len(x0)
        self.bounds = bounds
        self.mean = np.asarray(x0, dtype=float).copy()
        self.sigma = float(sigma0)
        self.population = int(population)
        self.rng = np.random.default_rng(seed)

        self.mu = max(1, self.population // 2)
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights**2)

        # Learning rates, from Hansen's tutorial defaults.
        self.c_sigma = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0.0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
        self.c_1 = 2 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff),
        )
        self.chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))

        self.path_sigma = np.zeros(self.dim)
        self.path_c = np.zeros(self.dim)
        self.covariance = np.eye(self.dim)
        self.generation = 0
        self._last_raw: list[np.ndarray] = []

    def ask(self) -> list[np.ndarray]:
        """The next generation of points to evaluate, clipped into the box."""
        values, vectors = np.linalg.eigh(self.covariance)
        values = np.sqrt(np.maximum(values, 1e-20))
        samples = []
        self._last_raw = []
        for _ in range(self.population):
            z = self.rng.standard_normal(self.dim)
            y = vectors @ (values * z)
            x = self.mean + self.sigma * y
            self._last_raw.append(x)
            samples.append(np.clip(x, self.bounds[:, 0], self.bounds[:, 1]))
        return samples

    def tell(self, scores: list[float]) -> None:
        """Update the distribution from the scores of the last `ask()`, higher being better."""
        order = np.argsort(-np.asarray(scores, dtype=float))
        selected = np.array([self._last_raw[i] for i in order[: self.mu]])
        old_mean = self.mean.copy()
        self.mean = np.clip(
            self.weights @ selected, self.bounds[:, 0], self.bounds[:, 1]
        )

        step = (self.mean - old_mean) / self.sigma
        values, vectors = np.linalg.eigh(self.covariance)
        inv_sqrt = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, 1e-20))) @ vectors.T

        self.path_sigma = (1 - self.c_sigma) * self.path_sigma + np.sqrt(
            self.c_sigma * (2 - self.c_sigma) * self.mu_eff
        ) * (inv_sqrt @ step)
        self.generation += 1
        h_sigma = float(
            np.linalg.norm(self.path_sigma)
            / np.sqrt(1 - (1 - self.c_sigma) ** (2 * self.generation))
            < (1.4 + 2 / (self.dim + 1)) * self.chi_n
        )
        self.path_c = (1 - self.c_c) * self.path_c + h_sigma * np.sqrt(
            self.c_c * (2 - self.c_c) * self.mu_eff
        ) * step

        rank_one = np.outer(self.path_c, self.path_c)
        deltas = (selected - old_mean) / self.sigma
        rank_mu = sum(w * np.outer(d, d) for w, d in zip(self.weights, deltas))
        self.covariance = (
            (1 - self.c_1 - self.c_mu) * self.covariance + self.c_1 * rank_one + self.c_mu * rank_mu
        )
        self.sigma *= float(
            np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.path_sigma) / self.chi_n - 1))
        )


# =============================================================================
# The command line
# =============================================================================


@click.command()
@click.option(
    "--setup",
    "setup_keys",
    multiple=True,
    type=click.Choice(SETUP_KEYS),
    help="Setup to run. Repeatable. Defaults to all seven, which is the baseline table.",
)
@click.option(
    "--search",
    type=click.Choice(["sweep", "none", "grid", "cmaes"]),
    default="sweep",
    help="'sweep' (the default) grids every parameter that applies to each setup, "
    "in stages, carrying the winner forward -- hours of rollouts, and the thing to "
    "run when you want coverage. 'none' runs each setup once at its own defaults, "
    "which is the quick comparison table. 'grid' and 'cmaes' search the dimensions "
    "named by --dim, and are worth pointing at one setup at a time.",
)
@click.option(
    "--dim",
    "dim_specs",
    multiple=True,
    help="A dimension to search, as name=lo:hi (cmaes) or name=lo:hi:steps (grid). "
    "Repeatable. --list-dims prints what is available.",
)
@click.option("--population", type=int, default=6, help="CMA-ES points per generation.")
@click.option("--generations", type=int, default=5, help="CMA-ES generations.")
@click.option("--seed", type=int, default=0, help="CMA-ES sampling seed.")
@click.option(
    "--checkpoint",
    default=None,
    help="A local DROID checkpoint. Defaults to fetching the released one from the Hub.",
)
@click.option(
    "--episode-steps",
    type=int,
    default=None,
    help="Steps per episode. Defaults to the benchmark's 20s at 15Hz, about 300.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("eval_output") / "retarget_params",
    help="Where the benchmark, the trials, the videos and the report go.",
)
@click.option(
    "--rebuild-benchmark",
    is_flag=True,
    help="Rebuild the four-episode benchmark even if it is already there.",
)
@click.option(
    "--exo-crop",
    default=None,
    help="Crop the exo frame to WxH, e.g. 640x360. The fisheye setups otherwise "
    "deliver the 400x640 upright portrait frame Stretch's sideways-mounted head "
    "camera actually produces; this trades field of view for the landscape shape "
    "the checkpoint was trained on.",
)
@click.option("--list-dims", is_flag=True, help="List the searchable dimensions and exit.")
@click.option("--list-setups", is_flag=True, help="List the seven setups and exit.")
def main(
    setup_keys: tuple[str, ...],
    search: str,
    dim_specs: tuple[str, ...],
    population: int,
    generations: int,
    seed: int,
    checkpoint: str | None,
    episode_steps: int | None,
    output_dir: Path,
    rebuild_benchmark: bool,
    exo_crop: str | None,
    list_dims: bool,
    list_setups: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if list_setups:
        for key in SETUP_KEYS:
            setup = SETUPS[key]
            click.echo(f"{key:20s} {setup.robot:8s} {setup.description}")
            click.echo(f"{'':20s} {setup.params.describe()}")
        return
    if list_dims:
        for name, dimension in DIMENSIONS.items():
            scope = "both robots" if len(dimension.robots) > 1 else f"{dimension.robots[0]} only"
            click.echo(
                f"{name:20s} {dimension.bounds[0]:>8.3f} .. {dimension.bounds[1]:<8.3f} {scope}"
            )
            click.echo(f"{'':20s} {dimension.description}")
            values = ", ".join(f"{value:g}" for value in dimension.sweep)
            click.echo(f"{'':20s} --search sweep tries: {values}")
        return

    keys = list(setup_keys) or list(SETUP_KEYS)
    crop = _parse_crop(exo_crop)
    axes = [parse_axis(spec) for spec in dim_specs]
    if search in ("grid", "cmaes") and not axes:
        raise click.UsageError(f"--search {search} needs at least one --dim. See --list-dims.")
    if search in ("none", "sweep") and axes:
        raise click.UsageError(
            f"--dim does not apply to --search {search}. Use --search grid or --search cmaes "
            "to search dimensions you name, or let --search sweep use its own."
        )
    if search in ("grid", "cmaes") and len(keys) > 1:
        click.secho(
            f"Searching {len(axes)} dimensions across {len(keys)} setups. Each trial is "
            f"{len(mini_benchmark.TARGETS)} rollouts; consider one --setup at a time.",
            fg="yellow",
        )

    # A dimension the setup's robot ignores would run identical trials and report
    # them as if they differed, which is worse than refusing.
    for axis in axes:
        inapplicable = [k for k in keys if SETUPS[k].robot not in axis.dimension.robots]
        if inapplicable:
            raise click.UsageError(
                f"--dim {axis.dimension.name} only does anything on "
                f"{'/'.join(axis.dimension.robots)} setups, and {', '.join(inapplicable)} "
                f"{'is' if len(inapplicable) == 1 else 'are'} not. Every trial would be "
                "identical."
            )

    if search == "sweep":
        _announce_sweep(keys)

    # MolmoBot is a clone rather than a dependency, so nothing puts its `olmo`
    # package on the import path. Done here so a missing checkout is a message
    # at the command line instead of an ImportError once the first scene has
    # loaded.
    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        MolmoBotSetupError,
        ensure_importable,
        inference_requirements_message,
        missing_inference_requirements,
    )

    try:
        ensure_importable()
    except MolmoBotSetupError as error:
        raise click.UsageError(str(error)) from error
    missing = missing_inference_requirements()
    if missing:
        raise click.UsageError(inference_requirements_message(missing))

    from examples.machine_learning.molmospaces.visualize import install_eval_video_hook

    output_dir.mkdir(parents=True, exist_ok=True)
    archived = preserve_previous_run(output_dir)
    if archived is not None:
        click.secho(
            f"{output_dir} already held a run's results; moved them to {archived} rather "
            "than overwriting them.",
            fg="yellow",
        )
    benchmark_dir = mini_benchmark.build(output_dir / "benchmark", force=rebuild_benchmark)

    # Both hooks wrap the same rollout, once, for the whole search: the recorder
    # writes into each evaluation's own output directory, and the probe is
    # emptied between trials.
    install_eval_video_hook()
    probe = install_probe()
    runner = TrialRunner(
        benchmark_dir=benchmark_dir,
        output_root=output_dir,
        probe=probe,
        episode_steps=episode_steps,
        checkpoint=checkpoint,
    )

    trials: list[TrialResult] = []

    def record(trial: TrialResult) -> TrialResult:
        """Keep a trial, and refresh the report with everything run so far.

        Rewritten after *every* trial rather than at the end, or even per setup:
        a grid over two dimensions is an hour or more, and being interrupted in
        the middle of one should still leave the rows it managed -- and let you
        watch the search decide while it runs. The files are small and rewritten
        whole, so there is no partial-file state to get wrong.
        """
        trials.append(trial)
        _write_outputs(trials, output_dir)
        return trial

    for setup_key in keys:
        base = SETUPS[setup_key].params
        if crop is not None:
            base = dataclasses.replace(base, exo=dataclasses.replace(base.exo, crop_to=crop))
        if search == "sweep":
            _run_sweep(runner, setup_key, base, record)
        elif search == "none":
            record(runner.run(setup_key, base, "default"))
        elif search == "grid":
            for index, point in enumerate(grid_points(axes)):
                record(runner.run(setup_key, apply_point(base, axes, point), _label(axes, point, index)))
        else:
            _run_cmaes(runner, setup_key, base, axes, population, generations, seed, record)

    targets = mini_benchmark.TARGET_KEYS
    click.echo("\n" + format_trial_table(trials, targets) + "\n")
    report = _write_outputs(trials, output_dir)
    click.secho(f"Wrote {report}", fg="green")


def sweep_axes(stage: str, robot: str) -> list[Axis]:
    """The axes a stage sweeps for a given robot, at their declared sweep values.

    Empty when no dimension in the stage applies -- the gripper stage on a
    Franka setup -- which the caller takes as "skip".
    """
    names = dict(SWEEP_STAGES)[stage]
    axes = []
    for name in names:
        dimension = DIMENSIONS[name]
        if robot not in dimension.robots:
            continue
        values = dimension.sweep
        axes.append(Axis(dimension, min(values), max(values), len(values), values))
    return axes


def sweep_plan(keys: list[str]) -> list[tuple[str, str, int]]:
    """`(setup, stage, trial count)` for every stage the sweep will run."""
    plan = []
    for setup_key in keys:
        robot = SETUPS[setup_key].robot
        for stage, _ in SWEEP_STAGES:
            axes = sweep_axes(stage, robot)
            if axes:
                plan.append((setup_key, stage, len(grid_points(axes))))
    return plan


def _run_sweep(
    runner: TrialRunner,
    setup_key: str,
    base: RetargetParams,
    record: Callable[[TrialResult], TrialResult],
) -> None:
    """Grid each stage in turn, carrying the best parameters forward.

    The winner of a stage is the best *usable* trial -- a stage whose trials all
    crashed carries its starting parameters forward rather than a meaningless
    one, so a GPU that filled up mid-sweep costs that stage rather than every
    stage after it.
    """
    robot = SETUPS[setup_key].robot
    best = base
    for stage, _ in SWEEP_STAGES:
        axes = sweep_axes(stage, robot)
        if not axes:
            log.info(f"[sweep] {setup_key}: no {stage} dimensions apply, skipping")
            continue
        points = grid_points(axes)
        log.info(
            f"[sweep] {setup_key} stage '{stage}': {len(points)} points over "
            f"{', '.join(axis.dimension.name for axis in axes)}"
        )
        results = [
            record(runner.run(setup_key, apply_point(best, axes, point), f"{stage}_{_label(axes, point, index)}"))
            for index, point in enumerate(points)
        ]
        usable = [trial for trial in results if trial.usable]
        if not usable:
            log.warning(
                f"[sweep] {setup_key} stage '{stage}': nothing usable, carrying the "
                "stage's starting parameters into the next one"
            )
            continue
        winner = max(usable, key=lambda trial: trial.score)
        best = apply_point(best, axes, points[results.index(winner)])
        log.info(
            f"[sweep] {setup_key} stage '{stage}' best {winner.score:.3f}: {best.describe()}"
        )


SECONDS_PER_ROLLOUT = 45
"""
Rough wall-clock per rollout, for the estimate printed before a sweep.

Measured on this machine over a few hundred episodes: a 20-second episode at
15Hz is ~300 policy steps, and one that grasps early stops sooner. Only ever
used to print an order of magnitude, so it does not need to be right.
"""


def _announce_sweep(keys: list[str]) -> None:
    """Print what the sweep will run before it starts, with a time estimate.

    A default that takes hours should say so on the first line rather than in
    the scrollback an hour later.
    """
    plan = sweep_plan(keys)
    trials = sum(count for _, _, count in plan)
    rollouts = trials * len(mini_benchmark.TARGETS)
    click.secho(f"Sweeping {len(keys)} setup(s): {trials} trials, {rollouts} rollouts.", bold=True)
    for setup_key, stage, count in plan:
        axes = sweep_axes(stage, SETUPS[setup_key].robot)
        names = ", ".join(
            f"{axis.dimension.name}({len(axis.grid_values())})" for axis in axes
        )
        click.echo(f"  {setup_key:20s} {stage:8s} {count:3d} trials   {names}")
    hours = rollouts * SECONDS_PER_ROLLOUT / 3600
    click.secho(
        f"Roughly {hours:.0f} hours. Results are written after every trial, so the run "
        "can be stopped at any point and what it has is already in report.md.",
        fg="yellow",
    )


def _parse_crop(text: str | None) -> tuple[int, int] | None:
    """`--exo-crop WxH` -> `(width, height)`."""
    if text is None:
        return None
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError as error:
        raise click.UsageError(f"--exo-crop {text!r} should look like 640x360.") from error
    return width, height


def _label(axes: list[Axis], point: np.ndarray, index: int) -> str:
    """A short, filesystem-safe name for a point, for its trial directory."""
    parts = [f"{axis.dimension.name}{value:g}" for axis, value in zip(axes, point)]
    return f"{index:03d}_" + "_".join(parts).replace(".", "p").replace("-", "m")


def _run_cmaes(
    runner: TrialRunner,
    setup_key: str,
    base: RetargetParams,
    axes: list[Axis],
    population: int,
    generations: int,
    seed: int,
    record: Callable[[TrialResult], TrialResult],
) -> list[TrialResult]:
    """Run CMA-ES over `axes`, starting from the setup's own defaults."""
    x0 = np.array([axis.dimension.read(base) for axis in axes], dtype=float)
    bounds = np.array([[axis.low, axis.high] for axis in axes], dtype=float)
    x0 = np.clip(x0, bounds[:, 0], bounds[:, 1])
    # A quarter of each range: wide enough to leave the setup's own
    # neighbourhood in the first generation, narrow enough that most of a small
    # population lands inside the box.
    sigma0 = float(np.mean((bounds[:, 1] - bounds[:, 0]) / 4.0))

    optimiser = SimpleCMAES(x0, sigma0, bounds, population=population, seed=seed)
    trials: list[TrialResult] = []
    evaluated = 0
    for generation in range(generations):
        points = optimiser.ask()
        scores = []
        for point in points:
            trial = record(
                runner.run(setup_key, apply_point(base, axes, point), _label(axes, point, evaluated))
            )
            trials.append(trial)
            # A trial that measured nothing is fed back as the generation's worst
            # score rather than as a zero. Zero is a claim about the parameters --
            # it would teach the distribution to avoid whatever was in flight when
            # the GPU filled up. This only keeps the generation's ranking intact,
            # which is all CMA-ES reads.
            scores.append(trial.score if trial.usable else None)
            evaluated += 1

        usable = [score for score in scores if score is not None]
        if not usable:
            log.error(
                f"[cmaes] generation {generation + 1}: every trial crashed, so there is "
                "nothing to learn from. Stopping the search; fix the cause and rerun."
            )
            break
        worst = min(usable)
        optimiser.tell([worst if score is None else score for score in scores])
        log.info(
            f"[cmaes] generation {generation + 1}/{generations}: best {max(usable):.3f}, "
            f"mean now {np.round(optimiser.mean, 4).tolist()}, sigma {optimiser.sigma:.4f}"
        )
    return trials


AGGREGATE_FILES = (
    "report.md",
    "trials.csv",
    "trials.jsonl",
    "episodes.csv",
    "analysis.md",
    "analysis.csv",
)
"""The files a run rewrites wholesale, and so the files a second run would destroy."""


def preserve_previous_run(output_dir: Path) -> Path | None:
    """Move a previous run's aggregate files aside before this one starts writing.

    `_write_outputs` rewrites each of `AGGREGATE_FILES` from scratch after every
    trial, holding only *this* run's trials -- so a second run pointed at a
    directory that already has them replaces a finished sweep with its own first
    trial, and nine hours of results are gone by the time anyone looks. The
    default `--output-dir` is a fixed path, so this is not an exotic mistake:
    running the command twice is enough.

    Returns the archive directory, or None if there was nothing to preserve. The
    per-trial subdirectories under `trials/` are left alone -- `run_evaluation`
    timestamps its own output inside them, so a repeated trial name adds a
    directory rather than overwriting one.
    """
    existing = [output_dir / name for name in AGGREGATE_FILES if (output_dir / name).is_file()]
    if not existing:
        return None

    stamp = datetime.datetime.fromtimestamp(
        max(path.stat().st_mtime for path in existing)
    ).strftime("%Y%m%d_%H%M%S")
    archive = output_dir / "previous" / stamp
    archive.mkdir(parents=True, exist_ok=True)
    for path in existing:
        path.rename(archive / path.name)
    return archive


def _write_outputs(trials: list[TrialResult], output_dir: Path) -> Path:
    """Refresh the two CSVs and the report from everything run so far."""
    write_episode_csv(trials, output_dir / "episodes.csv")
    write_trial_csv(trials, output_dir / "trials.csv")
    (output_dir / "trials.jsonl").write_text(
        "".join(
            json.dumps({"setup": t.setup, "score": t.score, "params": json.loads(t.params_json)})
            + "\n"
            for t in trials
        )
    )
    return write_report(trials, mini_benchmark.TARGET_KEYS, output_dir)


if __name__ == "__main__":
    main()
