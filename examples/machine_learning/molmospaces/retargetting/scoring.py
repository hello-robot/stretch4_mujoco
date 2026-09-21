"""
What a trial is scored on, and the report the search writes.

Success rate is the number that matters and the wrong thing to optimise. Four
episodes give five possible success rates, most settings score zero, and a
search that can only see zeros has nothing to climb: a camera that at least
brings the gripper to the object is indistinguishable from one pointed at the
ceiling. So every episode also gets a continuous score, and it is built out of
what a failed rollout still tells you:

    did the gripper get closer to the object than it started?   approach
    did it ever touch it?                                       contact
    did it come off the counter at all?                         lift

which is the order a grasp goes wrong in, so a setting that fails later scores
higher than one that fails earlier, and the search has a gradient to follow
long before anything succeeds. A success is 1.0 outright, so the ranking never
prefers a near-miss to a grasp.

None of that is recorded by the evaluation pipeline -- the H5 it writes has
`success` and `rewards` and neither says how close a failure came -- so
`GraspProbe` watches the rollout itself, through the same observer hook the MP4
recorder uses.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

APPROACH_WEIGHT = 0.45
CONTACT_WEIGHT = 0.25
LIFT_WEIGHT = 0.30
"""
How a failed episode's partial credit is split. They sum to 1.0, so a failure
can approach but never reach the 1.0 a success scores outright.

Weighted towards approach because that is the stage the camera decides: a policy
that cannot see the object does not get near it, and every setup in this study
differs in nothing but the camera. Contact and lift are downstream of the
gripper geometry, which `grasp_offset_m` and `wrist_tilt_deg` address.
"""

CONTACT_DISTANCE_M = 0.02
"""
How close the gripper has to get to count as having reached the object, in metres.

A fallback for the contact term, used alongside the contact list: an episode
where the gripper closed on thin air 5mm from a knife has clearly found the
object, and scoring it the same as one that never left home would throw away the
most informative near-miss there is.
"""


@dataclass
class EpisodeScore:
    """One episode of one trial: what happened, and what it is worth."""

    setup: str = ""
    target: str = ""
    scene: int = -1
    """`house_index` of the scene this episode ran in, or -1 if it could not be read."""

    instruction: str = ""
    success: bool = False
    steps: int = 0

    completed: bool = True
    """
    Whether the rollout finished and returned a verdict.

    False means it raised -- `run_single_rollout` returns a bool on every path,
    so the only way the outcome comes back as None is an exception, and by far
    the most common one is `torch.OutOfMemoryError` from the VLA when something
    else is holding the GPU. Such an episode says nothing about the parameters
    being trialled and is kept out of the score entirely; the MP4 for it is the
    one `EpisodeVideoRecorder` names `..._incomplete.mp4`.
    """

    start_distance_m: float = float("nan")
    """Gripper-to-object distance at reset. The denominator of `approach`."""

    min_distance_m: float = float("nan")
    """The closest the gripper ever got."""

    touched: bool = False
    """Whether the robot ever made contact with the object."""

    best_lift_m: float = 0.0
    """The furthest the object ever came off its starting height, in metres."""

    success_threshold_m: float = 0.01

    retarget_position_error_mean_m: float = float("nan")
    retarget_orientation_error_mean_rad: float = float("nan")
    """
    How far the commanded tool poses fell outside what the robot could reach.

    Read off the policy's own `get_info()`, so they are only present for the
    Stretch setups -- there is nothing to retarget on the Franka. A few
    centimetres is the retargeting working; a persistent 10cm+ means the policy
    is asking for somewhere the robot cannot go from where it is standing, which
    is a different failure from not being able to see the object.
    """

    video: str = ""
    """Path of the MP4 for this episode, if one was recorded."""

    @property
    def approach(self) -> float:
        """Fraction of the initial gripper-to-object gap that was closed, in [0, 1]."""
        if not np.isfinite(self.start_distance_m) or self.start_distance_m <= 1e-6:
            return 0.0
        closed = (self.start_distance_m - self.min_distance_m) / self.start_distance_m
        return float(np.clip(closed, 0.0, 1.0))

    @property
    def contact(self) -> float:
        return 1.0 if self.touched or self.min_distance_m <= CONTACT_DISTANCE_M else 0.0

    @property
    def lift(self) -> float:
        """Fraction of the success lift height reached, in [0, 1].

        Zero unless the robot touched the object, which is the same condition
        `PickTask.get_reward` puts on its own lift term and is here for the same
        reason: an object nudged upwards by its own settle, or by another object
        falling against it, has not been picked up by anybody. Measured on a
        do-nothing policy, the ungated term paid out 0.1 for a potato that
        rolled a few millimetres on its own.
        """
        if self.success_threshold_m <= 0 or not self.touched:
            return 0.0
        return float(np.clip(self.best_lift_m / self.success_threshold_m, 0.0, 1.0))

    @property
    def score(self) -> float:
        """This episode's contribution to the trial's objective, in [0, 1]."""
        if self.success:
            return 1.0
        return (
            APPROACH_WEIGHT * self.approach
            + CONTACT_WEIGHT * self.contact
            + LIFT_WEIGHT * self.lift
        )


@dataclass
class TrialResult:
    """One setup at one point in the search space, over all four objects."""

    setup: str
    params_description: str
    params_json: str
    episodes: list[EpisodeScore] = field(default_factory=list)
    output_dir: str = ""
    error: str = ""

    @property
    def scored_episodes(self) -> list[EpisodeScore]:
        """The episodes that actually ran to a verdict. See `EpisodeScore.completed`."""
        return [episode for episode in self.episodes if episode.completed]

    @property
    def incomplete(self) -> int:
        """How many episodes crashed rather than finishing."""
        return len(self.episodes) - len(self.scored_episodes)

    @property
    def successes(self) -> int:
        return sum(1 for episode in self.scored_episodes if episode.success)

    @property
    def success_rate(self) -> float:
        scored = self.scored_episodes
        return self.successes / len(scored) if scored else 0.0

    @property
    def usable(self) -> bool:
        """Whether this trial measured anything at all.

        False for a trial that errored outright and for one whose every episode
        crashed -- a run under a GPU that was already full, say. Such a trial has
        not tested its parameters, and `params_search` neither ranks it nor lets
        CMA-ES learn from it.
        """
        return not self.error and bool(self.scored_episodes)

    @property
    def score(self) -> float:
        """The objective a search maximises: the mean score of the episodes that ran.

        Crashed episodes are excluded rather than counted as zero. The
        distinction matters because it is the difference between "this camera
        cannot see the object" and "the GPU was full": counting a crash as a
        failure would have the search conclude that whichever parameters
        happened to be in flight when another process took the card are bad
        ones, and steer away from them for good.

        A trial where *nothing* ran scores zero, but `usable` is False and the
        search skips it instead of ranking it.
        """
        scored = self.scored_episodes
        if self.error or not scored:
            return 0.0
        return float(np.mean([episode.score for episode in scored]))


# =============================================================================
# Watching a rollout
# =============================================================================


class GraspProbe:
    """Records, per episode, how close the robot came to picking the object up.

    An observer in the sense `visualize._observed_rollout` means: `on_reset` and
    `log_step` are shadowed onto the task for the length of one rollout, so this
    sees the same steps the policy does with no hook of its own in the rollout
    loop.

    Deliberately robot-agnostic -- it reads the gripper's leaf frame and the
    object's body, both of which every `RobotView` here has -- because the whole
    point is comparing a Franka rollout with a Stretch one.
    """

    def __init__(self, sink: Path | None = None) -> None:
        self.sink = sink
        """
        Directory to append each finished episode to, as JSON, or None for memory only.

        This is what makes `--num-workers` more than 1 possible. The probe is a
        hook in whichever *process* runs the rollout, and with several workers
        that is not the process collecting results -- a worker's in-memory list
        dies with the worker. Writing one JSON line per episode into a shared
        directory, and having the parent read the directory once the evaluation
        returns, is the same trick `configs.VIDEO_EXPORT_ENV_VAR` uses to get MP4s
        out of workers, for the same reason.

        One file per process, appended to, so two workers never interleave
        writes into one file.
        """
        self.episodes: list[EpisodeScore] = []
        self._current: EpisodeScore | None = None
        self._object_name: str = ""
        self._start_height: float = 0.0
        self._scene: int = -1
        self._policy: Any = None

    # -- the observer protocol ------------------------------------------------

    def start_episode(self, episode_seed: int, task: Any, policy: Any = None) -> None:
        """Open a record. Called before `on_reset`, by the rollout hook."""
        self.finish_episode(success=None)
        self._current = EpisodeScore()
        self._policy = policy

    def on_reset(self, task: Any) -> None:
        """Latch what this episode is about, once the scene is built and settled."""
        if self._current is None:
            self._current = EpisodeScore()
        task_config = task.config.task_config
        self._object_name = task_config.pickup_obj_name
        self._scene = _house_index(task)
        self._start_height = float(task_config.pickup_obj_start_pose[2])
        self._current.instruction = _task_description(task)
        self._current.success_threshold_m = float(
            getattr(task_config, "succ_pos_threshold", 0.01) or 0.01
        )
        distance = self._distance(task)
        self._current.start_distance_m = distance
        self._current.min_distance_m = distance

    def log_step(self, step: int, task: Any, observation: Any, policy: Any = None) -> None:
        record = self._current
        if record is None:
            return
        record.steps = step

        distance = self._distance(task)
        if np.isfinite(distance):
            record.min_distance_m = float(np.nanmin([record.min_distance_m, distance]))
        if not np.isfinite(record.start_distance_m):
            record.start_distance_m = distance

        data = _mj_data(task)
        if data is None:
            return
        try:
            from molmo_spaces.env.data_views import MlSpacesObject

            pickup = MlSpacesObject(data=data, object_name=self._object_name)
            record.best_lift_m = max(
                record.best_lift_m, float(pickup.position[2]) - self._start_height
            )
            record.touched = record.touched or _robot_touches(task, data, pickup.body_id)
        except Exception as error:  # noqa: BLE001 - telemetry must not sink a rollout
            log.debug(f"[probe] step {step}: {error}")

    def finish_episode(self, success: bool | None) -> None:
        """Close the record. `success` is the rollout's own verdict, or None if it raised."""
        if self._current is None:
            return
        self._current.completed = success is not None
        self._current.success = bool(success)
        self._current.scene = self._scene
        self._current.retarget_position_error_mean_m = _policy_info(
            self._policy, "retarget_position_error_mean_m"
        )
        self._current.retarget_orientation_error_mean_rad = _policy_info(
            self._policy, "retarget_orientation_error_mean_rad"
        )
        self.episodes.append(self._current)
        if self.sink is not None:
            self._write(self._current)
        self._current = None

    def _write(self, episode: EpisodeScore) -> None:
        """Append one episode to this process's file in the sink directory."""
        try:
            self.sink.mkdir(parents=True, exist_ok=True)
            with (self.sink / f"{os.getpid()}.jsonl").open("a") as handle:
                handle.write(json.dumps(asdict(episode)) + "\n")
        except OSError as error:  # noqa: BLE001 - telemetry must not sink a rollout
            log.warning(f"[probe] could not record an episode: {error}")

    # -- the measurements -----------------------------------------------------

    def _distance(self, task: Any) -> float:
        """Metres from the gripper's grasp frame to the object's origin."""
        data = _mj_data(task)
        if data is None or not self._object_name:
            return float("nan")
        try:
            from molmo_spaces.env.data_views import MlSpacesObject

            gripper = task.env.current_robot.robot_view.get_move_group("gripper")
            tool = np.asarray(gripper.leaf_frame_to_world, dtype=float)[:3, 3]
            pickup = MlSpacesObject(data=data, object_name=self._object_name)
            return float(np.linalg.norm(tool - np.asarray(pickup.position, dtype=float)))
        except Exception as error:  # noqa: BLE001
            log.debug(f"[probe] distance: {error}")
            return float("nan")


def _house_index(task: Any) -> int:
    """Which scene this episode is in, for a benchmark with more than one."""
    for holder in (getattr(task, "config", None), getattr(getattr(task, "env", None), "config", None)):
        sampler = getattr(holder, "task_sampler_config", None)
        houses = getattr(sampler, "house_inds", None)
        if houses:
            return int(houses[0])
    return -1


def _mj_data(task: Any) -> Any:
    datas = getattr(getattr(task, "env", None), "mj_datas", None)
    return datas[0] if datas else None


def _task_description(task: Any) -> str:
    try:
        return str(task.get_task_description())
    except Exception:  # noqa: BLE001
        return ""


def _robot_touches(task: Any, data: Any, object_body_id: int) -> bool:
    """Whether any current contact is between the object and the robot.

    The same root-body test `PickTask.get_info` uses to decide whether a lift
    counts, minus the part that insists *nothing else* is touching -- here the
    question is only whether the robot found the object at all.
    """
    robot_root = task.env.current_robot.robot_view.base.root_body_id
    model = data.model
    for contact in data.contact:
        root1 = model.body_rootid[model.geom_bodyid[contact.geom1]]
        root2 = model.body_rootid[model.geom_bodyid[contact.geom2]]
        if (root1 == object_body_id) ^ (root2 == object_body_id):
            other = root1 if root1 != object_body_id else root2
            if other == robot_root:
                return True
    return False


def _policy_info(policy: Any, key: str) -> float:
    """One number off a policy's `get_info()`, or NaN if it has none."""
    if policy is None:
        return float("nan")
    try:
        return float(policy.get_info().get(key, float("nan")))
    except Exception:  # noqa: BLE001
        return float("nan")


def collect_probe_records(sink: Path) -> list[EpisodeScore]:
    """Every episode written into `sink`, by this process and by any workers.

    Sorted by the order they were written within each file and then by file, so
    a single-worker run comes back in rollout order; with several workers the
    order is per-worker rather than global, which is why nothing downstream
    relies on position -- episodes carry their own scene and instruction.
    """
    episodes: list[EpisodeScore] = []
    if not sink.is_dir():
        return episodes
    fields = {f.name for f in dataclasses.fields(EpisodeScore)}
    for path in sorted(sink.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                log.warning(f"[probe] skipping a malformed record in {path}")
                continue
            episodes.append(EpisodeScore(**{k: v for k, v in record.items() if k in fields}))
    return episodes


def install_probe(sink: Path | None = None) -> GraspProbe:
    """Watch every rollout in this process, and return the probe watching them.

    Shares the rollout hook the MP4 recorder and the Rerun stream use, so
    installing both costs one wrapper rather than two. That installer is private
    to `visualize.py` -- it is reached here rather than reimplemented so there is
    one place that knows how `JsonEvalRunner.run_single_rollout` is wrapped.
    """
    from examples.machine_learning.molmospaces.visualize import _install_eval_rollout_hook

    probe = GraspProbe(sink=sink)
    _install_eval_rollout_hook(probe)
    return probe


# =============================================================================
# The report
# =============================================================================

EPISODE_FIELDS = (
    "trial",
    "setup",
    "scene",
    "target",
    "instruction",
    "success",
    "completed",
    "score",
    "approach",
    "contact",
    "lift",
    "steps",
    "start_distance_m",
    "min_distance_m",
    "touched",
    "best_lift_m",
    "retarget_position_error_mean_m",
    "retarget_orientation_error_mean_rad",
    "video",
    "params",
)


def write_episode_csv(trials: list[TrialResult], path: Path) -> None:
    """One row per episode of every trial: the raw table everything else is a view of."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EPISODE_FIELDS))
        writer.writeheader()
        for index, trial in enumerate(trials):
            for episode in trial.episodes:
                row = {key: value for key, value in asdict(episode).items() if key in EPISODE_FIELDS}
                row.update(
                    trial=index,
                    score=round(episode.score, 4),
                    approach=round(episode.approach, 4),
                    contact=episode.contact,
                    lift=round(episode.lift, 4),
                    params=trial.params_description,
                )
                writer.writerow(row)


def write_trial_csv(trials: list[TrialResult], path: Path) -> None:
    """One row per trial: what a search ranked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["setup", "score", "successes", "episodes", "success_rate", "params", "output_dir", "error"]
        )
        for trial in trials:
            writer.writerow(
                [
                    trial.setup,
                    round(trial.score, 4),
                    trial.successes,
                    len(trial.episodes),
                    round(trial.success_rate, 4),
                    trial.params_description,
                    trial.output_dir,
                    trial.error,
                ]
            )


def format_trial_table(trials: list[TrialResult], targets: tuple[str, ...]) -> str:
    """The table printed at the end of a run: a trial per row, an object per column."""
    columns = "".join(f"{target[:9]:>10s}" for target in targets)
    header = f"{'setup':20s} {'score':>6s} {'succ':>6s}{columns}   parameters"
    lines = [header, "-" * len(header)]
    for trial in trials:
        if trial.error:
            lines.append(f"{trial.setup:20s} {'ERROR':>6s}  {trial.error[:70]}")
            continue
        by_target = {episode.target: episode for episode in trial.episodes}
        cells = ""
        for target in targets:
            episode = by_target.get(target)
            if episode is None:
                cells += f"{'-':>10s}"
            elif not episode.completed:
                cells += f"{'crashed':>10s}"
            elif episode.success:
                cells += f"{'PICKED':>10s}"
            else:
                cells += f"{episode.score:>10.2f}"
        scored = len(trial.scored_episodes)
        score = f"{trial.score:6.3f}" if trial.usable else f"{'n/a':>6s}"
        lines.append(
            f"{trial.setup:20s} {score} "
            f"{trial.successes:>3d}/{scored:<2d}{cells}   {trial.params_description}"
        )
    return "\n".join(lines)


def write_report(trials: list[TrialResult], targets: tuple[str, ...], output_dir: Path) -> Path:
    """Write `report.md` beside the CSVs, and return its path.

    Markdown because the interesting output of this search is a comparison
    somebody reads, not a number a program consumes -- the CSVs are there for
    the latter.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(trials, key=lambda trial: trial.score, reverse=True)

    lines = [
        "# Retargeting parameter search",
        "",
        "Each trial is one setup at one point in the parameter space, scored over "
        f"{len(targets)} objects ({', '.join(targets)}) in one kitchen from one robot pose.",
        "",
        "`score` is the mean episode score: 1.0 for a grasp, and for a failure a "
        "weighted sum of how far the gripper closed the gap to the object "
        f"({APPROACH_WEIGHT}), whether it touched it ({CONTACT_WEIGHT}) and how far it "
        f"lifted it ({LIFT_WEIGHT}). See `scoring.py`.",
        "",
        "| rank | setup | score | picked | " + " | ".join(targets) + " | parameters |",
        "|---:|---|---:|---:|" + "---:|" * len(targets) + "---|",
    ]
    crashed = sum(trial.incomplete for trial in trials)
    for rank, trial in enumerate(ranked, start=1):
        if trial.error:
            lines.append(
                f"| {rank} | {trial.setup} | - | - | "
                + " | ".join("-" for _ in targets)
                + f" | ERROR: {trial.error} |"
            )
            continue
        by_target = {episode.target: episode for episode in trial.episodes}
        cells = []
        for target in targets:
            episode = by_target.get(target)
            if episode is None:
                cells.append("-")
            elif not episode.completed:
                cells.append("crashed")
            elif episode.success:
                cells.append("**picked**")
            else:
                cells.append(f"{episode.score:.2f}")
        score = f"{trial.score:.3f}" if trial.usable else "n/a"
        lines.append(
            f"| {rank} | {trial.setup} | {score} | "
            f"{trial.successes}/{len(trial.scored_episodes)} | "
            + " | ".join(cells)
            + f" | `{trial.params_description}` |"
        )

    if crashed:
        lines += [
            "",
            f"**{crashed} episode(s) crashed rather than finishing** and are excluded "
            "from every score above, because a crash says nothing about the parameters. "
            "Their MP4s are the ones named `..._incomplete.mp4`. The usual cause is "
            "`torch.OutOfMemoryError` from the policy when another process is holding "
            "the GPU -- check `nvidia-smi` and the run's `running_log.log`. A trial "
            "whose episodes all crashed shows `n/a` and is not ranked.",
        ]

    lines += ["", "## Episodes", ""]
    for trial in ranked:
        if trial.error:
            continue
        lines.append(f"### {trial.setup} — score {trial.score:.3f}")
        lines.append("")
        lines.append(f"`{trial.params_description}`")
        lines.append("")
        lines.append(
            "| object | outcome | approach | touched | best lift | retarget pos err | video |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for episode in trial.episodes:
            outcome = (
                "crashed" if not episode.completed else "picked up" if episode.success else "failed"
            )
            residual = (
                f"{episode.retarget_position_error_mean_m:.3f} m"
                if np.isfinite(episode.retarget_position_error_mean_m)
                else "n/a"
            )
            video = f"[mp4]({episode.video})" if episode.video else ""
            lines.append(
                f"| {episode.target} | {outcome} | {episode.approach:.2f} | "
                f"{'yes' if episode.touched else 'no'} | {episode.best_lift_m * 100:.1f} cm | "
                f"{residual} | {video} |"
            )
        lines.append("")

    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
