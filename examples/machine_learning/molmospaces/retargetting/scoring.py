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

import click
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
    from examples.machine_learning.molmospaces.visualize import (
        _EVAL_OBSERVERS,
        _install_eval_rollout_hook,
    )

    # Idempotent, like `install_eval_video_hook`, and it has to be: this is called
    # from `setups.install_worker_hooks()` at module import, and that module is
    # imported again every time `run_evaluation` resolves an eval config from its
    # "module:Class" string -- once per setup. Installing a fresh probe each time
    # left every earlier probe registered and still writing, each to the sink it
    # was built with, so a run of eight setups wrote its first sink eight times
    # over: 160 records where 20 were run. Measured on
    # `eval_output/side_by_side_sept20`, whose first sink holds all eight setups'
    # episodes concatenated in run order.
    existing = next((o for o in _EVAL_OBSERVERS if isinstance(o, GraspProbe)), None)
    if existing is not None:
        if sink is not None:
            existing.sink = sink
        return existing

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


# =============================================================================
# What actually differs between trials
# =============================================================================

PARAM_LABELS = {
    "mount_body": "mount",
    "pos": "pos",
    "yaw_deg": "yaw",
    "pitch_deg": "pitch",
    "roll_deg": "roll",
    "fovy": "fovy",
    "render_size": "render",
    "fisheye": "lens",
    "quarter_turns": "turns",
    "virtual_pitch_deg": "vpitch",
    "crop_to": "crop",
    "grasp_offset_m": "grasp_off",
    "wrist_tilt_deg": "wrist_tilt",
    "target_z_offset_m": "z_offset",
}
"""Column headings for the parameter fields, short enough to put in a table."""

TOOL_FIELDS = ("grasp_offset_m", "wrist_tilt_deg", "target_z_offset_m")

PARAM_EXPLANATIONS = {
    "pitch_deg": (
        "Where the camera points, in degrees **up from straight down** — 0 looks at the "
        "floor, 90 at the horizon, so smaller numbers point further down. 43 is Stretch's "
        "head camera as built. This moves the camera itself, so a winner here is a claim "
        "about how the head *should* be mounted, not something that can be deployed on an "
        "existing robot — that is what `vpitch` is for."
    ),
    "fovy": (
        "Vertical field of view in degrees, as MuJoCo means it. Wider brings more of the "
        "workspace into frame at lower angular resolution on the object; narrower is the "
        "opposite trade. 71 is the DROID exo camera the checkpoint was trained on, 123 is "
        "Stretch's own head camera."
    ),
    "virtual_pitch_deg": (
        "Pitch **synthesised by sliding the crop window** up the rectified frame, with the "
        "camera left bolted where it is. A 123-degree fisheye sees far more than a 640x360 "
        "window needs, so a view that looks like it came from a differently-aimed camera can "
        "be cut out of the frame the hardware already produces. This is the only camera knob "
        "here that is deployable on the robot as built."
    ),
    "grasp_offset_m": (
        "Metres the commanded grasp centre is pushed **along Stretch's approach axis**. The "
        "policy drives `grasp_center_link` to the pose it asked for its Robotiq's grasp site, "
        "and those are not the same point: measured on a standing robot, Stretch's grasp "
        "centre sits about 1.5 cm past its own fingertips, so an uncorrected command leaves "
        "the object at or beyond the tips. Positive pulls the object deeper between the "
        "fingers. See `diagnose.py`."
    ),
    "wrist_tilt_deg": (
        "Extra pitch, in degrees, between the Franka's tool frame and Stretch's — applied "
        "about the tool y axis on top of the fixed -90 degree correction the retargeting "
        "already carries. 0 is the retargeting's own behaviour; positive tips the gripper "
        "further down than the policy asked for."
    ),
    "target_z_offset_m": (
        "**Metres** to raise every commanded target by. Stretch's lift runs out of travel "
        "where the Franka's does not, so raising targets buys clearance and stops the gripper "
        "dragging through the counter -- but only where the lift still has somewhere to go. "
        "0 is the default: this used to be a fraction of a shortfall measured at the Franka's "
        "home pose, which is above Stretch's ceiling, so it corrected targets that needed no "
        "correcting. Raise it too far and the gripper closes above the object."
    ),
}
"""One sentence on what each searched parameter actually does, for the report.

The marginal tables say which value won. Without this they do not say what was
won, and a mean-picked column is not self-explanatory to anyone who did not
write the retargeting.
"""


def _hashable(value: Any) -> Any:
    """Lists out of JSON turned into something that can go in a set."""
    return tuple(value) if isinstance(value, list) else value


def flat_params(trial: TrialResult) -> dict[str, Any]:
    """A trial's parameters as one flat `field -> value` mapping.

    The exo camera's fields and the tool correction's in a single namespace,
    because for the purpose of "what is different about this row" the nesting is
    noise -- nothing is named twice.
    """
    try:
        blob = json.loads(trial.params_json) if trial.params_json else {}
    except ValueError:
        return {}
    params = blob.get("params", blob)
    flat = {key: _hashable(value) for key, value in dict(params.get("exo", {})).items()}
    for key in TOOL_FIELDS:
        if key in params:
            flat[key] = params[key]
    return flat


def parameter_groups(
    trials: list[TrialResult],
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Split the parameter fields three ways: searched, setup-defining, constant.

    This is what makes the summary table readable. A trial's full description is
    fourteen fields long and all but two or three of them are identical down the
    whole table, so printing it per row buries the one number that changed --
    which is how two rows that differ in `vpitch` come to look like the same run
    repeated.

    * **searched** -- fields that take more than one value *within* a single
      setup. These are what the sweep moved, and they get a column each.
    * **setup-defining** -- fields constant within every setup but different
      between setups: the mount, the lens, the render size. The `setup` column
      already names these, so they go in a legend rather than in every row.
    * **constant** -- the same everywhere, stated once.

    Splitting within-setup from between-setup matters because the second kind is
    not a result. `stretch_fisheye` having a different `render_size` from
    `stretch_stretchcam` is the definition of those setups, not a finding about
    them.
    """
    usable = [trial for trial in trials if not trial.error]
    per_trial = [(trial.setup, flat_params(trial)) for trial in usable]
    fields = sorted({key for _, params in per_trial for key in params}, key=_field_order)

    by_setup: dict[str, list[dict[str, Any]]] = {}
    for setup, params in per_trial:
        by_setup.setdefault(setup, []).append(params)

    searched, setup_defining, constant = [], [], {}
    for field_name in fields:
        within = any(
            len({params.get(field_name) for params in group}) > 1 for group in by_setup.values()
        )
        across = len({params.get(field_name) for _, params in per_trial}) > 1
        if within:
            searched.append(field_name)
        elif across:
            setup_defining.append(field_name)
        elif per_trial:
            constant[field_name] = per_trial[0][1].get(field_name)
    return searched, setup_defining, constant


def _field_order(field_name: str) -> tuple[int, str]:
    """Camera fields before tool fields, each in the order they are applied."""
    order = list(PARAM_LABELS)
    return (order.index(field_name) if field_name in order else len(order), field_name)


def format_value(value: Any) -> str:
    """A parameter value, short: floats without trailing zeros, tuples as `a x b`."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, tuple):
        return "x".join(format_value(item) for item in value)
    return str(value)


def _target_cell(trial: TrialResult, target: str) -> str:
    """How a trial did on one object, across every scene it was tried in.

    A count rather than a verdict, because with more than one scene there is no
    single verdict to give: the same object is attempted once per scene and the
    interesting cases are the ones picked up in some scenes and not others.
    (The previous version of this kept only the last scene's episode per object,
    so a row could read "picked" on all four objects while its own total said
    15/20.)
    """
    episodes = [episode for episode in trial.episodes if episode.target == target]
    if not episodes:
        return "—"
    scored = [episode for episode in episodes if episode.completed]
    if not scored:
        return "crashed"
    picked = sum(1 for episode in scored if episode.success)
    if picked == len(scored):
        return f"**{picked}/{len(scored)}**"
    if picked:
        return f"{picked}/{len(scored)}"
    # Nothing picked: the partial score is the only signal left, so show it.
    return f"0/{len(scored)} ({np.mean([e.score for e in scored]):.2f})"


def format_trial_table(trials: list[TrialResult], targets: tuple[str, ...]) -> str:
    """The table printed at the end of a run: a trial per row, an object per column.

    Only the searched parameters are printed, for the reason `parameter_groups`
    gives -- on a console the full description is wider than the terminal and
    the part that differs is off the right-hand edge.
    """
    searched, _, _ = parameter_groups(trials)
    param_header = "".join(f"{PARAM_LABELS.get(f, f)[:10]:>11s}" for f in searched)
    columns = "".join(f"{target[:11]:>12s}" for target in targets)
    header = f"{'setup':20s} {'score':>6s} {'succ':>6s}{param_header}{columns}"
    lines = [header, "-" * len(header)]
    for trial in trials:
        if trial.error:
            lines.append(f"{trial.setup:20s} {'ERROR':>6s}  {trial.error[:70]}")
            continue
        params = flat_params(trial)
        cells = "".join(f"{format_value(params.get(f))[:10]:>11s}" for f in searched)
        cells += "".join(
            f"{_target_cell(trial, t).replace('**', '')[:11]:>12s}" for t in targets
        )
        scored = len(trial.scored_episodes)
        score = f"{trial.score:6.3f}" if trial.usable else f"{'n/a':>6s}"
        lines.append(
            f"{trial.setup:20s} {score} {trial.successes:>3d}/{scored:<2d}{cells}"
        )
    return "\n".join(lines)


def _rank(trials: list[TrialResult]) -> list[TrialResult]:
    """Best first, by objects picked and then by score.

    Picked first because that is what the benchmark asks. The score exists to
    give a search a gradient between settings that all pick nothing, and it pays
    up to 1.0 of partial credit for a near miss -- so ranking a finished sweep by
    it alone puts a trial that nearly picked four things above one that picked
    three.
    """
    return sorted(
        (trial for trial in trials if trial.usable),
        key=lambda trial: (trial.successes, trial.score),
        reverse=True,
    )


def _summary_table(trials: list[TrialResult], targets: tuple[str, ...], searched: list[str]) -> list[str]:
    """The ranked table: one row per trial, one column per searched parameter."""
    param_columns = [PARAM_LABELS.get(field, field) for field in searched]
    lines = [
        "| rank | setup | score | picked | "
        + " | ".join(param_columns + list(targets))
        + " |",
        "|---:|---|---:|---:|" + "---:|" * len(param_columns) + "---:|" * len(targets),
    ]
    for rank, trial in enumerate(_rank(trials), start=1):
        params = flat_params(trial)
        cells = [format_value(params.get(field)) for field in searched]
        cells += [_target_cell(trial, target) for target in targets]
        lines.append(
            f"| {rank} | {trial.setup} | {trial.score:.3f} | "
            f"{trial.successes}/{len(trial.scored_episodes)} | " + " | ".join(cells) + " |"
        )
    return lines


def _video_link(trial: TrialResult, run_dir: Path) -> str:
    """A markdown link to the directory holding this trial's MP4s.

    The directory rather than the clips: one trial is 20 rollouts, and twenty
    links in a table cell is not a table. Every episode of a trial records into
    the same `videos/` directory, so the link is unambiguous.

    Relative to the report, which sits in the run directory -- an absolute path
    would only open on the machine the sweep ran on.
    """
    videos = [episode.video for episode in trial.episodes if episode.video]
    if not videos:
        return "—"
    directory = Path(videos[0]).parent
    try:
        relative = directory.relative_to(run_dir)
    except ValueError:
        relative = directory
    return f"[{len(videos)} clips]({relative.as_posix()}/)"


def _stretch_section(
    trials: list[TrialResult],
    targets: tuple[str, ...],
    searched: list[str],
    run_dir: Path,
) -> list[str]:
    """The best configurations for Stretch 4, with the string to reproduce each.

    Separated from the main ranking because the Franka rows are a control, not a
    candidate: they measure what the checkpoint does on the robot it was trained
    on, and no amount of searching them produces something to run on a Stretch.
    Whoever is picking settings for the robot wants these rows and only these.
    """
    stretch = [trial for trial in trials if trial.setup.startswith("stretch") and trial.usable]
    if not stretch:
        return []

    lines = [
        "## Best configurations for Stretch 4",
        "",
        "The Franka rows above are the control -- what this checkpoint does on the robot it "
        "was trained on. These are the ones that can actually be run on a Stretch.",
        "",
        "### Best per setup",
        "",
        "One row per camera configuration: the best trial found for it. `--retarget-params` "
        "takes the description verbatim, so a row here can be replayed on a full benchmark "
        "with `run_benchmarks.py --policy molmobot_droid_retarget --retarget-setup <setup>`.",
        "",
    ]

    best_per_setup = []
    for trial in _rank(stretch):
        if trial.setup not in {t.setup for t in best_per_setup}:
            best_per_setup.append(trial)

    param_columns = [PARAM_LABELS.get(field, field) for field in searched]
    lines += [
        "| setup | picked | score | " + " | ".join(param_columns + list(targets)) + " |",
        "|---|---:|---:|" + "---:|" * (len(param_columns) + len(targets)),
    ]
    for trial in best_per_setup:
        params = flat_params(trial)
        cells = [format_value(params.get(field)) for field in searched]
        cells += [_target_cell(trial, target) for target in targets]
        lines.append(
            f"| {trial.setup} | {trial.successes}/{len(trial.scored_episodes)} | "
            f"{trial.score:.3f} | " + " | ".join(cells) + " |"
        )

    lines += ["", "The same rows as parameter strings:", ""]
    for trial in best_per_setup:
        lines += [
            f"**{trial.setup}** — {trial.successes}/{len(trial.scored_episodes)} picked",
            "",
            "```",
            trial.params_description,
            "```",
            "",
        ]

    top = _rank(stretch)[:10]
    lines += [
        "### Top 10 Stretch trials overall",
        "",
        "| rank | setup | picked | score | " + " | ".join(param_columns) + " | videos |",
        "|---:|---|---:|---:|" + "---:|" * len(param_columns) + "---|",
    ]
    for rank, trial in enumerate(top, start=1):
        params = flat_params(trial)
        lines.append(
            f"| {rank} | {trial.setup} | {trial.successes}/{len(trial.scored_episodes)} | "
            f"{trial.score:.3f} | "
            + " | ".join(format_value(params.get(field)) for field in searched)
            + f" | {_video_link(trial, run_dir)} |"
        )

    # What each searched parameter is worth on Stretch, marginalised over the rest.
    lines += ["", "### What each parameter is worth on Stretch", "",
              "Mean objects picked per trial, grouped by one parameter at a time. Every other "
              "parameter varies underneath each row, so these are marginals, not a recipe -- "
              "but a parameter whose rows are flat is one that did not matter.", ""]
    for field_name in searched:
        values: dict[Any, list[TrialResult]] = {}
        for trial in stretch:
            values.setdefault(flat_params(trial).get(field_name), []).append(trial)
        if len(values) < 2:
            continue
        lines += [
            f"**{PARAM_LABELS.get(field_name, field_name)}** (`{field_name}`)",
            "",
            PARAM_EXPLANATIONS.get(field_name, ""),
            "",
            "| value | trials | mean picked | mean score |",
            "|---|---:|---:|---:|",
        ]
        for value, group in sorted(values.items(), key=lambda kv: str(kv[0])):
            mean_picked = float(np.mean([t.successes for t in group]))
            mean_score = float(np.mean([t.score for t in group]))
            lines.append(
                f"| {format_value(value)} | {len(group)} | {mean_picked:.2f} | {mean_score:.3f} |"
            )
        lines.append("")
    return lines


def write_report(trials: list[TrialResult], targets: tuple[str, ...], output_dir: Path) -> Path:
    """Write `report.md` beside the CSVs, and return its path.

    Markdown because the interesting output of this search is a comparison
    somebody reads, not a number a program consumes -- the CSVs are there for
    the latter.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    searched, setup_defining, constant = parameter_groups(trials)
    ranked = _rank(trials)
    scenes = sorted({episode.scene for trial in trials for episode in trial.episodes})
    per_trial = max((len(trial.episodes) for trial in trials), default=0)

    lines = [
        "# Retargeting parameter search",
        "",
        f"{len(trials)} trials. Each is one setup at one point in the parameter space, "
        f"scored over {len(targets)} objects ({', '.join(targets)}) in "
        f"{len(scenes)} scene(s) — {per_trial} rollouts per trial.",
        "",
        "Ranked by **objects picked, then score**. `score` is the mean episode score: 1.0 "
        "for a grasp, and for a failure a weighted sum of how far the gripper closed the gap "
        f"to the object ({APPROACH_WEIGHT}), whether it touched it ({CONTACT_WEIGHT}) and how "
        f"far it lifted it ({LIFT_WEIGHT}). See `scoring.py`. The score is the right thing for "
        "a search to climb and the wrong thing to read a winner off, because a trial that "
        "misses four times by a millimetre can outscore one that picks three things up.",
        "",
        "Object columns are **picked / attempted** across scenes, bold when every scene "
        "succeeded; a `0/n (0.42)` cell gives the mean partial score instead.",
        "",
    ]

    if searched:
        lines += [
            "## What varied",
            "",
            "Only the parameters the sweep actually moved get a column below. "
            + ", ".join(f"`{PARAM_LABELS.get(f, f)}`" for f in searched)
            + ".",
            "",
        ]
    if setup_defining:
        lines += [
            "Constant within each setup but different between them — these are what the "
            "`setup` column *means*, not results:",
            "",
            "| setup | " + " | ".join(PARAM_LABELS.get(f, f) for f in setup_defining) + " |",
            "|---|" + "---|" * len(setup_defining),
        ]
        seen: dict[str, dict[str, Any]] = {}
        for trial in trials:
            if not trial.error:
                seen.setdefault(trial.setup, flat_params(trial))
        for setup, params in seen.items():
            lines.append(
                f"| {setup} | "
                + " | ".join(format_value(params.get(f)) for f in setup_defining)
                + " |"
            )
        lines.append("")
    if constant:
        lines += [
            "Held constant across every trial: "
            + ", ".join(
                f"`{PARAM_LABELS.get(f, f)}={format_value(v)}`" for f, v in constant.items()
            )
            + ".",
            "",
        ]

    lines += ["## All trials", ""]
    lines += _summary_table(trials, targets, searched)

    errored = [trial for trial in trials if trial.error or not trial.usable]
    if errored:
        lines += ["", "### Trials that measured nothing", "", "| setup | parameters | why |", "|---|---|---|"]
        for trial in errored:
            why = trial.error or "every episode crashed"
            lines.append(f"| {trial.setup} | `{trial.params_description}` | {why} |")

    crashed = sum(trial.incomplete for trial in trials)
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

    lines += [""]
    lines += _stretch_section(trials, targets, searched, output_dir)

    lines += ["", "## Episodes", ""]
    for trial in ranked:
        if trial.error:
            continue
        lines.append(
            f"### {trial.setup} — {trial.successes}/{len(trial.scored_episodes)} picked, "
            f"score {trial.score:.3f}"
        )
        lines.append("")
        lines.append(f"`{trial.params_description}`")
        lines.append("")
        lines.append(
            "| scene | object | outcome | approach | touched | best lift | retarget pos err | video |"
        )
        lines.append("|---:|---|---|---:|---:|---:|---:|---|")
        for episode in sorted(trial.episodes, key=lambda e: (e.scene, e.target)):
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
                f"| {episode.scene} | {episode.target} | {outcome} | {episode.approach:.2f} | "
                f"{'yes' if episode.touched else 'no'} | {episode.best_lift_m * 100:.1f} cm | "
                f"{residual} | {video} |"
            )
        lines.append("")

    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# =============================================================================
# Rebuilding a finished run
# =============================================================================


def load_trials(run_dir: Path) -> list[TrialResult]:
    """Reconstruct a finished sweep's trials from the files it wrote.

    `write_report` runs at the end of a search, from objects that only exist in
    that process -- so changing how the report is laid out would otherwise mean
    re-running the rollouts to see the new layout, which for this sweep is nine
    hours. `episodes.csv` and `trials.jsonl` between them hold everything the
    report reads, so it can be rebuilt instead.

    `trial` in the CSV is the index into `trials.jsonl`; both are written
    together by `params_search._write_outputs` after every trial.
    """
    trials_path = run_dir / "trials.jsonl"
    records = [json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()]

    episodes: dict[int, list[EpisodeScore]] = {}
    descriptions: dict[int, str] = {}
    with (run_dir / "episodes.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["trial"])
            episodes.setdefault(index, []).append(_episode_from_row(row))
            descriptions.setdefault(index, row.get("params", ""))

    trials = []
    for index, record in enumerate(records):
        params = record.get("params", {})
        trials.append(
            TrialResult(
                setup=record.get("setup", ""),
                params_description=descriptions.get(index, ""),
                params_json=json.dumps(params),
                episodes=episodes.get(index, []),
                error=record.get("error", ""),
            )
        )
    return trials


def _episode_from_row(row: dict[str, str]) -> EpisodeScore:
    """One `episodes.csv` row back into an `EpisodeScore`.

    Only the recorded fields are restored; `score`, `approach`, `contact` and
    `lift` are properties and are recomputed from them, which is also a check
    that the two agree.
    """

    def number(key: str, default: float = float("nan")) -> float:
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            return default

    def flag(key: str) -> bool:
        return str(row.get(key, "")).strip().lower() in ("true", "1", "yes")

    return EpisodeScore(
        setup=row.get("setup", ""),
        target=row.get("target", ""),
        scene=int(number("scene", -1)) if row.get("scene") else -1,
        instruction=row.get("instruction", ""),
        success=flag("success"),
        steps=int(number("steps", 0)),
        completed=flag("completed"),
        start_distance_m=number("start_distance_m"),
        min_distance_m=number("min_distance_m"),
        touched=flag("touched"),
        best_lift_m=number("best_lift_m", 0.0),
        retarget_position_error_mean_m=number("retarget_position_error_mean_m"),
        retarget_orientation_error_mean_rad=number("retarget_orientation_error_mean_rad"),
        video=row.get("video", ""),
    )


def rebuild_report(run_dir: Path, targets: tuple[str, ...]) -> Path:
    """Re-render `report.md` for a finished run, without re-running anything."""
    trials = load_trials(run_dir)
    return write_report(trials, targets, run_dir)


@click.command()
@click.option(
    "--run-dir",
    type=click.Path(path_type=Path, exists=True),
    default=Path("eval_output") / "retarget_params",
    help="A directory `params_search` wrote: trials.jsonl and episodes.csv.",
)
def main(run_dir: Path) -> None:
    """Re-render `report.md` for a finished run, without re-running the rollouts."""
    from examples.machine_learning.molmospaces.retargetting import mini_benchmark

    targets = tuple(target.key for target in mini_benchmark.TARGETS)
    path = rebuild_report(run_dir, targets)
    click.secho(f"Wrote {path}", fg="green")


if __name__ == "__main__":
    main()
