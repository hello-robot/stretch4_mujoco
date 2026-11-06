"""
Search for Stretch 4's grasp offset (`--grasp-offset-mm`) on the real robot, with the policy in
the loop: MolmoBot-DROID tries to pick up the ball at one offset after another, you say how each
attempt went, and the next offset to try is worked out from that.

Each trial:

1. Stretch waits at the start pose for the trial's offset (where a fresh run of
   run_stretch4_real.py with that offset would start: the virtual Franka's pedestal and yaw are
   worked out again for it). Put the ball in place; Enter runs the instruction. Type `x y z`
   (mm) instead to try another offset: Stretch moves to its start pose first. Any key stops a
   move to a start pose; what was typed is kept for the prompt. A start pose Stretch did not
   reach is reported, with the joints that are off.
2. Enter or Space stops it, when it has grasped the ball or clearly will not.
3. Say how it went, with Stretch still where it stopped: `y` it grasped the ball, `s` skip (the
   attempt says nothing about the offset), or `n` it missed, with what went wrong:

       f      the gripper went too far: past the ball, into the table   → less x
       s      the gripper stopped short of the ball                     → more x
       l / r  the ball was left / right of the gap between the fingers  → more / less y
       u / d  the ball was above / below the gap                        → more / less z

   left/right/above/below as seen in the wrist view (Rerun's `wrist`, or `gripper_raw`). E.g.
   `n f`, `n f u`, or `n` alone when it missed for some other reason. Anything after that is
   kept as a note. Typing the answer instead of Enter/Space also stops the robot.
4. Stretch goes back to the start pose for the next offset, holding the ball if it has it;
   Enter then opens the gripper, and Enter again starts the next trial.

The next offset moves each axis that was off: by FIRST_STEP_MM, doubling while it stays off the
same way, and by bisection once one attempt was off one way and another the other. An attempt that grasped the ball, or missed with no direction, is tried again at the same
offset. The directions are the open-loop ones: an offset moves Stretch's tool, not where the
policy steers it, so it takes a few trials to settle.

Every trial goes to `trials.jsonl` as it ends; Ctrl+C (or `q` at the suggestion) stops the
robot and writes `report.md` next to it. `--report <dir>` writes the report again from a
directory's `trials.jsonl`, without the robot.

Usage:
    python -m examples.vla.molmobot_droid.franka_retarget.test.measure_gripper_offset
    python -m examples.vla.molmobot_droid.franka_retarget.test.measure_gripper_offset --grasp-offset-mm 33,21,17
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import click
import numpy as np

from examples.vla.molmobot_droid import rerun_scene
from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS, GRIPPER_CLOSED, load_policy, unload_policy
from examples.vla.molmobot_droid.droid import RobotPose, franka_link0_height_for_object
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    CUSTOM_START_QPOS,
    custom_start_link0_height,
    franka_yaw_for_start,
    params_from_kwargs,
    retarget_options,
)
from examples.vla.molmobot_droid.rollout import StepInfo, run_rollout
from examples.vla.molmobot_droid.run_stretch4_real import (
    Console,
    ImageAndJointReceiver,
    LiveView,
    RealStretch4Env,
    RobotModel,
    connect,
)

INSTRUCTION = "Pick up the yellow tennis ball"
OUTPUT_ROOT = Path("outputs/molmobot_droid/gripper_offset")

DEFAULTS = {
    "execute_first_n": 8,
    "wait_for_arrival": False,
    "grasp_offset_mm": "4,0,0",
}
"""This script's defaults for `retarget_options()` flags (the start guess lines the fingertips
up). The flags can still be given."""

AXES = ("x", "y", "z")

DIRECTIONS = {
    "f": (0, -1, "gripper went too far"),
    "s": (0, +1, "gripper stopped short"),
    "l": (1, +1, "ball left of the gap"),
    "r": (1, -1, "ball right of the gap"),
    "u": (2, +1, "ball above the gap"),
    "d": (2, -1, "ball below the gap"),
}
"""What went wrong → (axis, sign) of the change to the offset that moves Stretch's tool to put
it right: more x takes the tool further along the approach, and in the gripper camera's view
(both of them, from the PG4 model) +y is to the image's left and +z up."""

FIRST_STEP_MM = (10.0, 6.0, 6.0)
"""How far an axis moves past its bound the first time it is off (x, y, z). It doubles each time a
trial pushes the bound further the same way; a trial off the same way but not past the bound
(an earlier, larger guess) leaves it."""

RESOLUTION_MM = 1.0
"""An axis bracketed this tightly is done: bisecting further is below what one trial can tell."""

OFFSET_LIMITS_MM = ((-40.0, 100.0), (-40.0, 40.0), (-40.0, 40.0))
"""Suggestions stay inside these (x, y, z); a suggestion held at one is reported."""

START_POSE_TOLERANCE_M = 0.01
START_POSE_TOLERANCE_RAD = math.radians(3)
"""How close lift and arm (m) and the wrist (rad) have to get to the start pose to count as there."""

OUTCOMES = {"y": "grasped", "n": "missed", "s": "skipped"}

MAX_DIRECTION_WORD = 3
"""After `n`, a word this short is taken for directions, and refused if it has other letters: a
note is longer."""


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


@dataclass
class AxisSearch:
    """One axis of the offset: the values found too low and too high so far, and the step."""

    first_step: float
    limits: tuple[float, float]
    too_low: float | None = None
    """The largest value the ball was off on the + side at: the offset should be above it."""
    too_high: float | None = None
    step: float = 0.0
    last_sign: int = 0

    def update(self, value: float, sign: int) -> tuple[float, str | None]:
        """The next value after the ball was off on the `sign` side at `value`, and a note when
        that contradicts the bracket (which then starts again from this trial)."""
        note = None
        bound = self.too_low if sign > 0 else self.too_high
        moved = bound is None or (value > bound if sign > 0 else value < bound)
        if sign > 0:
            self.too_low = value if self.too_low is None else max(self.too_low, value)
        else:
            self.too_high = value if self.too_high is None else min(self.too_high, value)
        if self.too_low is not None and self.too_high is not None and self.too_low >= self.too_high:
            note = f"contradicts an earlier trial (too low at {self.too_low:g}, too high at {self.too_high:g}); bracket reset"
            self.too_low, self.too_high = (value, None) if sign > 0 else (None, value)
        if self.too_low is not None and self.too_high is not None:
            next_value = (self.too_low + self.too_high) / 2
        else:
            # On from the one bound there is (the tightest so far, not necessarily `value`), by a
            # step that doubles each time a trial pushes that bound further the same way.
            if not self.step or sign != self.last_sign:
                self.step = self.first_step
            elif moved:
                self.step *= 2
            next_value = (self.too_low if sign > 0 else self.too_high) + sign * self.step
        self.last_sign = sign
        clipped = float(np.clip(next_value, *self.limits))
        if clipped != next_value and clipped == value:
            limit = f"{clipped:g} mm, the {'top' if sign > 0 else 'bottom'} of the search range"
            note = (note + "; " if note else "") + (
                f"at {limit}, and still off the same way: the offset cannot go further, and moving it "
                "this far has not changed which way it is off")
        return clipped, note

    @property
    def settled(self) -> bool:
        return self.too_low is not None and self.too_high is not None and self.too_high - self.too_low <= 2 * RESOLUTION_MM

    def describe(self) -> str:
        low = "…" if self.too_low is None else f"{self.too_low:g}"
        high = "…" if self.too_high is None else f"{self.too_high:g}"
        return f"({low}, {high})" + (" settled" if self.settled else "")


@dataclass
class Trial:
    number: int
    offset_mm: tuple[float, float, float]
    started: str
    outcome: str = "interrupted"
    """grasped, missed, skipped, or interrupted (Ctrl+C during it)."""
    directions: str = ""
    note: str = ""
    steps: int = 0
    seconds: float = 0.0
    link0_height_m: float = 0.0
    franka_yaw_deg: float = 0.0
    ik_twisted: int = 0
    ik_unreachable: int = 0
    ik_clamped: int = 0
    recoveries: int = 0
    stale_observations: int = 0
    close_step: int | None = None
    """The step the policy first closed the gripper at."""
    tool_at_close_m: list[float] | None = None
    """Where Stretch's tool was then, in the world (the start footprint's frame)."""
    joints_at_close: dict[str, float] | None = None
    next_offset_mm: tuple[float, float, float] | None = None
    search_notes: list[str] = field(default_factory=list)


class OffsetSearch:
    """The suggestions, from the trials so far."""

    def __init__(self, start_mm):
        self.axes = [AxisSearch(step, limits) for step, limits in zip(FIRST_STEP_MM, OFFSET_LIMITS_MM)]
        self.next_mm = tuple(float(v) for v in start_mm)

    def record(self, trial: Trial) -> list[str]:
        """Take `trial`'s outcome into account; returns notes on contradictions."""
        notes = []
        next_mm = list(trial.offset_mm)
        if trial.outcome == "missed":
            for letter in dict.fromkeys(trial.directions):  # each direction once, in order
                if letter not in DIRECTIONS:
                    continue
                axis, sign, _ = DIRECTIONS[letter]
                next_mm[axis], note = self.axes[axis].update(trial.offset_mm[axis], sign)
                if note:
                    notes.append(f"{AXES[axis]}: {note}")
        self.next_mm = tuple(float(round(v)) for v in next_mm)
        return notes

    def describe(self) -> str:
        return " · ".join(f"{name} {axis.describe()}" for name, axis in zip(AXES, self.axes))


def parse_outcome(text: str) -> tuple[str, str, str] | None:
    """`y [note]`, `s [note]`, `n [directions] [note]` → (outcome, directions, note)."""
    words = text.split()
    if not words or words[0][0].lower() not in OUTCOMES:
        return None
    outcome = OUTCOMES[words[0][0].lower()]
    rest = [words[0][1:]] + words[1:] if len(words[0]) > 1 else words[1:]
    rest = [word for word in rest if word]
    directions = ""
    while outcome == "missed" and rest and all(c in DIRECTIONS for c in rest[0].lower()):
        directions += rest.pop(0).lower()
    if outcome == "missed" and rest and len(rest[0]) <= MAX_DIRECTION_WORD and rest[0].isalpha():
        return None  # e.g. 'n c u': not a direction, and too short to be the start of a note
    return outcome, directions, " ".join(rest)


def parse_offset(text: str) -> tuple[float, float, float] | None:
    try:
        values = [float(v) for v in text.replace(",", " ").split()]
    except ValueError:
        return None
    return tuple(values) if len(values) == 3 else None


def describe_direction(letter: str) -> str:
    return DIRECTIONS[letter][2] if letter in DIRECTIONS else f"'{letter}'"


def format_offset(offset_mm) -> str:
    return ", ".join(f"{v:g}" for v in offset_mm)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def load_trials(directory: Path) -> list[Trial]:
    path = directory / "trials.jsonl"
    if not path.exists():
        return []
    return [Trial(**{**row, "offset_mm": tuple(row["offset_mm"])}) for row in map(json.loads, path.read_text().splitlines())]


def write_report(directory: Path, session: dict, trials: list[Trial], search: OffsetSearch | None) -> Path:
    """`report.md` in `directory`: what each offset did, and which to use."""
    counted = [t for t in trials if t.outcome in ("grasped", "missed")]
    by_offset: dict[tuple, list[Trial]] = {}
    for trial in counted:
        by_offset.setdefault(tuple(trial.offset_mm), []).append(trial)

    def rate(group: list[Trial]) -> float:
        return sum(t.outcome == "grasped" for t in group) / len(group)

    # The most reliable offset: the best success rate, more trials breaking a tie.
    ranked = sorted(by_offset.items(), key=lambda item: (rate(item[1]), len(item[1])), reverse=True)
    lines = [
        "# Grasp offset search",
        "",
        f"{session.get('started', '')} · Stretch 4 at {session.get('robot_ip', '?')} · {session.get('tool', '?')}"
        f" · \"{session.get('instruction', INSTRUCTION)}\"",
        "",
        "Flags: " + " ".join(f"`{flag}`" for flag in session.get("flags", [])),
        "",
        "Offsets are `--grasp-offset-mm` x, y, z: x along the approach, y between the fingers, z across.",
        "",
        "## Result",
        "",
    ]
    if ranked:
        best, group = ranked[0]
        grasped = sum(t.outcome == "grasped" for t in group)
        lines.append(f"**Most reliable so far: `--grasp-offset-mm {','.join(f'{v:g}' for v in best)}`**, "
                     f"{grasped} of {len(group)} grasped.")
        if len(group) < 3:
            lines.append("")
            lines.append("Fewer than 3 attempts there: try it a few more times before relying on it.")
    else:
        lines.append("No attempt was counted yet.")
    if search is not None:
        lines += ["", f"Search brackets (mm, too low … too high): {search.describe()}",
                  f"Next suggestion: `{format_offset(search.next_mm)}`"]
    lines += ["", "## By offset", "", "| Offset (mm) | Attempts | Grasped | What went wrong (misses) |", "|---|---|---|---|"]
    for offset, group in ranked:
        tally: dict[str, int] = {}
        for trial in group:
            for letter in trial.directions:
                tally[letter] = tally.get(letter, 0) + 1
        where = ", ".join(f"{describe_direction(letter)} ×{count}" for letter, count in tally.items()) or "–"
        grasped = sum(t.outcome == "grasped" for t in group)
        lines.append(f"| {format_offset(offset)} | {len(group)} | {grasped} ({100 * grasped / len(group):.0f} %) | {where} |")
    lines += [
        "",
        "## Trials",
        "",
        "| # | Started | Offset (mm) | Outcome | What went wrong | Steps | s | Closed at step | Twisted / unreachable / recovered | Note |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in trials:
        where = ", ".join(describe_direction(letter) for letter in t.directions) or "–"
        notes = "; ".join([t.note] + t.search_notes if t.note else t.search_notes)
        lines.append(
            f"| {t.number} | {t.started[11:19]} | {format_offset(t.offset_mm)} | {t.outcome} | {where} | {t.steps} | "
            f"{t.seconds:.0f} | {t.close_step if t.close_step is not None else '–'} | "
            f"{t.ik_twisted} / {t.ik_unreachable} / {t.recoveries} | {notes} |"
        )
    lines += [
        "",
        "Too far / short → x down / up; the ball left / right of the gap → y up / down; above / below it → "
        "z up / down (left/right/above/below as seen in the wrist view).",
        "",
    ]
    path = directory / "report.md"
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# On the robot
# ---------------------------------------------------------------------------


def set_franka_yaw(robot_model: RobotModel, yaw: float) -> None:
    """Turn the virtual Franka to `yaw` about the start footprint: for retargeting (its
    `FrankaSpawn`, which the retargeter shares) and its ghost (a mocap body)."""
    spawn = robot_model.franka
    spawn.robot_pose = RobotPose(spawn.robot_pose.x, spawn.robot_pose.y, yaw)
    model = robot_model.model
    base = model.body(spawn.base_name)
    model.body_quat[base.id] = spawn.robot_pose.quat_wxyz
    if base.mocapid[0] >= 0:
        robot_model.data.mocap_quat[base.mocapid[0]] = spawn.robot_pose.quat_wxyz


def apply_offset(env: RealStretch4Env, robot_model: RobotModel, offset_mm, custom_start: bool, link0_height: float,
                 start_q7) -> tuple[float, float]:
    """
    Use `offset_mm` from now on, with the virtual Franka where run_stretch4_real.py would put it
    for that offset: with --custom_franka_start_pose its pedestal depends on the offset, and its
    yaw (so the start pose needs no base turn) always does. Returns (link0 height, yaw).
    """
    env.retargeter.set_grasp_offset(offset_mm)
    if custom_start:
        link0_height = custom_start_link0_height(env.params, floor_z=0.0)
        robot_model.set_franka_pedestal_height(link0_height)
    yaw = franka_yaw_for_start(env.params, link0_height, start_q7)
    set_franka_yaw(robot_model, yaw)
    return link0_height, yaw


class Session:
    """The trials on the robot: running them, asking how they went, keeping the record."""

    def __init__(self, env: RealStretch4Env, robot_model: RobotModel, policy, console: Console, directory: Path,
                 search: OffsetSearch, instruction: str, max_steps: int, custom_start: bool, link0_height: float,
                 start_q7, rerun: bool, log_frame):
        self.env, self.robot_model, self.policy, self.console = env, robot_model, policy, console
        self.directory, self.search, self.instruction, self.max_steps = directory, search, instruction, max_steps
        self.custom_start, self.link0_height, self.start_q7 = custom_start, link0_height, start_q7
        self.rerun, self.log_frame = rerun, log_frame
        self.trials: list[Trial] = load_trials(directory)
        """Those of an `--output` directory continued, then this session's."""
        for trial in self.trials:
            search.record(trial)
        self.current: Trial | None = None
        """The trial running or being asked about: saved as it is if Ctrl+C cuts it short."""
        self.home: tuple[tuple, float, float] | None = None
        """(offset, link0 height, yaw) of the start pose Stretch is at; None when it is elsewhere."""
        self.typed_ahead: tuple[str, str | None] | None = None
        """What was typed while Stretch was moving to a start pose: the next answer to `ask()`."""

    def say(self, message: str, color: str | None = None) -> None:
        self.console.event(message, color)
        if self.rerun:
            rerun_scene.log_text(f"{rerun_scene.LOG_PATH}/offset", message)

    def ask(self, prompt: str) -> tuple[str, str | None]:
        if self.typed_ahead is not None:
            command, self.typed_ahead = self.typed_ahead, None
            return command
        self.console.status(click.style(prompt, fg="cyan"))
        command = self.console.get()
        self.console.status("")
        return command

    def save(self, trial: Trial) -> None:
        with (self.directory / "trials.jsonl").open("a") as file:
            file.write(json.dumps(asdict(trial)) + "\n")

    # -- one trial -----------------------------------------------------------

    def run(self) -> None:
        """Start pose, wait, run, ask, record; until `q` (or Ctrl+C)."""
        offset = self.search.next_mm
        while True:
            if self.home is None or self.home[0] != tuple(offset):
                self.go_home(offset)
            holding = bool(self.env._gripper_closed)
            self.say(f"Trial {len(self.trials) + 1}: offset {format_offset(offset)} mm"
                     + (" · Stretch is at its start pose; reset the ball" if self.home else ""), "cyan")
            if self.home is None:
                prompt = "Enter: go to the start pose · x y z: another offset · q: finish › "
            elif holding:
                prompt = "Enter: open the gripper (take the ball) · x y z: another offset · q: finish › "
            else:
                prompt = "Enter: run it · x y z: another offset · q: finish › "
            kind, text = self.ask(prompt)
            if kind == "text":
                if text in ("q", "quit", "exit"):
                    return
                typed = parse_offset(text)
                if typed is None:
                    self.say(f"Not an offset: '{text}' (three numbers, mm)", "red")
                else:
                    offset = typed  # its start pose next, then asked again
                continue
            if self.home is None:
                continue  # to the start pose, at the top of the loop
            if holding:
                self.env.send_gripper(False)
                self.env.robot.push_command()
                continue
            trial = self.run_trial(offset)
            self.home = None  # Stretch is where the trial left it
            self.ask_outcome(trial)
            notes = self.search.record(trial)
            trial.search_notes = notes
            trial.next_offset_mm = self.search.next_mm
            self.current = None
            self.trials.append(trial)
            self.save(trial)
            for note in notes:
                self.say(f"  {note}", "yellow")
            self.say(f"  {trial.outcome}" + (f" ({trial.directions})" if trial.directions else "")
                     + f" · brackets {self.search.describe()}", "green" if trial.outcome == "grasped" else "yellow")
            offset = self.search.next_mm

    def go_home(self, offset) -> None:
        """To the start pose for `offset`, the gripper as it is (holding the ball, if it has it)."""
        env = self.env
        link0_height, yaw = apply_offset(env, self.robot_model, offset, self.custom_start, self.link0_height, self.start_q7)
        self.say(f"  to the start pose for {format_offset(offset)} mm: fr3_link0 at {link0_height:.3f} m, "
                 f"turned {math.degrees(yaw):.1f}°", "white")
        self.console.status(click.style("◆ going to the start pose", fg="yellow"))
        try:
            env.go_home(keep_gripper=True)
        except RuntimeError as error:
            self.say(f"  {error}", "red")
            return
        finally:
            self.console.status("")
        if self.console.pending():  # a key cut the move short
            kind, text = self.console.get()
            env.stop_motion()
            if kind == "text":  # an offset, or q: for the prompt
                self.typed_ahead = (kind, text)
            self.say("  stopped on the way to the start pose", "yellow")
            return
        off = self.off_start_pose()
        if off:
            self.say(f"  did not reach the start pose ({off}); Enter tries again, or type another offset", "red")
            return
        self.home = (tuple(offset), link0_height, yaw)
        self.log_frame(env.observe())

    def off_start_pose(self) -> str:
        """The joints not where the start pose put them, if any (a move timed out, or a guarded
        contact stopped a joint)."""
        targets = self.env.retargeter.last_targets
        if targets is None:
            return ""
        live = self.env.live_joint_state()
        now = {"lift": live["lift"]["height"], "arm": live["arm"]["extension"],
               **{joint: live[joint]["angle"] for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll")}}
        off = []
        for name, value in now.items():
            target = getattr(targets, name)
            if name in ("lift", "arm") and abs(value - target) > START_POSE_TOLERANCE_M:
                off.append(f"{name} at {value * 1000:.0f} of {target * 1000:.0f} mm")
            elif name not in ("lift", "arm") and abs(value - target) > START_POSE_TOLERANCE_RAD:
                off.append(f"{name} at {math.degrees(value):.0f}° of {math.degrees(target):.0f}°")
        return ", ".join(off)

    def run_trial(self, offset) -> Trial:
        """The instruction, from the start pose Stretch is at, until a key stops it."""
        env, retargeter = self.env, self.env.retargeter
        _, link0_height, yaw = self.home
        trial = Trial(len(self.trials) + 1, tuple(offset), datetime.now().isoformat(timespec="seconds"),
                      link0_height_m=link0_height, franka_yaw_deg=math.degrees(yaw))
        self.current = trial
        recoveries, stale = env.recoveries, env.stale_observations
        start = time.monotonic()

        def on_step(info: StepInfo) -> None:
            self.log_frame(info.observation)
            if trial.close_step is None and info.action[7] >= GRIPPER_CLOSED / 2:
                footprint, joints = env._state(env.last_message)
                trial.close_step = info.step
                trial.tool_at_close_m = [round(v, 4) for v in retargeter.stretch_tool_world(footprint, joints)[:3, 3]]
                trial.joints_at_close = {k: round(v, 4) for k, v in asdict(joints).items()}
                self.say(f"  gripper closing at step {info.step}", "white")
            self.console.status(
                click.style(f"▶ trial {trial.number} · {format_offset(offset)} mm", fg="green")
                + click.style(
                    f"  step {info.step}"
                    + (f" · {retargeter.ik_twisted} twisted" if retargeter.ik_twisted else "")
                    + (f" · {retargeter.ik_failures} unreachable" if retargeter.ik_failures else "")
                    + "  · Enter/Space stops (or type the outcome)",
                    dim=True,
                )
            )

        self.say(f"▶ {self.instruction}", "green")
        result = run_rollout(env, self.policy, self.instruction, env.params.execute_horizon, env.params.execute_first_n,
                             self.max_steps if self.max_steps > 0 else 10**12, on_step, should_stop=self.console.pending)
        env.stop_motion()
        trial.steps, trial.seconds = result.steps, round(time.monotonic() - start, 1)
        trial.ik_twisted, trial.ik_unreachable, trial.ik_clamped = (
            retargeter.ik_twisted, retargeter.ik_failures, retargeter.ik_clamped)
        trial.recoveries = env.recoveries - recoveries
        trial.stale_observations = env.stale_observations - stale
        # What stopped it: a bare Enter/Space, or the outcome typed straight away.
        if self.console.pending():
            kind, text = self.console.get()
            if kind == "text" and (parsed := parse_outcome(text)) is not None:
                trial.outcome, trial.directions, trial.note = parsed
                return trial
        trial.outcome = ""
        return trial

    def ask_outcome(self, trial: Trial) -> None:
        if trial.outcome:
            return
        self.say("  y grasped · s skip · n missed, + what went wrong: f gripper went too far, s stopped short, "
                 "l/r/u/d ball left/right/above/below the gap in the wrist view (e.g. 'n f u'); then an optional note",
                 "white")
        while True:
            kind, text = self.ask("How did it go? › ")
            parsed = parse_outcome(text) if kind == "text" else None
            if parsed is not None:
                trial.outcome, trial.directions, trial.note = parsed
                return
            self.say("  y, n [f s l r u d] [note], or s: directions are f s l r u d only", "red")


@click.command(context_settings={"default_map": DEFAULTS})
@click.option("--robot_ip", "--robot-ip", "robot_ip", default="100.71.110.52", show_default=True)
@click.option("--object-height", type=float, default=0.9, show_default=True,
              help="Height of the ball above the floor (m); places the virtual Franka without "
              "--custom_franka_start_pose.")
@click.option("--custom_franka_start_pose/--no-custom_franka_start_pose", "custom_franka_start_pose", default=True,
              show_default="on", help="As run_stretch4_real.py's --custom_franka_start_pose.")
@click.option("--instruction", default=INSTRUCTION, show_default=True)
@click.option("--max-steps", type=int, default=0, show_default=True,
              help="Stop a trial after this many steps (15/s). 0: when Enter or Space is pressed.")
@click.option("--output", type=click.Path(path_type=Path), default=None,
              help=f"Directory for trials.jsonl and report.md (default: a new one under {OUTPUT_ROOT}). "
              "An existing one is continued.")
@click.option("--report", "report_only", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Only write the report for this directory's trials.jsonl, without the robot.")
@click.option("--checkpoint", default=None)
@click.option("--rerun/--no-rerun", default=True, show_default=True)
@retarget_options(gripper_option=False)
def main(robot_ip, object_height, custom_franka_start_pose, instruction, max_steps, output, report_only,
         checkpoint, rerun, **kwargs):
    if report_only is not None:
        session = json.loads((report_only / "session.json").read_text()) if (report_only / "session.json").exists() else {}
        path = write_report(report_only, session, load_trials(report_only), None)
        click.echo(f"Wrote {path}")
        return
    params = params_from_kwargs(kwargs)
    if params.exo_camera == "droid":
        raise click.BadParameter("the real robot has no DROID exo camera; use left, right or center", param_hint="--exo_camera")

    directory = output or OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    directory.mkdir(parents=True, exist_ok=True)

    click.secho(f"Connecting to Stretch 4 at {robot_ip}...", dim=True)
    robot, gripper = connect(robot_ip)
    if not robot.is_homed():
        robot.stop()
        raise click.ClickException("Home the robot first")
    params.use_parallel_gripper = gripper == "parallel_gripper"
    start_mm = params.effective_grasp_offset_mm
    start_q7 = np.array(CUSTOM_START_QPOS if custom_franka_start_pose else FRANKA_HOME_QPOS)
    link0_height = (custom_start_link0_height(params, floor_z=0.0) if custom_franka_start_pose
                    else franka_link0_height_for_object(object_height))
    robot_model = RobotModel(link0_height, params.tool_name, franka_yaw_for_start(params, link0_height, start_q7))
    session = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "robot_ip": robot_ip,
        "tool": params.tool_name,
        "instruction": instruction,
        "flags": [f"--object-height {object_height:g}"] * (not custom_franka_start_pose)
        + ["--custom_franka_start_pose"] * custom_franka_start_pose
        + [f"--execute-horizon {params.execute_horizon}", f"--execute-horizon-do-only-first-n-steps {params.execute_first_n}",
           "--wait-for-arrival" if params.wait_for_arrival else "--no-wait-for-arrival",
           f"--exo_camera {params.exo_camera}", f"--gripper_camera {params.gripper_camera}",
           f"--head-crop {params.head_crop}", f"--grasp-offset-deg {','.join(f'{v:g}' for v in params.grasp_offset_deg)}"]
        + ["--slow"] * params.slow,
        "start_offset_mm": list(start_mm),
    }
    (directory / "session.json").write_text(json.dumps(session, indent=2))

    receiver = ImageAndJointReceiver(robot_ip)
    env = live = console = runner = None
    search = OffsetSearch(start_mm)
    try:
        env = RealStretch4Env(robot, receiver, params, robot_model.franka)
        env.start_q7 = start_q7
        click.echo(click.style("Stretch 4 ", bold=True) + f"{robot_ip} · {gripper.replace('_', ' ')}\n"
                   + click.style("Search   ", bold=True) + f"from {format_offset(start_mm)} mm · \"{instruction}\"\n"
                   + click.style("Saving   ", bold=True) + str(directory))
        step_count = 0
        if rerun:
            rerun_scene.init_rerun("MolmoBot-DROID grasp offset", ["exo", "wrist", "head_raw", "gripper_raw"],
                                   text_log=True, main_cameras=["exo", "wrist"])
            live = LiveView(env, robot_model, step=lambda: step_count)

        def log_frame(observation) -> None:
            nonlocal step_count
            step_count += 1
            if not rerun:
                return
            rerun_scene.set_step(step_count)
            rerun_scene.rr.set_time("time", timestamp=time.time())
            rerun_scene.log_cameras({"exo": observation.exo_rgb, "wrist": observation.wrist_rgb})
            footprint, joints = env._state(env.last_message)
            rerun_scene.log_tool_poses(env.retargeter.last_target_tool_world, env.retargeter.stretch_tool_world(footprint, joints))

        click.secho("Loading MolmoBot-DROID...", dim=True)
        policy = load_policy(checkpoint)
        if not click.confirm(click.style("Raise the lift and move Stretch 4 to the start pose?", bold=True), default=True):
            return

        console = Console()
        env.interrupted = console.pending
        env.say = console.event
        runner = Session(env, robot_model, policy, console, directory, search, instruction, max_steps,
                         custom_franka_start_pose, link0_height, start_q7, rerun, log_frame)
        runner.run()
    except KeyboardInterrupt:
        pass
    finally:
        if console is not None:
            console.close()
        click.secho("Stopping the robot.", dim=True)
        if env is not None:
            env.stop_motion()
        if runner is not None and runner.current is not None:
            runner.current.outcome = runner.current.outcome or "interrupted"
            runner.save(runner.current)  # the trial Ctrl+C cut short, as "interrupted"
            runner.trials.append(runner.current)
        if live is not None:
            live.close()
        unload_policy()
        receiver.close()
        robot.stop()
        trials = runner.trials if runner is not None else load_trials(directory)
        path = write_report(directory, session, trials, search)
        click.secho(f"Report: {path}", fg="green", bold=True)



if __name__ == "__main__":
    main()
