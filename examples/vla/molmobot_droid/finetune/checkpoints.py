"""
The fine-tuned checkpoints `run_finetune.py` saves, and how the `*_finetuned_*` runners drive
Stretch 4 the way each was trained.

A run directory (`<out>/<run name>/`) holds MolmoBot's checkpoints: `step<N>/` (sharded, with
the optimizer state) and the final `step<N>-unsharded/`. Either loads with
`checkpoint.load_policy()`; each one's `config.yaml` carries Stretch's normalization. The run's
`spec.json` says which of Stretch's cameras it was trained on and with which gripper
(`apply_to_params()`), and its `dataset/` which start pose its episodes began from
(`training_start()`).

`--checkpoint` takes a step directory or a run directory (its newest checkpoint); without it the
newest checkpoint under MODELS_DIR is used, and with --interactive the runner lists them all to
pick from.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

import click
from click.core import ParameterSource

from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import PARALLEL_GRIPPER_TOOL, RetargetParams

MODELS_DIR = Path("outputs/molmobot_droid/finetuning/models")
"""Where `run_finetune.py` saves its runs (its --out default)."""

STEP_DIR = re.compile(r"step(\d+)(-unsharded)?")


@dataclass(frozen=True)
class FinetunedCheckpoint:
    path: Path
    """The step directory: what `load_policy()` loads."""
    step: int
    unsharded: bool
    saved_at: float
    """When it was finished (its last file written), as `time.time()`."""

    @property
    def run_dir(self) -> Path:
        return self.path.parent

    @property
    def name(self) -> str:
        """<run name>/<step directory>, for run names and messages."""
        return f"{self.run_dir.name}/{self.path.name}"

    @cached_property
    def spec(self) -> dict:
        """The run's `spec.json` (see `run_finetune.py`)."""
        return json.loads((self.run_dir / "spec.json").read_text())

    @property
    def exo_camera(self) -> str:
        """The camera its exo view was trained on: fisheye_<left, right or tool>[_rectified]."""
        return self.spec["cameras"][0]

    @property
    def wrist_camera(self) -> str:
        """The camera its wrist view was trained on: gripper_<left or right>."""
        return self.spec["cameras"][1]

    @property
    def tool_name(self) -> str:
        return self.spec["tool_name"]

    def describe(self) -> str:
        """One line: the checkpoint, when it was saved, and what it was trained on."""
        saved = datetime.fromtimestamp(self.saved_at).strftime("%Y-%m-%d %H:%M")
        return (f"{self.name}  ({saved}, exo {self.exo_camera}, wrist {self.wrist_camera}, "
                f"{self.tool_name.rsplit('_', 1)[-1].upper()})")


def _finished_at(path: Path, unsharded: bool) -> float | None:
    """When the checkpoint at `path` was finished, or None while it is still being written (or
    never was): the sharded one's `.metadata` is written last, the unsharded one's model.pt."""
    marker = path / ("model.pt" if unsharded else "model_and_optim/.metadata")
    if not marker.exists() or not (path / "config.yaml").exists():
        return None
    return marker.stat().st_mtime


def run_checkpoints(run_dir: Path) -> list[FinetunedCheckpoint]:
    """The finished checkpoints of the `run_finetune.py` run in `run_dir`, oldest first."""
    if not (run_dir / "spec.json").exists():
        return []
    found = []
    for path in run_dir.iterdir():
        match = STEP_DIR.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        unsharded = match.group(2) is not None
        saved_at = _finished_at(path, unsharded)
        if saved_at is not None:
            found.append(FinetunedCheckpoint(path, int(match.group(1)), unsharded, saved_at))
    return sorted(found, key=lambda c: c.saved_at)


def find_checkpoints(models_dir: Path = MODELS_DIR) -> list[FinetunedCheckpoint]:
    """Every finished checkpoint of every run under `models_dir`, oldest first."""
    if not models_dir.is_dir():
        return []
    found = [c for run_dir in models_dir.iterdir() if run_dir.is_dir() for c in run_checkpoints(run_dir)]
    return sorted(found, key=lambda c: c.saved_at)


def choose_checkpoint(checkpoints: list[FinetunedCheckpoint]) -> FinetunedCheckpoint:
    """Ask which of `checkpoints` to use, newest first and the default."""
    newest_first = checkpoints[::-1]
    for number, checkpoint in enumerate(newest_first, 1):
        click.echo(f"  {number:2d}. {checkpoint.describe()}")
    number = click.prompt("Checkpoint", type=click.IntRange(1, len(newest_first)), default=1)
    return newest_first[number - 1]


def resolve_checkpoint(checkpoint: str | None, interactive: bool) -> FinetunedCheckpoint:
    """
    The checkpoint --checkpoint and --interactive name: a step directory as it is, a run
    directory's newest (with --interactive, any of its), or without --checkpoint the newest under
    MODELS_DIR (with --interactive, any of them).
    """
    if checkpoint is not None:
        path = Path(checkpoint)
        match = STEP_DIR.fullmatch(path.name)
        if match is not None:
            unsharded = match.group(2) is not None
            saved_at = _finished_at(path, unsharded)
            if saved_at is None or not (path.parent / "spec.json").exists():
                raise click.BadParameter(f"{path} is not a finished checkpoint of a run_finetune.py run",
                                         param_hint="--checkpoint")
            return FinetunedCheckpoint(path, int(match.group(1)), unsharded, saved_at)
        checkpoints = run_checkpoints(path)
        where = str(path)
    else:
        checkpoints = find_checkpoints()
        where = str(MODELS_DIR)
    if not checkpoints:
        raise click.ClickException(f"No finished fine-tuned checkpoints under {where}")
    chosen = choose_checkpoint(checkpoints) if interactive else checkpoints[-1]
    click.secho(f"Checkpoint: {chosen.describe()}", fg="green")
    return chosen


def finetuned_checkpoint_options(function):
    """--checkpoint and --interactive (see `resolve_checkpoint()`)."""
    options = [
        click.option("--checkpoint", default=None, type=click.Path(exists=True, file_okay=False),
                     help=f"A run_finetune.py checkpoint (step<N>/ or step<N>-unsharded/), or a run directory "
                     f"for its newest. Default: the newest under {MODELS_DIR}."),
        click.option("--interactive", "-i", is_flag=True,
                     help="List the checkpoints (of --checkpoint's run, or all of them) and ask which to use."),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def apply_to_params(checkpoint: FinetunedCheckpoint, params: RetargetParams) -> None:
    """
    Show the policy Stretch's cameras as `checkpoint` was trained on them: the exo view from the
    head fisheye on the tool's side (fisheye_tool) or a fixed one, DROID-framed or rectified,
    and the wrist view from its gripper camera; and give Stretch the gripper it was trained with.
    """
    match = re.fullmatch(r"fisheye_(left|right|tool)(_rectified)?", checkpoint.exo_camera)
    wrist = re.fullmatch(r"gripper_(left|right)", checkpoint.wrist_camera)
    if match is None or wrist is None:
        raise click.ClickException(f"{checkpoint.name} was trained on cameras {checkpoint.spec['cameras']}, "
                                   "which these runners cannot show it")
    side, rectified = match.groups()
    params.head_crop = "droid"
    params.head_follows_tool = side == "tool"
    if side != "tool":
        params.exo_camera = side
    params.rectify = rectified is not None
    params.gripper_camera = wrist.group(1)
    params.use_parallel_gripper = checkpoint.tool_name == PARALLEL_GRIPPER_TOOL
    params.__post_init__()


@dataclass
class TrainingStart:
    """The start pose the episodes a checkpoint was trained on began from: the follow-the-Franka
    runs' --custom_franka_start_pose, --grasp-offset-mm and --grasp-offset-deg."""

    custom_franka_start_pose: bool
    grasp_offset_mm: tuple[float, float, float]
    grasp_offset_deg: tuple[float, float, float]

    def describe(self) -> str:
        return ("--custom_franka_start_pose " if self.custom_franka_start_pose else "") + (
            f"--grasp-offset-mm {','.join(f'{v:g}' for v in self.grasp_offset_mm)} "
            f"--grasp-offset-deg {','.join(f'{v:g}' for v in self.grasp_offset_deg)}"
        )


def training_start(checkpoint: FinetunedCheckpoint) -> TrainingStart | None:
    """The start pose `checkpoint`'s training episodes began from (an h5 of each of its runs,
    through the train split's index), or None if they did not all begin from the same one."""
    import h5py

    train = checkpoint.run_dir / "dataset" / "train"
    index = json.loads((train / "valid_trajectory_index.json").read_text())
    starts = set()
    for episodes in index.values():
        with h5py.File(train / next(iter(episodes)), "r") as f:
            flags = json.loads(f.attrs["flags"])
            starts.add((
                bool(flags.get("custom_franka_start_pose", False)),
                tuple(float(v) for v in json.loads(f.attrs["grasp_offset_mm"])),
                tuple(float(v) for v in json.loads(f.attrs["grasp_offset_deg"])),
            ))
    return TrainingStart(*starts.pop()) if len(starts) == 1 else None


def apply_training_start(checkpoint: FinetunedCheckpoint, params: RetargetParams,
                         custom_franka_start_pose: bool) -> bool:
    """
    Start Stretch where `checkpoint`'s training episodes did, for each of --custom_franka_start_pose,
    --grasp-offset-mm and --grasp-offset-deg not given on the command line; says what it took and
    warns when a given one differs. Returns --custom_franka_start_pose.
    """
    start = training_start(checkpoint)
    if start is None:
        click.secho("The training episodes began from different start poses: the command line's is used", fg="yellow")
        return custom_franka_start_pose
    context = click.get_current_context()

    def given(name: str) -> bool:
        return context.get_parameter_source(name) not in (ParameterSource.DEFAULT, None)

    if not given("custom_franka_start_pose"):
        custom_franka_start_pose = start.custom_franka_start_pose
    if not given("grasp_offset_mm"):
        params.grasp_offset_mm = start.grasp_offset_mm
    if not given("grasp_offset_deg"):
        params.grasp_offset_deg = start.grasp_offset_deg
    used = TrainingStart(custom_franka_start_pose, tuple(params.effective_grasp_offset_mm), tuple(params.grasp_offset_deg))
    if used == start:
        click.secho(f"Start pose as in training: {start.describe()}", fg="green")
    else:
        click.secho(f"Start pose {used.describe()}, not training's {start.describe()}", fg="yellow")
    return custom_franka_start_pose
