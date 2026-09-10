"""
What a MolmoBot checkpoint says about the run that produced it.

A fine-tuned checkpoint is not just weights: `train_molmobot.py` writes a
`config.yaml` beside them that records the model it built *and*, under
`runtime_data.args`, the exact argument list it was launched with. That second
part is what makes an evaluation self-configuring, and it matters more than it
sounds like it should.

The failure it prevents is silent. MolmoBot is trained on a chosen set of
cameras in a chosen order (`--camera_names head_camera_right
wrist_camera_right`) and served the images in the same order at inference. Serve
it a different set -- four cameras where it was trained on two, or the same two
the other way round -- and nothing raises: the images are the right dtype and
the right shape, the model consumes them, and the policy produces confident
actions for a scene it is not really looking at. The same is true of
`--action_type`: `joint_pos_rel` weights served as `joint_pos` move the arm
smoothly to the wrong place.

So the checkpoint gets to say. `policies/molmobot_policy.py` reads these values
and prefers them over its own defaults, announcing any disagreement rather than
resolving it quietly, and `run_benchmarks.py --policy molmobot` therefore needs
no `--cameras` flag: the answer is already on disk.

Everything here is best-effort by design. A checkpoint from a MolmoBot old
enough not to record `runtime_data`, or one assembled by hand, yields an empty
`TrainingArgs` and leaves the policy config's own values in force -- which is
exactly the behaviour that existed before this module.
"""

from __future__ import annotations

import json
import logging
import shlex
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_FILENAME = "config.yaml"
"""What `train_molmobot.py` writes beside every checkpoint it saves."""

TRAINING_METRICS_FILENAME = "training_metrics.json"
"""
What `molmobot_repo.ensure_metrics_log`'s patch writes inside each checkpoint.

The state training was in when those weights were saved: the step, the training
loss, the best validation loss and where it happened, and the learning rates as
they had decayed by then. Written by the trainer, read here, so an evaluation
can say what it is about to run rather than only where it came from. Absent from
a checkpoint saved before that patch existed, which is not an error -- it is the
same "the checkpoint did not say" as everything else in this module.
"""

_LIST_FLAGS = ("--camera_names", "--action_move_groups", "--cameras_to_warp")
"""
Trainer flags declared `nargs="+"`, whose values run on until the next flag.

Listed rather than inferred because `shlex.split` gives back a flat list and
`--camera_names a b --action_type c` and `--camera_names a --action_type b c`
are indistinguishable without knowing which of the two takes several values.
"""


@dataclass(frozen=True)
class TrainingArgs:
    """The parts of a checkpoint's training configuration an evaluation has to match.

    Every field is optional: this is read from a file written by a program this
    repository does not own, and a missing value means "the checkpoint did not
    say", never "the checkpoint said no". Callers fall back to their own
    configuration for anything `None`.
    """

    camera_names: list[str] | None = None
    """`--camera_names`, in order. The order is part of the model's input layout."""

    action_type: str | None = None
    """`--action_type`: `joint_pos_rel` (deltas) or `joint_pos` (absolute targets)."""

    action_preset: str | None = None
    """`--action_preset`, e.g. `stretch_jointdelta`. Names the per-move-group widths."""

    action_move_groups: list[str] | None = None
    """`--action_move_groups`, when the run named them instead of using a preset."""

    action_dim: int | None = None
    """`model.action_dim`: the total width of the action vector the head emits."""

    states_mode: str | None = None
    """`model.states_mode`: how proprioception enters the model, e.g. `cross_attn`."""

    source: Path | None = None
    """The `config.yaml` this came from, for error messages that can be acted on."""

    def __bool__(self) -> bool:
        return any(
            value is not None
            for name, value in vars(self).items()
            if name != "source"
        )


@dataclass(frozen=True)
class TrainingState:
    """Where training had got to when a checkpoint was written."""

    step: int | None = None
    max_steps: int | None = None
    kind: str | None = None
    """`bestfit` for a `step<N>_bestfit/`, `periodic` for an ordinary `step<N>/`."""

    written: str | None = None
    train_loss: float | None = None
    best_metric: str | None = None
    best_loss: float | None = None
    best_step: int | None = None
    evals_since_improvement: int | None = None
    eval_losses: dict[str, float] = field(default_factory=dict)
    """The last validation loss each evaluator reported, by label."""

    learning_rates: dict[str, float] = field(default_factory=dict)
    source: Path | None = None

    def __bool__(self) -> bool:
        return self.step is not None

    @property
    def converged(self) -> bool | None:
        """Whether the best validation loss was still moving when this was saved.

        None when the run recorded no best-fit bookkeeping. False means the best
        loss was this step's -- so training had not finished improving, and a
        longer run would have produced a better checkpoint.
        """
        if self.evals_since_improvement is None:
            return None
        return self.evals_since_improvement > 0

    def summary(self) -> str:
        """One line, for the log an evaluation starts with."""
        parts = [f"step {self.step:,}" + (f"/{self.max_steps:,}" if self.max_steps else "")]
        if self.kind:
            parts.append(self.kind)
        if self.train_loss is not None:
            parts.append(f"train loss {self.train_loss:.5f}")
        if self.best_loss is not None:
            best = f"best {self.best_metric or 'val'} {self.best_loss:.5f}"
            if self.best_step is not None:
                best += f" @ step {self.best_step:,}"
            parts.append(best)
        if self.evals_since_improvement:
            parts.append(f"{self.evals_since_improvement} evals without improvement")
        elif self.converged is False:
            parts.append("still improving when saved")
        return ", ".join(parts)


def read_training_state(checkpoint_path: str | Path) -> TrainingState:
    """Read the `training_metrics.json` a checkpoint carries, if it has one.

    Never raises: a checkpoint from a run that predates the patch, or one whose
    file will not parse, yields an empty `TrainingState` that is falsey.
    """
    path = Path(checkpoint_path) / TRAINING_METRICS_FILENAME
    if not path.is_file():
        return TrainingState()
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        log.warning(f"[molmobot] could not read {path}: {error}")
        return TrainingState()
    if not isinstance(payload, dict):
        return TrainingState()

    train = payload.get("train") or {}
    best = payload.get("best") or {}
    return TrainingState(
        step=_as_int(payload.get("step")),
        max_steps=_as_int(payload.get("max_steps")),
        kind=payload.get("kind") if isinstance(payload.get("kind"), str) else None,
        written=payload.get("written") if isinstance(payload.get("written"), str) else None,
        train_loss=_as_float(train.get("loss")),
        best_metric=best.get("metric") if isinstance(best.get("metric"), str) else None,
        best_loss=_as_float(best.get("loss")),
        best_step=_as_int(best.get("step")),
        evals_since_improvement=_as_int(best.get("evals_since_improvement")),
        eval_losses=_eval_losses(payload.get("eval"), best.get("metric")),
        learning_rates={
            name: float(value)
            for name, value in (payload.get("learning_rates") or {}).items()
            if isinstance(value, (int, float))
        },
        source=path,
    )


def _eval_losses(evaluations: object, preferred: object) -> dict[str, float]:
    """The one loss per evaluator worth quoting, out of its whole metrics dump."""
    if not isinstance(evaluations, dict):
        return {}
    losses: dict[str, float] = {}
    for label, record in evaluations.items():
        metrics = (record or {}).get("metrics") if isinstance(record, dict) else None
        if not isinstance(metrics, dict):
            continue
        names = [name for name in metrics if "loss" in name.lower()]
        if isinstance(preferred, str) and preferred in metrics:
            names = [preferred]
        if names:
            losses[str(label)] = float(metrics[sorted(names)[0]])
    return losses


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def config_path_for(checkpoint_path: str | Path) -> Path:
    """The `config.yaml` belonging to a checkpoint directory."""
    return Path(checkpoint_path) / CONFIG_FILENAME


def read_training_args(checkpoint_path: str | Path) -> TrainingArgs:
    """Read what `checkpoint_path` records about how it was trained.

    Never raises for a checkpoint that simply does not say: a missing file, a
    file that will not parse, or one without `runtime_data` all return an empty
    `TrainingArgs`. The one thing worth logging is the parse failure, because a
    `config.yaml` that exists and cannot be read is a different situation from
    one that was never written.
    """
    path = config_path_for(checkpoint_path)
    if not path.is_file():
        return TrainingArgs()

    try:
        import yaml

        config = yaml.safe_load(path.read_text())
    except Exception as error:  # noqa: BLE001 - any parse failure is the same non-answer
        log.warning(
            f"[molmobot] could not read {path}, so its training settings are unknown: {error}"
        )
        return TrainingArgs()

    if not isinstance(config, dict):
        return TrainingArgs()

    model = config.get("model") or {}
    runtime = config.get("runtime_data") or {}
    flags = _parse_trainer_args(runtime.get("args"))

    action_dim = model.get("action_dim") if isinstance(model, dict) else None
    return TrainingArgs(
        camera_names=flags.get("--camera_names"),
        action_type=_single(flags.get("--action_type")),
        action_preset=_single(flags.get("--action_preset")),
        action_move_groups=flags.get("--action_move_groups"),
        action_dim=int(action_dim) if isinstance(action_dim, (int, float)) else None,
        states_mode=model.get("states_mode") if isinstance(model, dict) else None,
        source=path,
    )


def _parse_trainer_args(args: object) -> dict[str, list[str]]:
    """`launch_scripts/train_molmobot.py ... --action_type joint_pos_rel ...` -> a flag table.

    Handles both spellings the trainer accepts -- `--flag value` and
    `--flag=value` -- because its generated launch line uses each in places
    (`--exp_name=...` next to `--seq_len 528`), and treats the flags in
    `_LIST_FLAGS` as taking every value up to the next `--`.
    """
    if not isinstance(args, str):
        return {}

    try:
        tokens = shlex.split(args)
    except ValueError:
        return {}

    flags: dict[str, list[str]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.startswith("--"):
            continue
        if "=" in token:
            name, _, value = token.partition("=")
            flags[name] = [value]
            continue
        values: list[str] = []
        while index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
            if values and token not in _LIST_FLAGS:
                break
        flags[token] = values
    return flags


def _single(values: list[str] | None) -> str | None:
    """The one value of a scalar flag, or None if it was absent or empty."""
    return values[0] if values else None
