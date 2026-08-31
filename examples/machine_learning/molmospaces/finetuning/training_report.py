"""
What a MolmoBot fine-tune did, read back from the metrics it wrote.

A fine-tune is eleven hours of one number scrolling past. The run either
converged or it did not, the learning rates were either right or they were not,
and by the end the terminal holds a thousand screens of `[step=...]` blocks that
answer those questions only if you scroll through all of them. This reads the
JSON lines `molmobot_repo.ensure_metrics_log` has the trainer write and says what
happened:

    python -m examples.machine_learning.molmospaces.finetuning.training_report \\
        data/stretch_pick/rollouts/molmobot/checkpoints/stretch4_pick

    # several runs side by side -- what a hyperparameter probe produces
    python -m examples.machine_learning.molmospaces.finetuning.training_report \\
        data/stretch_pick/rollouts/probe/* --compare

    # and the curves, if matplotlib is around
    python -m examples.machine_learning.molmospaces.finetuning.training_report \\
        <run> --plot report.png

Three things are worth knowing about what it reads.

**The validation loss is the number that matters, and MolmoBot nearly hides
it.** Its `loss_eval` logs to the console and to Weights & Biases; the console
copy is a bare `<label>` header followed by indented values, hundreds of lines
after the training block it belongs with. The metrics file keeps the pairing.

**The learning rates and gradient norms are in the file and *not* in the
console.** `log_metrics_to_console` drops everything under `optim/` except
`optim/total_grad_norm`, which this trainer never emits -- it emits
`optim/<group>_grad_norm`, one per parameter group. Those are the numbers that
tell an LR that is too high from one that is merely slow, which is why the patch
records the metrics dict before that filter rather than parsing the log after
it.

**Nothing here is a verdict.** The diagnostics are stated as what was measured
and what usually follows from it; a plateau at a loss you are happy with is a
finished run, and the same plateau at a loss you are not is a reason to change
something. Deliberately stdlib-only, with matplotlib optional, so it runs under
this repository's interpreter or the training venv's without either having to
grow a dependency for a report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

METRICS_FILENAME = "metrics.jsonl"
"""What the trainer patch writes, in the save folder, unless told otherwise."""

PLOT_BESIDE_METRICS = Path("__beside_the_metrics__")
"""
Sentinel for `--plot` with no argument: put `training.png` beside `metrics.jsonl`.

Which is where it wants to be for a run in progress -- next to the checkpoints,
in the save folder, at a path that does not change between runs.
"""

CHECKPOINT_METRICS_FILENAME = "training_metrics.json"
"""
What the same patch writes *inside* each checkpoint directory.

Spelled out here rather than imported from `molmobot_repo`, because this file
is run by path under MolmoBot's own interpreter, where this repository is not
importable. `tests/test_stretch_finetuning.py` holds the two in step.
"""

SPIKE_RATIO = 5.0
"""
How far above the median a gradient norm has to be to count as a spike.

A ratio rather than a threshold because the norms of a frozen-backbone
fine-tune and a full one differ by orders of magnitude, and it is the *shape*
that carries the signal: a handful of excursions many times the typical norm is
what an optimizer stepping too far looks like from the outside.
"""

PLATEAU_IMPROVEMENT = 0.01
"""Relative improvement over the last third of the run below which it has flattened out."""

LONGER_RUN_ADVICE = (
    "Raise MAX_STEPS and start again from this run's best weights, into a save folder of "
    "its own:\n"
    "      CHECKPOINT=<save folder>/step<best>_bestfit SAVE_FOLDER=<save folder>_v2 "
    "MAX_STEPS=<more> bash run_molmobot.sh\n"
    "    Not a resume in place: MAX_STEPS is the learning-rate horizon, so resuming into a "
    "longer one re-expands the schedule and jumps the rate back up mid-run -- and with "
    "save_final_optim=False there is no optimizer state in the checkpoint to resume from "
    "anyway. Passed as CHECKPOINT the weights arrive as initial_model_checkpoint, which "
    "resets the optimizer and the step counter, so the new run gets one clean schedule."
)
"""
What to do about a run that stopped while it was still improving.

One string because two different diagnostics reach the same conclusion -- the
training loss still falling, and the validation loss bottoming out at the last
evaluation -- and the wrong version of this advice ("just resume") costs a day
of GPU to discover.
"""


# =============================================================================
# Reading
# =============================================================================


@dataclass
class Record:
    """One metrics dump: a training step's, or one evaluator's."""

    step: int
    max_steps: int
    split: str
    label: str | None
    time: datetime | None
    metrics: dict[str, float]


@dataclass
class Run:
    """One training run's metrics, as read off disk."""

    name: str
    path: Path
    train: list[Record] = field(default_factory=list)
    evals: dict[str, list[Record]] = field(default_factory=dict)
    settings: dict[str, object] = field(default_factory=dict)
    """
    What the run was configured with, from the `config.yaml` beside the metrics.

    Needed because the learning rates are not always *in* the metrics:
    `LRMonitor.check()` contributes `optim/<group>_lr` only when the optimizer
    reports its groups, and the 8-bit optimizer this repository defaults to
    yields nothing there -- so a sweep would compare runs it could not name. The
    trainer writes its whole resolved config next to the checkpoints, which does
    say, and it is the same file `--checkpoint` is later pointed at.
    """

    @property
    def steps(self) -> int:
        return max((record.step for record in self.train), default=0)

    @property
    def max_steps(self) -> int:
        return max((record.max_steps for record in self.train), default=0)

    @property
    def elapsed(self) -> float | None:
        """Seconds between the first and last dump, or None if they are not timed."""
        times = [record.time for record in self.train if record.time is not None]
        if len(times) < 2:
            return None
        return (max(times) - min(times)).total_seconds()


def find_metrics_file(target: Path) -> Path | None:
    """The metrics file for `target`, which may be the file, a save folder, or a checkpoint.

    A checkpoint directory is accepted because that is the path most likely to
    be on the clipboard -- it is what `--checkpoint` takes -- and the metrics
    file for it sits one level up, beside the `step<N>/` directories.
    """
    if target.is_file():
        return target
    for candidate in (target / METRICS_FILENAME, target.parent / METRICS_FILENAME):
        if candidate.is_file():
            return candidate
    return None


def read_run(target: Path) -> Run:
    """Load one run's metrics, skipping any line that will not parse.

    A truncated last line is normal: the file is appended to while training
    runs, and a report asked for mid-run should describe the run so far rather
    than refuse.
    """
    path = find_metrics_file(target)
    if path is None:
        raise FileNotFoundError(
            f"No {METRICS_FILENAME} at or beside {target}. It is written by the metrics "
            "patch in the MolmoBot checkout, which finetune.py applies and the generated "
            f"run script switches on -- a run started before that, or with the variable "
            "unset, leaves no metrics behind. See finetuning/README.md."
        )

    run = Run(name=_run_name(path), path=path, settings=read_settings(path.parent))
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        record = Record(
            step=int(payload.get("step", 0)),
            max_steps=int(payload.get("max_steps", 0)),
            split=str(payload.get("split", "train")),
            label=payload.get("label"),
            time=_parse_time(payload.get("time")),
            metrics={
                name: float(value)
                for name, value in (payload.get("metrics") or {}).items()
                if isinstance(value, (int, float))
            },
        )
        if record.split == "train":
            run.train.append(record)
        else:
            run.evals.setdefault(record.label or "eval", []).append(record)
    return run


SETTING_KEYS = {
    "action_expert_lr": ("optimizer", "action_expert_learning_rate"),
    "vit_lr": ("optimizer", "vit_learning_rate"),
    "llm_lr": ("optimizer", "llm_learning_rate"),
    "optimizer": ("optimizer", "name"),
    "global_batch": ("global_train_batch_size",),
    "max_steps": ("max_duration",),
    "seq_len": ("model", "max_sequence_length"),
    "ft_vit": ("ft_vit",),
    "ft_llm": ("ft_llm",),
    "ft_connector": ("ft_connector",),
}
"""Where each reported setting lives in the trainer's resolved config."""


def read_settings(save_folder: Path) -> dict[str, object]:
    """The handful of `config.yaml` values worth putting beside a loss.

    Optional in every direction: no file, no PyYAML, or a config whose fields
    have moved all yield `{}`, and the report simply reports less. PyYAML is a
    MolmoBot dependency and is in this repository's environment too, so in
    practice this works from either interpreter.
    """
    path = save_folder / "config.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        config = yaml.safe_load(path.read_text())
    except Exception:  # noqa: BLE001 - a config that will not parse is not an error here
        return {}
    if not isinstance(config, dict):
        return {}

    settings: dict[str, object] = {}
    for name, keys in SETTING_KEYS.items():
        value: object = config
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            settings[name] = value
    return settings


def read_checkpoints(save_folder: Path) -> list[dict]:
    """Every checkpoint in `save_folder` that recorded the state it was saved in.

    A run's checkpoints outlive its terminal and get copied around on their own,
    so each carries its own summary; this is the view from the run's side --
    which of them exist, and which one is worth evaluating.
    """
    found = []
    for directory in sorted(save_folder.glob("step*")):
        path = directory / CHECKPOINT_METRICS_FILENAME
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            continue
        if isinstance(payload, dict):
            payload["directory"] = directory
            found.append(payload)
    return sorted(found, key=lambda payload: payload.get("step") or 0)


def _run_name(path: Path) -> str:
    """A name for the run: the save folder's, since the file is always `metrics.jsonl`."""
    return path.parent.name or str(path.parent)


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# =============================================================================
# Picking the numbers out
# =============================================================================


def loss_key(metrics: dict[str, float], prefix: str = "") -> str | None:
    """The metric to treat as *the* loss.

    MolmoBot's action model reports `action_flow_loss`, and reports it under
    `train/` while training and bare while evaluating. Other losses come and go
    with the model configuration (`frame_score_loss`, auxiliary terms), so the
    preference is: the flow loss, then anything called `total`, then the first
    thing with `loss` in its name -- and never a `_wt` or `_scale` term, which
    are settings rather than measurements.
    """
    candidates = [
        name
        for name in metrics
        if name.startswith(prefix)
        and "loss" in name.lower()
        and not name.endswith(("_wt", "_scale", "_weight"))
    ]
    if not candidates:
        return None
    for preference in ("action_flow_loss", "total_loss", "loss"):
        for name in candidates:
            if name[len(prefix) :] == preference:
                return name
    return sorted(candidates)[0]


def series(records: list[Record], key: str) -> list[tuple[int, float]]:
    """`(step, value)` for every record that reported `key`, in step order."""
    return sorted(
        ((record.step, record.metrics[key]) for record in records if key in record.metrics),
        key=lambda pair: pair[0],
    )


def learning_rates(run: Run) -> dict[str, list[tuple[int, float]]]:
    """Per parameter group, the learning rate over time.

    The trainer emits `optim/<group>_lr` for every group it has, which for a
    MolmoBot fine-tune is the six of `initial_lr_dict` -- so this is also the
    only record of which groups were actually being trained.
    """
    names = {
        name
        for record in run.train
        for name in record.metrics
        if name.startswith("optim/") and name.endswith("_lr")
    }
    return {name[len("optim/") : -len("_lr")]: series(run.train, name) for name in sorted(names)}


def grad_norms(run: Run) -> dict[str, list[tuple[int, float]]]:
    """Per parameter group, the gradient norm over time (before clipping)."""
    names = {
        name
        for record in run.train
        for name in record.metrics
        if name.startswith("optim/") and name.endswith("_grad_norm")
    }
    return {
        name[len("optim/") : -len("_grad_norm")]: series(run.train, name) for name in sorted(names)
    }


def trend(points: list[tuple[int, float]], fraction: float = 1 / 3) -> float | None:
    """Relative improvement over the last `fraction` of a series.

    Positive means it went down (better). Measured between the medians of the
    first and second half of that window rather than between two single points,
    because one noisy evaluation should not read as a trend.
    """
    if len(points) < 4:
        return None
    window = points[-max(4, int(len(points) * fraction)) :]
    half = len(window) // 2
    early = statistics.median(value for _, value in window[:half])
    late = statistics.median(value for _, value in window[half:])
    if early == 0:
        return None
    return (early - late) / abs(early)


# =============================================================================
# The report
# =============================================================================


def format_run(run: Run) -> str:
    """One run's report, as text."""
    lines = [f"{run.name}  ({run.path})", "=" * 78, ""]
    lines += _progress_lines(run)
    lines += _loss_lines(run)
    lines += _optimizer_lines(run)
    lines += _checkpoint_lines(run)
    lines += _diagnostic_lines(run)
    return "\n".join(lines)


def _progress_lines(run: Run) -> list[str]:
    if not run.train:
        return ["No training metrics recorded.", ""]
    lines = ["progress", "--------"]
    step_text = f"{run.steps:,}" + (f" of {run.max_steps:,}" if run.max_steps else "")
    lines.append(f"  steps          {step_text}")
    elapsed = run.elapsed
    if elapsed:
        hours, remainder = divmod(int(elapsed), 3600)
        lines.append(f"  wall clock     {hours}:{remainder // 60:02d}:{remainder % 60:02d}")
        if run.steps:
            lines.append(f"  average        {run.steps / elapsed:.2f} steps/s")
    for name, label in (
        ("throughput/device/batches_per_second", "batches/s"),
        ("System/Peak GPU Memory (MB)", "peak GPU MB"),
    ):
        points = series(run.train, name)
        if points:
            lines.append(f"  {label:<14} {points[-1][1]:,.1f}")
    return lines + [""]


def _loss_lines(run: Run) -> list[str]:
    """The training loss and each evaluator's, on one aligned row apiece.

    The names are padded to the widest of them rather than to a constant,
    because an evaluator's label comes from the dataset (`synthmanip_val`) and a
    fixed column either wastes half the line or is overrun by it.
    """
    rows: list[tuple[str, list[tuple[int, float]]]] = []
    train_key = loss_key(_merged_metrics(run.train), prefix="train/")
    if train_key:
        rows.append((train_key, series(run.train, train_key)))
    for label, records in sorted(run.evals.items()):
        key = loss_key(_merged_metrics(records))
        if key:
            rows.append((f"{label}/{key}", series(records, key)))

    lines = ["loss", "----"]
    if not train_key:
        lines.append("  no training loss recorded")
    width = max((len(name) for name, _ in rows), default=0)
    for name, points in rows:
        if not points:
            continue
        best_step, best = min(points, key=lambda pair: pair[1])
        lines.append(
            f"  {name:<{width}}  first {points[0][1]:.5f}   last {points[-1][1]:.5f}   "
            f"best {best:.5f} @ step {best_step:,}"
        )
    if len(rows) < 2:
        lines.append("  no validation loss recorded -- see the diagnostics below")
    return lines + [""]


def _optimizer_lines(run: Run) -> list[str]:
    rates = learning_rates(run)
    norms = grad_norms(run)
    if not rates and not norms and not run.settings:
        return []
    lines = ["optimizer", "---------"]
    lines += _settings_lines(run)
    for group, points in rates.items():
        if not points:
            continue
        peak = max(value for _, value in points)
        if peak == 0:
            # A group at zero throughout was frozen; saying so is more useful
            # than a row of zeros, because it is the setting people forget.
            lines.append(f"  {group:<22} lr 0 (frozen)")
            continue
        lines.append(
            f"  {group:<22} lr peak {peak:.2e}   last {points[-1][1]:.2e}"
            + _grad_norm_summary(norms.get(group, []))
        )
    for group, points in norms.items():
        if group not in rates and points:
            lines.append(f"  {group:<22} {_grad_norm_summary(points).lstrip()}")
    return lines + [""]


def _checkpoint_lines(run: Run) -> list[str]:
    """What is on disk, and which of it to evaluate."""
    checkpoints = read_checkpoints(run.path.parent)
    if not checkpoints:
        return []

    lines = ["checkpoints", "-----------"]
    width = max(len(payload["directory"].name) for payload in checkpoints)
    best_step = None
    for payload in checkpoints:
        best = payload.get("best") or {}
        train = (payload.get("train") or {}).get("loss")
        parts = [f"  {payload['directory'].name:<{width}}"]
        if train is not None:
            parts.append(f"train {train:.5f}")
        if best.get("loss") is not None:
            best_step = best.get("step", best_step)
            parts.append(
                f"best {best.get('metric', 'val')} {best['loss']:.5f}"
                + (f" @ step {best['step']:,}" if best.get("step") is not None else "")
            )
        since = best.get("evals_since_improvement")
        if since == 0:
            parts.append("(saved on an improvement)")
        elif since:
            parts.append(f"({since} evals without improvement by then)")
        lines.append("   ".join(parts))

    if best_step is not None:
        # A `step<N>_bestfit/` is preferred over a `step<N>/` at the same step:
        # it was saved *because* the loss improved, and it is the one the
        # trainer keeps when retention starts deleting the others.
        at_best = [payload for payload in checkpoints if payload.get("step") == best_step]
        at_best.sort(key=lambda payload: payload.get("kind") != "bestfit")
        if at_best:
            lines.append(f"  -> evaluate {at_best[0]['directory']}")
    return lines + [""]


def _settings_lines(run: Run) -> list[str]:
    """What the run was configured with, when the metrics do not carry it.

    The learning rates come from the config rather than the metrics whenever
    `LRMonitor` reported nothing -- see `Run.settings` -- and the trainable
    components come from there always, because no metric records them.
    """
    if not run.settings:
        return []
    lines = []
    configured = ", ".join(
        f"{name.replace('_lr', '')} {run.settings[name]:.1e}"
        for name in ("action_expert_lr", "vit_lr", "llm_lr")
        if isinstance(run.settings.get(name), (int, float))
    )
    if configured:
        lines.append(f"  {'configured lr':<22} {configured}")
    trainable = [
        component
        for component, key in (("vit", "ft_vit"), ("llm", "ft_llm"), ("connector", "ft_connector"))
        if run.settings.get(key)
    ]
    lines.append(
        f"  {'trainable':<22} action expert"
        + (f" + {', '.join(trainable)}" if trainable else " only (everything else frozen)")
        + (f"   optimizer {run.settings['optimizer']}" if run.settings.get("optimizer") else "")
    )
    return lines


def _grad_norm_summary(points: list[tuple[int, float]]) -> str:
    if not points:
        return ""
    values = [value for _, value in points]
    median = statistics.median(values)
    spikes = sum(1 for value in values if median > 0 and value > SPIKE_RATIO * median)
    text = f"   |grad| median {median:.3g}  max {max(values):.3g}"
    if spikes:
        text += f"  ({spikes} spikes >{SPIKE_RATIO:g}x)"
    return text


def _merged_metrics(records: list[Record]) -> dict[str, float]:
    """Every metric name these records ever reported, for choosing a loss key."""
    merged: dict[str, float] = {}
    for record in records:
        merged.update(record.metrics)
    return merged


# =============================================================================
# Diagnostics
# =============================================================================


def diagnose(run: Run) -> list[tuple[str, str]]:
    """`(observation, what usually follows)` for the things worth acting on.

    Each entry states a measurement first, because the measurement is the part
    that is true regardless of what anyone does about it. The advice is the
    conventional reading of that measurement for a VLA fine-tune, not a rule.
    """
    findings: list[tuple[str, str]] = []
    if not run.train:
        return [("No training metrics were recorded.", "Nothing can be said about this run.")]

    train_key = loss_key(_merged_metrics(run.train), prefix="train/")
    train_points = series(run.train, train_key) if train_key else []
    findings += _diagnose_train_loss(train_points)
    findings += _diagnose_validation(run, train_points)
    findings += _diagnose_gradients(run)
    findings += _diagnose_schedule(run)
    return findings


def _diagnose_train_loss(points: list[tuple[int, float]]) -> list[tuple[str, str]]:
    if len(points) < 4:
        return []
    findings = []
    values = [value for _, value in points]
    if any(math.isnan(value) or math.isinf(value) for value in values):
        findings.append(
            (
                "The training loss went NaN or infinite.",
                "Lower ACTION_EXPERT_LR (and VIT_LR if the tower is unfrozen) and restart "
                "from the last good checkpoint; the trainer also dumps the offending batch "
                "into the save folder.",
            )
        )
    early = statistics.median(values[: max(2, len(values) // 10)])
    late = statistics.median(values[-max(2, len(values) // 10) :])
    if early > 0 and (early - late) / early < 0.02:
        findings.append(
            (
                f"The training loss barely moved: {early:.5f} -> {late:.5f}.",
                "Either the learning rate is far too low, or nothing is actually being "
                "trained -- check the optimizer section above for a group whose lr is 0, "
                "and TRAINABLE in the run script.",
            )
        )
    final = trend(points)
    if final is not None and final > 0.05:
        findings.append(
            (
                f"The training loss was still falling at the end ({final:.0%} over the last "
                "third).",
                "The run stopped before it converged. " + LONGER_RUN_ADVICE,
            )
        )
    return findings


def _diagnose_validation(run: Run, train_points: list[tuple[int, float]]) -> list[tuple[str, str]]:
    # Not every non-training dump is a validation pass: the trainer logs a
    # "Pre-train system metrics" block through the same call, carrying peak GPU
    # memory and no loss at all. Judged by whether a loss was reported rather
    # than by the label, which is the evaluator's name and varies per dataset.
    if not any(loss_key(_merged_metrics(records)) for records in run.evals.values()):
        return [
            (
                "No validation loss was recorded.",
                "Without it there is no way to tell learning from memorising. Set "
                "EVAL_INTERVAL (the generated script defaults to 500) and make sure every "
                "task has a non-empty val/ split -- finetune.py warns when one does not.",
            )
        ]

    findings = []
    for label, records in sorted(run.evals.items()):
        key = loss_key(_merged_metrics(records))
        points = series(records, key) if key else []
        if len(points) < 3:
            continue
        best_step, best = min(points, key=lambda pair: pair[1])
        last_step, last = points[-1]

        if best > 0 and (last - best) / best > 0.05:
            findings.append(
                (
                    f"{label}: validation bottomed out at {best:.5f} (step {best_step:,}) and "
                    f"is {last:.5f} by step {last_step:,}.",
                    "That is overfitting past the best fit. The step<N>_bestfit/ checkpoint "
                    "is the one to evaluate; to spend the extra steps better, generate more "
                    "episodes or turn on --img_aug.",
                )
            )
        elif best_step >= 0.8 * (run.steps or 1):
            findings.append(
                (
                    f"{label}: validation was still improving at the end (best {best:.5f} at "
                    f"step {best_step:,} of {run.steps:,}).",
                    "There is more to gain from a longer run. " + LONGER_RUN_ADVICE,
                )
            )
        else:
            improvement = trend(points)
            if improvement is not None and improvement < PLATEAU_IMPROVEMENT:
                findings.append(
                    (
                        f"{label}: validation flattened out around {last:.5f} "
                        f"({improvement:.1%} over the last third).",
                        "More steps at these settings will not help. What usually does, in "
                        "order: more data, then TRAINABLE=vision if it is not already on, "
                        "then a higher ACTION_EXPERT_LR.",
                    )
                )

        if train_points:
            train_last = train_points[-1][1]
            if train_last > 0 and last / train_last > 2.0:
                findings.append(
                    (
                        f"{label}: validation ({last:.5f}) is more than twice the training "
                        f"loss ({train_last:.5f}).",
                        "The model fits the demonstrations it has seen far better than the "
                        "ones it has not, which for a procedurally generated dataset means "
                        "the episodes are too alike -- vary the houses and objects rather "
                        "than adding more episodes of the same.",
                    )
                )
    return findings


def _diagnose_gradients(run: Run) -> list[tuple[str, str]]:
    findings = []
    for group, points in grad_norms(run).items():
        values = [value for _, value in points]
        if len(values) < 8:
            continue
        median = statistics.median(values)
        if median <= 0:
            continue
        spikes = [value for value in values if value > SPIKE_RATIO * median]
        if len(spikes) > max(2, 0.05 * len(values)):
            findings.append(
                (
                    f"{group}: {len(spikes)} of {len(values)} gradient norms were more than "
                    f"{SPIKE_RATIO:g}x the median ({median:.3g}), peaking at {max(values):.3g}.",
                    "Steps that large are the usual sign of a learning rate above what this "
                    f"group can take; halve {_lr_variable(group)} before looking anywhere else.",
                )
            )
    return findings


def _diagnose_schedule(run: Run) -> list[tuple[str, str]]:
    findings = []
    for group, points in learning_rates(run).items():
        if len(points) < 4:
            continue
        peak = max(value for _, value in points)
        if peak == 0:
            continue
        if points[-1][1] > 0.5 * peak and run.max_steps and run.steps < run.max_steps:
            findings.append(
                (
                    f"{group}: the learning rate was still at {points[-1][1]:.2e} of a "
                    f"{peak:.2e} peak when the metrics end.",
                    "The run stopped before its schedule decayed, so these weights are mid-"
                    "descent rather than settled -- expect the last checkpoint to be worse "
                    "than a completed run's.",
                )
            )
    return findings


def _lr_variable(group: str) -> str:
    """The run script's environment variable for a parameter group's learning rate."""
    return {
        "action_expert": "ACTION_EXPERT_LR",
        "vit": "VIT_LR",
        "llm": "LLM_LR",
    }.get(group, f"the {group} learning rate")


def _diagnostic_lines(run: Run) -> list[str]:
    findings = diagnose(run)
    if not findings:
        return ["diagnostics", "-----------", "  Nothing stood out.", ""]

    # Two measurements often lead to the same recommendation -- a training loss
    # still falling and a validation loss still improving are the same run seen
    # twice -- and printing the paragraph again buries the second observation.
    lines = ["diagnostics", "-----------"]
    quoted = False
    for observation, action in findings:
        if LONGER_RUN_ADVICE in action:
            action = action.replace(
                LONGER_RUN_ADVICE, "Same recipe as above." if quoted else LONGER_RUN_ADVICE
            )
            quoted = True
        lines.append(f"  * {observation}")
        lines.append(f"    {action}")
    return lines + [""]


# =============================================================================
# Comparing runs, which is what a sweep produces
# =============================================================================


def format_comparison(runs: list[Run]) -> str:
    """One row per run: what it was trained at, and what it got."""
    header = (
        f"{'run':<28} {'steps':>8} {'best val':>10} {'@step':>8} {'last train':>11} "
        f"{'action_expert':>14} {'vit':>10}"
    )
    lines = [header, "-" * len(header)]
    for run in sorted(runs, key=_best_validation_sort_key):
        best_value, best_step = _best_validation(run)
        train_key = loss_key(_merged_metrics(run.train), prefix="train/")
        train_points = series(run.train, train_key) if train_key else []
        rates = learning_rates(run)
        lines.append(
            f"{run.name[:28]:<28} {run.steps:>8,} "
            f"{('-' if best_value is None else f'{best_value:.5f}'):>10} "
            f"{('-' if best_step is None else f'{best_step:,}'):>8} "
            f"{(f'{train_points[-1][1]:.5f}' if train_points else '-'):>11} "
            f"{_peak_lr(run, rates, 'action_expert'):>14} {_peak_lr(run, rates, 'vit'):>10}"
        )
    lines.append("")
    lines.append(
        "Sorted by best validation loss. A run whose best is at its last step has not "
        "finished improving, so its row understates it."
    )
    return "\n".join(lines)


def _best_validation(run: Run) -> tuple[float | None, int | None]:
    """The lowest validation loss anywhere in the run, and where it happened."""
    best: tuple[float, int] | None = None
    for records in run.evals.values():
        key = loss_key(_merged_metrics(records))
        if not key:
            continue
        for step, value in series(records, key):
            if best is None or value < best[0]:
                best = (value, step)
    return (None, None) if best is None else best


def _best_validation_sort_key(run: Run) -> float:
    value, _ = _best_validation(run)
    return math.inf if value is None else value


def _peak_lr(run: Run, rates: dict[str, list[tuple[int, float]]], group: str) -> str:
    """The group's learning rate: as measured if it was logged, else as configured."""
    points = rates.get(group) or []
    peak = max((value for _, value in points), default=0.0)
    if peak:
        return f"{peak:.1e}"
    configured = run.settings.get(f"{group}_lr")
    return f"{configured:.1e}" if isinstance(configured, (int, float)) else "-"


# =============================================================================
# Plots
# =============================================================================


MOVE_GROUP_DIMENSIONS: tuple[tuple[str, int], ...] = (
    ("base", 3),
    ("lift", 1),
    ("arm", 1),
    ("wrist", 3),
    ("gripper", 2),
)
"""
Stretch's move groups and their widths, in the order the action vector packs them.

The trainer reports `train/flow_loss_dim_<i>` per action dimension, which is the
most useful thing in the whole dump and the least readable: dimension 7 means
nothing until you know it is the third wrist joint. Averaging the dimensions of
each group turns ten anonymous curves into five that say *what* the policy is
failing to learn -- a gripper loss that will not come down is a different
problem from a base loss that will not.

Duplicated from `policies/molmobot_policy.STRETCH_ACTION_SPEC` rather than
imported, for the same reason as `CHECKPOINT_METRICS_FILENAME`: this file runs
under MolmoBot's interpreter. `tests/test_stretch_finetuning.py` holds the two
in step.
"""


def move_group_dimensions() -> dict[str, list[int]]:
    """`{"base": [0, 1, 2], "lift": [3], ...}` -- which action dimensions each group owns."""
    groups: dict[str, list[int]] = {}
    index = 0
    for group, width in MOVE_GROUP_DIMENSIONS:
        groups[group] = list(range(index, index + width))
        index += width
    return groups


def smoothed(points: list[tuple[int, float]], span: int | None = None) -> list[tuple[int, float]]:
    """An exponential moving average over a series, for a curve worth looking at.

    The raw training loss of a flow-matching model is *noisy* -- each step
    samples its own flow timesteps, so consecutive values differ by more than a
    thousand steps of progress does. The raw series is still drawn underneath;
    this is the line the eye should follow.
    """
    if not points:
        return []
    span = span or max(5, len(points) // 50)
    alpha = 2 / (span + 1)
    average = points[0][1]
    output = []
    for step, value in points:
        average += alpha * (value - average)
        output.append((step, average))
    return output


def plot_targets(run: Run) -> dict[str, list[tuple[int, float]]]:
    """The per-move-group action loss for one run, or `{}` if it did not report one."""
    dimensions = move_group_dimensions()
    series_by_group: dict[str, list[tuple[int, float]]] = {}
    for group, indices in dimensions.items():
        columns = [series(run.train, f"train/flow_loss_dim_{index}") for index in indices]
        columns = [column for column in columns if column]
        if len(columns) != len(indices):
            continue
        # Same steps in every column -- they come out of one metrics dump.
        merged = [
            (step, sum(column[position][1] for column in columns) / len(columns))
            for position, (step, _) in enumerate(columns[0])
        ]
        series_by_group[group] = merged
    return series_by_group


def write_plot(runs: list[Run], path: Path) -> bool:
    """Draw the five plots that say whether a fine-tune is working.

    Loss first, because that is the question. Then the per-move-group action
    loss, which says *what* is not being learned; the learning rates, which is
    the schedule actually applied rather than the one configured; the gradient
    norms, whose spikes are what an over-large rate looks like from outside; and
    throughput with peak memory, which is how a run that has slowed down or is
    about to run out of memory announces itself.

    Returns False rather than raising when matplotlib is not installed: the text
    report is the product, and the picture is a convenience.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axes = plt.subplots(5, 1, figsize=(11, 17), sharex=True)
    loss, groups, rates, norms, throughput = axes
    single = len(runs) == 1

    for index, run in enumerate(runs):
        prefix = "" if single else f"{run.name} "
        colour = f"C{index}"

        train_key = loss_key(_merged_metrics(run.train), prefix="train/")
        if train_key:
            points = series(run.train, train_key)
            loss.plot(*_xy(points), color=colour, alpha=0.25, linewidth=0.8)
            loss.plot(
                *_xy(smoothed(points)),
                color=colour,
                linewidth=1.8,
                label=f"{prefix}train ({train_key.split('/')[-1]})",
            )
        for offset, (label, records) in enumerate(sorted(run.evals.items())):
            key = loss_key(_merged_metrics(records))
            if not key:
                continue
            points = series(records, key)
            style = "o-" if offset == 0 else "s--"
            line = loss.plot(
                *_xy(points),
                style,
                color=colour,
                markersize=4,
                linewidth=1.4,
                markerfacecolor="none",
                label=f"{prefix}{label}",
            )
            best_step, best = min(points, key=lambda pair: pair[1])
            loss.plot([best_step], [best], "*", color=line[0].get_color(), markersize=14)
            loss.annotate(
                f"best {best:.4f}\n@ {best_step:,}",
                (best_step, best),
                textcoords="offset points",
                xytext=(6, -14),
                fontsize=7,
            )

        if single:
            for group, points in plot_targets(run).items():
                groups.plot(*_xy(smoothed(points)), linewidth=1.2, label=group)

        for group, points in learning_rates(run).items():
            if points and max(value for _, value in points) > 0:
                rates.plot(*_xy(points), linewidth=1.2, label=f"{prefix}{group}")
        for group, points in grad_norms(run).items():
            if points:
                # The raw trace takes the next colour from the cycle and the smoothed
                # line reuses it, so a faint curve and the bold one drawn over it are
                # visibly the same group rather than two unrelated colours.
                raw = norms.plot(*_xy(points), linewidth=0.8, alpha=0.4)
                norms.plot(
                    *_xy(smoothed(points)),
                    color=raw[0].get_color(),
                    linewidth=1.4,
                    label=f"{prefix}{group}",
                )

        rate_points = series(run.train, "throughput/device/batches_per_second")
        if rate_points:
            throughput.plot(
                *_xy(rate_points), color=colour, linewidth=1.2, label=f"{prefix}batches/s"
            )
        memory = series(run.train, "System/Peak GPU Memory (MB)")
        if memory:
            twin = getattr(throughput, "_stretch4_twin", None) or throughput.twinx()
            throughput._stretch4_twin = twin
            twin.plot(*_xy(memory), color=colour, linestyle=":", linewidth=1.2)
            twin.set_ylabel("peak GPU MB")

    _configure(loss, "loss", log=True, empty="no loss recorded")
    _configure(
        groups,
        "action loss by move group",
        log=True,
        empty="per-move-group loss needs several runs to be legible"
        if not single
        else "no per-dimension action loss recorded",
    )
    _configure(
        rates,
        "learning rate",
        empty="no learning rates recorded -- runs from before the tensor-valued lr fix "
        "have none; see molmobot_repo._stretch4_number",
    )
    _configure(norms, "gradient norm", log=True, empty="no gradient norms recorded")
    _configure(throughput, "throughput (batches/s)", empty="no throughput recorded")
    throughput.set_xlabel("step")
    figure.suptitle(
        runs[0].name if single else f"{len(runs)} runs", fontsize=12, y=0.995
    )
    figure.tight_layout()
    # Written beside itself and moved into place, so a viewer that reloads the
    # file on change never catches a half-written PNG -- which is exactly what
    # --watch invites. The pid is in the name because the watcher and the
    # end-of-run report can both be drawing the same picture for a moment.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=110)
    plt.close(figure)
    temporary.replace(path)
    return True


def _xy(points: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
    return [step for step, _ in points], [value for _, value in points]


def _configure(axis, ylabel: str, log: bool = False, empty: str | None = None) -> None:
    """Label one panel, and say why it is blank when it is.

    An empty panel with no explanation reads as a broken plot; most of the ways
    one ends up empty here are "the run did not record that", which is a fact
    about the run worth stating on the picture.
    """
    axis.set_ylabel(ylabel)
    if not axis.has_data():
        if empty:
            axis.text(
                0.5,
                0.5,
                empty,
                ha="center",
                va="center",
                fontsize=8,
                alpha=0.6,
                wrap=True,
                transform=axis.transAxes,
            )
        axis.set_yticks([])
        return
    if log:
        axis.set_yscale("log")
    axis.grid(alpha=0.3)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=7, ncol=2)


def watch_plot(targets: list[Path], path: Path, interval: float) -> int:
    """Redraw `path` whenever the metrics files grow. Runs until interrupted.

    Polling rather than a filesystem watch: the writer is a training run that
    appends a line every log interval, the reader wants a picture every few
    minutes, and a poll on file size is both dependency-free and immune to the
    ways inotify behaves differently over NFS and inside containers.
    """
    print(f"Watching {', '.join(str(target) for target in targets)} -> {path}", flush=True)
    signatures: list[tuple[int, float]] = []
    try:
        while True:
            files = [find_metrics_file(target) for target in targets]
            current = [
                (file.stat().st_size, file.stat().st_mtime) if file and file.is_file() else (0, 0.0)
                for file in files
            ]
            if current != signatures and any(size for size, _ in current):
                signatures = current
                runs = []
                for target in targets:
                    try:
                        runs.append(read_run(target))
                    except FileNotFoundError:
                        continue
                if runs and write_plot(runs, path):
                    print(f"  {_watch_status(runs)} -> {path}", flush=True)
                elif runs:
                    print("matplotlib is not installed, so no plot was written.", file=sys.stderr)
                    return 1
            time.sleep(interval)
    except KeyboardInterrupt:
        return 0


def _watch_status(runs: list[Run]) -> str:
    """`step 1,240/30,000  train 0.0421  val 0.0518` for the line each redraw prints."""
    run = runs[0]
    parts = [f"step {run.steps:,}" + (f"/{run.max_steps:,}" if run.max_steps else "")]
    train_key = loss_key(_merged_metrics(run.train), prefix="train/")
    points = series(run.train, train_key) if train_key else []
    if points:
        parts.append(f"train {points[-1][1]:.5f}")
    value, step = _best_validation(run)
    if value is not None:
        parts.append(f"best val {value:.5f} @ {step:,}")
    if len(runs) > 1:
        parts.append(f"(+{len(runs) - 1} more)")
    return "  ".join(parts)


# =============================================================================
# CLI
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Save folders, checkpoint directories or metrics.jsonl files.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="One row per run instead of a report each. What a hyperparameter probe wants.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        nargs="?",
        # A bare `--plot` yields this sentinel, resolved below against the run
        # being reported on. Named rather than empty because argparse puts
        # `const` through `type`, and `Path("")` is `Path(".")`.
        const=str(PLOT_BESIDE_METRICS),
        default=None,
        help="Write the curves here (PNG). Bare --plot, or --watch, defaults to "
        "training.png beside the first run's metrics.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Redraw the plot whenever the metrics file grows, until interrupted. What to "
        "leave running beside a fine-tune.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Seconds between --watch polls. The trainer appends every LOG_INTERVAL steps, "
        "so anything under a minute redraws as often as there is anything new.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write the training loss series as CSV, for a spreadsheet or another tool.",
    )
    args = parser.parse_args(argv)

    plot_path = args.plot
    if args.watch and plot_path is None:
        plot_path = PLOT_BESIDE_METRICS
    if plot_path == PLOT_BESIDE_METRICS:
        first = find_metrics_file(args.runs[0])
        if first is None:
            print(f"No {METRICS_FILENAME} at or beside {args.runs[0]}.", file=sys.stderr)
            return 1
        plot_path = first.parent / "training.png"

    if args.watch:
        # Deliberately before reading anything: a watch started at the same
        # moment as the run has no metrics to read yet, and waiting for the
        # first ones is the whole job.
        return watch_plot(args.runs, plot_path, args.interval)

    runs = []
    for target in args.runs:
        try:
            runs.append(read_run(target))
        except FileNotFoundError as error:
            print(f"skipping {target}: {error}", file=sys.stderr)
    if not runs:
        return 1

    if args.compare:
        print(format_comparison(runs))
    else:
        print("\n\n".join(format_run(run) for run in runs))

    if args.csv:
        _write_csv(runs, args.csv)
        print(f"Wrote {args.csv}")
    if plot_path is not None:
        if write_plot(runs, plot_path):
            print(f"Wrote {plot_path}")
        else:
            print("matplotlib is not installed, so no plot was written.", file=sys.stderr)
    return 0


def _write_csv(runs: list[Run], path: Path) -> None:
    """Every recorded series, long-form: one row per run, split, step and metric."""
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "split", "label", "step", "metric", "value"])
        for run in runs:
            for record in run.train:
                for name, value in sorted(record.metrics.items()):
                    writer.writerow([run.name, "train", "", record.step, name, value])
            for label, records in sorted(run.evals.items()):
                for record in records:
                    for name, value in sorted(record.metrics.items()):
                        writer.writerow([run.name, "eval", label, record.step, name, value])


if __name__ == "__main__":
    raise SystemExit(main())
