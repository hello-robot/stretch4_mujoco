"""
Plot a `run_finetune.py` run's metrics across steps, from the `metrics.jsonl` its trainer appends
to (`finetune/train.py`), as one figure, `metrics.png` in the run directory: a panel per loss
with the train curve (raw and smoothed) and the validation one, the flow loss per joint and per
step of the action horizon, the learning rates, the gradient norms, and each of the other metrics
the trainer logs (throughput, peak GPU memory, batch statistics).

`run_finetune.py` redraws it every --plot-interval steps while training and when training ends or
stops; this draws it any time.

Usage:
    python -m examples.vla.molmobot_droid.plot_finetune outputs/molmobot_droid/finetuning/models/<run>
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import click

from examples.vla.molmobot_droid.finetune.stretch4_joints import JOINTS

METRICS_FILE = "metrics.jsonl"
FIGURE_FILE = "metrics.png"
COLUMNS = 3
SKIPPED = {"throughput/total_tokens"}  # cumulative: a straight line
FIRST = "action_flow_loss"  # the loss the action expert trains on; the LLM's (CrossEntropyLoss, ...) follow
LAST = ("batch/",)  # batch statistics after throughput and memory
# The dataviz reference palette: its categorical order (train blue, val orange, ...), and the ends
# of its blue ramp for the ordered horizon steps.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RAMP = ("#86b6ef", "#0d366b")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PER_INDEX = re.compile(r"^(?P<loss>.+)_(?P<axis>dim|time)_(?P<index>\d+)$")
"""Losses logged per action dimension (a joint slot) or per step of the action horizon."""


def load_metrics(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    """{"train": {step: metrics}, "val": {step: metrics}}; a step logged again (a resumed run) keeps
    its last values."""
    runs: dict[str, dict[int, dict[str, float]]] = {"train": {}, "val": {}}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line still being written
            continue
        runs.setdefault(record["kind"], {}).setdefault(record["step"], {}).update(record["metrics"])
    return runs


def latest_step(path: Path) -> int:
    """The last train step in a metrics file, 0 when it has none (yet)."""
    if not path.exists():
        return 0
    return max(load_metrics(path)["train"], default=0)


def series(by_step: dict[int, dict[str, float]], name: str) -> tuple[list[int], list[float]]:
    steps = sorted(s for s, m in by_step.items() if name in m and math.isfinite(m[name]))
    return steps, [by_step[s][name] for s in steps]


def smoothed(values: list[float]) -> list[float]:
    """An exponential moving average spanning about 1/50 of the points."""
    alpha = 2 / (max(1, len(values) // 50) + 1)
    out, average = [], values[0]
    for value in values:
        average = alpha * value + (1 - alpha) * average
        out.append(average)
    return out


def ramp(count: int) -> list[str]:
    """`count` colors from light to dark along RAMP, for ordered series."""
    (r0, g0, b0), (r1, g1, b1) = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in RAMP]
    return [f"#{round(r0 + (r1 - r0) * t):02x}{round(g0 + (g1 - g0) * t):02x}{round(b0 + (b1 - b0) * t):02x}"
            for t in ((i / max(1, count - 1)) for i in range(count))]


def per_index_panels(names: list[str], kind: str) -> tuple[list[dict], set[str]]:
    """A panel per loss logged per joint or per horizon step, in `kind`'s names."""
    families: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for name in names:
        prefix, _, metric = name.partition("/")
        if match := PER_INDEX.match(metric):
            families.setdefault((f"{prefix}/{match['loss']}", match["axis"]), []).append((int(match["index"]), name))
    out = []
    for (loss, axis), members in families.items():
        members.sort()
        if axis == "dim":  # identity: categorical, every joint in the legend
            lines = [(JOINTS[i] if i < len(JOINTS) else f"dim {i}", n, kind, c)
                     for (i, n), c in zip(members, COLORS)]
            title = f"{loss} per joint"
        else:  # order: light (first step) to dark (last)
            last = members[-1][0]
            lines = [(f"step {i}" if i in (0, last) else None, n, kind, c)
                     for (i, n), c in zip(members, ramp(len(members)))]
            title = f"{loss} per horizon step"
        out.append({"title": title, "lines": lines, "log": False})
    return out, {n for members in families.values() for _, n in members}


def panels(runs: dict[str, dict[int, dict[str, float]]]) -> list[dict]:
    """What each panel shows: {"title", "lines": [(label or None, name, kind, color)], "log"}."""
    train_names = list(dict.fromkeys(n for m in runs["train"].values() for n in m if n not in SKIPPED))
    val_names = list(dict.fromkeys(n for m in runs["val"].values() for n in m))
    train_indexed, used = per_index_panels(train_names, "train")
    val_indexed, val_used = per_index_panels(val_names, "val")
    used |= val_used
    out = []
    train_names.sort(key=lambda n: n != f"train/{FIRST}")
    for name in train_names:  # each train loss, with the validation losses of the same name
        if not name.startswith("train/") or name in used:
            continue
        metric = name.split("/", 1)[1]
        matching = [v for v in val_names if v.split("/", 1)[-1] == metric and v not in used]
        out.append({"title": metric, "log": False, "lines": [("train", name, "train", COLORS[0])]
                    + [(v.split("/", 1)[0], v, "val", c) for v, c in zip(matching, COLORS[1:])]})
        used.update([name, *matching])
    out += [{"title": v, "lines": [(None, v, "val", COLORS[1])], "log": False} for v in val_names if v not in used]
    out += train_indexed + val_indexed
    rates = [n for n in train_names if n.startswith("optim/") and n.endswith("_lr")]
    if rates:
        out.append({"title": "learning rate", "log": True, "lines": [
            (n[len("optim/"):-len("_lr")], n, "train", c) for n, c in zip(rates, COLORS)]})
    norms = [n for n in train_names if n.startswith("optim/") and n.endswith("grad_norm")]
    if norms:
        out.append({"title": "gradient norm", "log": True, "lines": [
            (n[len("optim/"):].removesuffix("grad_norm").rstrip("_") or "total", n, "train", c)
            for n, c in zip(norms, COLORS)]})
    used.update(rates + norms)
    rest = sorted((n for n in train_names if n not in used), key=lambda n: n.startswith(LAST))
    out += [{"title": n, "lines": [(None, n, "train", COLORS[0])], "log": False} for n in rest]
    return out


def plot_metrics(run_dir: Path) -> Path | None:
    """`run_dir`'s metrics as one figure, `metrics.png`; None when it has no metrics yet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = run_dir / METRICS_FILE
    if not path.exists():
        return None
    runs = load_metrics(path)
    layout = panels(runs)
    if not layout:
        return None
    rows = math.ceil(len(layout) / COLUMNS)
    figure, axes = plt.subplots(rows, COLUMNS, figsize=(6.4 * COLUMNS, 3.8 * rows), squeeze=False,
                                facecolor=SURFACE)
    for ax, panel in zip(axes.flat, layout):
        ax.set_facecolor(SURFACE)
        for label, name, kind, color in panel["lines"]:
            steps, values = series(runs[kind], name)
            if not steps:
                continue
            raw = kind == "train" and len(values) > 50 and len(panel["lines"]) == 1
            if raw:  # noisy per step: raw faint, smoothed on top (several lines: smoothed only)
                ax.plot(steps, values, color=color, linewidth=0.8, alpha=0.25)
            if kind == "train" and len(values) > 50:
                values = smoothed(values)
            ax.plot(steps, values, color=color, linewidth=2 if len(panel["lines"]) <= 8 else 1.4, label=label,
                    marker="o" if kind == "val" else None, markersize=4)
        ax.set_title(panel["title"], color=INK, fontsize=11, loc="left")
        if panel["log"]:
            ax.set_yscale("log")
        else:
            ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_xlabel("step", color=MUTED, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        if len(panel["lines"]) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, ncols=2 if len(panel["lines"]) > 4 else 1)
    for ax in axes.flat[len(layout):]:
        ax.set_visible(False)
    last = max(runs["train"], default=0)
    figure.suptitle(f"{run_dir.name}: step {last}", color=INK, fontsize=13, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.35 / (3.8 * rows)))  # room for the title
    out = run_dir / FIGURE_FILE
    figure.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(figure)
    return out


@click.command()
@click.argument("run_dirs", nargs=-1, required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def main(run_dirs):
    for run_dir in run_dirs:
        out = plot_metrics(run_dir)
        click.echo(f"wrote {out}" if out else f"No {METRICS_FILE} in {run_dir} yet")


if __name__ == "__main__":
    main()
