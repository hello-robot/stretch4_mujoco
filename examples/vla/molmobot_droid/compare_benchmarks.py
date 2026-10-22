"""
Compare a Franka DROID benchmark run with one or more Stretch 4 ones: `comparison_report.md`
with the runs' flags, success rates, head-to-head differences and per-episode results, and with
--videos, each episode's grid videos stacked top to bottom (Franka first).

Usage:
    python -m examples.vla.molmobot_droid.compare_benchmarks \\
        --franka outputs/molmobot_droid/franka_exo-droid_eh-8_n-2 \\
        --stretch4 outputs/molmobot_droid/stretch4_exo-left_grip-left_eh-8_n-2_...
    # Several Stretch 4 runs, e.g. with and without a fix, against the same Franka run:
    python -m examples.vla.molmobot_droid.compare_benchmarks --franka <run> \\
        --stretch4 <run> --stretch4 <run> --videos
    # Every run in a directory (one Franka run; pass --franka too if there are several):
    python -m examples.vla.molmobot_droid.compare_benchmarks --all outputs/molmobot_droid
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import click
import cv2
import numpy as np

from examples.vla.molmobot_droid.checkpoint import POLICY_HZ
from examples.vla.molmobot_droid.molmospaces.benchmark import (
    FLAG_ABBREVIATIONS,
    STRETCH_MAST_HEIGHT,
    above_mast,
    benchmark_description,
    episode_name,
    reachable_cell,
    scene_table,
    video_path,
)

LABEL_HEIGHT = 40


def load_run(run_dir: Path, label: str) -> dict:
    results = run_dir / "results.json"
    if not results.exists():
        raise click.ClickException(f"No results.json in {run_dir}")
    run = json.loads(results.read_text())
    run["dir"] = run_dir
    run["label"] = label
    run["by_index"] = {r["index"]: r for r in run["results"]}
    return run


def stretch_labels(flags: list[dict]) -> list[str]:
    """ "Stretch 4" for one run; for several, numbered and with the flags that set each apart."""
    if len(flags) == 1:
        return ["Stretch 4"]
    keys = dict.fromkeys(k for f in flags for k in f)
    differing = [k for k in keys if len({repr(f.get(k)) for f in flags}) > 1]
    labels = []
    for n, f in enumerate(flags, 1):
        parts = [f"{FLAG_ABBREVIATIONS.get(k, k)}-{int(f[k]) if isinstance(f[k], bool) else f[k]}"
                 for k in differing if k in f]
        labels.append(f"Stretch 4 #{n}" + (f" ({', '.join(parts)})" if parts else ""))
    return labels


def episode_video(run: dict, index: int, camera: str) -> Path | None:
    prefix = f"{run['run_name']}_{episode_name(index)}"
    # Runs from before grids moved to their own folder have them next to the other videos.
    for path in (video_path(run["dir"], prefix, camera), run["dir"] / f"{prefix}_{camera}.mp4"):
        if path.exists():
            return path
    return None


def read_frames(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    with imageio.get_reader(path) as reader:
        return [frame for frame in reader]


def labelled(frame: np.ndarray, label: str, success: bool, width: int) -> np.ndarray:
    """`frame` at `width`, under a bar naming its run and outcome."""
    if frame.shape[1] != width:
        frame = cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])))
    bar = np.zeros((LABEL_HEIGHT, width, 3), np.uint8)
    color = (60, 220, 60) if success else (230, 60, 60)
    cv2.putText(bar, f"{label}: {'success' if success else 'fail'}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return np.vstack([bar, frame])


def write_stacked(videos: list[Path], out: Path, labels: list[str], successes: list[bool]) -> None:
    """The videos one above the other at the first one's width; the shorter ones hold their last frame."""
    import imageio.v2 as imageio

    clips = [read_frames(v) for v in videos]
    width = clips[0][0].shape[1]
    length = max(len(c) for c in clips)
    with imageio.get_writer(out, fps=POLICY_HZ, codec="libx264", quality=7, macro_block_size=8) as writer:
        size = None
        for i in range(length):
            frame = np.vstack(
                [labelled(c[min(i, len(c) - 1)], label, ok, width) for c, label, ok in zip(clips, labels, successes)]
            )
            if size is None:
                size = (frame.shape[1] - frame.shape[1] % 8, frame.shape[0] - frame.shape[0] % 8)
            writer.append_data(cv2.resize(frame, size))


def flags_table(runs: list[dict]) -> list[str]:
    keys = list(dict.fromkeys(k for run in runs for k in run["flags"]))
    lines = ["| Flag | " + " | ".join(run["label"] for run in runs) + " |", "|---" * (len(runs) + 1) + "|"]
    for key in keys:
        lines.append(f"| `--{key}` | " + " | ".join(_cell(run["flags"].get(key)) for run in runs) + " |")
    return lines


def _cell(value) -> str:
    return "—" if value is None else f"`{value}`"


def _rate(run: dict, indices: list[int]) -> tuple[int, float]:
    hits = sum(run["by_index"][i]["success"] for i in indices)
    return hits, (hits / len(indices) if indices else 0.0)


def difference(a: dict, b: dict, common: list[int]) -> str:
    """How `b` does against `a` on the episodes both ran."""
    both = sum(a["by_index"][i]["success"] and b["by_index"][i]["success"] for i in common)
    only_a = [i for i in common if a["by_index"][i]["success"] and not b["by_index"][i]["success"]]
    only_b = [i for i in common if b["by_index"][i]["success"] and not a["by_index"][i]["success"]]
    return (
        f"Success rate difference ({b['label']} − {a['label']}): "
        f"**{(_rate(b, common)[1] - _rate(a, common)[1]) * 100:+.0f} points**. "
        f"Both succeeded on {both}; only {a['label']} on {len(only_a)} {only_a or ''}; "
        f"only {b['label']} on {len(only_b)} {only_b or ''}."
    )


def comparison_report(franka: dict, stretches: list[dict], common: list[int], videos: dict[int, str]) -> str:
    runs = [franka, *stretches]

    def row(name: str, values) -> str:
        return f"| {name} | " + " | ".join(values) + " |"

    def success(run: dict, indices: list[int]) -> str:
        hits, rate = _rate(run, indices)
        return f"{hits}/{len(indices)} ({rate:.0%})"

    reachable = [i for i in common if not above_mast(franka["by_index"][i].get("scene"))]

    per_stretch = lambda title: [f"{title}" if len(stretches) == 1 else f"{s['label']} {title[0].lower()}{title[1:]}"
                                 for s in stretches]
    benchmarks = list(dict.fromkeys(run["benchmark"] for run in runs))
    lines = [
        "# MolmoBot-DROID: Franka DROID vs. Stretch 4",
        "",
        *[f"**Benchmark:** {benchmark_description(b)}" for b in benchmarks],
        *(["", "> **The runs used different benchmarks**, so their episodes are not the same."]
          if len(benchmarks) > 1 else []),
        "",
        f"On the {len(common)} episodes {'both runs have' if len(runs) == 2 else 'all runs have'}:",
        "",
        "| | " + " | ".join(run["label"] for run in runs) + " |",
        "|---" * (len(runs) + 1) + "|",
        row("Success", [success(run, common) for run in runs]),
        row(f"Success when reachable (target below Stretch 4's {STRETCH_MAST_HEIGHT} m mast)",
            [success(run, reachable) for run in runs]),
        row("Run", [f"`{run['run_name']}`" for run in runs]),
        row("Benchmark", [f"`{run['benchmark']}`" for run in runs]),
        # Runs from before `scene_variant` was recorded used the "base" scenes.
        row("Scenes", [f"`{run.get('scene_variant', 'base')}`" for run in runs]),
        row("Checkpoint", [f"`{run['checkpoint']}`" for run in runs]),
        "",
        *[line for s in stretches for line in (difference(franka, s, common), "")],
        *[line for a, b in combinations(stretches, 2) for line in (difference(a, b, common), "")],
        "## Flags",
        "",
        *flags_table(runs),
        "",
        "## Episodes",
        "",
        "| # | Episode | Scene | Instruction | Reachable | " + " | ".join(run["label"] for run in runs) + " | "
        + " | ".join(f"{run['label']} steps" for run in runs) + " | "
        + " | ".join(per_stretch("Stretch IK failures") + per_stretch("Stretch IK clamped")) + " | Videos |",
        "|---" * (2 * len(runs) + 2 * len(stretches) + 6) + "|",
    ]
    for i in common:
        results = [run["by_index"][i] for run in runs]
        lines.append(
            f"| {i} | {episode_name(i)} | {results[0]['scene_id']} | {results[0]['instruction']} | "
            f"{reachable_cell(results[0].get('scene'))} | "
            + " | ".join(_outcome(r) for r in results) + " | "
            + " | ".join(str(r["steps"]) for r in results) + " | "
            + " | ".join([_metric(r, "ik_failures") for r in results[1:]] + [_metric(r, "ik_clamped") for r in results[1:]])
            + " | "
            + (f"[grid]({videos[i]})" if i in videos else "")
            + " |"
        )
    missing = sorted(set().union(*(run["by_index"] for run in runs)) - set(common))
    if missing:
        lines += ["", f"Episodes not in every run (not compared): {missing}"]
    lines += [
        "",
        "## Scenes",
        "",
        "Where each episode happens, as the Franka run built it. Stretch 4's footprint and the furniture "
        "removed for it are in each Stretch 4 run's `report.md`. To rerun episodes (e.g. `--episodes 3,7`):",
        "",
        *[f"- {run['label']}: `{run['command']} --episodes <#>`" if run.get("command")
          else f"- {run['label']}: (not recorded; the run predates it)" for run in runs],
        "",
        *scene_table([franka["by_index"][i] for i in common]),
    ]
    return "\n".join(lines) + "\n"


def _outcome(result: dict) -> str:
    if result.get("error"):
        return "error"
    return "success" if result["success"] else "fail"


def _metric(result: dict, name: str) -> str:
    value = result.get("metrics", {}).get(name)
    return "" if value is None else f"{value:g}"


def default_out_dir(franka: dict, stretches: list[dict]) -> Path:
    parent = stretches[0]["dir"].parent
    if len(stretches) == 1:
        return parent / f"compare__{franka['run_name']}__vs__{stretches[0]['run_name']}"
    # Several run names are too long for one directory name.
    digest = hashlib.sha1("\n".join(s["run_name"] for s in stretches).encode()).hexdigest()[:8]
    return parent / f"compare__{franka['run_name']}__vs__{len(stretches)}_stretch4_runs_{digest}"


def find_runs(directory: Path) -> dict[str, list[Path]]:
    """The runs in `directory`'s subfolders, {robot: [run dir, ...]}, from their `results.json`."""
    runs: dict[str, list[Path]] = {}
    for results in sorted(directory.glob("*/results.json")):
        runs.setdefault(json.loads(results.read_text())["robot"], []).append(results.parent)
    return runs


@click.command()
@click.option("--franka", "franka_dir", default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--stretch4", "stretch_dirs", multiple=True, type=click.Path(exists=True, file_okay=False),
              help="A Stretch 4 run. Give it more than once to compare several.")
@click.option("--all", "all_dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Compare every run in this directory's subfolders, in place of --stretch4 (and of --franka, "
              "unless there are several Franka runs).")
@click.option("--out", default=None, type=click.Path(), help="Default: next to the runs.")
@click.option("--videos", is_flag=True, help="Also stack each episode's grid videos, one run above the other.")
def main(franka_dir, stretch_dirs, all_dir, out, videos):
    if all_dir:
        if stretch_dirs:
            raise click.UsageError("--all finds the Stretch 4 runs itself; leave out --stretch4")
        found = find_runs(Path(all_dir))
        stretch_dirs = found.get("stretch4", [])
        if franka_dir is None:
            frankas = found.get("franka", [])
            if len(frankas) != 1:
                raise click.UsageError(
                    f"--all needs exactly one Franka run in {all_dir}, found {len(frankas)}"
                    + (f": {[f.name for f in frankas]}. Pick one with --franka." if frankas else ".")
                )
            franka_dir = frankas[0]
        click.echo(f"Franka: {Path(franka_dir).name}\n" + "\n".join(f"Stretch 4: {Path(d).name}" for d in stretch_dirs))
    if franka_dir is None or not stretch_dirs:
        raise click.UsageError("Need --franka and at least one --stretch4, or --all")
    franka = load_run(Path(franka_dir), "Franka")
    stretch_flags = [json.loads((Path(d) / "results.json").read_text())["flags"]
                     if (Path(d) / "results.json").exists() else {} for d in stretch_dirs]
    stretches = [load_run(Path(d), label) for d, label in zip(stretch_dirs, stretch_labels(stretch_flags))]
    runs = [franka, *stretches]
    out_dir = Path(out) if out else default_out_dir(franka, stretches)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = sorted(set.intersection(*(set(run["by_index"]) for run in runs)))
    if not common:
        raise click.ClickException("The runs have no episodes in common")

    stacked: dict[int, str] = {}
    if videos:
        for i in common:
            paths = [p for run in runs if (p := episode_video(run, i, "grid")) is not None]
            if len(paths) < len(runs):
                continue
            name = f"{episode_name(i)}_grid_stacked.mp4"
            write_stacked(
                paths, out_dir / name, [run["label"] for run in runs], [run["by_index"][i]["success"] for run in runs]
            )
            stacked[i] = name
            click.echo(f"wrote {out_dir / name}")
    else:  # link the ones an earlier --videos wrote
        stacked = {i: name for i in common if (out_dir / (name := f"{episode_name(i)}_grid_stacked.mp4")).exists()}

    report = out_dir / "comparison_report.md"
    report.write_text(comparison_report(franka, stretches, common, stacked))
    click.secho(f"Report: {report}", fg="green")


if __name__ == "__main__":
    main()
