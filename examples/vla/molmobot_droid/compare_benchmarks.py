"""
Compare a Franka DROID benchmark run with a Stretch 4 one: side-by-side videos for every
episode both ran, and `comparison_report.md` with both runs' flags, success rates and
per-episode results.

Usage:
    python -m examples.vla.molmobot_droid.compare_benchmarks \\
        --franka outputs/molmobot_droid/franka_exo-droid_eh-8_n-2 \\
        --stretch4 outputs/molmobot_droid/stretch4_exo-center_grip-right_eh-8_n-2_...
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import cv2
import numpy as np

from examples.vla.molmobot_droid.checkpoint import POLICY_HZ
from examples.vla.molmobot_droid.molmospaces.benchmark import episode_name

SIDE_BY_SIDE_CAMERAS = ("grid", "scene")
TILE_HEIGHT = 540


def load_run(run_dir: Path) -> dict:
    results = run_dir / "results.json"
    if not results.exists():
        raise click.ClickException(f"No results.json in {run_dir}")
    run = json.loads(results.read_text())
    run["dir"] = run_dir
    run["by_index"] = {r["index"]: r for r in run["results"]}
    return run


def episode_video(run: dict, index: int, camera: str) -> Path | None:
    path = run["dir"] / f"{run['run_name']}_{episode_name(index)}_{camera}.mp4"
    return path if path.exists() else None


def read_frames(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    with imageio.get_reader(path) as reader:
        return [frame for frame in reader]


def labelled(frame: np.ndarray, label: str, success: bool) -> np.ndarray:
    tile = cv2.resize(frame, (round(frame.shape[1] * TILE_HEIGHT / frame.shape[0]), TILE_HEIGHT))
    color = (60, 220, 60) if success else (230, 60, 60)
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(tile, f"{label}: {'success' if success else 'fail'}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return tile


def write_side_by_side(left: Path, right: Path, out: Path, left_label, right_label, left_ok, right_ok) -> None:
    """Both videos next to each other; the shorter one holds its last frame."""
    import imageio.v2 as imageio

    a, b = read_frames(left), read_frames(right)
    length = max(len(a), len(b))
    with imageio.get_writer(out, fps=POLICY_HZ, codec="libx264", quality=7, macro_block_size=8) as writer:
        size = None
        for i in range(length):
            frame = np.hstack(
                [labelled(a[min(i, len(a) - 1)], left_label, left_ok), labelled(b[min(i, len(b) - 1)], right_label, right_ok)]
            )
            if size is None:
                size = (frame.shape[1] - frame.shape[1] % 8, frame.shape[0] - frame.shape[0] % 8)
            writer.append_data(cv2.resize(frame, size))


def flags_table(franka: dict, stretch: dict) -> list[str]:
    keys = list(dict.fromkeys([*franka["flags"], *stretch["flags"]]))
    lines = ["| Flag | Franka | Stretch 4 |", "|---|---|---|"]
    for key in keys:
        lines.append(f"| `--{key}` | {_cell(franka['flags'].get(key))} | {_cell(stretch['flags'].get(key))} |")
    return lines


def _cell(value) -> str:
    return "—" if value is None else f"`{value}`"


def comparison_report(franka: dict, stretch: dict, common: list[int], videos: dict[int, list[str]]) -> str:
    def rate(run, indices):
        hits = sum(run["by_index"][i]["success"] for i in indices)
        return hits, (hits / len(indices) if indices else 0.0)

    f_hits, f_rate = rate(franka, common)
    s_hits, s_rate = rate(stretch, common)
    both = sum(franka["by_index"][i]["success"] and stretch["by_index"][i]["success"] for i in common)
    only_franka = [i for i in common if franka["by_index"][i]["success"] and not stretch["by_index"][i]["success"]]
    only_stretch = [i for i in common if stretch["by_index"][i]["success"] and not franka["by_index"][i]["success"]]

    lines = [
        "# MolmoBot-DROID: Franka DROID vs. Stretch 4",
        "",
        f"On the {len(common)} episodes both runs have:",
        "",
        "| | Franka | Stretch 4 |",
        "|---|---|---|",
        f"| Success | {f_hits}/{len(common)} ({f_rate:.0%}) | {s_hits}/{len(common)} ({s_rate:.0%}) |",
        f"| Run | `{franka['run_name']}` | `{stretch['run_name']}` |",
        f"| Benchmark | `{franka['benchmark']}` | `{stretch['benchmark']}` |",
        f"| Checkpoint | `{franka['checkpoint']}` | `{stretch['checkpoint']}` |",
        "",
        f"Success rate difference (Stretch 4 − Franka): **{(s_rate - f_rate) * 100:+.0f} points**. "
        f"Both succeeded on {both}; only the Franka on {len(only_franka)} {only_franka or ''}; "
        f"only Stretch 4 on {len(only_stretch)} {only_stretch or ''}.",
        "",
        "## Flags",
        "",
        *flags_table(franka, stretch),
        "",
        "## Episodes",
        "",
        "| # | Instruction | Franka | Stretch 4 | Franka steps | Stretch 4 steps | Stretch IK failures | "
        "Stretch IK clamped | Videos |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i in common:
        f, s = franka["by_index"][i], stretch["by_index"][i]
        lines.append(
            f"| {i} | {f['instruction']} | {_outcome(f)} | {_outcome(s)} | {f['steps']} | {s['steps']} | "
            f"{_metric(s, 'ik_failures')} | {_metric(s, 'ik_clamped')} | "
            + " ".join(f"[{v.split('_')[1]}]({v})" for v in videos.get(i, []))
            + " |"
        )
    missing = sorted(set(franka["by_index"]) ^ set(stretch["by_index"]))
    if missing:
        lines += ["", f"Episodes in only one run (not compared): {missing}"]
    return "\n".join(lines) + "\n"


def _outcome(result: dict) -> str:
    if result.get("error"):
        return "error"
    return "success" if result["success"] else "fail"


def _metric(result: dict, name: str) -> str:
    value = result.get("metrics", {}).get(name)
    return "" if value is None else f"{value:g}"


@click.command()
@click.option("--franka", "franka_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--stretch4", "stretch_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--out", default=None, type=click.Path(), help="Default: next to the two runs.")
@click.option("--no-videos", is_flag=True, help="Only write the report.")
def main(franka_dir, stretch_dir, out, no_videos):
    franka, stretch = load_run(Path(franka_dir)), load_run(Path(stretch_dir))
    out_dir = Path(out) if out else Path(stretch_dir).parent / f"compare__{franka['run_name']}__vs__{stretch['run_name']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    common = sorted(set(franka["by_index"]) & set(stretch["by_index"]))
    if not common:
        raise click.ClickException("The runs have no episodes in common")

    videos: dict[int, list[str]] = {}
    if not no_videos:
        for i in common:
            for camera in SIDE_BY_SIDE_CAMERAS:
                left, right = episode_video(franka, i, camera), episode_video(stretch, i, camera)
                if left is None or right is None:
                    continue
                name = f"{episode_name(i)}_{camera}_side_by_side.mp4"
                write_side_by_side(
                    left, right, out_dir / name, "Franka", "Stretch 4",
                    franka["by_index"][i]["success"], stretch["by_index"][i]["success"],
                )
                videos.setdefault(i, []).append(name)
                click.echo(f"wrote {out_dir / name}")

    report = out_dir / "comparison_report.md"
    report.write_text(comparison_report(franka, stretch, common, videos))
    click.secho(f"Report: {report}", fg="green")


if __name__ == "__main__":
    main()
