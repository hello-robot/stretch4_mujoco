"""
Turn a finished benchmark run into artifacts you can show someone.

`run_benchmarks.py` leaves behind what MolmoSpaces' rollout pipeline writes: one
HDF5 per house with per-step JSON blobs, and one MP4 per camera per episode. That
is complete but not presentable -- the numbers are in string blobs, the videos
are unlabelled, and nothing ties an episode's footage to whether it succeeded.

This reads that output and writes, per episode:

    episode_XXXXXXXX_review.mp4   cameras tiled side by side, captioned with the
                                  step, the outcome and the instruction
    episode_XXXXXXXX.csv          per-step joint positions, commanded targets,
                                  tool pose and base pose
    summary.json / summary.md     per-episode outcomes and the run's success rate

    python -m examples.machine_learning.molmospaces.report eval_output/stretch4/<run>

Point it at any level of a run: a single `house_*` directory, a benchmark's
output directory, or the whole sweep.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import click
import numpy as np

log = logging.getLogger(__name__)

_TRAJECTORY_FILE_PATTERN = re.compile(r"^trajectories(?P<suffix>.*)\.h5$")
_TRAJECTORY_KEY_PATTERN = re.compile(r"^traj_(?P<index>\d+)$")

CAPTION_HEIGHT_PX = 64
CAPTION_COLOUR_SUCCESS = (90, 200, 90)
CAPTION_COLOUR_FAILURE = (90, 90, 220)


@dataclass
class EpisodeReport:
    """What one episode produced."""

    house: str
    episode: int
    steps: int
    success: bool
    instruction: str = ""
    video: str = ""
    telemetry: str = ""
    final_reward: float = 0.0
    extra: dict = field(default_factory=dict)


def decode_json_blob(row: np.ndarray) -> dict:
    """MolmoSpaces stores dict observations as NUL-padded UTF-8 in a uint8 row."""
    text = bytes(row).rstrip(b"\x00").decode("utf-8", "replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def build_report(
    run_dir: Path,
    output_dir: Path | None = None,
    video: bool = True,
    max_episodes: int | None = None,
) -> list[EpisodeReport]:
    """Write artifacts for every episode under `run_dir`.

    Args:
        run_dir: anything containing `house_*/trajectories*.h5` -- one house, one
            benchmark's output, or a whole sweep.
        output_dir: where artifacts go. Defaults to `<run_dir>/report`.
        video: render the captioned review videos. Turning this off makes the
            pass fast when only the numbers are wanted.
        max_episodes: stop after this many episodes.

    Returns:
        One `EpisodeReport` per episode, in the order they were found.
    """
    import h5py

    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[EpisodeReport] = []
    for h5_path in sorted(run_dir.rglob("trajectories*.h5")):
        match = _TRAJECTORY_FILE_PATTERN.match(h5_path.name)
        if match is None:
            continue
        batch_suffix = match.group("suffix")
        house = h5_path.parent.name

        with h5py.File(h5_path, "r") as h5_file:
            for traj_key in sorted(h5_file.keys()):
                key_match = _TRAJECTORY_KEY_PATTERN.match(traj_key)
                if key_match is None:
                    continue
                if max_episodes is not None and len(reports) >= max_episodes:
                    break

                episode_index = int(key_match.group("index"))
                report = _report_episode(
                    trajectory=h5_file[traj_key],
                    episode_dir=h5_path.parent,
                    house=house,
                    episode_index=episode_index,
                    batch_suffix=batch_suffix,
                    output_dir=output_dir,
                    render_video=video,
                )
                reports.append(report)
                log.info(
                    f"[report] {house}/{traj_key}: {report.steps} steps, "
                    f"{'SUCCESS' if report.success else 'failure'}"
                )

    _write_summary(reports, output_dir)
    return reports


def _report_episode(
    trajectory,
    episode_dir: Path,
    house: str,
    episode_index: int,
    batch_suffix: str,
    output_dir: Path,
    render_video: bool,
) -> EpisodeReport:
    qpos_rows = trajectory["obs/agent/qpos"][:]
    action_rows = trajectory["actions/joint_pos"][:]
    num_steps = len(qpos_rows)
    success_flags = (
        trajectory["success"][:] if "success" in trajectory else np.zeros(num_steps, bool)
    )
    success = bool(np.any(success_flags))

    stem = f"{house}_episode_{episode_index:08d}"
    report = EpisodeReport(
        house=house,
        episode=episode_index,
        steps=num_steps,
        success=success,
        instruction=_instruction(trajectory),
        final_reward=float(trajectory["rewards"][-1]) if "rewards" in trajectory else 0.0,
    )

    telemetry_path = output_dir / f"{stem}.csv"
    _write_telemetry(trajectory, qpos_rows, action_rows, num_steps, telemetry_path)
    report.telemetry = telemetry_path.name

    if render_video:
        video_path = _render_review_video(
            episode_dir, episode_index, batch_suffix, num_steps, report, output_dir / f"{stem}.mp4"
        )
        report.video = video_path.name if video_path is not None else ""
    return report


def _instruction(trajectory) -> str:
    """The episode's language instruction, if the task recorded one.

    It lives in the scalar `obs_scene` dataset -- a JSON string holding the
    task type, the description and the referral expressions -- rather than in
    the per-step `task_info` blob, which carries only progress metrics.
    """
    if "obs_scene" not in trajectory:
        return ""
    raw = trajectory["obs_scene"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        scene = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    description = scene.get("task_description")
    return description if isinstance(description, str) else ""


def _write_telemetry(trajectory, qpos_rows, action_rows, num_steps: int, path: Path) -> None:
    """Flatten the per-step blobs and pose arrays into one CSV.

    Column names are taken from the recorded move groups rather than fixed, so
    the same code reports a Stretch run and a Franka one.
    """
    import csv

    tcp_poses = trajectory["obs/extra/tcp_pose"][:] if "obs/extra/tcp_pose" in trajectory else None
    base_poses = (
        trajectory["obs/extra/robot_base_pose"][:]
        if "obs/extra/robot_base_pose" in trajectory
        else None
    )
    rewards = trajectory["rewards"][:] if "rewards" in trajectory else None
    successes = trajectory["success"][:] if "success" in trajectory else None

    rows = []
    for step in range(num_steps):
        row: dict[str, float | int | str] = {"step": step}
        for group, values in decode_json_blob(qpos_rows[step]).items():
            for index, value in enumerate(np.ravel(values)):
                row[f"qpos_{group}_{index}"] = round(float(value), 6)
        for group, values in decode_json_blob(action_rows[step]).items():
            for index, value in enumerate(np.ravel(values)):
                row[f"cmd_{group}_{index}"] = round(float(value), 6)
        if tcp_poses is not None:
            row.update(
                {f"tcp_{axis}": round(float(tcp_poses[step][i]), 6) for i, axis in enumerate("xyz")}
            )
        if base_poses is not None:
            row.update(
                {
                    f"base_{axis}": round(float(base_poses[step][i]), 6)
                    for i, axis in enumerate("xyz")
                }
            )
        if rewards is not None:
            row["reward"] = round(float(rewards[step]), 6)
        if successes is not None:
            row["success"] = int(bool(successes[step]))
        rows.append(row)

    # Union the keys: a step where the policy commanded nothing has no cmd_*
    # columns of its own, and DictWriter refuses unknown keys later on.
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _render_review_video(
    episode_dir: Path,
    episode_index: int,
    batch_suffix: str,
    num_steps: int,
    report: EpisodeReport,
    path: Path,
) -> Path | None:
    """Tile this episode's camera MP4s side by side and caption every frame."""
    import cv2

    video_paths = sorted(episode_dir.glob(f"episode_{episode_index:08d}_*{batch_suffix}.mp4"))
    if not video_paths:
        log.warning(f"[report] no camera videos beside {episode_dir} for episode {episode_index}")
        return None

    captures = [cv2.VideoCapture(str(p)) for p in video_paths]
    names = [
        p.name.replace(f"episode_{episode_index:08d}_", "").replace(f"{batch_suffix}.mp4", "")
        for p in video_paths
    ]
    writer = None
    try:
        for step in range(num_steps):
            panels = []
            for capture, name in zip(captures, names):
                ok, frame = capture.read()
                if not ok:
                    break
                panels.append(_label_panel(frame, name))
            if len(panels) != len(captures):
                break

            # Cameras can differ in resolution (the head camera is wider than the
            # wrist one), so match heights before stacking horizontally.
            target_height = max(panel.shape[0] for panel in panels)
            panels = [
                cv2.resize(
                    panel,
                    (int(panel.shape[1] * target_height / panel.shape[0]), target_height),
                    interpolation=cv2.INTER_AREA,
                )
                for panel in panels
            ]
            tiled = np.hstack(panels)
            frame = _add_caption(tiled, step, num_steps, report)

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (width, height)
                )
            writer.write(frame)
    finally:
        for capture in captures:
            capture.release()
        if writer is not None:
            writer.release()

    return path if writer is not None else None


def _label_panel(frame: np.ndarray, name: str) -> np.ndarray:
    import cv2

    frame = frame.copy()
    cv2.putText(frame, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(
        frame, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    return frame


def _add_caption(frame: np.ndarray, step: int, num_steps: int, report: EpisodeReport) -> np.ndarray:
    """A banner under the frame: outcome, progress and the instruction."""
    import cv2

    width = frame.shape[1]
    banner = np.zeros((CAPTION_HEIGHT_PX, width, 3), dtype=np.uint8)
    banner[:] = CAPTION_COLOUR_SUCCESS if report.success else CAPTION_COLOUR_FAILURE

    outcome = "SUCCESS" if report.success else "FAILURE"
    headline = f"{report.house} ep{report.episode:04d}  {outcome}  step {step + 1}/{num_steps}"
    cv2.putText(
        banner, headline, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )
    if report.instruction:
        instruction = report.instruction
        # Truncate rather than wrap: the banner is one line by construction, and
        # a wrapped caption would change the frame height mid-video.
        max_characters = max(10, int(width / 9))
        if len(instruction) > max_characters:
            instruction = instruction[: max_characters - 1] + "…"
        cv2.putText(
            banner,
            instruction,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([frame, banner])


def _write_summary(reports: list[EpisodeReport], output_dir: Path) -> None:
    successes = sum(report.success for report in reports)
    total = len(reports)
    rate = successes / total if total else 0.0

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "episodes": total,
                "successes": successes,
                "success_rate": rate,
                "results": [asdict(report) for report in reports],
            },
            indent=2,
        )
    )

    lines = [
        "# Evaluation report",
        "",
        f"**{successes}/{total} successful** ({rate:.1%})",
        "",
        "| house | episode | steps | outcome | instruction | video | telemetry |",
        "| ----- | ------- | ----- | ------- | ----------- | ----- | --------- |",
    ]
    for report in reports:
        lines.append(
            f"| {report.house} | {report.episode} | {report.steps} "
            f"| {'success' if report.success else 'failure'} | {report.instruction} "
            f"| [{report.video}]({report.video}) | [{report.telemetry}]({report.telemetry}) |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


@click.command()
@click.argument("run_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write artifacts. Defaults to <run_dir>/report.",
)
@click.option(
    "--no-video",
    is_flag=True,
    help="Skip the captioned review videos and write only telemetry and summaries.",
)
@click.option("--max-episodes", type=int, default=None, help="Stop after this many episodes.")
def main(run_dir: Path, output_dir: Path | None, no_video: bool, max_episodes: int | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    reports = build_report(
        run_dir, output_dir=output_dir, video=not no_video, max_episodes=max_episodes
    )
    if not reports:
        raise click.ClickException(f"No trajectories found under {run_dir}")

    destination = output_dir or run_dir / "report"
    successes = sum(report.success for report in reports)
    click.secho(
        f"{successes}/{len(reports)} successful. Report written to {destination}", fg="green"
    )


if __name__ == "__main__":
    main()
