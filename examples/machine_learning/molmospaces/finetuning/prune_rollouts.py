"""
A rollout run cleaned by deleting videos -> HDF5 files that agree with what is left.

Reviewing a run means watching the MP4s, and rejecting a run means deleting its
MP4s -- that part needs no tooling. What it leaves behind is a directory whose
two halves disagree: the videos of the bad episodes are gone, and the
trajectories that describe them are still in `house_*/trajectories*.h5`,
pointing at files that no longer exist. This script removes those trajectories.

The removal is not just a `del f["traj_7"]`, for two reasons that are only
visible from the consumer's side:

**Trajectory indices have to stay consecutive.** MolmoBot's
`validate_trajectories.py` sorts the `traj_*` keys and checks them against
`range(len(keys))`; when they do not match it prints one line and writes an
*all-false* `valid_traj_mask`, so a single hole in the numbering silently
discards every remaining episode in that house. So the survivors are renumbered
`traj_0 .. traj_{n-1}`, in their original order.

**Renumbering breaks the `episode_%08d` convention.** The format's rule --
`hdf5_layout.video_filename` -- is that `traj_7`'s video is
`episode_00000007_<camera><batch suffix>.mp4`, and renumbering the survivors
without moving their files ends that: `traj_0` may well hold
`episode_00000003_*.mp4`. Nothing in training minds, because every consumer
reads the filename out of `obs/sensor_data` rather than deriving it -- but
`ensure_sensor_data_paths` *does* derive it, for a camera whose entry is missing
(which is how a newly derived camera gets picked up, e.g. after
`depth_to_gray.sh`). So it must not be run against a pruned run: it would write
filenames belonging to a different episode. `--rename-videos` is the way out if
you need that -- it renames the surviving episodes' files to match their new
indices and rewrites `obs/sensor_data` accordingly, restoring the convention --
but by default nothing outside the HDF5 files is touched.

Each file is rebuilt into a temporary file and swapped in with `os.replace`,
rather than edited in place: `h5py` cannot reclaim the space of a deleted group,
so an in-place prune would leave every file at its original size, and a crash
mid-edit would leave it truncated. The original file is never opened for
writing. `--backup` keeps it beside the new one as `<name>.h5.bak`.

    python -m examples.machine_learning.molmospaces.finetuning.prune_rollouts \
        data/stretch_potato/rollouts/potato_cleaned --dry-run

`--interactive` does the reviewing too, so the deleting-by-hand step is not
needed at all: every trajectory's cameras play as one looping grid -- the layout
`generate_montage.sh` uses -- at 5x, until you press `a` to keep it or `d` to
delete its videos and the trajectory with them. `s` toggles down to 1x for the
ones 5x cannot settle, `q` stops. A verdict is acted on immediately: `d` deletes
the MP4s, and the prune that follows the house finds an episode with no videos
and drops it, which is the same path the by-hand route takes. Accepted
trajectories are marked in the HDF5, so a session can be stopped and resumed
without re-watching them, and `--start-at-house 100` starts at `house_100` and
carries on upwards.

    python -m examples.machine_learning.molmospaces.finetuning.prune_rollouts \
        data/stretch_potato/rollouts/potato_cleaned --interactive --start-at-house 100

Two things downstream of the HDF5 are stale afterwards and this does not try to
fix them, because both are generated files with a command that regenerates them:
the `valid_traj_mask` (subsetted here, so it stays truthful, but a dropped
episode may have been why you rejected the run in the first place) and any
`stats` group written by `calculate_stats.py`, whose per-trajectory entries are
renumbered alongside the trajectories but whose `aggregated_stats.json` still
pools the episodes that are gone. Re-run both after pruning, and rebuild any
`train/`+`val/` split that symlinks these houses, since its
`valid_trajectory_index.json` lists trajectory keys by name.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import click
import cv2
import numpy as np

from examples.machine_learning.molmospaces.hdf5_layout import (
    TRAJECTORY_FILE_PATTERN,
    TRAJECTORY_KEY_PATTERN,
    decode_video_path,
    encode_video_path,
    trajectory_files,
    video_filename,
)

log = logging.getLogger(__name__)

VALID_TRAJECTORY_KEY = "valid_traj_mask"
"""MolmoBot's per-file validity mask, one entry per trajectory in index order."""

STATS_KEY = "stats"
"""`calculate_stats.py`'s output group, keyed by trajectory name."""

EPISODE_FILE_PATTERN = re.compile(r"^episode_(?P<index>\d{8})_(?P<rest>.+)$")
"""An episode's side-car file, split so the index can be rewritten in place.

Deliberately not restricted to `.mp4`: an episode's files are whatever was
written beside the HDF5 with that prefix -- eight videos per episode in the
current pipeline, including the derived depth and grayscale ones -- and a rename
that moved only the ones this script knows about would scatter an episode across
two indices.
"""

HOUSE_DIR_PATTERN = re.compile(r"^house_(?P<index>\d+)$")

SOURCE_EPISODE_ATTR = "source_episode_index"
"""Written on every trajectory a prune keeps: the episode its videos are named after.

Renumbering costs a trajectory the one identifier it had. This is the one that
survives: `traj_0` with `source_episode_index = 3` says its videos are
`episode_00000003_*`, which stays true through later prunes and is what makes a
trajectory findable again after several rounds of cleaning.
"""

TEMP_SUFFIX = ".pruning.tmp"
BACKUP_SUFFIX = ".bak"

DEFAULT_REVIEW_SPEED = 5.0
"""How fast `--interactive` plays a clip before `s` slows it down.

A pick that went wrong is obvious at 5x -- the arm misses, or the object never
moves -- and a rollout is a couple of hundred frames, so at 1x a run of 500 is
several hours of watching. The slow speed is for the ones where 5x is not enough
to tell.
"""

BASE_CELL_SIZE = 240
"""The cell size the grid's text sizes and paddings are written against.

Everything drawn over the video -- the banner strip, the camera labels, their
font sizes and offsets -- is expressed as a multiple of `cell_size /
BASE_CELL_SIZE`, so a bigger `--cell` scales the whole layout rather than
enlarging the video and leaving unreadable text stranded in the corners.
"""

DEFAULT_CELL_SIZE = 960
"""Pixels per side of one camera's cell in the review grid.

Six cells in a 3x2 grid, so the window is about 2880x1964 -- most of a 4K screen
and legible from a normal viewing distance, which is what a review at 5x needs.
`--cell` sizes it for a smaller display: 480 halves each side, 240 quarters it.
"""


class PruneError(RuntimeError):
    """A file cannot be pruned safely, so it is left exactly as it was.

    Raised per file and caught by the caller, so one unpruneable house does not
    abandon the rest of the run half-done.
    """


@dataclass
class TrajectoryFate:
    """What is to become of one `traj_*` group, and why."""

    old_index: int
    new_index: int | None
    """`None` for a trajectory being dropped."""
    episode_index: int = 0
    """Which `episode_%08d_*` on disk is this trajectory's, read from its own videos.

    Not the same number as `old_index` once a run has been pruned before: dropping
    a trajectory renumbers the ones after it while their files keep the names they
    were written with, so `traj_0` can perfectly well hold `episode_00000002`.
    Deriving the filename from the index instead -- which is what the *format*
    says, and what `ensure_sensor_data_paths` does -- then reads the wrong episode,
    or none at all. `obs/sensor_data` is the only thing that knows.
    """
    videos_present: list[str] = field(default_factory=list)
    videos_missing: list[str] = field(default_factory=list)
    reviewed: bool = False
    """Accepted in an earlier `--interactive` session, per `REVIEWED_ATTR`."""

    @property
    def dropped(self) -> bool:
        return self.new_index is None

    @property
    def partial(self) -> bool:
        """Some of its videos survived the cleaning and some did not."""
        return bool(self.videos_present) and bool(self.videos_missing)


@dataclass
class FilePlan:
    """Everything one `trajectories*.h5` needs, decided before anything is touched."""

    path: Path
    batch_suffix: str
    fates: list[TrajectoryFate]
    renames: dict[Path, Path] = field(default_factory=dict)

    @property
    def kept(self) -> list[TrajectoryFate]:
        return [fate for fate in self.fates if not fate.dropped]

    @property
    def dropped(self) -> list[TrajectoryFate]:
        return [fate for fate in self.fates if fate.dropped]

    @property
    def partial(self) -> list[TrajectoryFate]:
        return [fate for fate in self.fates if fate.partial and not fate.dropped]

    @property
    def renumbered(self) -> list[TrajectoryFate]:
        """Survivors whose `traj_*` key changes, which is what forces a rewrite."""
        return [fate for fate in self.kept if fate.new_index != fate.old_index]

    @property
    def misaligned(self) -> list[TrajectoryFate]:
        """Survivors whose videos are named after a different index than their key.

        Not the same question as `renumbered`: a run pruned once already has
        `traj_0` holding `episode_00000002`, and this pass may renumber nothing at
        all while every file is still misnamed. `--rename-videos` acts on this one.
        """
        return [fate for fate in self.kept if fate.episode_index != fate.new_index]

    @property
    def changed(self) -> bool:
        return bool(self.dropped) or bool(self.renumbered) or bool(self.renames)


@dataclass
class PruneSummary:
    """Counts for the run, for the closing report.

    The trajectory counts are what *happened*, not what was planned: a file
    whose rewrite failed contributes all of its trajectories to
    `trajectories_remaining` and none to `trajectories_removed`, because that
    file is still on disk exactly as it was. Under `--dry-run` they are the plan,
    since nothing happens at all.
    """

    files_scanned: int = 0
    files_rewritten: int = 0
    files_failed: int = 0
    trajectories_before: int = 0
    """How many trajectories the files held when the run started."""
    trajectories_removed: int = 0
    trajectories_remaining: int = 0
    trajectories_partial: int = 0
    videos_renamed: int = 0
    trajectories_accepted: int = 0
    """Reviewed and kept, under `--interactive`."""
    trajectories_rejected: int = 0
    """Reviewed and deleted, under `--interactive`. They are part of `trajectories_removed`."""
    already_reviewed: int = 0
    """Skipped by `--interactive` because an earlier session accepted them."""
    review_stopped: bool = False
    houses_skipped: int = 0
    """Houses below `--start-at-house`, neither reviewed nor pruned."""
    houses_emptied: list[str] = field(default_factory=list)
    houses_without_trajectories: list[str] = field(default_factory=list)


# =============================================================================
# Planning
# =============================================================================


def _referenced_videos(trajectory, house_dir: Path) -> tuple[list[str], list[str]]:
    """An episode's videos as the trajectory itself names them, split by existence.

    `obs/sensor_data` is the authority rather than a glob of the directory,
    because it is what MolmoBot reads: a trajectory whose entries all point at
    deleted files is unusable no matter what else sits beside it.
    """
    group = trajectory.get("obs/sensor_data")
    if group is None:
        return [], []
    present, missing = [], []
    for camera in sorted(group.keys()):
        filename = decode_video_path(group[camera][:])
        (present if (house_dir / filename).exists() else missing).append(filename)
    return present, missing


def _adjacent_episode_files(house_dir: Path, episode_index: int) -> list[Path]:
    """Every file written beside the HDF5 for one episode, whatever its camera."""
    return sorted(house_dir.glob(f"episode_{episode_index:08d}_*"))


def _episode_index_of(filenames: list[str], default: int) -> int:
    """The `episode_%08d` a trajectory's videos are named after.

    They all share one prefix -- the videos of an episode were written together --
    so the first parsable name settles it. `default` covers a trajectory with no
    videos named at all, where the trajectory's own index is the only guess
    available and also the one the format intends.
    """
    for filename in filenames:
        match = EPISODE_FILE_PATTERN.match(filename)
        if match is not None:
            return int(match.group("index"))
    return default


def plan_file(h5_path: Path, drop_partial: bool = False, rename_videos: bool = False) -> FilePlan:
    """Decide which trajectories a file keeps, their new indices, and the renames.

    Args:
        h5_path: one `trajectories*.h5`.
        drop_partial: also drop a trajectory that kept only *some* of its videos.
            Off by default: a partial deletion is more likely a slip than a
            verdict, and saying so is more useful than acting on it.
        rename_videos: plan the episode-file renames that keep `episode_%08d`
            aligned with the new trajectory indices. Off by default -- pruning
            is a change to the HDF5 files and nothing else.
    """
    import h5py

    match = TRAJECTORY_FILE_PATTERN.match(h5_path.name)
    if match is None:
        raise PruneError(f"{h5_path} is not a trajectories*.h5 file")
    batch_suffix = match.group("suffix")
    house_dir = h5_path.parent

    fates: list[TrajectoryFate] = []
    with h5py.File(h5_path, "r") as h5_file:
        indices = sorted(
            int(key_match.group("index"))
            for key in h5_file
            if (key_match := TRAJECTORY_KEY_PATTERN.match(key)) is not None
        )
        for old_index in indices:
            trajectory = h5_file[f"traj_{old_index}"]
            present, missing = _referenced_videos(trajectory, house_dir)
            if not present and not missing:
                # No sensor_data to go on -- pre-`ensure_sensor_data_paths`, or a
                # camera-less run. Fall back to what is on disk under the
                # episode's own prefix, which is the same question one level out.
                names = [path.name for path in _adjacent_episode_files(house_dir, old_index)]
                present = names
            keep = bool(present) and not (missing and drop_partial)
            fates.append(
                TrajectoryFate(
                    # Numbered by `renumber` below: a survivor's new index is how
                    # many survivors precede it, which is not known yet.
                    old_index=old_index,
                    new_index=0 if keep else None,
                    episode_index=_episode_index_of(present + missing, default=old_index),
                    videos_present=present,
                    videos_missing=missing,
                    reviewed=bool(trajectory.attrs.get(REVIEWED_ATTR)),
                )
            )

    renumber(fates)
    plan = FilePlan(path=h5_path, batch_suffix=batch_suffix, fates=fates)
    if rename_videos:
        plan.renames = _plan_renames(plan, house_dir)
    return plan


def renumber(fates: list[TrajectoryFate]) -> None:
    """Assign the survivors consecutive indices from 0, in their original order.

    Called again after an interactive review, because rejecting a trajectory
    changes every later survivor's index -- and a hole in the numbering is the
    one thing `validate_trajectories.py` turns into a whole discarded house.
    """
    survivors = 0
    for fate in fates:
        if not fate.dropped:
            fate.new_index = survivors
            survivors += 1


# =============================================================================
# Review
# =============================================================================

REVIEW_GRID_CAMERAS = (
    "head_camera_left",
    "head_camera",
    "head_camera_right",
    "wrist_camera_left",
    "wrist_camera_stereo_depth",
    "wrist_camera_right",
)
"""The cameras the review grid shows, in row-major order, and why these six.

The same set and the same layout as `generate_montage.sh`, which exists to make
a run watchable: the three head views across the top and the three wrist views
below, so a failure reads the same way here as it does in a montage. A run
missing some of them shows whatever it has -- and if it has none of them, every
video of the episode, which is the honest fallback for a run with other cameras.
"""

REVIEW_GRID_COLUMNS = 3

REVIEWED_ATTR = "reviewed"
"""Set on an accepted `traj_*` group so a later session does not re-play it.

Reviewing 500 episodes is not one sitting, and quitting halfway is expected. The
mark lives in the HDF5 rather than a side-car file because it has to survive the
renumbering that pruning does -- an index is not an identity once trajectories
are dropped, but an attribute moves with its group. `--re-review` ignores it.
"""

ACCEPT_KEYS = frozenset(b"aA")
DELETE_KEYS = frozenset(b"dD")
SLOW_KEYS = frozenset(b"sS")
QUIT_KEYS = frozenset(b"qQ\x1b")

REVIEW_WINDOW = "prune_rollouts -- a=keep  d=delete  s=1x/5x  q=quit"


class ReviewQuit(Exception):
    """The reviewer asked to stop. Whatever was decided so far still counts."""


def _decode_frame(frame: np.ndarray, camera: str) -> np.ndarray:
    """One decoded video frame, made watchable.

    The depth stream is not an image: MolmoSpaces packs a 16-bit metric depth
    into R (high byte) and G (low byte), which plays back as green banding. Its R
    channel is already the normalized depth, so showing that as grey is the same
    move `depth_to_gray.sh` makes permanent, and it is what `generate_montage.sh`
    puts in the grid.
    """
    if camera.endswith("_depth"):
        return cv2.cvtColor(frame[:, :, 2], cv2.COLOR_GRAY2BGR)
    return frame


def _letterbox(frame: np.ndarray, size: int) -> np.ndarray:
    """A frame fitted into a square cell without distorting it.

    The head cameras are portrait and the wrist cameras landscape, so a grid of
    plain resizes would stretch half the cells and change what the reviewer is
    judging.
    """
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    cell = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    cell[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return cell


def _label(cell: np.ndarray, text: str, scale: float) -> np.ndarray:
    """The camera's name in the corner of its cell, so the views are identifiable."""
    origin = (round(6 * scale), round(16 * scale))
    font_scale = 0.4 * scale
    outline = max(2, round(3 * scale))
    cv2.putText(cell, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), outline)
    cv2.putText(
        cell,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(1, round(scale)),
        cv2.LINE_AA,
    )
    return cell


def _grid(frames: list[tuple[str, np.ndarray]], cell_size: int) -> np.ndarray:
    """The cameras of one instant, tiled row-major and padded to a full rectangle."""
    scale = cell_size / BASE_CELL_SIZE
    cells = [
        _label(_letterbox(_decode_frame(frame, camera), cell_size), camera, scale)
        for camera, frame in frames
    ]
    columns = min(REVIEW_GRID_COLUMNS, len(cells))
    while len(cells) % columns:
        cells.append(np.zeros((cell_size, cell_size, 3), dtype=np.uint8))
    rows = [cv2.hconcat(cells[start : start + columns]) for start in range(0, len(cells), columns)]
    return cv2.vconcat(rows) if len(rows) > 1 else rows[0]


def _banner(grid: np.ndarray, lines: tuple[str, str], scale: float) -> np.ndarray:
    """A strip above the grid naming what is playing and how to answer."""
    strip = np.zeros((round(44 * scale), grid.shape[1], 3), dtype=np.uint8)
    for index, text in enumerate(lines):
        cv2.putText(
            strip,
            text,
            (round(8 * scale), round((18 + index * 18) * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * scale,
            (255, 255, 255) if index == 0 else (170, 220, 170),
            max(1, round(scale)),
            cv2.LINE_AA,
        )
    return cv2.vconcat([strip, grid])


def _ensure_window(grid: np.ndarray) -> None:
    """Create the review window at the grid's size, the first time only.

    `WINDOW_NORMAL` rather than the `WINDOW_AUTOSIZE` window `imshow` creates for
    itself, so the reviewer can drag or maximise it. That also means the size has
    to be set explicitly -- and only once, or every new clip would undo a window
    the reviewer had just resized by hand.
    """
    try:
        if cv2.getWindowProperty(REVIEW_WINDOW, cv2.WND_PROP_VISIBLE) >= 0:
            return
        cv2.namedWindow(REVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(REVIEW_WINDOW, grid.shape[1], grid.shape[0])
    except cv2.error:  # a build or session with no GUI; `imshow` will say so
        pass


def _review_videos(house_dir: Path, episode_index: int, batch_suffix: str) -> list[Path]:
    """The videos to put in the grid for one episode, in `REVIEW_GRID_CAMERAS` order."""
    preferred = [
        house_dir / video_filename(episode_index, camera, batch_suffix)
        for camera in REVIEW_GRID_CAMERAS
    ]
    present = [path for path in preferred if path.exists()]
    if present:
        return present
    return [
        path for path in _adjacent_episode_files(house_dir, episode_index) if path.suffix == ".mp4"
    ]


def _camera_of(path: Path, episode_index: int, batch_suffix: str) -> str:
    """`episode_00000003_head_camera_batch_1_of_1.mp4` -> `head_camera`."""
    stem = path.name[len(f"episode_{episode_index:08d}_") :]
    return stem[: -len(f"{batch_suffix}.mp4")] if batch_suffix else path.stem


def play_until_decided(
    videos: list[Path],
    heading: str,
    fast: float,
    slow: float = 1.0,
    cell_size: int = 240,
) -> bool:
    """Loop an episode's videos as one grid until the reviewer accepts or rejects it.

    Playback is streamed from the files and rewound at the end rather than
    decoded into a buffer: these clips decode at several hundred frames a second,
    so looping costs nothing, and a review session that held every frame of a
    long episode in memory would be the only part of this script that cares how
    long an episode is.

    Args:
        videos: the episode's videos, in the order they should be tiled.
        heading: what is playing -- house, trajectory and position in the run.
        fast: the speed multiple playback starts at.
        slow: the speed multiple `s` switches to, and back.
        cell_size: pixels per side of one camera's cell in the grid.

    Returns:
        True to keep the trajectory, False to delete it.

    Raises:
        ReviewQuit: the reviewer pressed `q`, or closed the window.
    """
    captures = [cv2.VideoCapture(str(path)) for path in videos]
    opened = [capture for capture in captures if capture.isOpened()]
    if not opened:
        for capture in captures:
            capture.release()
        raise PruneError(f"none of {len(videos)} video(s) could be opened: {videos[0].parent}")

    cameras = [
        _camera_of(path, *_episode_and_suffix(path))
        for path, capture in zip(videos, captures)
        if capture.isOpened()
    ]
    source_fps = opened[0].get(cv2.CAP_PROP_FPS) or 30.0
    speed = fast
    try:
        while True:
            frames = []
            for camera, capture in zip(cameras, opened):
                read, frame = capture.read()
                if read:
                    frames.append((camera, frame))
            if len(frames) < len(opened):
                # End of the shortest video: rewind them all and loop, which is
                # what makes the clip repeat until a verdict is given.
                for capture in opened:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            grid = _banner(
                _grid(frames, cell_size),
                (
                    heading,
                    f"{speed:g}x    a = keep    d = delete    s = {slow:g}x/{fast:g}x    q = quit",
                ),
                cell_size / BASE_CELL_SIZE,
            )
            _ensure_window(grid)
            cv2.imshow(REVIEW_WINDOW, grid)
            key = cv2.waitKey(max(1, round(1000.0 / (source_fps * speed))))
            if key == -1:
                if cv2.getWindowProperty(REVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    raise ReviewQuit
                continue
            key &= 0xFF
            if key in ACCEPT_KEYS:
                return True
            if key in DELETE_KEYS:
                return False
            if key in SLOW_KEYS:
                speed = slow if speed == fast else fast
            if key in QUIT_KEYS:
                raise ReviewQuit
    finally:
        for capture in captures:
            capture.release()


def _episode_and_suffix(path: Path) -> tuple[int, str]:
    """An episode video's index and batch suffix, read back off its filename."""
    match = EPISODE_FILE_PATTERN.match(path.name)
    if match is None:
        return 0, ""
    index = int(match.group("index"))
    rest = match.group("rest")
    suffix_match = re.search(r"(_batch_\d+_of_\d+)\.mp4$", rest)
    return index, suffix_match.group(1) if suffix_match else ""


def _count_reviewable(files: list[Path], re_review: bool) -> tuple[int, int, int]:
    """What a review session has ahead of it, counted before it starts.

    Counted up front by planning every file, which costs a fraction of a second
    over a whole run and is worth it twice over: "17/541" tells the reviewer how
    long they are in for, and the two other numbers explain an empty session --
    a run where every clip has already been judged, or one whose videos are gone,
    otherwise just opens no window and says nothing.

    Returns:
        `(to_review, already_reviewed, without_videos)`.
    """
    to_review = already_reviewed = without_videos = 0
    for h5_path in files:
        try:
            plan = plan_file(h5_path)
        except (PruneError, OSError):
            continue
        without_videos += len(plan.dropped)
        for fate in plan.kept:
            if fate.reviewed and not re_review:
                already_reviewed += 1
            else:
                to_review += 1
    return to_review, already_reviewed, without_videos


def review_house(
    h5_path: Path,
    batch_suffix: str,
    fates: list[TrajectoryFate],
    reviewed: int,
    removed_so_far: int,
    total: int,
    fast: float,
    re_review: bool,
    cell_size: int,
) -> tuple[int, int, int]:
    """Play every undecided trajectory of one house and act on the verdicts.

    Rejecting a trajectory deletes its videos immediately, which is exactly the
    manual cleaning this script was written to follow: the prune that runs next
    then finds an episode with no videos and drops it. So the interactive and the
    by-hand routes end in the same place, and neither has to trust the other.

    Args:
        h5_path: the house's trajectory file. Opened `a` only to mark acceptances.
        batch_suffix: the file's `_batch_n_of_m`, for building video names.
        fates: the house's trajectories, as `plan_file` found them.
        reviewed: how many trajectories were already reviewed before this house.
        removed_so_far: how many the session has removed before this house, so
            each verdict can be logged with the running total.
        total: how many are in the run, for the position indicator.
        fast: the speed multiple playback starts at.
        re_review: play trajectories that were accepted in an earlier session.
        cell_size: pixels per side of one camera's cell.

    Returns:
        `(accepted, deleted, quit_flag)`; `quit_flag` is 1 if the reviewer stopped.
    """
    import h5py

    accepted = deleted = 0
    house_dir = h5_path.parent
    # Whose verdict is still open, decided before the file is opened for writing:
    # a house with nothing left to watch should not be touched at all, and a
    # whole run of them -- which is what a finished review looks like -- should
    # not rewrite 136 files' modification times to say so.
    pending = [
        (fate, _review_videos(house_dir, fate.episode_index, batch_suffix))
        for fate in fates
        if not fate.dropped and (re_review or not fate.reviewed)
    ]
    pending = [(fate, videos) for fate, videos in pending if videos]
    if not pending:
        return 0, 0, 0

    with h5py.File(h5_path, "a") as h5_file:
        for fate, videos in pending:
            trajectory = h5_file[f"traj_{fate.old_index}"]

            position = reviewed + accepted + deleted + 1
            heading = (
                f"{house_dir.name}  traj_{fate.old_index}  "
                f"(episode {fate.episode_index})  "
                f"({position}/{total})  {len(videos)} camera(s)"
            )
            try:
                keep = play_until_decided(videos, heading, fast=fast, cell_size=cell_size)
            except ReviewQuit:
                return accepted, deleted, 1
            except PruneError as error:
                log.error(f"[review] {error}")
                continue

            if keep:
                trajectory.attrs[REVIEWED_ATTR] = True
                accepted += 1
                log.info(
                    f"[review] {house_dir.name} traj_{fate.old_index}: kept "
                    f"({removed_so_far + deleted} removed so far)"
                )
                continue

            for path in _adjacent_episode_files(house_dir, fate.episode_index):
                path.unlink()
            fate.new_index = None  # the prune below sees an episode with no videos
            renumber(fates)  # so the survivors stay consecutive as verdicts come in
            deleted += 1
            log.info(
                f"[review] {house_dir.name} traj_{fate.old_index}: deleted its "
                f"{len(videos)} video(s) and the trajectory with them "
                f"({removed_so_far + deleted} removed so far)"
            )
    return accepted, deleted, 0


def _plan_renames(plan: FilePlan, house_dir: Path) -> dict[Path, Path]:
    """`old path -> new path` for every file of a renumbered episode.

    Applied in ascending index order these can never collide: episode numbers
    rise with the trajectories they belong to, so the k-th survivor's episode is
    numbered at least k, and its target was either freed by a dropped episode or
    already vacated by an earlier rename. `apply_plan` checks that anyway before
    it moves anything -- a target that still exists means a file of a dropped
    episode was left behind, and overwriting it would mix two runs together under
    one index.
    """
    renames: dict[Path, Path] = {}
    for fate in plan.misaligned:
        for path in _adjacent_episode_files(house_dir, fate.episode_index):
            match = EPISODE_FILE_PATTERN.match(path.name)
            if match is None:  # unreachable: the glob supplied the prefix
                continue
            renames[path] = path.with_name(f"episode_{fate.new_index:08d}_{match.group('rest')}")
    return renames


# =============================================================================
# Rewriting
# =============================================================================


def _rewrite(plan: FilePlan, destination: Path) -> None:
    """Write the pruned copy of `plan.path` to `destination`.

    A copy rather than an edit because `h5py` never reclaims a deleted group's
    space: pruning half a file in place leaves it as large as it was, and the
    source is only ever opened read-only.
    """
    import h5py

    renamed = {old.name: new.name for old, new in plan.renames.items()}
    # Whose files this pass moves, so their recorded episode is the one they will
    # answer to afterwards rather than the one they arrived with.
    moved = {id(fate) for fate in plan.misaligned} if plan.renames else set()
    with h5py.File(plan.path, "r") as old, h5py.File(destination, "w") as new:
        for name, value in old.attrs.items():
            new.attrs[name] = value

        # Everything that is not per-trajectory, carried over untouched. The two
        # exceptions are the ones keyed by trajectory: the mask is subsetted and
        # `stats` is renumbered, both below.
        for key in old:
            if TRAJECTORY_KEY_PATTERN.match(key) is not None:
                continue
            if key in (VALID_TRAJECTORY_KEY, STATS_KEY):
                continue
            old.copy(key, new, name=key)

        old_stats = old.get(STATS_KEY)
        new_stats = new.create_group(STATS_KEY) if old_stats is not None else None
        for fate in plan.kept:
            old_key, new_key = f"traj_{fate.old_index}", f"traj_{fate.new_index}"
            old.copy(old_key, new, name=new_key)
            # The episode rather than the old key, because the episode number is
            # what the videos on disk are named after and it does not move when a
            # later prune renumbers the trajectories again.
            new[new_key].attrs[SOURCE_EPISODE_ATTR] = (
                fate.new_index if id(fate) in moved else fate.episode_index
            )
            _repoint_videos(new[new_key], renamed)
            if old_stats is not None and old_key in old_stats:
                old_stats.copy(old_key, new_stats, name=new_key)

        if VALID_TRAJECTORY_KEY in old:
            mask = old[VALID_TRAJECTORY_KEY][:]
            kept_mask = np.array(
                [
                    bool(mask[fate.old_index]) if fate.old_index < len(mask) else True
                    for fate in plan.kept
                ],
                dtype=bool,
            )
            new.create_dataset(VALID_TRAJECTORY_KEY, data=kept_mask)


def _repoint_videos(trajectory, renamed: dict[str, str]) -> None:
    """Rewrite `obs/sensor_data` entries whose file was renamed.

    The datasets are a fixed 100-byte array each, so a changed name is a delete
    and a re-create rather than an assignment.
    """
    group = trajectory.get("obs/sensor_data")
    if group is None:
        return
    for camera in list(group.keys()):
        filename = decode_video_path(group[camera][:])
        new_filename = renamed.get(filename)
        if new_filename is None:
            continue
        del group[camera]
        group.create_dataset(camera, data=encode_video_path(new_filename), dtype=np.uint8)


def apply_plan(plan: FilePlan, backup: bool = False) -> int:
    """Rename the surviving episodes' files and swap in the pruned HDF5.

    The videos move first and the HDF5 second, so the moment of inconsistency is
    a file whose `obs/sensor_data` names files that have already moved -- rather
    than an HDF5 naming files that do not exist yet. Either way the window is one
    `os.replace` wide, and the pruned copy is fully written before either
    happens.

    Returns:
        How many episode files were renamed.
    """
    for old, new in plan.renames.items():
        if new.exists() and new not in plan.renames:
            raise PruneError(
                f"{new} already exists, so renaming {old.name} to it would overwrite a file "
                "from a rejected run. Delete the leftover files of the deleted episodes, or "
                "drop --rename-videos and leave the filenames alone."
            )

    temporary = plan.path.with_name(plan.path.name + TEMP_SUFFIX)
    try:
        _rewrite(plan, temporary)
        for old in sorted(plan.renames, key=lambda path: path.name):
            os.replace(old, plan.renames[old])
        if backup:
            shutil.copy2(plan.path, plan.path.with_name(plan.path.name + BACKUP_SUFFIX))
        os.replace(temporary, plan.path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return len(plan.renames)


# =============================================================================
# Driver
# =============================================================================


def house_number(path: Path) -> int | None:
    """`house_101` -> 101, for whichever of a path's parts names a house."""
    for part in (path.name, *(parent.name for parent in path.parents)):
        match = HOUSE_DIR_PATTERN.match(part)
        if match is not None:
            return int(match.group("index"))
    return None


def _in_house_order(paths: list[Path]) -> list[Path]:
    """Trajectory files ordered by house *number*, not by name.

    `sorted()` gives house_0, house_1, house_10, house_100 -- fine for a batch
    prune, wrong for anything a person follows along with: a review session walks
    the run in front of a human, and `--start-at-house 100` means the hundredth
    house, not the position of `house_100` in a lexicographic list.
    """
    return sorted(
        paths, key=lambda path: (house_number(path) is None, house_number(path) or 0, path.name)
    )


def prune_rollout_dir(
    rollout_dir: Path,
    dry_run: bool = False,
    rename_videos: bool = False,
    drop_partial: bool = False,
    backup: bool = False,
    interactive: bool = False,
    start_at_house: int | None = None,
    speed: float = DEFAULT_REVIEW_SPEED,
    re_review: bool = False,
    cell_size: int = DEFAULT_CELL_SIZE,
) -> PruneSummary:
    """Prune every `house_*/trajectories*.h5` under a run to the videos that remain.

    Args:
        rollout_dir: a run of `house_*/` directories, each holding one
            `trajectories*.h5` and its episode MP4s.
        dry_run: report what would change without writing anything. Ignored for
            the review itself, which cannot ask a person to decide and then throw
            the answer away -- `--interactive` and `--dry-run` are refused
            together by the command.
        rename_videos: also rename the surviving episodes' files, so
            `episode_%08d` matches the renumbered trajectories again.
        drop_partial: also drop trajectories that kept only some of their videos.
        backup: leave the original file as `<name>.h5.bak`.
        interactive: play each undecided trajectory and prune it or keep it by
            the reviewer's verdict.
        start_at_house: skip houses numbered below this one, then carry on
            upwards. For resuming a review, or re-doing part of a run.
        speed: the speed multiple the review starts playback at.
        re_review: play trajectories accepted in an earlier session too.
        cell_size: pixels per side of one camera's cell in the review grid.
    """
    rollout_dir = Path(rollout_dir)
    summary = PruneSummary()

    files = _in_house_order(trajectory_files(rollout_dir))
    if start_at_house is not None:
        started = [path for path in files if (house_number(path) or 0) >= start_at_house]
        summary.houses_skipped = len(files) - len(started)
        files = started
    total_to_review = 0
    if interactive:
        total_to_review, summary.already_reviewed, without_videos = _count_reviewable(
            files, re_review
        )
        log.info(
            f"[review] {total_to_review} trajectory(ies) to review over {len(files)} house(s) "
            f"({summary.already_reviewed} already reviewed, {without_videos} with no videos)"
        )

    stopped_early = False
    for h5_path in files:
        if stopped_early:
            # Stop where the reviewer stopped, the house they were in having been
            # pruned by the iteration they stopped in. Carrying on through the
            # rest of the run would be a prune they did not ask for in a session
            # they just ended.
            break
        summary.files_scanned += 1
        try:
            plan = plan_file(h5_path, drop_partial=drop_partial, rename_videos=False)
        except PruneError as error:
            summary.files_failed += 1
            log.error(f"[prune] {error}")
            continue

        if interactive and not summary.review_stopped:
            accepted, deleted, stopped = review_house(
                h5_path,
                batch_suffix=plan.batch_suffix,
                fates=plan.fates,
                reviewed=summary.trajectories_accepted + summary.trajectories_rejected,
                removed_so_far=summary.trajectories_removed,
                total=total_to_review,
                fast=speed,
                re_review=re_review,
                cell_size=cell_size,
            )
            summary.trajectories_accepted += accepted
            summary.trajectories_rejected += deleted
            summary.review_stopped = bool(stopped)
            stopped_early = bool(stopped)

        if rename_videos:
            # Planned here rather than in `plan_file`, because a review that just
            # deleted an episode's videos has changed which renames there are.
            plan.renames = _plan_renames(plan, h5_path.parent)

        summary.trajectories_before += len(plan.fates)
        summary.trajectories_partial += len(plan.partial)
        for fate in plan.partial:
            log.warning(
                f"[prune] {h5_path}: traj_{fate.old_index} kept {len(fate.videos_present)} "
                f"video(s) and lost {len(fate.videos_missing)} "
                f"({', '.join(fate.videos_missing)}) -- keeping it; --drop-partial drops it"
            )
        if not plan.kept and plan.dropped:
            summary.houses_emptied.append(plan.path.parent.name)

        if not plan.changed:
            summary.trajectories_remaining += len(plan.kept)
            log.debug(f"[prune] {h5_path}: nothing to do ({len(plan.kept)} trajectories)")
            continue

        removed = ", ".join(f"traj_{fate.old_index}" for fate in plan.dropped)
        log.info(
            f"[prune] {h5_path.parent.name}: "
            f"{'would remove' if dry_run else 'removing'} {len(plan.dropped)} of "
            f"{len(plan.fates)} ({removed or 'none'}), {len(plan.kept)} remaining, "
            f"renumbering {len(plan.renumbered)}, renaming {len(plan.renames)} episode file(s)"
        )
        if dry_run:
            summary.trajectories_removed += len(plan.dropped)
            summary.trajectories_remaining += len(plan.kept)
            continue

        try:
            summary.videos_renamed += apply_plan(plan, backup=backup)
            summary.files_rewritten += 1
            summary.trajectories_removed += len(plan.dropped)
            summary.trajectories_remaining += len(plan.kept)
        except (PruneError, OSError) as error:
            # The file is untouched, so its trajectories are all still there.
            summary.files_failed += 1
            summary.trajectories_remaining += len(plan.fates)
            log.error(f"[prune] {h5_path} left unchanged: {error}")

    if interactive:
        cv2.destroyAllWindows()

    # Videos with no trajectory file at all: a house whose run was interrupted
    # before the HDF5 was written. Nothing here can prune them, and they are
    # worth naming because they explain a video count that exceeds the
    # trajectory count. Judged against every house that has a trajectory file,
    # not only the ones this call got to, so stopping early does not turn the
    # houses left over into a scary-looking report.
    houses_with_h5 = {path.parent for path in trajectory_files(rollout_dir)}
    for house_dir in _in_house_order(sorted(rollout_dir.glob("house_*"))):
        if house_dir not in houses_with_h5 and any(house_dir.glob("episode_*")):
            summary.houses_without_trajectories.append(house_dir.name)

    return summary


@click.command()
@click.argument(
    "rollout_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report what would change without writing anything.",
)
@click.option(
    "--rename-videos",
    is_flag=True,
    help="Also rename the surviving episodes' files so episode_%08d matches the renumbered "
    "trajectories again. Off by default: only the .h5 files are touched, which leaves "
    "trajectory indices no longer matching episode numbers -- fine for training, but "
    "ensure_sensor_data_paths must not be run against the run afterwards.",
)
@click.option(
    "--drop-partial",
    is_flag=True,
    help="Also drop trajectories that lost only some of their videos. "
    "By default those are reported and kept.",
)
@click.option(
    "--backup",
    is_flag=True,
    help="Keep each original file beside the pruned one as <name>.h5.bak.",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Review each trajectory instead of reading the verdict off the disk: its cameras play "
    "as one looping grid until you press 'a' to keep it or 'd' to delete its videos and the "
    "trajectory with them. 's' toggles between 1x and the review speed, 'q' stops.",
)
@click.option(
    "--start-at-house",
    "--start_at_house",
    "start_at_house",
    type=int,
    default=None,
    metavar="N",
    help="Skip houses numbered below N and carry on upwards -- e.g. 100 starts at house_100. "
    "Houses are walked in numeric order, so this resumes where a session left off.",
)
@click.option(
    "--speed",
    type=float,
    default=DEFAULT_REVIEW_SPEED,
    show_default=True,
    help="Speed multiple --interactive starts playback at. 's' toggles it against 1x.",
)
@click.option(
    "--re-review",
    is_flag=True,
    help="Play trajectories accepted in an earlier session too, rather than skipping them.",
)
@click.option(
    "--cell",
    "cell_size",
    type=int,
    default=DEFAULT_CELL_SIZE,
    show_default=True,
    help="Pixels per side of one camera's cell in the review grid.",
)
@click.option("--verbose", is_flag=True, help="Log every file, including unchanged ones.")
def main(
    rollout_dir: Path,
    dry_run: bool,
    rename_videos: bool,
    drop_partial: bool,
    backup: bool,
    interactive: bool,
    start_at_house: int | None,
    speed: float,
    re_review: bool,
    cell_size: int,
    verbose: bool,
) -> None:
    """Drop the trajectories whose videos you deleted from a cleaned rollout run.

    ROLLOUT_DIR is a run of house_* directories, e.g.
    data/stretch_potato/rollouts/potato_cleaned.

    With --interactive, review the run first: each trajectory's cameras play as
    one looping grid and your verdict decides whether it stays.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )
    if interactive and dry_run:
        # Asking someone to watch 500 clips and then discarding their answers is
        # not a dry run of anything.
        raise click.UsageError("--interactive and --dry-run cannot be combined.")
    if speed <= 0:
        raise click.UsageError("--speed must be positive.")

    summary = prune_rollout_dir(
        rollout_dir,
        dry_run=dry_run,
        rename_videos=rename_videos,
        drop_partial=drop_partial,
        backup=backup,
        interactive=interactive,
        start_at_house=start_at_house,
        speed=speed,
        re_review=re_review,
        cell_size=cell_size,
    )

    click.echo("")
    click.secho(
        f"{'Would remove' if dry_run else 'Removed'} {summary.trajectories_removed} "
        f"trajectory(ies) of {summary.trajectories_before}; "
        f"{summary.trajectories_remaining} "
        f"{'would remain' if dry_run else 'remaining'} "
        f"across {summary.files_scanned} trajectory file(s)",
        fg="green" if not summary.files_failed else "yellow",
    )
    if not dry_run:
        click.echo(f"  rewrote {summary.files_rewritten} file(s) in place")
    if summary.videos_renamed:
        click.echo(f"  renamed {summary.videos_renamed} episode file(s)")
    if interactive:
        watched = summary.trajectories_accepted + summary.trajectories_rejected
        click.echo(
            f"  reviewed {watched}: kept {summary.trajectories_accepted}, "
            f"deleted {summary.trajectories_rejected}"
        )
        if not watched and summary.already_reviewed:
            # The window never opened, and without this the run looks broken
            # rather than finished.
            click.secho(
                f"  no clips played: all {summary.already_reviewed} trajectory(ies) here are "
                "already marked as reviewed by an earlier session. Pass --re-review to watch "
                "them again, or --start-at-house N to jump to a house you have not reached",
                fg="yellow",
            )
        elif not watched:
            click.secho(
                "  no clips played: nothing in this run has videos left to review",
                fg="yellow",
            )
    if summary.houses_skipped:
        click.echo(f"  skipped {summary.houses_skipped} house(s) below --start-at-house")
    if summary.review_stopped:
        click.secho(
            "  review stopped early; the houses after it were left alone. "
            "Re-run with --start-at-house to pick up where you left off",
            fg="yellow",
        )
    if summary.trajectories_partial:
        click.secho(
            f"  {summary.trajectories_partial} trajectory(ies) lost only some of their videos "
            "and were kept; see the warnings above",
            fg="yellow",
        )
    if summary.houses_emptied:
        click.secho(
            f"  {len(summary.houses_emptied)} house(s) have no trajectories left "
            f"({', '.join(summary.houses_emptied)}) -- delete the directories if that is right",
            fg="yellow",
        )
    if summary.houses_without_trajectories:
        click.secho(
            f"  {len(summary.houses_without_trajectories)} house(s) have videos but no "
            f"trajectories*.h5 ({', '.join(summary.houses_without_trajectories)}) -- "
            "an interrupted run; nothing to prune there",
            fg="yellow",
        )
    if summary.files_failed:
        click.secho(f"  {summary.files_failed} file(s) left unchanged; see the errors", fg="red")

    if summary.files_rewritten:
        click.echo("")
        click.echo(
            "The generated files that describe these trajectories are now stale. Re-run\n"
            "  <MolmoBot>/data_scripts/validate_trajectories.py <split> --overwrite\n"
            "  <MolmoBot>/data_scripts/calculate_stats.py <split>/train --keys ...\n"
            "and rebuild any train/+val/ split that symlinks these houses, since its\n"
            "valid_trajectory_index.json lists trajectories by name."
        )


if __name__ == "__main__":
    main()
