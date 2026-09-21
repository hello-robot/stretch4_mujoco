"""
Watch the Franka and Stretch attempt the same grasp, side by side, in one video.

`params_search.py` scores setups and writes a table. A table is the wrong
instrument for the question this answers: when `stretch_fisheye` picks up two
objects and `franka_fisheye` picks up four with the *same camera*, the difference
is somewhere in a 300-step rollout, and no column of `report.md` says where. This
runs both halves of a matched pair over the same episodes and puts them in one
frame:

    +---------------------------+---------------------------+
    |      franka, scene        |      stretch, scene       |
    +-------------+-------------+-------------+-------------+
    | franka exo  | franka wrist| stretch exo |stretch wrist|
    +-------------+-------------+-------------+-------------+

The top row is a third-person view from an *identical* free camera -- same
lookat, azimuth, elevation and distance on both sides, aimed at the spot the
robot is standing at rather than at the robot, because the two have different
heights and a camera framed on each would not be one view of one task. The
bottom row is what each policy actually saw, which is the other half of the
question: a Stretch that reaches for the wrong place may be looking at a frame
the checkpoint cannot read.

Matched pairs, and why they are pairs
------------------------------------
"Same camera config" is not something this script arranges -- `setups.py`
already defines it. `franka_stretchcam` and `stretch_stretchcam` carry identical
`ExoCameraParams` (fovy 71, pitch 43, 640x360) and differ only in the robot and
the mount body it hangs off; likewise the fisheye and rectified pairs. So the
pair *is* the controlled comparison, and `MATCHED_PAIRS` just names them.

    python -m examples.machine_learning.molmospaces.retargetting.params_search_side_by_side

    # a different pair, and the parameters a sweep found
    python -m ...params_search_side_by_side --pair fisheye \\
        --param grasp_offset_m=0.09 --param target_z_offset_m=0.05

    # one scene and one object, to check the plumbing before committing an hour
    python -m ...params_search_side_by_side --scenes 1 --episode-steps 40

What it costs
-------------
Two evaluations rather than one, run *sequentially* -- which is not a detail to
optimise away. Each rollout worker loads its own copy of the DROID checkpoint and
peaks near 17 GiB, so two policies live at once do not fit on a 32 GiB card; see
`params_search.WORKER_VRAM_GIB`. Running them one after the other means one
policy is resident at a time, at the price of the two halves not being
frame-synchronised to the same action stream. They are synchronised to the same
*episode* -- same house, same object, same start -- which is what the comparison
needs.

What it is not
--------------
Not a search: it runs one point per setup and scores it exactly as
`params_search` would, so the numbers are comparable, but nothing is optimised.
Use `params_search.py` to find parameters and this to see what they do.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

# MuJoCo binds the backend named by MUJOCO_GL when it is first imported, which
# the imports below trigger -- so this has to come before them.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# See `params_search`: read by CUDA's caching allocator when torch first
# initialises it, so it has to be set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402

from examples.machine_learning.molmospaces.policies import franka_retarget as fr  # noqa: E402
from examples.machine_learning.molmospaces.retargetting import mini_benchmark  # noqa: E402
from examples.machine_learning.molmospaces.retargetting.cameras import (  # noqa: E402
    RetargetParams,
)
from examples.machine_learning.molmospaces.retargetting.params_search import (  # noqa: E402
    DIMENSIONS,
    affordable_workers,
)
from examples.machine_learning.molmospaces.retargetting.scoring import (  # noqa: E402
    EpisodeScore,
    collect_probe_records,
    install_probe,
)
from examples.machine_learning.molmospaces.retargetting.setups import (  # noqa: E402
    PROBE_SINK_ENV_VAR,
    SETUPS,
    publish_params,
    qualified_config_name,
)
from examples.machine_learning.molmospaces.visualize import (  # noqa: E402
    SCENE_PANEL_SIZE,
    EpisodeVideoRecorder,
)

log = logging.getLogger(__name__)


# =============================================================================
# Which setups make a pair
# =============================================================================

MATCHED_PAIRS: dict[str, tuple[str, str]] = {
    "stretchcam": ("franka_stretchcam", "stretch_stretchcam"),
    "fisheye": ("franka_fisheye", "stretch_fisheye"),
    "rectified": ("franka_rectified", "stretch_rectified"),
}
"""
The `(franka, stretch)` setups that differ only in the robot.

Each pair carries the same `ExoCameraParams`, so a difference between its halves
is the robot and the retargeting rather than the view. `franka_baseline` has no
partner on purpose: it is the DROID shoulder camera, which no Stretch setup
reproduces, and pairing it with one would put the lens back into a comparison
this exists to take it out of.
"""

DEFAULT_PAIR = "stretchcam"
"""
The pinhole pair, which is the one to look at first.

It is the pair with the *fewest* differences left -- an upright pinhole at
Stretch's head-camera height on both robots -- so whatever is visible here is
the retargeting and the robot, with no lens to blame. The fisheye pairs add the
lens back on both sides.
"""


# =============================================================================
# Recording the two panels
# =============================================================================

SCENE_LOOKAT_HEIGHT_M = 1.05
"""
The height the shared scene camera looks at, above the floor.

A constant rather than the robot's own base: `EpisodeVideoRecorder` aims at the
base and adds a fixed offset, which is right when there is one robot and wrong
here, because the Franka stands on a 0.58m pedestal and Stretch on the floor --
the same rule would frame the two at different heights and the split screen
would stop being one view of one task. 1.05m is a little above this benchmark's
counter, so the grasp is in frame on both sides.
"""

SCENE_AZIMUTH_OFFSET_DEG = 90.0
SCENE_ELEVATION = -20.0
SCENE_DISTANCE = 2.4
"""
The shared scene camera, as an offset from the robot's own yaw.

Derived per episode rather than fixed, and the first version of this script had
it fixed, which was wrong. The mini benchmark turns the robot to face whatever
object that scene is about (`setups._point_base_at`), so every scene has a
different yaw -- and a constant azimuth that frames the counter in the
hand-tuned kitchen renders a wall in the next house. Measured on the first
smoke run: at a fixed 150 degrees, 40% of both panels was a pillar.

A quarter turn off the robot's facing is the offset that works: it puts the robot
side-on with the counter and its objects across the frame, rather than behind the
robot looking at its back (yaw + 180, which is into a wall here) or over its
shoulder into the worktop.

This is still one camera for both halves. Both robots are placed by the same rule
from the same episode, so they share a yaw, so they share an azimuth -- which is
what the split screen needs. What it gives up is comparability of the *camera*
between episodes, which nothing here asks for.
"""


GRASP_FRAME_BALL_RADIUS = 0.020
GRASP_FRAME_AXIS_LENGTH = 0.15
GRASP_FRAME_AXIS_RADIUS = 0.010
"""
How big the grasp-centre coordinate frame is drawn on a scene panel, in metres.

Larger than `tests/test_retargeting.py` uses, because this camera sits further
back (`SCENE_DISTANCE` 2.4m against 1.15m) and a marker that reads at one
distance is a smudge at the other.

Drawn because "the two grippers are in the same place" is not the whole question
and the frame is what answers the rest of it. Two grasp centres can coincide
while the hands are oriented a quarter turn apart -- which is exactly what
`FRANKA_TO_STRETCH_TOOL` exists to prevent and what a mis-signed tool rotation
would produce -- and a ball on its own cannot show that. The red/green/blue
arrows on x/y/z can.

Both panels draw their axes in *one* convention -- Stretch's -- so the same
colour means the same direction across the split screen; see
`SplitPanelRecorder.in_stretch_convention` for why the Franka's are rotated and
why each panel says so.
"""


class SplitPanelRecorder(EpisodeVideoRecorder):
    """Records each episode as two MP4s -- the scene, and the policy's cameras.

    `EpisodeVideoRecorder` writes one file with the scene beside the cameras,
    which is the right layout for watching one robot and the wrong one here: the
    two halves of this comparison have to be tiled against *another robot's*
    two halves, so they are kept apart until `compose_pair` puts all four
    together.

    Everything else is inherited, deliberately. The scene renderer, the camera
    extraction, the per-house episode numbering and the observer protocol are the
    ones the normal recorder uses, so what this writes is the same footage the
    benchmark's own videos would show -- which matters, because the point is to
    debug the benchmark rather than a re-implementation of it.

    Two things are overridden. `_build_camera` returns the shared fixed camera
    (see `SCENE_AZIMUTH`), and `_scene_frame` renders through it *without*
    re-aiming at the robot each step, which the base class does and which would
    undo the sharing.
    """

    def __init__(self, panel_dir: Path, scene_panel_size=SCENE_PANEL_SIZE) -> None:
        super().__init__(output_dir=panel_dir, scene_panel_size=scene_panel_size)
        self.panel_dir = Path(panel_dir)
        self.episodes: dict[str, dict[str, Any]] = {}
        self.marker_color = fr.FRANKA_TOOL_COLOR
        self.in_stretch_convention = False
        """
        Whether this run's grasp frame is rotated into Stretch's tool convention.

        Set for the Franka half. The Robotiq reaches along its tool +z and
        Stretch's gripper along +x, so two true frames put a blue arrow into the
        counter on one panel and a red one on the other -- the same direction,
        read as a 180 degree error. Rotating the Franka's by
        `FRANKA_TO_STRETCH_TOOL` makes one colour mean one direction across the
        split screen; the panel's note says it has been done.
        """
        """
        The colour this run's grasp-centre frame is drawn in.

        Per robot, and the same two colours `tests/test_retargeting.py` and
        `replay.py` use -- green for the Franka, orange for Stretch -- so a ball
        means the same robot in every visualisation this package produces.
        """
        self._scene_writer: Any = None
        self._camera_writer: Any = None
        self._key: str | None = None

    def set_panel_dir(
        self,
        panel_dir: Path,
        marker_color: tuple | None = None,
        in_stretch_convention: bool | None = None,
    ) -> None:
        """Point the recorder at a new run's directory, and start its numbering afresh.

        `_episodes_per_house` has to be cleared, not just `episodes`. It is the
        base class's per-house counter and it is what names the files, so a
        recorder reused for a second run carries on counting: the first run's
        episodes come out `house_0_ep0000..0003` and the second's
        `house_0_ep0004..0007`, no key appears in both, and `compose_pair`
        silently finds nothing to pair. That is precisely what the first smoke
        run of this script did.
        """
        self.panel_dir = Path(panel_dir)
        self._output_dir = Path(panel_dir)
        self.episodes = {}
        self._episodes_per_house = {}
        if marker_color is not None:
            self.marker_color = marker_color
        if in_stretch_convention is not None:
            self.in_stretch_convention = in_stretch_convention

    # -- the shared camera ---------------------------------------------------

    def _build_camera(self, task: Any) -> Any:
        """A free camera at the fixed shared pose, aimed where the robot stands."""
        import mujoco

        from examples.machine_learning.molmospaces.visualize import _base_pose_of

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.fixedcamid = -1
        camera.elevation = SCENE_ELEVATION
        camera.distance = SCENE_DISTANCE

        pose = _base_pose_of(task)
        if pose is None:
            log.debug("[side-by-side] no robot pose; keeping MuJoCo's default framing")
            return camera
        position, yaw = pose
        camera.azimuth = math.degrees(float(yaw)) + SCENE_AZIMUTH_OFFSET_DEG
        # The robot's xy but *not* its z: see `SCENE_LOOKAT_HEIGHT_M`.
        camera.lookat[:] = [float(position[0]), float(position[1]), SCENE_LOOKAT_HEIGHT_M]
        return camera

    def _scene_frame(self, task: Any):
        """The third-person panel, rendered through the fixed camera as built.

        The base class re-aims `lookat` at the robot on every step, which is what
        keeps a driving robot in frame when there is only one of them. Here it
        would pull the two sides apart the moment either base moved, so the
        camera is left exactly where `_build_camera` put it.
        """
        if self._renderer is None:
            return None
        if self._camera is None:
            self._camera = self._build_camera(task)
        try:
            self._renderer.update_scene(self._episode_data(task), camera=self._camera)
            self._draw_grasp_frame(task)
            frame = np.ascontiguousarray(self._renderer.render()[..., ::-1])
            self._draw_axes_note(frame)
        except Exception as error:  # noqa: BLE001 - the camera panels are still worth writing
            log.debug(f"[side-by-side] scene render failed: {error}")
            return None
        self._scene_panel = frame
        return frame

    def _draw_axes_note(self, frame: np.ndarray) -> None:
        """Say, on the panel, which convention its axes are drawn in.

        The Franka half is rotated into Stretch's convention so that one colour
        means one direction across the split screen; silently doing that and not
        saying so would trade one confusion for a worse one. Stretch's half says
        its axes are its own, so the pair reads as a choice rather than an
        unexplained asymmetry.
        """
        import cv2

        note = (
            "franka axes drawn in Stretch's tool convention: x/red = approach"
            if self.in_stretch_convention
            else "stretch axes are its own: x/red = approach"
        )
        origin = (12, frame.shape[0] - 14)
        for thickness, shade in ((3, (0, 0, 0)), (1, (235, 235, 235))):
            cv2.putText(
                frame, note, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.52, shade,
                thickness, cv2.LINE_AA,
            )

    def _draw_grasp_frame(self, task: Any) -> None:
        """Draw this robot's grasp-centre coordinate frame into the updated scene.

        `get_move_group("gripper").leaf_frame_to_world` is the grasp centre on
        *both* robots -- `grasp_center_link` on Stretch and `gripper/grasp_site`
        on the Franka -- and is the same frame the retargeting drives and reads.
        So one accessor serves both halves, and what the two panels show is
        exactly the pair of frames `FRANKA_TO_STRETCH_TOOL` maps between rather
        than some nearby body drawn for illustration.

        Failure here is swallowed to a debug line: a missing move group is a
        reason to lose a marker, not an episode's footage.
        """
        try:
            robot_view = task.env.current_robot.robot_view
            pose = np.asarray(
                robot_view.get_move_group("gripper").leaf_frame_to_world, dtype=float
            )
        except Exception as error:  # noqa: BLE001 - the panel is still worth writing
            log.debug(f"[side-by-side] no grasp frame to draw: {error}")
            return
        if self.in_stretch_convention:
            pose = pose.copy()
            pose[:3, :3] = pose[:3, :3] @ fr.FRANKA_TO_STRETCH_TOOL
        fr.add_frame_marker(
            self._renderer.scene,
            pose,
            color=self.marker_color,
            label="grasp centre",
            ball_radius=GRASP_FRAME_BALL_RADIUS,
            axis_length=GRASP_FRAME_AXIS_LENGTH,
            axis_radius=GRASP_FRAME_AXIS_RADIUS,
        )

    # -- two files instead of one -------------------------------------------

    def start_episode(self, episode_seed: int, task: Any, policy: Any = None) -> None:
        super().start_episode(episode_seed, task, policy=policy)
        self._close_writers()
        house = self._house_label(task)
        index = max(0, self._episodes_per_house.get(house, 1) - 1)
        self._key = f"{house}_ep{index:04d}"
        self.episodes[self._key] = {
            "key": self._key,
            "house": house,
            "index": index,
            "seed": episode_seed,
            "instruction": self._instruction,
            "steps": 0,
            "success": None,
        }
        self.panel_dir.mkdir(parents=True, exist_ok=True)

    def log_step(self, step_idx: int, task: Any, observation: Any = None, policy: Any = None):
        """Write one frame to each of the two files."""
        if self._key is None:
            return
        try:
            scene = self._scene_frame(task)
            cameras = self._camera_grid_panel(self._camera_frames(observation))
        except Exception as error:  # noqa: BLE001 - a dropped frame must not sink the episode
            log.debug(f"[side-by-side] dropped a frame at step {step_idx}: {error}")
            return

        if scene is not None:
            self._scene_writer = self._ensure_writer(
                self._scene_writer, self.panel_dir / f"{self._key}_scene.mp4", scene
            )
            if self._scene_writer is not None:
                self._scene_writer.write(scene)
        if cameras is not None:
            self._camera_writer = self._ensure_writer(
                self._camera_writer, self.panel_dir / f"{self._key}_cams.mp4", cameras
            )
            if self._camera_writer is not None:
                self._camera_writer.write(cameras)
        self.episodes[self._key]["steps"] = int(step_idx) + 1

    def finish_episode(self, success: bool | None = None):
        if self._key is not None:
            self.episodes[self._key]["success"] = success
            (self.panel_dir / "episodes.json").write_text(
                json.dumps(list(self.episodes.values()), indent=2)
            )
        self._close_writers()
        self._key = None
        # The base class holds an outcome banner and renames its file; there is
        # no single file here, so its bookkeeping is reset rather than run.
        self._writer = None
        return super().finish_episode(success=success)

    def _camera_grid_panel(self, cameras):
        """The camera row, at the scene panel's width so the two tile cleanly."""
        if not cameras:
            return None
        width = self._scene_panel_size[0]
        height = max(1, int(round(width / max(1, len(cameras)) * 0.6)))
        return self._camera_grid(cameras, width, height)

    def _ensure_writer(self, writer, path: Path, frame: np.ndarray):
        """Open `path` sized to `frame` the first time, then hand the writer back."""
        import cv2

        if writer is not None:
            return writer
        height, width = frame.shape[:2]
        opened = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self._episode_fps, (width, height)
        )
        if not opened.isOpened():
            log.warning(f"[side-by-side] could not open {path}")
            return None
        return opened

    def _close_writers(self) -> None:
        for writer in (self._scene_writer, self._camera_writer):
            if writer is not None:
                writer.release()
        self._scene_writer = None
        self._camera_writer = None


# =============================================================================
# Running one setup
# =============================================================================


@dataclass
class RunResult:
    """One setup's run: where its panels are, and what each episode did."""

    setup: str
    panel_dir: Path
    episodes: list[EpisodeScore] = field(default_factory=list)
    error: str | None = None

    def outcome_by_key(self) -> dict[str, EpisodeScore]:
        """Probe records, keyed the way `SplitPanelRecorder` keys its panels.

        Matched on instruction rather than order, for the reason
        `params_search._label_episodes` gives: with several workers the records
        arrive per worker, in whichever order they finished.
        """
        return {episode.instruction: episode for episode in self.episodes}


def run_setup(
    setup_key: str,
    params: RetargetParams,
    benchmark_dir: Path,
    output_root: Path,
    recorder: SplitPanelRecorder,
    checkpoint: str | None,
    episode_steps: int | None,
    num_workers: int,
) -> RunResult:
    """Evaluate one setup, recording each episode's two panels."""
    import shutil

    from molmo_spaces.evaluation import run_evaluation

    setup = SETUPS[setup_key]
    run_dir = output_root / "runs" / setup_key
    panel_dir = run_dir / "panels"
    if panel_dir.is_dir():
        shutil.rmtree(panel_dir, ignore_errors=True)
    is_franka = setup.robot == "franka"
    recorder.set_panel_dir(
        panel_dir,
        marker_color=fr.FRANKA_TOOL_COLOR if is_franka else fr.STRETCH_TOOL_COLOR,
        in_stretch_convention=is_franka,
    )

    sink = run_dir / "probe"
    if sink.is_dir():
        shutil.rmtree(sink, ignore_errors=True)
    os.environ[PROBE_SINK_ENV_VAR] = str(sink)

    # Before the config class is resolved, exactly as in `params_search`: the
    # experiment config is built from a "module:Class" string and reads the
    # trial back out of the environment as it constructs.
    publish_params(setup_key, params)
    probe = install_probe()
    probe.sink = sink
    probe.episodes.clear()

    result = RunResult(setup=setup_key, panel_dir=panel_dir)
    log.info(f"[side-by-side] {setup_key}: {params.describe()}")
    try:
        run_evaluation(
            eval_config_cls=qualified_config_name(setup.eval_config),
            benchmark_dir=benchmark_dir,
            checkpoint_path=checkpoint,
            output_dir=run_dir,
            max_episodes=None,
            num_workers=num_workers,
            task_horizon_steps=episode_steps,
            use_wandb=False,
        )
    # One half failing must still leave the other watchable.
    except Exception as error:  # noqa: BLE001
        result.error = f"{type(error).__name__}: {error}"
        log.error(f"[side-by-side] {setup_key} failed: {result.error}")
        log.debug("", exc_info=True)
        return result

    result.episodes = collect_probe_records(sink)
    return result


# =============================================================================
# Tiling the two runs together
# =============================================================================

CAPTION_HEIGHT = 34
"""Height of the caption bar over each half, in pixels."""


class _Stream:
    """One panel MP4, read a frame at a time and holding its last frame at the end.

    Streamed rather than decoded into a list, which is what the first version
    did. The two halves are the same episode on different robots and an episode
    stops when the policy succeeds or runs out of horizon, so they are routinely
    different lengths -- and holding the last frame is the honest way to pad the
    shorter one: that robot *is* standing there, having finished.

    The reason not to hold the frames in memory is the real episode length. Four
    streams of 300 frames at 960x540x3 is about 1.9 GiB per episode, which is a
    lot to ask for a tiling job that only ever needs one frame from each at a
    time.
    """

    def __init__(self, path: Path) -> None:
        import cv2

        self._capture = cv2.VideoCapture(str(path)) if path.is_file() else None
        self._last: np.ndarray | None = None
        self.length = 0
        if self._capture is not None and self._capture.isOpened():
            self.length = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        elif self._capture is not None:
            self._capture.release()
            self._capture = None

    def next(self) -> np.ndarray | None:
        """The next frame, or the last one again once the file runs out."""
        if self._capture is not None:
            ok, frame = self._capture.read()
            if ok:
                self._last = frame
                return frame
            self.close()
        return self._last

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "_Stream":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


def _caption(width: int, text: str, success: bool | None) -> np.ndarray:
    """A caption bar, tinted by outcome."""
    import cv2

    colour = (40, 90, 40) if success else ((40, 40, 90) if success is not None else (60, 60, 60))
    bar = np.full((CAPTION_HEIGHT, width, 3), colour, dtype=np.uint8)
    cv2.putText(
        bar, text[:120], (10, CAPTION_HEIGHT - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (235, 235, 235), 1, cv2.LINE_AA,
    )
    return bar


def _half(
    scene: np.ndarray | None, cameras: np.ndarray | None, width: int, caption: np.ndarray
) -> np.ndarray:
    """One robot's column: caption, scene, cameras -- all at `width`."""
    import cv2

    panels = [caption]
    for panel in (scene, cameras):
        if panel is None:
            continue
        if panel.shape[1] != width:
            scale = width / panel.shape[1]
            panel = cv2.resize(panel, (width, max(1, int(round(panel.shape[0] * scale)))))
        panels.append(panel)
    return np.vstack(panels)


def compose_pair(
    franka: RunResult,
    stretch: RunResult,
    output_dir: Path,
    fps: float = 15.0,
    panel_width: int = SCENE_PANEL_SIZE[0],
) -> list[Path]:
    """Tile each episode's four panels into one MP4. Returns what it wrote.

    Episodes are paired on the key `SplitPanelRecorder` assigns -- house and
    index within that house -- which both runs assign identically because they
    ran the same benchmark in the same order. The instruction is carried through
    to the caption and checked: a mismatch means the pairing is wrong, and a
    silently mispaired video comparing two different grasps would be worse than
    no video.
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    franka_episodes = _panel_index(franka.panel_dir)
    stretch_episodes = _panel_index(stretch.panel_dir)
    franka_outcomes = franka.outcome_by_key()
    stretch_outcomes = stretch.outcome_by_key()

    written: list[Path] = []
    for key in sorted(set(franka_episodes) & set(stretch_episodes)):
        left_meta, right_meta = franka_episodes[key], stretch_episodes[key]
        if left_meta.get("instruction") != right_meta.get("instruction"):
            log.warning(
                f"[side-by-side] {key}: the two runs disagree about the task "
                f"({left_meta.get('instruction')!r} vs {right_meta.get('instruction')!r}); "
                "skipping rather than pairing two different grasps"
            )
            continue

        instruction = left_meta.get("instruction") or ""
        left_score = franka_outcomes.get(instruction)
        right_score = stretch_outcomes.get(instruction)
        left_caption = _caption(
            panel_width, _describe(franka.setup, left_meta, left_score), _picked(left_score)
        )
        right_caption = _caption(
            panel_width, _describe(stretch.setup, right_meta, right_score), _picked(right_score)
        )

        path = output_dir / f"{key}_{_slug(instruction)}.mp4"
        writer = None
        with (
            _Stream(franka.panel_dir / f"{key}_scene.mp4") as left_scene,
            _Stream(franka.panel_dir / f"{key}_cams.mp4") as left_cams,
            _Stream(stretch.panel_dir / f"{key}_scene.mp4") as right_scene,
            _Stream(stretch.panel_dir / f"{key}_cams.mp4") as right_cams,
        ):
            streams = (left_scene, left_cams, right_scene, right_cams)
            length = max(stream.length for stream in streams)
            if not length:
                log.warning(f"[side-by-side] {key}: no frames on either side, skipping")
                continue
            for _ in range(length):
                left = _half(left_scene.next(), left_cams.next(), panel_width, left_caption)
                right = _half(right_scene.next(), right_cams.next(), panel_width, right_caption)
                height = max(left.shape[0], right.shape[0])
                left = np.pad(left, ((0, height - left.shape[0]), (0, 0), (0, 0)))
                right = np.pad(right, ((0, height - right.shape[0]), (0, 0), (0, 0)))
                frame = np.hstack([left, right])
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():
                        log.warning(f"[side-by-side] could not open {path}")
                        break
                writer.write(frame)
        if writer is not None:
            writer.release()
            written.append(path)
            log.info(f"[side-by-side] wrote {path.name} ({length} frames)")
    return written


def _panel_index(panel_dir: Path) -> dict[str, dict[str, Any]]:
    """`episodes.json` from a run, keyed by episode key."""
    path = panel_dir / "episodes.json"
    if not path.is_file():
        return {}
    try:
        return {entry["key"]: entry for entry in json.loads(path.read_text())}
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        log.warning(f"[side-by-side] could not read {path}: {error}")
        return {}


def _picked(score: EpisodeScore | None) -> bool | None:
    """Whether the object was picked up, or None when there is no record."""
    if score is None:
        return None
    return bool(score.success)


def _describe(setup: str, meta: dict[str, Any], score: EpisodeScore | None) -> str:
    """The caption text for one half.

    The retargeting residual is on the caption because it is the number that
    separates the two explanations for a failed grasp: a Stretch that missed with
    a residual of millimetres was aimed at the wrong place by the policy, and one
    that missed with a residual of centimetres was asked for somewhere it could
    not reach. It is blank on the Franka half, which has no retargeting to
    measure.
    """
    instruction = meta.get("instruction") or "?"
    if score is None:
        return f"{setup}  |  {instruction}  |  no probe record"
    outcome = "PICKED UP" if _picked(score) else "not picked up"
    parts = [setup, instruction, f"{outcome} (score {score.score:.2f})"]
    residual = score.retarget_position_error_mean_m
    if residual == residual:  # not NaN, which is how the Franka half reports it
        parts.append(f"retarget miss {residual * 1000:.0f}mm")
    if not score.completed:
        parts.append("CRASHED")
    return "  |  ".join(parts)


def _slug(text: str) -> str:
    """A filesystem-safe fragment of an instruction."""
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")[:40] or "episode"


# =============================================================================
# The command line
# =============================================================================


def _apply_params(base: RetargetParams, specs: tuple[str, ...]) -> RetargetParams:
    """`--param name=value`, applied through the same handles `params_search` searches."""
    params = base
    for spec in specs:
        name, _, value = spec.partition("=")
        if name not in DIMENSIONS:
            raise click.UsageError(
                f"Unknown parameter {name!r}. Available: {', '.join(sorted(DIMENSIONS))}."
            )
        if not value:
            raise click.UsageError(f"--param {spec!r} should be name=value.")
        params = DIMENSIONS[name].write(params, float(value))
    return params


@click.command()
@click.option(
    "--pair",
    type=click.Choice(sorted(MATCHED_PAIRS)),
    default=DEFAULT_PAIR,
    show_default=True,
    help="Which matched (franka, stretch) pair to run. See MATCHED_PAIRS.",
)
@click.option(
    "--param",
    "param_specs",
    multiple=True,
    help="Set a retargeting parameter, as name=value. Repeatable. Applied to both halves "
    "where it applies -- the gripper parameters exist only on the Stretch side. The names "
    "are params_search's own; see its --list-dims.",
)
@click.option(
    "--scenes",
    "scene_count",
    type=int,
    default=mini_benchmark.DEFAULT_SCENE_COUNT,
    show_default=True,
    help="How many scenes to run. Each contributes one episode per object.",
)
@click.option(
    "--episode-steps",
    type=int,
    default=None,
    help="Steps per episode. Defaults to the benchmark's 20s at 15Hz, about 300. "
    "A small number is the way to check the plumbing cheaply.",
)
@click.option(
    "--checkpoint",
    default=None,
    help="A local DROID checkpoint. Defaults to fetching the released one from the Hub.",
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Rollout workers per run. Defaults as params_search does: whichever is smaller of "
    "one per scene or as many as fit in free GPU memory.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("eval_output") / "side_by_side",
    show_default=True,
    help="Where the benchmark, the two runs and the composed videos go.",
)
@click.option(
    "--rebuild-benchmark", is_flag=True, help="Rebuild the benchmark even if it is there."
)
@click.option(
    "--replay-as-stretch4",
    "--replay_as_stretch4",
    "replay_as_stretch4",
    is_flag=True,
    help="Skip the evaluation and instead replay the Franka trajectories already recorded "
    "under --output-dir through the retargeting, as Stretch 4. No policy runs, so it is "
    "seconds rather than minutes -- the fast loop for changing a retargeting parameter and "
    "seeing what it does to actions that are known to work. See retargetting/replay.py.",
)
@click.option(
    "--replay-z-offset",
    type=float,
    default=0.0,
    show_default=True,
    help="target_z_offset to replay with, in metres. The default is 0, which is also "
    "what a rollout now applies unless told otherwise.",
)
@click.option(
    "--replay-limit",
    type=int,
    default=None,
    help="Replay at most this many episodes. Handy on a full run, which has dozens.",
)
@click.option(
    "--replay-no-video",
    is_flag=True,
    help="Measure without rendering, which is much faster when you only want the numbers.",
)
@click.option(
    "--compose-only",
    is_flag=True,
    help="Skip both evaluations and re-tile the panels already under --output-dir. For "
    "changing the layout without paying for the rollouts again.",
)
def main(
    pair: str,
    param_specs: tuple[str, ...],
    scene_count: int,
    episode_steps: int | None,
    checkpoint: str | None,
    num_workers: int | None,
    output_dir: Path,
    rebuild_benchmark: bool,
    compose_only: bool,
    replay_as_stretch4: bool,
    replay_z_offset: float,
    replay_limit: int | None,
    replay_no_video: bool,
) -> None:
    """Run a matched pair over the same episodes and tile them into one video each."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    franka_key, stretch_key = MATCHED_PAIRS[pair]
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"

    if replay_as_stretch4:
        # Before anything expensive: a replay needs no checkpoint, no GPU and no
        # MolmoBot checkout, so it must not be gated behind their setup checks.
        from examples.machine_learning.molmospaces.retargetting import replay as replay_mod

        destination = replay_mod.replay_output_dir(output_dir)
        results = replay_mod.replay_run(
            output_dir,
            destination,
            render=not replay_no_video,
            limit=replay_limit,
            target_z_offset=replay_z_offset,
        )
        replay_mod.report(results, destination, rendered=not replay_no_video)
        return

    if compose_only:
        franka = RunResult(franka_key, output_dir / "runs" / franka_key / "panels")
        stretch = RunResult(stretch_key, output_dir / "runs" / stretch_key / "panels")
        for result in (franka, stretch):
            probe = result.panel_dir.parent / "probe"
            if probe.is_dir():
                result.episodes = collect_probe_records(probe)
        written = compose_pair(franka, stretch, videos_dir)
        _report(written, videos_dir)
        return

    # MolmoBot is a clone rather than a dependency, so nothing puts its `olmo`
    # package on the import path. Checked here so a missing checkout is a message
    # at the command line rather than an ImportError once a scene has loaded.
    from examples.machine_learning.molmospaces.finetuning.molmobot_repo import (
        MolmoBotSetupError,
        ensure_importable,
        inference_requirements_message,
        missing_inference_requirements,
    )

    try:
        ensure_importable()
    except MolmoBotSetupError as error:
        raise click.UsageError(str(error)) from error
    missing = missing_inference_requirements()
    if missing:
        raise click.UsageError(inference_requirements_message(missing))

    workers = num_workers if num_workers is not None else affordable_workers(scene_count)
    benchmark_dir = mini_benchmark.build(
        output_dir / "benchmark", force=rebuild_benchmark, scene_count=scene_count
    )

    recorder = SplitPanelRecorder(output_dir / "runs" / franka_key / "panels")
    _install(recorder)

    click.secho(
        f"Running {franka_key} then {stretch_key} over {scene_count} scene(s) x "
        f"{len(mini_benchmark.TARGETS)} objects, {workers} worker(s) each. "
        "Sequentially, because two policies do not fit on one card.",
        bold=True,
    )

    results = []
    for setup_key in (franka_key, stretch_key):
        base = SETUPS[setup_key].params
        params = _apply_params(base, param_specs)
        results.append(
            run_setup(
                setup_key,
                params,
                benchmark_dir=benchmark_dir,
                output_root=output_dir,
                recorder=recorder,
                checkpoint=checkpoint,
                episode_steps=episode_steps,
                num_workers=workers,
            )
        )

    franka, stretch = results
    for result in results:
        if result.error:
            click.secho(f"{result.setup} failed: {result.error}", fg="red")

    written = compose_pair(franka, stretch, videos_dir)
    _report(written, videos_dir)


def _install(recorder: SplitPanelRecorder) -> None:
    """Register the recorder as the evaluation's only rollout observer.

    `install_eval_video_hook` is deliberately *not* used: it would add the normal
    one-file-per-episode recorder alongside this one, which would render every
    scene twice and write footage nobody reads.
    """
    from examples.machine_learning.molmospaces import visualize

    visualize._EVAL_OBSERVERS.clear()
    visualize._install_eval_rollout_hook(recorder)


def _report(written: list[Path], videos_dir: Path) -> None:
    if not written:
        click.secho(
            "No videos were composed. If the runs completed, check that both "
            f"{videos_dir.parent / 'runs'}/*/panels hold an episodes.json.",
            fg="red",
        )
        return
    click.secho(f"Wrote {len(written)} side-by-side video(s) to {videos_dir}", fg="green")
    for path in written:
        click.echo(f"  {path.name}")


if __name__ == "__main__":
    main()
