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

Replaying instead of running the Stretch half
--------------------------------------------
`--replay-as-stretch4` produces the same split screen without the second
evaluation. The left column is the Franka run's own recorded panels, read back
off disk; the right is the recorded actions pushed through the retargeting
kinematically, with the setup's exo camera and Stretch's wrist camera rendered
underneath it. Both bars say which is which -- RECORDED against REPLAY -- and
the replayed half carries the retargeting residual and the unreachable-step
count in place of an outcome it does not have. Seconds rather than an hour, and
no checkpoint or GPU, which makes it the loop to change a retargeting parameter
in. See `replay_pair` and `retargetting/replay.py`.

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
from collections.abc import Callable
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
    _label_episodes,
    affordable_workers,
    preserve_previous_run,
)
from examples.machine_learning.molmospaces.retargetting.scoring import (  # noqa: E402
    EpisodeScore,
    TrialResult,
    collect_probe_records,
    format_trial_table,
    install_probe,
    write_episode_csv,
    write_report,
    write_trial_csv,
)
from examples.machine_learning.molmospaces.retargetting.setups import (  # noqa: E402
    PROBE_SINK_ENV_VAR,
    SETUPS,
    StretchCameraChoices,
    params_to_json,
    publish_params,
    publish_stretch_camera_choices,
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
    "baseline": ("franka_baseline", "stretch_baseline"),
    "stretchcam": ("franka_stretchcam", "stretch_stretchcam"),
    "fisheye": ("franka_fisheye", "stretch_fisheye"),
    "rectified": ("franka_rectified", "stretch_rectified"),
}
"""
The `(franka, stretch)` setups that differ only in the robot.

Every pair carries identical `ExoCameraParams`, so a difference between its
halves is the robot and the retargeting rather than the view. That is the whole
design: four cameras, each mounted at the same height in the room on both
robots, and nothing else varying.

    baseline    the DROID shoulder camera MolmoBot ships with, 71 degrees
    stretchcam  an upright pinhole at Stretch's head-camera pose, 71 degrees
    fisheye     Stretch's real 123-degree fisheye, distorted
    rectified   that fisheye rectified, whole frame

`baseline` used to have no Stretch half, on the grounds that nothing reproduced
the DROID camera on Stretch. It does now: the camera is mounted on `base_link` at
`STRETCH_BASELINE_HEIGHT`, which is the same 1.41 m above the floor it sits at on
the Franka's pedestal. The one thing the transplant cannot preserve is that the
Franka's hangs off `fr3_link0` and turns with the arm; Stretch has no link that
moves that way, so its copy is fixed to the base.

`rectified` carries no crop and no synthesised pitch. Both were there to trade
field of view for the checkpoint's training shape, and both made the pair carry
two changes instead of one.
"""

ALL_PAIRS = "all"
"""`--pair all`: run every pair in one go, which is eight setups and four videos."""

DEFAULT_PAIR = ALL_PAIRS
"""
Every pair, because the four of them are the experiment rather than four
alternatives.

Read in order they walk from the camera the policy knows to the one it does not:
`baseline` is the DROID camera on both robots, `stretchcam` moves to Stretch's
head-camera pose, `fisheye` puts the real lens on, `rectified` takes the
distortion back out. A difference that appears at one step and not the one before
it is attributable to that step.
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
            "seed": int(episode_seed),
            "instruction": self._instruction,
            "steps": 0,
            "success": None,
        }
        try:
            self.panel_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            # Same rule as `_write_index`: this runs inside the rollout, so it
            # must not be able to end one. Losing the panels is bad; losing the
            # episode they were recording is worse.
            log.warning(f"[side-by-side] cannot write to {self.panel_dir}: {error}")
            self._key = None

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
            # `bool(...)`, not the value as handed over. The pipeline reports
            # success as a `numpy.bool`, which `json.dumps` refuses -- and the
            # refusal is maddening to read, because NumPy 2 renamed `np.bool_` to
            # `numpy.bool` whose `__class__.__name__` is `"bool"`, so the error
            # says "Object of type bool is not JSON serializable" about a type
            # that is not Python's bool. That exception used to escape this
            # method and take the whole rollout with it.
            self.episodes[self._key]["success"] = None if success is None else bool(success)
            self._write_index()
        self._close_writers()
        self._key = None
        # The base class holds an outcome banner and renames its file; there is
        # no single file here, so its bookkeeping is reset rather than run.
        self._writer = None
        return super().finish_episode(success=success)

    def _write_index(self) -> None:
        """Write `episodes.json`, and never let a failure to do so end an episode.

        This is an observer: the pipeline calls it from inside the rollout, so an
        exception raised here surfaces as "rollout error" and loses the episode --
        which is what a stray `numpy.bool` did before the coercion above. The
        panels are already on disk by this point, so the worst a failure here
        should cost is the index that pairs them, and `compose_pair` says so
        clearly when it finds none.
        """
        try:
            (self.panel_dir / "episodes.json").write_text(
                json.dumps(list(self.episodes.values()), indent=2)
            )
        except (TypeError, ValueError, OSError) as error:
            log.warning(
                f"[side-by-side] could not write {self.panel_dir / 'episodes.json'}: {error}. "
                f"The panels are written; the pairing index is not, so --compose-only will "
                f"find nothing to tile for this run."
            )

    def _camera_grid_panel(self, cameras):
        """The camera row, at the scene panel's width so the two tile cleanly.

        `replay.camera_row` rather than a local formula, because a replayed
        Stretch's camera row is tiled against a recorded Franka's and the two
        have to be laid out identically.
        """
        if not cameras:
            return None
        from examples.machine_learning.molmospaces.retargetting import replay as replay_mod

        return replay_mod.camera_row(cameras, self._scene_panel_size[0])

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

    params: RetargetParams | None = None
    """The parameters this setup ran at, so the report can say what it measured."""

    output_dir: str = ""
    """Where `run_evaluation` wrote, for the report's per-trial link."""

    def as_trial(self) -> TrialResult:
        """This run as a `scoring.TrialResult`, which is what the report is built from.

        One setup at one point is exactly what `params_search` calls a trial, so
        the report machinery is reused rather than reimplemented: the same
        scoring, the same columns, the same meaning of `score`. What differs is
        only that a side-by-side run has one trial per setup instead of one per
        searched point.
        """
        params = self.params if self.params is not None else SETUPS[self.setup].params
        return TrialResult(
            setup=self.setup,
            params_description=params.describe(),
            params_json=params_to_json(self.setup, params),
            episodes=_label_episodes(self.episodes, self.setup),
            output_dir=self.output_dir,
            error=self.error or "",
        )

    def outcome_by_key(self) -> dict[tuple[int, str], EpisodeScore]:
        """Probe records, keyed the way `SplitPanelRecorder` keys its panels.

        Keyed on scene *and* instruction rather than on order, for the reason
        `params_search._label_episodes` gives: with several workers the records
        arrive per worker, in whichever order they finished.

        The scene has to be in the key. A benchmark runs the same four
        instructions in every house, so keying on the instruction alone collapsed
        twenty episodes onto four and left each caption showing whichever scene
        happened to be written last -- which is how `house_1011`'s salt shaker
        came to be captioned "PICKED UP (score 1.00)" in a run whose own
        `episodes.csv` records it as a failure. A video that disagrees with the
        report about what happened is worse than no video, because the report is
        what gets checked second.
        """
        return {(episode.scene, episode.instruction): episode for episode in self.episodes}


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

    result = RunResult(setup=setup_key, panel_dir=panel_dir, params=params)
    log.info(f"[side-by-side] {setup_key}: {params.describe()}")
    try:
        evaluation = run_evaluation(
            eval_config_cls=qualified_config_name(setup.eval_config),
            benchmark_dir=benchmark_dir,
            checkpoint_path=checkpoint,
            output_dir=run_dir,
            max_episodes=None,
            num_workers=num_workers,
            task_horizon_steps=episode_steps,
            use_wandb=False,
        )
        result.output_dir = str(evaluation.output_dir)
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


CAPTION_FONT_SCALE = 0.5
CAPTION_MIN_FONT_SCALE = 0.34
"""How small a caption may be shrunk before it is clipped instead.

Shrunk rather than clipped because the end of a caption is where its numbers
are: a replay's bar ends in the retargeting residual and the unreachable-step
count, which are the two things it exists to say, and a fixed font drops
exactly those off the right edge. Below `CAPTION_MIN_FONT_SCALE` the text stops
being readable at video resolution, so past that it is truncated after all.
"""


def _caption(width: int, text: str, success: bool | None) -> np.ndarray:
    """A caption bar, tinted by outcome and sized so the whole line fits."""
    import cv2

    colour = (40, 90, 40) if success else ((40, 40, 90) if success is not None else (60, 60, 60))
    bar = np.full((CAPTION_HEIGHT, width, 3), colour, dtype=np.uint8)
    room = width - 20
    scale = CAPTION_FONT_SCALE
    while (
        scale > CAPTION_MIN_FONT_SCALE
        and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > room
    ):
        scale -= 0.02
    cv2.putText(
        bar, text[:160], (10, CAPTION_HEIGHT - 11), cv2.FONT_HERSHEY_SIMPLEX, scale,
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


Captioner = Callable[["RunResult", dict[str, Any], "EpisodeScore | None"], tuple[str, bool | None]]
"""How one half's caption bar is written: its text, and how the bar is tinted.

A hook rather than a fixed rule because a replay's right-hand half has no
outcome and no probe record to describe -- what it has is a retargeting
residual and a count of steps it could not reach -- and captioning it "no probe
record" would read as a broken run rather than a different experiment. See
`_replay_captioner`.
"""


def _default_caption(
    run: "RunResult", meta: dict[str, Any], score: "EpisodeScore | None"
) -> tuple[str, bool | None]:
    """What a rollout's own half says: the setup, the task and the outcome."""
    return _describe(run.setup, meta, score), _picked(score)


def compose_pair(
    franka: RunResult,
    stretch: RunResult,
    output_dir: Path,
    fps: float = 15.0,
    panel_width: int = SCENE_PANEL_SIZE[0],
    caption_for: Captioner = _default_caption,
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
        scene = _scene_of(left_meta)
        left_score = franka_outcomes.get((scene, instruction))
        right_score = stretch_outcomes.get((_scene_of(right_meta), instruction))
        left_caption = _caption(panel_width, *caption_for(franka, left_meta, left_score))
        right_caption = _caption(panel_width, *caption_for(stretch, right_meta, right_score))

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


def _scene_of(meta: dict[str, Any]) -> int:
    """The house index behind a panel record's `house` field (`"house_1011"` -> 1011).

    `SplitPanelRecorder` writes the name; `EpisodeScore.scene` holds the number,
    so one of the two has to be converted before they can be matched.
    """
    house = str(meta.get("house") or "")
    digits = house.rsplit("_", 1)[-1]
    return int(digits) if digits.isdigit() else -1


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
# The same tiling, from a replay rather than a second rollout
# =============================================================================


def _replay_captioner(stretch_setup: str, results: dict[str, Any]) -> Captioner:
    """Caption both halves of a replayed pair, saying which is which.

    The two halves of a replay are not two rollouts and must not be captioned as
    though they were. The left is a recording -- a real Franka rollout, with a
    real outcome, played back from the MP4 it was recorded to. The right never
    ran a policy: it is those same actions pushed through the retargeting.

    What the right half may say about the object depends on which kind of replay
    it was. A kinematic one writes joint positions and runs `mj_forward`, so the
    object cannot move and the bar stays grey with the residual and the
    unreachable-step count on it -- the numbers that are the reason to look at a
    kinematic replay at all. A physics one steps the scene through the
    evaluation's own controllers, so the object is grasped or it is not, and that
    verdict is worth the same tint a rollout's is. It is still not a rollout's
    verdict: the actions are the Franka's, replayed blind, so it says the
    retargeting can or cannot hold a grasp the policy already found, and nothing
    about whether the policy finds it through Stretch's camera.
    """

    def caption(
        run: RunResult, meta: dict[str, Any], score: EpisodeScore | None
    ) -> tuple[str, bool | None]:
        if run.setup != stretch_setup:
            return f"RECORDED  |  {_describe(run.setup, meta, score)}", _picked(score)

        instruction = meta.get("instruction") or "?"
        result = results.get(meta.get("key", ""))
        grasp = getattr(result, "grasp", None)
        kind = "physics" if grasp is not None else "kinematic"
        parts = [f"REPLAY ({kind})  |  {run.setup}", instruction]
        if grasp is not None:
            parts.append(
                "PICKED UP"
                if grasp.success
                else f"no grasp (lifted {grasp.best_lift_m * 1000:.0f}mm)"
            )
        if result is not None and len(result.position_error_m):
            parts.append(
                f"retarget miss {result.position_error_m.mean() * 1000:.0f}mm mean, "
                f"{result.position_error_m.max() * 1000:.0f}mm max"
            )
            parts.append(f"unreachable {result.unreachable_steps}/{result.episode.steps}")
        return "  |  ".join(parts), (grasp.success if grasp is not None else None)

    return caption


def replay_pair(
    pair: str,
    output_dir: Path,
    params: RetargetParams,
    scene_count: int,
    target_z_offset: float,
    limit: int | None = None,
    fps: float = 15.0,
    physics: bool = True,
) -> tuple[list[Path], list[Any]]:
    """Replay one pair's recorded Franka episodes and tile them as the rollouts are.

    The left half is not re-rendered -- it *is* the Franka run's own panels,
    read back off disk. That is the whole economy of the thing: the expensive
    half of a side-by-side is the rollout, the recording of it is already there,
    and a replay only has to produce the other column. It also means the Franka
    column is the same footage the rollout's own video shows, rather than a
    second render of the same episode that could differ.

    Episodes are matched to that footage by `RecordedEpisode.panel_key`, and the
    match is checked on the step count before anything is rendered: a trajectory
    file and a panel that disagree about how long the episode was are not the
    same episode, and tiling them would produce a confident-looking comparison
    of two different rollouts.
    """
    from examples.machine_learning.molmospaces.retargetting import replay as replay_mod

    franka_key, stretch_key = MATCHED_PAIRS[pair]
    franka = RunResult(setup=franka_key, panel_dir=output_dir / "runs" / franka_key / "panels")
    probe = franka.panel_dir.parent / "probe"
    if probe.is_dir():
        franka.episodes = own_records(collect_probe_records(probe), franka_key)

    index = _panel_index(franka.panel_dir)
    if not index:
        log.warning(
            f"[replay] no episodes.json under {franka.panel_dir}; there is no Franka footage "
            f"to put a replay beside. Run the pair first, without --replay-as-stretch4."
        )
        return [], []

    destination = replay_mod.replay_output_dir(output_dir) / pair
    panel_dir = destination / "panels"
    results: list[Any] = []
    by_key: dict[str, Any] = {}
    written_index: list[dict[str, Any]] = []

    for path in replay_mod.latest_run_trajectories(output_dir / "runs" / franka_key):
        for episode in replay_mod.load_episodes(path):
            if limit is not None and len(results) >= limit:
                break
            key = episode.panel_key
            meta = index.get(key)
            if meta is None:
                log.warning(
                    f"[replay] {key} is in {path.name} but not in the recorded panels; "
                    f"skipping rather than tiling it against nothing."
                )
                continue
            if int(meta.get("steps", -1)) != episode.steps:
                log.warning(
                    f"[replay] {key}: the trajectory has {episode.steps} steps and the "
                    f"recorded panel {meta.get('steps')}; these are not the same episode, "
                    f"so it is skipped rather than mispaired."
                )
                continue

            episode.instruction = meta.get("instruction", "")
            log.info(f"[replay] {key}: {episode.steps} recorded steps -- {episode.instruction}")
            result = replay_mod.replay_to_panels(
                episode,
                panel_dir,
                key,
                exo=params.exo,
                # The episode's own object, from the benchmark the run used, so
                # the two halves show the same grasp rather than one hand on a
                # bowl and one on an empty counter.
                stage=replay_mod.episode_staging(
                    output_dir, episode.house, int(meta.get("index", 0)), episode.instruction
                ),
                fps=fps,
                target_z_offset=target_z_offset,
                tool_correction=(params.wrist_tilt_deg, params.grasp_offset_m),
                aperture_m=params.aperture_m or None,
                scene_count=scene_count,
                physics=physics,
            )
            results.append(result)
            by_key[key] = result
            # The recorded episode's own record, minus the outcome: a replay has
            # none, and `compose_pair` pairs on the key and the instruction.
            written_index.append({**meta, "success": None})
            log.info(f"[replay] {result.summary()}")

    if not results:
        return [], []

    panel_dir.mkdir(parents=True, exist_ok=True)
    (panel_dir / "episodes.json").write_text(json.dumps(written_index, indent=2))

    stretch = RunResult(setup=stretch_key, panel_dir=panel_dir)
    written = compose_pair(
        franka, stretch, destination, fps=fps, caption_for=_replay_captioner(stretch_key, by_key)
    )
    return written, results


# =============================================================================
# The report
# =============================================================================


def write_outputs(runs: list[RunResult], output_dir: Path) -> Path:
    """Write the same report, CSVs and JSONL `params_search` writes.

    Deliberately the *same* files with the same columns, produced by the same
    `scoring` functions: a side-by-side run and a search run then score
    identically and their rows can be read against each other. The only
    difference is what a row is -- here one setup at its own parameters, there
    one searched point.
    """
    trials = [run.as_trial() for run in runs]
    write_episode_csv(trials, output_dir / "episodes.csv")
    write_trial_csv(trials, output_dir / "trials.csv")
    (output_dir / "trials.jsonl").write_text(
        "".join(
            json.dumps({"setup": t.setup, "score": t.score, "params": json.loads(t.params_json)})
            + "\n"
            for t in trials
        )
    )
    return write_report(trials, mini_benchmark.TARGET_KEYS, output_dir)


def own_records(episodes: list[EpisodeScore], setup: str) -> list[EpisodeScore]:
    """The setup's *own* episodes, when its sink also caught later setups'.

    A run made before `install_probe` was idempotent left every earlier sink
    still being written to, so the first setup's file holds all eight setups'
    episodes concatenated in run order and the last holds only its own. See the
    comment in `scoring.install_probe`.

    Detected from the data rather than from a count, so it needs no flag to be
    told how big a run was: every setup sees the same episodes in the same order,
    so the block ends where the `(scene, instruction)` of the first record comes
    round again. A correctly-written sink has no repeat and is returned whole.
    """
    if not episodes:
        return episodes
    signature = lambda e: (e.scene, e.instruction)  # noqa: E731
    first = signature(episodes[0])
    for index in range(1, len(episodes)):
        if signature(episodes[index]) == first:
            log.warning(
                f"[side-by-side] {setup}'s probe holds {len(episodes)} episodes but repeats "
                f"after {index}; taking the first {index} as its own. This run predates the "
                f"fix to `install_probe`, which left one sink per setup all being written at "
                f"once -- see `scoring.install_probe`."
            )
            return episodes[:index]
    return episodes


def runs_from_disk(
    output_dir: Path, setup_keys: list[str], param_specs: tuple[str, ...]
) -> list[RunResult]:
    """Rebuild each setup's `RunResult` from what a finished run left on disk.

    The probe records are the run's own scoring output and are all the report
    needs; the parameters are recomputed the way the run computed them, from the
    setup's defaults and the same `--param` overrides. That means a report built
    after the fact has to be asked for with the same `--param` flags the run
    used, which is why the report names them in its own table.
    """
    runs = []
    for key in setup_keys:
        probe = output_dir / "runs" / key / "probe"
        episodes = own_records(collect_probe_records(probe), key) if probe.is_dir() else []
        if not episodes:
            log.warning(f"[side-by-side] no probe records under {probe}; {key} will be empty")
        runs.append(
            RunResult(
                setup=key,
                panel_dir=output_dir / "runs" / key / "panels",
                episodes=episodes,
                params=_apply_params(SETUPS[key].params, param_specs, key),
            )
        )
    return runs


# =============================================================================
# The command line
# =============================================================================


def _typed(name: str) -> bool:
    """Whether `name` was given on the command line, rather than left at its default.

    The overrides that default to "whatever the setup already says" all need this
    distinction, and a sentinel default cannot carry it: 0.0 is a legitimate
    value for every one of them. Click's `get_parameter_source` is what knows.
    Outside a click context -- a test, an importer -- nothing was typed.
    """
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    source = ctx.get_parameter_source(name)
    return source is not None and source.name != "DEFAULT"


def _apply_params(
    base: RetargetParams, specs: tuple[str, ...], setup_key: str
) -> RetargetParams:
    """A setup's params as this invocation wants them: `--param`, then the flags.

    `--stretch4-grasp-offset` is read off the click context rather than passed
    in, so that every path that rebuilds a setup's params picks it up: the run
    loop, the replay, and `runs_from_disk`, which is what `--report-only` and
    `--compose-only` label their tables from. A report naming the setup's default
    offset while the run it describes used another one would be quietly wrong,
    and quietly is the problem -- nobody re-checks a number the tool printed.

    `setup_key` is what decides whether it applies at all. `grasp_offset_m` is a
    correction to the Franka-to-Stretch tool transform (`apply_tool_correction`),
    so it means nothing on a Franka setup -- writing it there would put a number
    in that row of `trials.csv` that describes nothing the run did. Hence the
    name: it is the Stretch side's offset, and the Franka half of a pair keeps
    its own 0.

    Applied after `--param` so it wins over a `--param grasp_offset_m=` naming
    the same number; the dedicated flag is the more specific statement. Note the
    asymmetry that leaves: `--param` is applied to whichever setup it is handed,
    Franka included, which is what its own help means by "where it applies".
    """
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
    if _typed("stretch4_grasp_offset") and SETUPS[setup_key].robot == "stretch":
        offset = click.get_current_context().params["stretch4_grasp_offset"]
        params = DIMENSIONS["grasp_offset_m"].write(params, float(offset))
    return params


@click.command()
@click.option(
    "--pair",
    type=click.Choice([ALL_PAIRS, *MATCHED_PAIRS]),
    default=DEFAULT_PAIR,
    show_default=True,
    help="Which matched (franka, stretch) pair to run, or 'all' for every one of them "
    "-- eight setups, four split-screen videos per episode. See MATCHED_PAIRS.",
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
    "seeing what it does to actions that are known to work. Writes the same split screen a "
    "run does -- the recorded Franka on the left, the replayed Stretch and its cameras on "
    "the right, captioned as a replay -- under <output-dir>/replay_as_stretch4/<pair>. See "
    "retargetting/replay.py.",
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
    "--stretch4-grasp-offset",
    "--stretch4_grasp_offset",
    "stretch4_grasp_offset",
    type=float,
    default=0.0,
    show_default=True,
    help="grasp_offset_m for every Stretch setup this invocation touches, in metres -- "
    "how far along the approach axis Stretch's commanded grasp centre sits from where "
    "the retargeting puts the Franka's. Applies to the evaluation runs and to a replay "
    "alike, and overrides --param grasp_offset_m. Stretch only, because it is a term in "
    "the Franka-to-Stretch tool transform and means nothing on a Franka: the Franka half "
    "of a pair keeps its own 0. Left untyped, each setup keeps its own offset. The one "
    "parameter a grasp is most sensitive to; see setups.apply_tool_correction.",
)
@click.option(
    "--replay-kinematic",
    "replay_kinematic",
    is_flag=True,
    help="Replay without the physics: write the retargeted joint positions and run "
    "mj_forward, so nothing the gripper touches moves. Faster, and the way to read the "
    "retargeting's reach on its own -- the residual and the unreachable-step count are "
    "the same either way. The default steps the scene through the evaluation's own "
    "controllers instead, so the replay can actually pick the object up and say so.",
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
    help="Measure without rendering, which is much faster when you only want the numbers. "
    "Nothing is tiled, so this replays every Franka trajectory under --output-dir rather "
    "than only the ones the pair's own run recorded.",
)
@click.option(
    "--report-only",
    is_flag=True,
    help="Write report.md and the CSVs from a finished run's probe records and exit. "
    "No rollouts and no tiling, so it is seconds -- the way to get a report for a run "
    "that predates this flag, or to refresh one after changing how a score is computed. "
    "Pass the same --param flags the run used.",
)
@click.option(
    "--compose-only",
    is_flag=True,
    help="Skip both evaluations and re-tile the panels already under --output-dir. For "
    "changing the layout without paying for the rollouts again.",
)
@click.option(
    "--change_franka_start_pose_flip_wrist",
    "change_franka_start_pose_flip_wrist",
    is_flag=True,
    help="Start the Franka rolled half a turn about its approach axis. The grasp is "
    "identical either way round; what swings round is the hand, and the wrist camera "
    "bolted off to one side of it. See `franka_retarget.PoseConventions`.",
)
@click.option(
    "--change_franka_start_pose_limit_height",
    "change_franka_start_pose_limit_height",
    is_flag=True,
    help="Cap the Franka's start tool height at Stretch's own reach ceiling, so the "
    "Stretch condition does not begin every episode with its lift already at its stop. "
    "See `franka_retarget.PoseConventions`.",
)
@click.option(
    "--change_stretch_start_pose_flip_wrist",
    "change_stretch_start_pose_flip_wrist",
    is_flag=True,
    help="Spawn Stretch with its own wrist rolled half a turn, the counterpart of "
    "--change_franka_start_pose_flip_wrist. Overwritten by the snap to the Franka's home "
    "unless snap_to_franka_home is off. See `franka_retarget.PoseConventions`.",
)
@click.option(
    "--match_stretch_spawn_pose_to_franka",
    "match_stretch_spawn_pose_to_franka",
    is_flag=True,
    help="Stand Stretch back far enough that its spawn gripper pose is the Franka's, "
    "cancelling the retreat in the virtual Franka's mount so the frame is unchanged. "
    "Costs most of the arm's remaining reach and moves the base-mounted exo camera with "
    "it -- see `fr.stretch_spawn_base_offset_xy` for both numbers.",
)
@click.option(
    "--map_franka_wrist_to_flipped_stretch4_wrist",
    "map_franka_wrist_to_flipped_stretch4_wrist",
    is_flag=True,
    help="Retarget every pose onto the half-turned branch of Stretch's wrist, by folding "
    "the turn into the tool transform itself -- so it holds for the whole episode and both "
    "directions carry it, unlike jaw_mode. See `franka_retarget.PoseConventions`.",
)
@click.option(
    "--use_left_gripper_camera",
    "use_left_gripper_camera",
    is_flag=True,
    help="Feed the policy's wrist channel Stretch's left gripper camera instead of its "
    "right. The two are a stereo pair 20mm apart on the same side of the hand, both 241mm "
    "from the grasp centre and 10mm either side of it, so this is a parallax check rather "
    "than a new viewpoint. Stretch only. See `setups.StretchCameraChoices`.",
)
@click.option(
    "--use_left_fisheye_camera",
    "use_left_fisheye_camera",
    is_flag=True,
    help="Feed the policy's exo channel Stretch's real left head fisheye, warped and "
    "turned as the hardware produces it, in place of the exo camera under test -- which "
    "leaves every camera parameter of the trial inert. Stretch only, so the Franka half of "
    "the pair keeps its own camera. See `setups.StretchCameraChoices`.",
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
    report_only: bool,
    replay_as_stretch4: bool,
    replay_z_offset: float,
    stretch4_grasp_offset: float,
    replay_limit: int | None,
    replay_no_video: bool,
    replay_kinematic: bool,
    change_franka_start_pose_flip_wrist: bool,
    change_franka_start_pose_limit_height: bool,
    change_stretch_start_pose_flip_wrist: bool,
    map_franka_wrist_to_flipped_stretch4_wrist: bool,
    match_stretch_spawn_pose_to_franka: bool,
    use_left_gripper_camera: bool,
    use_left_fisheye_camera: bool,
) -> None:
    """Run a matched pair over the same episodes and tile them into one video each."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conventions = fr.PoseConventions(
        change_franka_start_pose_flip_wrist=change_franka_start_pose_flip_wrist,
        change_franka_start_pose_limit_height=change_franka_start_pose_limit_height,
        change_stretch_start_pose_flip_wrist=change_stretch_start_pose_flip_wrist,
        map_franka_wrist_to_flipped_stretch4_wrist=map_franka_wrist_to_flipped_stretch4_wrist,
        match_stretch_spawn_pose_to_franka=match_stretch_spawn_pose_to_franka,
    )
    # Before anything runs, and every variable written in both directions: the
    # point of a matched pair is that its two halves differ in exactly one thing,
    # so a convention left over from a previous run in the same shell would undo
    # the pairing. See `publish_pose_conventions`.
    fr.publish_pose_conventions(conventions)
    if conventions:
        log.info(f"[pose] conventions: {conventions.describe()}")

    # Published the same way and for the same reason: a camera left selected by
    # an earlier run in the same shell would put the two halves of a pair on
    # different lenses. See `setups.publish_stretch_camera_choices`.
    camera_choices = StretchCameraChoices(
        use_left_gripper_camera=use_left_gripper_camera,
        use_left_fisheye_camera=use_left_fisheye_camera,
    )
    publish_stretch_camera_choices(camera_choices)
    if camera_choices:
        log.info(f"[camera] Stretch reads: {camera_choices.describe()}")

    pair_names = list(MATCHED_PAIRS) if pair == ALL_PAIRS else [pair]
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"

    if replay_as_stretch4:
        # Before anything expensive: a replay needs no checkpoint, no GPU and no
        # MolmoBot checkout, so it must not be gated behind their setup checks.
        from examples.machine_learning.molmospaces.retargetting import replay as replay_mod

        destination = replay_mod.replay_output_dir(output_dir)
        if replay_no_video:
            # Nothing to tile, so nothing to pair: every Franka trajectory under
            # the directory is replayed for its numbers, whichever run left it.
            # The tool correction still has to come from somewhere, and it is the
            # first pair's Stretch setup plus whatever --param says -- otherwise
            # this measures a retargeting no setup uses and `--param
            # grasp_offset_m=...` would be silently ignored, which is the one
            # thing anybody runs this flag to sweep.
            stretch_key = MATCHED_PAIRS[pair_names[0]][1]
            params = _apply_params(SETUPS[stretch_key].params, param_specs, stretch_key)
            z_offset = replay_z_offset if _typed("replay_z_offset") else params.target_z_offset_m
            log.info(f"[replay] {params.describe()}, target_z_offset={z_offset:+.3f}m")
            results = replay_mod.replay_run(
                output_dir,
                destination,
                render=False,
                limit=replay_limit,
                target_z_offset=z_offset,
                tool_correction=(params.wrist_tilt_deg, params.grasp_offset_m),
                aperture_m=params.aperture_m or None,
                # The run's own --scenes, so `scene_for_house` searches a list
                # that contains the houses the trajectories were recorded in.
                scene_count=scene_count,
                physics=not replay_kinematic,
            )
            replay_mod.report(results, destination, rendered=False)
            return

        written: list[Path] = []
        results = []
        for name in pair_names:
            stretch_key = MATCHED_PAIRS[name][1]
            params = _apply_params(SETUPS[stretch_key].params, param_specs, stretch_key)
            # The setup's own z offset unless the flag was actually typed, so a
            # replay of `stretch_baseline` retargets the way that setup does
            # rather than the way the flag's default happens to.
            z_offset = replay_z_offset if _typed("replay_z_offset") else params.target_z_offset_m
            log.info(
                f"[replay] {name}: replaying {MATCHED_PAIRS[name][0]}'s recorded episodes as "
                f"{stretch_key} -- {params.describe()}, target_z_offset={z_offset:+.3f}m"
            )
            pair_videos, pair_results = replay_pair(
                name,
                output_dir,
                params,
                scene_count=scene_count,
                target_z_offset=z_offset,
                limit=replay_limit,
                physics=not replay_kinematic,
            )
            written += pair_videos
            results += pair_results
        _report(written, destination)
        replay_mod.report(results, destination, rendered=False)
        return

    setup_keys = [key for name in pair_names for key in MATCHED_PAIRS[name]]

    if report_only:
        runs = runs_from_disk(output_dir, setup_keys, param_specs)
        report = write_outputs(runs, output_dir)
        table = format_trial_table([r.as_trial() for r in runs], mini_benchmark.TARGET_KEYS)
        click.echo("\n" + table)
        click.secho(f"\nWrote {report}", fg="green")
        return

    if compose_only:
        written = []
        for name in pair_names:
            franka_key, stretch_key = MATCHED_PAIRS[name]
            halves = [
                RunResult(key, output_dir / "runs" / key / "panels")
                for key in (franka_key, stretch_key)
            ]
            for result in halves:
                probe = result.panel_dir.parent / "probe"
                if probe.is_dir():
                    result.episodes = collect_probe_records(probe)
            written += compose_pair(*halves, videos_dir / name)
        _report(written, videos_dir)
        report = write_outputs(runs_from_disk(output_dir, setup_keys, param_specs), output_dir)
        click.secho(f"Wrote {report}", fg="green")
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

    # One recorder, re-pointed per run; `set_panel_dir` is what keeps each run's
    # panels and episode numbering separate.
    recorder = SplitPanelRecorder(output_dir / "runs")
    _install(recorder)

    setups_to_run = setup_keys
    episodes = scene_count * len(mini_benchmark.TARGETS)
    click.secho(
        f"Running {len(setups_to_run)} setup(s) over {scene_count} scene(s) x "
        f"{len(mini_benchmark.TARGETS)} objects = {episodes} episodes each, "
        f"{workers} worker(s). Sequentially, because two policies do not fit on one card. "
        f"Output: {len(pair_names)} split-screen video(s) per episode.",
        bold=True,
    )
    for name in pair_names:
        click.echo(f"  {name:11s} {' vs '.join(MATCHED_PAIRS[name])}")

    # Every setup first, then the tiling: a pair cannot be composed until both
    # its halves have run, and running them pair by pair would reload the
    # checkpoint for each half anyway.
    runs: dict[str, RunResult] = {}
    for setup_key in setups_to_run:
        base = SETUPS[setup_key].params
        params = _apply_params(base, param_specs, setup_key)
        runs[setup_key] = run_setup(
            setup_key,
            params,
            benchmark_dir=benchmark_dir,
            output_root=output_dir,
            recorder=recorder,
            checkpoint=checkpoint,
            episode_steps=episode_steps,
            num_workers=workers,
        )
        if runs[setup_key].error:
            click.secho(f"{setup_key} failed: {runs[setup_key].error}", fg="red")

    written = []
    for name in pair_names:
        franka_key, stretch_key = MATCHED_PAIRS[name]
        # A pair whose halves both failed has nothing to tile; one that lost a
        # half still writes what it has, held against the surviving side.
        written += compose_pair(runs[franka_key], runs[stretch_key], videos_dir / name)
    _report(written, videos_dir)

    ordered = [runs[key] for key in setups_to_run]
    table = format_trial_table([r.as_trial() for r in ordered], mini_benchmark.TARGET_KEYS)
    click.echo("\n" + table)
    report = write_outputs(ordered, output_dir)
    click.secho(f"\nWrote {report}", fg="green")


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
            "No videos were composed. If the runs completed, check that both halves of a "
            f"pair left an episodes.json under {videos_dir.parent / 'runs'}/*/panels.",
            fg="red",
        )
        return
    click.secho(f"Wrote {len(written)} side-by-side video(s) under {videos_dir}", fg="green")
    for path in written:
        # `<pair>/<episode>.mp4`, so the pair is visible without the full path.
        click.echo(f"  {path.parent.name}/{path.name}")


if __name__ == "__main__":
    main()
