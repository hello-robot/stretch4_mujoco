"""
Record teleoperated Stretch episodes from the live simulator, in MolmoSpaces format.

`generate_dataset.py` gets demonstrations from the simple_ik expert, which is
fast, unbounded and limited to what the expert can plan. This is the other
source: a person driving the robot in `examples/molmo_environment.py`, writing
the *same* on-disk format, so both feed the same trainers with no branching
downstream.

    python -m examples.molmo_environment --dataset procthor-10k --house-index 0 \\
        --keyboard --record_dataset data/teleop_pick --record-task "pick up the mug"

Everything about the format is in `../hdf5_layout.py`; this module is about the two
things live recording has to decide for itself.

**What counts as the action.** A teleop session has no commanded target vector
to record -- the operator nudges velocities, and the position controllers chase
whatever falls out. So the action recorded for step *t* is the state observed at
step *t+1*: for a position-controlled arm that *is* the command, retrospectively,
and it is exactly what `networks.encode_action` and MolmoBot's
`actions/joint_pos` both mean by one. `actions/joint_pos_rel` is written from
the same pair as a difference, because MolmoBot's dataset prefers it. The last
frame of an episode has no successor and is dropped rather than duplicated.

**Where an episode begins and ends.** The operator says, with a keypress
(`R` to start, `T` to finish and keep, `X` to discard). Recording continuously
and slicing later would fill a dataset with the minutes spent driving between
objects, and a demonstration of "reach for the mug" that starts thirty seconds
before the reach teaches a policy to wait.

Nothing here is on the control path: `record_step()` appends to lists and hands
a frame to a video writer, so switching recording on does not change how the
robot behaves.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from examples.machine_learning.molmospaces.hdf5_layout import (
    encode_json_blob,
    encode_video_path,
    video_filename,
)

log = logging.getLogger(__name__)

TRAINED_CAMERA_MJCF_NAMES = {
    "head_camera": "camera_center_link",
    "wrist_camera_left": "gripper_camera_left_rgb",
    "wrist_camera_right": "gripper_camera_right_rgb",
    "head_camera_left": "camera_left_link",
    "head_camera_right": "camera_right_link",
}
"""
The camera names a trained policy uses, and the MJCF cameras behind them.

Duplicated from `stretch/config.py`'s `HEAD_CAMERA_MJCF_NAME` and
`WRIST_CAMERA_MJCF_NAME` -- and asserted equal to them in
`tests/test_stretch_finetuning.py` -- so that recording a dataset does not drag
the MolmoSpaces dependency into `examples/molmo_environment.py`, which runs
happily on a bare `--scene`.

The head entry is the *centre* camera of Stretch 4's three-camera head, at
1.62m and pitched 35 degrees down. The left and right pair sit 7.5cm either side
and look 47 degrees down; they are the stereo pair, aimed at the workspace, and
are not what the benchmark or these datasets use.
"""


def camera_frames_for(camera_names: list[str]) -> dict[str, object]:
    """Trained camera name -> the `StretchCameras` member that renders it.

    The pairing is derived from the MJCF camera name rather than written out,
    because the two stacks name the same physical camera differently and getting
    it wrong is invisible: a dataset recorded from the *right* gripper camera
    under the name `wrist_camera` looks entirely plausible and trains a policy
    that is quietly 2cm off. Same reasoning as
    `live_policy._camera_for_mjcf_name`, which cannot be reused here without
    pulling in MolmoSpaces.
    """
    from stretch4_mujoco.enums.stretch_cameras import StretchCameras

    frames = {}
    for name in camera_names:
        mjcf_name = TRAINED_CAMERA_MJCF_NAMES[name]
        matches = [
            camera
            for camera in StretchCameras.all_stretch4()
            if camera.camera_name_in_mjcf == mjcf_name
        ]
        if not matches:
            raise ValueError(
                f"No StretchCameras member renders MJCF camera {mjcf_name!r} for {name!r}; "
                f"available: {sorted(c.camera_name_in_mjcf for c in StretchCameras.all_stretch4())}"
            )
        frames[name] = matches[0]
    return frames


def qpos_from_status(status) -> dict[str, list[float]]:
    """A `StatusStretchJoints` -> the per-move-group qpos dict the format stores.

    The group names and widths are MolmoSpaces' (`Stretch4RobotView`), and the
    units line up without conversion: `arm.pos` is the tendon length, i.e. total
    telescoping extension, and the finger positions are raw URDF joint angles.
    Both gripper channels are recorded because the MJCF models the one
    `stretch_gripper` actuator as a mirrored pair -- see
    `lerobot_export.GRIPPER_CHANNEL_NAMES`.
    """
    return {
        "base": [status.base.x, status.base.y, status.base.theta],
        "lift": [status.lift.pos],
        "arm": [status.arm.pos],
        "wrist": [status.wrist_yaw.pos, status.wrist_pitch.pos, status.wrist_roll.pos],
        "gripper": [status.gripper_right_finger.pos, status.gripper_left_finger.pos],
    }


def pose7_from_matrix(pose: np.ndarray) -> np.ndarray:
    """A 4x4 pose -> `[x, y, z, qw, qx, qy, qz]`, MolmoSpaces' pose convention."""
    from scipy.spatial.transform import Rotation as R

    pose = np.asarray(pose, dtype=float)
    quaternion = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
    return np.concatenate([pose[:3, 3], quaternion]).astype(np.float32)


DEFAULT_TASK_DESCRIPTION = "complete the demonstrated task"
"""
What goes in `obs_scene["task_description"]` when the operator gave no `--record-task`.

Deliberately generic rather than clever. The instruction is the model's whole
conditioning signal, so a wrong one is worse than a vague one -- and a dataset
full of this string is a visible reminder to pass `--record-task`.
"""


@dataclass
class RecordedEpisode:
    """One demonstration, accumulating in memory until it is written."""

    task_description: str
    qpos: list[dict] = field(default_factory=list)
    base_pose: list[np.ndarray] = field(default_factory=list)
    tcp_pose: list[np.ndarray] = field(default_factory=list)
    sim_time: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.qpos)


class LiveDatasetRecorder:
    """Writes teleop demonstrations into a MolmoSpaces-format rollout directory.

    The output is laid out exactly as a generated run -- `house_teleop_NNN/`
    holding `trajectories.h5` plus one MP4 per camera per episode -- so
    `arrange_train_val_split()`, `lerobot_export.py` and MolmoBot's trainer all
    read it without knowing where it came from.

    Args:
        output_dir: root to write the rollout directory under.
        camera_names: cameras to record, in the names the *trained* policy will
            use them by (`head_camera`, `wrist_camera`) rather than the
            simulator's MJCF names. `LiveDatasetRecorder` is handed already-named
            frames, so the mapping is the caller's -- see
            `live_policy.CAMERA_FOR_TRAINED_NAME`.
        fps: frame rate to stamp on the MP4s. Must be the rate `record_step()`
            is actually called at, or every video plays at the wrong speed and
            every timestamp in a downstream dataset is wrong.
        task_description: the language instruction for these demonstrations.
    """

    def __init__(
        self,
        output_dir: Path,
        camera_names: list[str],
        fps: float = 15.0,
        task_description: str = DEFAULT_TASK_DESCRIPTION,
        house_name: str = "house_teleop_000",
    ) -> None:
        self.output_dir = Path(output_dir) / house_name
        self.camera_names = list(camera_names)
        self.fps = float(fps)
        self.task_description = task_description or DEFAULT_TASK_DESCRIPTION

        self._episode: RecordedEpisode | None = None
        self._writers: dict[str, object] = {}
        self._frame_counts: dict[str, int] = {}
        self._episode_index = 0
        self._kept: list[int] = []
        self._last_frame_time: float | None = None

    # =========================================================================
    # Episode boundaries
    # =========================================================================

    @property
    def recording(self) -> bool:
        return self._episode is not None

    @property
    def kept_episodes(self) -> int:
        return len(self._kept)

    @property
    def current_length(self) -> int:
        return len(self._episode) if self._episode is not None else 0

    def start_episode(self, task_description: str | None = None) -> None:
        """Begin a demonstration. A second call while recording is ignored."""
        if self._episode is not None:
            log.warning("[record] already recording; ignoring start")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episode = RecordedEpisode(task_description=task_description or self.task_description)
        self._open_writers()
        log.info(
            f"[record] episode {self._episode_index} started "
            f"({self._episode.task_description!r})"
        )

    def finish_episode(self, success: bool = True) -> Path | None:
        """Close the videos, append the trajectory to the HDF5, and keep it.

        Returns the HDF5 path, or None if the episode was too short to be a
        demonstration. Two frames is the floor: the action for a frame is the
        *next* frame's state, so a one-frame episode has no actions in it at all.
        """
        if self._episode is None:
            return None
        episode, self._episode = self._episode, None
        self._close_writers()

        if len(episode) < 2:
            log.warning(
                f"[record] episode {self._episode_index} has {len(episode)} frames; "
                "discarding (an action is the next frame's state, so 2 is the minimum)"
            )
            self._discard_videos()
            return None

        path = self._write_trajectory(episode, success=success)
        self._kept.append(self._episode_index)
        log.info(
            f"[record] episode {self._episode_index} kept: {len(episode) - 1} transitions -> {path}"
        )
        self._episode_index += 1
        return path

    def discard_episode(self) -> None:
        """Throw the current demonstration away, videos included."""
        if self._episode is None:
            return
        length = len(self._episode)
        self._episode = None
        self._close_writers()
        self._discard_videos()
        log.info(f"[record] episode {self._episode_index} discarded ({length} frames)")

    def close(self) -> None:
        """Finish anything in progress and report. Safe to call twice."""
        if self._episode is not None:
            self.finish_episode()
        if self._kept:
            log.info(
                f"[record] {len(self._kept)} episodes written to {self.output_dir}. "
                "Export or train with:\n"
                "  python -m examples.machine_learning.molmospaces.finetuning.finetune "
                f"--rollouts {self.output_dir.parent}"
            )
        else:
            log.info(f"[record] no episodes kept; {self.output_dir} may be empty")

    # =========================================================================
    # Per-step capture
    # =========================================================================

    def record_step(
        self,
        qpos: dict[str, list[float]],
        base_pose7: np.ndarray,
        tcp_pose7: np.ndarray,
        images: dict[str, np.ndarray],
        sim_time: float = 0.0,
        frame_time: float | None = None,
    ) -> None:
        """Append one frame. A no-op unless an episode is open and the frame is new.

        Args:
            qpos: per-move-group joint positions, in MolmoSpaces' convention --
                `{"base": [x, y, theta], "lift": [...], "arm": [...],
                "wrist": [...], "gripper": [...]}`.
            base_pose7: base pose as `[x, y, z, qw, qx, qy, qz]`.
            tcp_pose7: tool centre pose **in the base frame**, same layout. That
                frame is not a choice: `TCPPoseSensor` records it base-relative
                and `lerobot_export.py` composes it with the base pose, so a
                world-frame tool pose here would silently double the base offset.
            images: frames keyed by the camera names this recorder was built
                with. A missing camera skips the whole frame -- a half-recorded
                step would put the cameras out of sync with the state.
            sim_time: simulator time, kept for diagnostics.
            frame_time: the camera frame's own timestamp. Repeats are dropped:
                a render loop spins faster than the cameras produce frames, and
                padding an episode with the same image against advancing joint
                states would be a demonstration of the robot moving while blind.
                Pass None to record unconditionally.
        """
        if self._episode is None:
            return
        if frame_time is not None:
            if frame_time == self._last_frame_time:
                return
            self._last_frame_time = frame_time
        missing = [name for name in self.camera_names if images.get(name) is None]
        if missing:
            log.debug(f"[record] skipping frame; cameras not ready: {missing}")
            return

        self._episode.qpos.append(
            {group: [float(v) for v in values] for group, values in qpos.items()}
        )
        self._episode.base_pose.append(np.asarray(base_pose7, dtype=np.float32).reshape(7))
        self._episode.tcp_pose.append(np.asarray(tcp_pose7, dtype=np.float32).reshape(7))
        self._episode.sim_time.append(float(sim_time))
        for name in self.camera_names:
            self._append_frame(name, images[name])

    # =========================================================================
    # Video
    # =========================================================================

    def _open_writers(self) -> None:
        import cv2

        self._writers = {}
        self._frame_counts = {name: 0 for name in self.camera_names}
        # Writers are created lazily on the first frame, because the frame size
        # is not known until one arrives -- the simulator's camera resolution is
        # a property of the model, not of this class.
        del cv2

    def _append_frame(self, camera: str, frame: np.ndarray) -> None:
        import cv2

        writer = self._writers.get(camera)
        if writer is None:
            height, width = frame.shape[:2]
            path = self.output_dir / video_filename(self._episode_index, camera)
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open {path} for writing")
            self._writers[camera] = writer
        writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        self._frame_counts[camera] = self._frame_counts.get(camera, 0) + 1

    def _close_writers(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers = {}

    def _discard_videos(self) -> None:
        for camera in self.camera_names:
            path = self.output_dir / video_filename(self._episode_index, camera)
            path.unlink(missing_ok=True)

    # =========================================================================
    # HDF5
    # =========================================================================

    def _write_trajectory(self, episode: RecordedEpisode, success: bool) -> Path:
        """Append one trajectory to `trajectories.h5`, in MolmoSpaces' layout."""
        import h5py

        path = self.output_dir / "trajectories.h5"
        # The action for frame t is frame t+1's state, so the last observed frame
        # contributes an observation with nothing to imitate and is dropped.
        length = len(episode) - 1

        with h5py.File(path, "a") as h5_file:
            group = h5_file.create_group(f"traj_{self._episode_index}")

            observations = group.create_group("obs")
            agent = observations.create_group("agent")
            agent.create_dataset(
                "qpos",
                data=np.stack([encode_json_blob(episode.qpos[i]) for i in range(length)]),
                dtype=np.uint8,
            )

            extra = observations.create_group("extra")
            extra.create_dataset("robot_base_pose", data=np.stack(episode.base_pose[:length]))
            extra.create_dataset("tcp_pose", data=np.stack(episode.tcp_pose[:length]))

            sensor_data = observations.create_group("sensor_data")
            for camera in self.camera_names:
                sensor_data.create_dataset(
                    camera,
                    data=encode_video_path(video_filename(self._episode_index, camera)),
                    dtype=np.uint8,
                )

            actions = group.create_group("actions")
            actions.create_dataset(
                "joint_pos",
                data=np.stack([encode_json_blob(episode.qpos[i + 1]) for i in range(length)]),
                dtype=np.uint8,
            )
            actions.create_dataset(
                "joint_pos_rel",
                data=np.stack(
                    [
                        encode_json_blob(_delta(episode.qpos[i + 1], episode.qpos[i]))
                        for i in range(length)
                    ]
                ),
                dtype=np.uint8,
            )

            # A teleoperated demonstration is a demonstration: the operator
            # stopped when it was done. `--record-task` names what was done, and
            # `X` discards the ones that were not.
            successes = np.zeros(length, dtype=bool)
            successes[-1] = success
            group.create_dataset("success", data=successes)
            group.create_dataset("fail", data=~successes)
            terminated = np.zeros(length, dtype=bool)
            terminated[-1] = True
            group.create_dataset("terminated", data=terminated)
            group.create_dataset("truncated", data=np.zeros(length, dtype=bool))
            rewards = np.zeros(length, dtype=np.float32)
            rewards[-1] = 1.0 if success else 0.0
            group.create_dataset("rewards", data=rewards)
            group.create_dataset(
                "obs_scene",
                data=json.dumps(
                    {
                        "task_type": "teleop",
                        "task_description": episode.task_description,
                        "source": "examples/molmo_environment.py --record_dataset",
                        "fps": self.fps,
                        "sim_time_start": episode.sim_time[0] if episode.sim_time else 0.0,
                    }
                ),
            )
        return path


def _delta(later: dict, earlier: dict) -> dict:
    """Per-move-group difference, for `actions/joint_pos_rel`.

    Including the base, whose delta is a *world-frame* displacement here. That
    matches what MolmoSpaces' own `JointRelPosController` adds to the current
    position, and it is the caller's job not to feed a relative base command to
    a model trained on base-frame steps -- `networks.world_base_step_to_local`
    is the conversion, and `lerobot_export.py` applies it for the `stretch`
    action space.
    """
    return {
        group: [float(a) - float(b) for a, b in zip(later[group], earlier[group])]
        for group in later
        if group in earlier
    }
