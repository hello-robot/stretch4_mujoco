"""
Live telemetry for a policy running in the simulator: view it, and keep proof.

`report.py` covers the after-the-fact case -- turning a finished benchmark run
into artifacts. This covers the during case, for `live_policy.py`: stream what
the policy sees and does to a Rerun viewer, and optionally write the same thing
to disk as MP4s plus a CSV.

The two outputs answer different questions. Rerun is for watching: camera feeds
beside time-series of every joint and every command, scrubbable. The CSV and
MP4s are for showing someone afterwards that a specific run did a specific
thing, without them needing this repository.

Recording is deliberately cheap per step -- a CSV row and a frame append -- so
that turning it on does not change how the policy behaves by slowing the control
loop down.
"""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

STATE_COLUMNS = (
    "lift",
    "arm",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
    "gripper_right",
    "gripper_left",
)
"""Names for the seven proprioception numbers, in `networks.STATE_GROUPS` order."""


class LiveTelemetry:
    """Records and/or streams one policy step at a time.

    Args:
        camera_names: the cameras that will be handed to `record()`.
        use_rerun: stream to a Rerun viewer, spawning one on first use.
        output_dir: write `telemetry.csv` and `<camera>.mp4` here. None to skip.
        video_fps: frame rate to stamp on the written MP4s. Should match the
            policy control rate so the video plays back in real time.
    """

    def __init__(
        self,
        camera_names: list[str],
        use_rerun: bool = False,
        output_dir: Path | None = None,
        video_fps: float = 15.0,
    ) -> None:
        self.camera_names = list(camera_names)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.video_fps = video_fps

        self._lock = threading.Lock()
        self._rerun = _RerunStream(self.camera_names) if use_rerun else None
        self._writers: dict[str, "cv2.VideoWriter"] = {}  # noqa: F821 - cv2 imported lazily
        self._csv_file = None
        self._csv_writer: csv.DictWriter | None = None

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        step: int,
        sim_time: float,
        state: np.ndarray,
        base_xytheta: np.ndarray,
        commanded: dict[str, float],
        images: dict[str, np.ndarray],
    ) -> None:
        """One control step. Safe to call from the policy thread."""
        row = {"step": step, "sim_time": round(float(sim_time), 4)}
        row.update(
            {name: round(float(value), 5) for name, value in zip(STATE_COLUMNS, np.ravel(state))}
        )
        row.update(
            {
                "base_x": round(float(base_xytheta[0]), 5),
                "base_y": round(float(base_xytheta[1]), 5),
                "base_theta": round(float(base_xytheta[2]), 5),
            }
        )
        row.update({f"cmd_{name}": round(float(value), 5) for name, value in commanded.items()})

        with self._lock:
            if self.output_dir is not None:
                self._write_row(row)
                self._write_frames(images)
        if self._rerun is not None:
            self._rerun.log(step, row, images)

    def close(self) -> None:
        with self._lock:
            for writer in self._writers.values():
                writer.release()
            self._writers.clear()
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
        if self.output_dir is not None:
            log.info(f"[telemetry] wrote {self.output_dir}")

    # =========================================================================
    # Disk
    # =========================================================================

    def _write_row(self, row: dict) -> None:
        if self._csv_writer is None:
            # The column set is only known once the first row exists, since it
            # depends on which actuators the policy actually commands.
            self._csv_file = (self.output_dir / "telemetry.csv").open("w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)

    def _write_frames(self, images: dict[str, np.ndarray]) -> None:
        import cv2

        for name, pixels in images.items():
            frame = np.ascontiguousarray(pixels)
            if name not in self._writers:
                height, width = frame.shape[:2]
                self._writers[name] = cv2.VideoWriter(
                    str(self.output_dir / f"{name}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.video_fps,
                    (width, height),
                )
            self._writers[name].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


class _RerunStream:
    """Rerun logging for a policy rollout: camera feeds plus every scalar.

    Kept separate from `examples/rerun_utils.py`'s `RerunLogger`, which is built
    around `StatusStretchCameras` objects and a lidar-and-cameras layout. This
    one takes plain arrays and lays out cameras beside the policy's own
    time-series, which is what you want when the question is "what is the policy
    doing", not "what does the robot see".
    """

    def __init__(self, camera_names: list[str]) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        rr.init("Stretch4 policy", spawn=True)
        rr.send_blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    *[
                        rrb.Spatial2DView(origin=f"policy/cameras/{name}", name=name)
                        for name in camera_names
                    ],
                    name="What the policy sees",
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(origin="policy/state", name="Joint positions"),
                    rrb.TimeSeriesView(origin="policy/command", name="Commanded"),
                    name="Telemetry",
                ),
                column_shares=[1, 1],
            )
        )

    def log(self, step: int, row: dict, images: dict[str, np.ndarray]) -> None:
        rr = self._rr
        # Index on the policy step rather than wall clock so the timeline lines
        # up with the CSV, which is indexed the same way.
        rr.set_time("policy_step", sequence=step)

        for name, pixels in images.items():
            rr.log(f"policy/cameras/{name}", rr.Image(pixels))
        for key, value in row.items():
            if key in ("step", "sim_time"):
                continue
            branch = "command" if key.startswith("cmd_") else "state"
            rr.log(f"policy/{branch}/{key.removeprefix('cmd_')}", rr.Scalars(value))
