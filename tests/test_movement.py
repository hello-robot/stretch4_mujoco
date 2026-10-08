"""Movement test suite for Stretch 4 in MuJoCo.

Every move the suite makes is also recorded and plotted, so the trapezoidal
profiles the joints now run can be eyeballed rather than only asserted on:
`tests/motion_profile_plots/<variant>/` gets one figure per tested move
(position, velocity and acceleration against sim time) plus a whole-run
overview.
"""

import re
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-safe: these figures are only ever written to disk

import matplotlib.pyplot as plt  # noqa: E402

from stretch4_mujoco import config  # noqa: E402
from stretch4_mujoco.datamodels.status_stretch_joints import StatusStretchJoints  # noqa: E402
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator  # noqa: E402

PLOT_ROOT = Path(__file__).parent / "motion_profile_plots"

SAMPLE_PERIOD_S = 0.005
"""Poll faster than the 100 Hz control loop publishes; repeats are dropped."""

SMOOTHING_SAMPLES = 5
"""Width of the moving average applied before differentiating (~50 ms)."""


@dataclass(frozen=True)
class _Channel:
    """One plottable degree of freedom, and where to read it off the status.

    `read_vel` is left `None` where the status has no velocity in the same frame
    as the position it reports -- the base pose and the gripper aperture -- and
    the velocity is differentiated from the position instead, so each figure's
    three panels are always in one consistent set of units.
    """

    label: str
    unit: str
    read_pos: Callable[[StatusStretchJoints], float]
    read_vel: Callable[[StatusStretchJoints], float] | None = None
    limits: tuple[float, float] | None = None
    unwrap: bool = False


_BASE_XY_LIMITS, _BASE_THETA_LIMITS = config.get_base_motion_limits()

CHANNELS: dict[str, _Channel] = {
    "base_x": _Channel("Base x", "m", lambda s: s.base.x, limits=_BASE_XY_LIMITS),
    "base_y": _Channel("Base y", "m", lambda s: s.base.y, limits=_BASE_XY_LIMITS),
    "base_theta": _Channel(
        "Base yaw",
        "rad",
        lambda s: s.base.theta,
        lambda s: s.base.theta_vel,
        _BASE_THETA_LIMITS,
        unwrap=True,
    ),
    "arm": _Channel(
        "Arm", "m", lambda s: s.arm.pos, lambda s: s.arm.vel, config.get_actuator_motion_limits("arm")
    ),
    "lift": _Channel(
        "Lift", "m", lambda s: s.lift.pos, lambda s: s.lift.vel, config.get_actuator_motion_limits("lift")
    ),
    "wrist_yaw": _Channel(
        "Wrist yaw",
        "rad",
        lambda s: s.wrist_yaw.pos,
        lambda s: s.wrist_yaw.vel,
        config.get_actuator_motion_limits("wrist_yaw"),
    ),
    "wrist_pitch": _Channel(
        "Wrist pitch",
        "rad",
        lambda s: s.wrist_pitch.pos,
        lambda s: s.wrist_pitch.vel,
        config.get_actuator_motion_limits("wrist_pitch"),
    ),
    "wrist_roll": _Channel(
        "Wrist roll",
        "rad",
        lambda s: s.wrist_roll.pos,
        lambda s: s.wrist_roll.vel,
        config.get_actuator_motion_limits("wrist_roll"),
    ),
    "gripper": _Channel(
        "Gripper aperture",
        "rad",
        lambda s: s.gripper.pos,
        None,
        config.get_actuator_motion_limits("gripper"),
    ),
}


@dataclass
class _Segment:
    """A tested move: which channels it exercises, and when it ran in sim time."""

    name: str
    channels: tuple[str, ...]
    t_start: float
    t_end: float = 0.0


def _slug(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


def _smooth(values: np.ndarray, window: int = SMOOTHING_SAMPLES) -> np.ndarray:
    """Moving average that keeps the array length and does not droop at the ends.

    Differentiating the sampled velocity twice over would otherwise bury the
    acceleration plateau in quantisation noise.
    """
    if window < 2 or values.size < window:
        return values
    pad = window // 2
    kernel = np.ones(window) / window
    return np.convolve(np.pad(values, pad, mode="edge"), kernel, mode="valid")[: values.size]


class MotionProfileRecorder:
    """Samples the simulator's joint state so each move can be plotted afterwards.

    The profiles are what make a move take the time it does, so the picture worth
    saving is position/velocity/acceleration against *sim* time, taken from the
    same status the assertions below read. A background thread polls faster than
    the control loop publishes and drops repeated timestamps, which gives one
    sample per control step without assuming the sim runs in real time.

    Recording is independent of plotting: a recorder that was never started still
    accepts `segment()` calls and simply has nothing to draw.
    """

    def __init__(self, sim, plot_dir: Path | str, sample_period_s: float = SAMPLE_PERIOD_S):
        self._sim = sim
        self.plot_dir = Path(plot_dir)
        self._sample_period_s = sample_period_s

        self._times: list[float] = []
        self._pos: dict[str, list[float]] = {key: [] for key in CHANNELS}
        self._vel: dict[str, list[float]] = {key: [] for key in CHANNELS}
        self._segments: list[_Segment] = []

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- recording ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _sample_loop(self) -> None:
        last_time: float | None = None
        while not self._stop_event.is_set():
            try:
                status = self._sim.pull_status()
            except Exception:  # the sim is shutting down underneath us
                return
            sim_time = float(status.time)
            if last_time is None or sim_time > last_time:
                last_time = sim_time
                with self._lock:
                    self._times.append(sim_time)
                    for key, channel in CHANNELS.items():
                        self._pos[key].append(float(channel.read_pos(status)))
                        self._vel[key].append(
                            float(channel.read_vel(status)) if channel.read_vel else np.nan
                        )
            self._stop_event.wait(self._sample_period_s)

    def _now(self) -> float:
        with self._lock:
            if self._times:
                return self._times[-1]
        try:
            return float(self._sim.pull_status().time)
        except Exception:
            return 0.0

    @contextmanager
    def segment(self, name: str, *channels: str):
        """Tags the sim-time window of one tested move, for its own figure.

        Exits via `finally` so a failed assertion still leaves a plotted segment
        -- a move that missed its target is exactly the one worth looking at.
        """
        segment = _Segment(name=name, channels=channels, t_start=self._now())
        self._segments.append(segment)
        try:
            yield segment
        finally:
            segment.t_end = self._now()

    # -- plotting -----------------------------------------------------------

    def save_plots(self) -> list[Path]:
        """Writes one figure per recorded segment, plus a whole-run overview."""
        with self._lock:
            times = np.asarray(self._times, dtype=float)
            positions = {key: np.asarray(values, dtype=float) for key, values in self._pos.items()}
            velocities = {key: np.asarray(values, dtype=float) for key, values in self._vel.items()}

        if times.size < 2:
            click.secho("No motion samples recorded; skipping motion profile plots.", fg="red")
            return []

        series = {
            key: self._series(key, times, positions[key], velocities[key]) for key in CHANNELS
        }

        self.plot_dir.mkdir(parents=True, exist_ok=True)
        written = [
            self._plot_segment(index, segment, times, series)
            for index, segment in enumerate(self._segments, start=1)
        ]
        overview = self._plot_overview(times, series)
        if overview is not None:
            written.append(overview)

        click.secho(f"\nSaved {len(written)} motion profile plots to {self.plot_dir}", fg="cyan")
        return written

    def _series(
        self, key: str, times: np.ndarray, position: np.ndarray, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position, velocity and acceleration for one channel over the whole run.

        Derivatives are taken across the full run rather than per segment, so a
        segment's plot has no artificial edge at its own boundaries.
        """
        channel = CHANNELS[key]
        if channel.unwrap:
            position = np.unwrap(position)
        if channel.read_vel is None:
            velocity = _smooth(np.gradient(position, times))
        acceleration = _smooth(np.gradient(_smooth(velocity), times))
        return position, velocity, acceleration

    def _plot_segment(
        self,
        index: int,
        segment: _Segment,
        times: np.ndarray,
        series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> Path:
        # Pad either side so the joint is visibly at rest before and after, but
        # never into a neighbouring segment -- the next move's ramp showing up on
        # this move's figure is the one thing that misreads here.
        pad_s = 0.3
        previous = self._segments[index - 2] if index >= 2 else None
        following = self._segments[index] if index < len(self._segments) else None
        window_start = segment.t_start - pad_s
        window_end = segment.t_end + pad_s
        if previous is not None:
            window_start = max(window_start, previous.t_end)
        if following is not None:
            window_end = min(window_end, following.t_start)

        mask = (times >= window_start) & (times <= window_end)
        t_rel = times[mask] - segment.t_start

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8.5))
        for key in segment.channels:
            channel = CHANNELS[key]
            position, velocity, acceleration = series[key]
            for ax, values in zip(axes, (position[mask], velocity[mask], acceleration[mask])):
                ax.plot(t_rel, values, lw=1.4, label=channel.label)

        unit = CHANNELS[segment.channels[0]].unit
        axes[0].set_ylabel(f"position [{unit}]")
        axes[1].set_ylabel(f"velocity [{unit}/s]")
        axes[2].set_ylabel(f"acceleration [{unit}/s²]")
        axes[2].set_xlabel("sim time since command [s]")

        # Only meaningful against a single channel: on a diagonal base move the
        # limit applies to the resultant, not to x and y separately.
        limits = CHANNELS[segment.channels[0]].limits if len(segment.channels) == 1 else None
        if limits is not None:
            max_vel, max_accel = limits
            for ax, bound, name in ((axes[1], max_vel, "vel"), (axes[2], max_accel, "accel")):
                ax.axhline(bound, color="firebrick", ls="--", lw=0.9, label=f"max {name} ±{bound:.3g}")
                ax.axhline(-bound, color="firebrick", ls="--", lw=0.9)

        for position_in_fig, ax in enumerate(axes):
            ax.axvspan(
                0.0,
                max(segment.t_end - segment.t_start, 0.0),
                color="0.93",
                zorder=0,
                label="command window" if position_in_fig == 0 else None,
            )
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        fig.suptitle(f"{index:02d}. {segment.name}")
        fig.tight_layout()
        path = self.plot_dir / f"{index:02d}_{_slug(segment.name)}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path

    def _plot_overview(
        self, times: np.ndarray, series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> Path | None:
        moved = [key for key in CHANNELS if np.ptp(series[key][0]) > 1e-4]
        if not moved:
            return None

        fig, axes = plt.subplots(
            len(moved), 1, sharex=True, figsize=(11, 1.7 * len(moved) + 1), squeeze=False
        )
        for ax, key in zip(axes[:, 0], moved):
            channel = CHANNELS[key]
            ax.plot(times, series[key][1], lw=1.0)
            ax.set_ylabel(f"{channel.label}\n[{channel.unit}/s]", fontsize=8)
            ax.grid(alpha=0.3)
            for segment in self._segments:
                ax.axvline(segment.t_start, color="0.8", lw=0.8, zorder=0)

        for index, segment in enumerate(self._segments, start=1):
            axes[0, 0].annotate(
                str(index),
                xy=(segment.t_start, 1.03),
                xycoords=("data", "axes fraction"),
                fontsize=7,
                ha="center",
            )
        axes[-1, 0].set_xlabel("sim time [s]")
        fig.suptitle("Joint velocities over the whole run (numbers mark each tested move)")
        fig.tight_layout()
        path = self.plot_dir / "00_overview.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path


def wait_for_sim_time(sim, duration_s):
    t_start = sim.pull_status().time
    while sim.pull_status().time - t_start < duration_s:
        time.sleep(0.01)


def get_perf_str(sim):
    """Returns formatted sim FPS and real-time-ratio string."""
    status = sim.pull_status()
    fps = status.fps
    rtr = status.sim_to_real_time_ratio_msg or "1.0x"
    return f"FPS: {fps:.1f}, RTR: {rtr}"


def run_movement_tests(sim, test_cameras=False, recorder: MotionProfileRecorder | None = None):
    """Runs the core movement test suite on an active simulator instance."""
    # An unstarted recorder records the segments and has nothing to plot, so the
    # suite reads the same whether or not the caller wants figures out of it.
    recorder = recorder or MotionProfileRecorder(sim, PLOT_ROOT / "unused")

    # ==========================================
    # 1. Base Tests
    # ==========================================
    click.secho("\n--- Testing Base Movement ---", fg="yellow")

    # 1a. Base translate_by
    with recorder.segment("Base translate_by(0.20 m)", "base_x", "base_y"):
        start_x = sim.base.status.x
        sim.base.translate_by(0.20)
        time.sleep(0.3)
        assert sim.wait_command(timeout=5.0, position_tolerance=0.001)
        end_x = sim.base.status.x
        disp_x = end_x - start_x
        click.secho(
            f"Base translate_by(0.20): start={start_x:.4f}, end={end_x:.4f}, disp={disp_x:.4f}m [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_x, 0.20, atol=0.04), f"Base translate_by failed: disp={disp_x:.4f}m"

    # 1b. Base rotate_by
    with recorder.segment("Base rotate_by(90 deg)", "base_theta"):
        start_theta = sim.base.status.theta
        rotate_by = np.radians(90)
        sim.base.rotate_by(rotate_by)
        assert sim.wait_command(timeout=5.0, position_tolerance=0.001)
        end_theta = sim.base.status.theta
        disp_theta = end_theta - start_theta
        click.secho(
            f"Base rotate_by({rotate_by:.4f}): start={start_theta:.4f}, end={end_theta:.4f}, disp={disp_theta:.4f}rad [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_theta, rotate_by, atol=0.08), f"Base rotate_by failed: disp={disp_theta:.4f}rad"

    # ==========================================
    # Omnibase Velocity & Error/Bias Metrics
    # ==========================================
    click.secho("\n--- Testing Omnibase Velocity & Error Metrics ---", fg="yellow")

    # Helper for converting world delta to local frame relative to initial heading
    def world_to_local(dx_world, dy_world, th0):
        cos_th = np.cos(th0)
        sin_th = np.sin(th0)
        disp_x_local = dx_world * cos_th + dy_world * sin_th
        disp_y_local = -dx_world * sin_th + dy_world * cos_th
        return disp_x_local, disp_y_local

    # 1c. Base set_velocity X (Forward 0.20 m/s for 4.0s -> Expected 0.80m)
    with recorder.segment("Base set_velocity vx=0.20 m/s for 4 s", "base_x", "base_y"):
        start_x = sim.base.status.x
        start_y = sim.base.status.y
        start_th = sim.base.status.theta
        sim.base.set_velocity(vx_m=0.20, vy_m=0.0, w_r=0.0)
        wait_for_sim_time(sim, 4.0)
        sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.0)
        time.sleep(0.5)
        end_x = sim.base.status.x
        end_y = sim.base.status.y
        end_th = sim.base.status.theta

        disp_x, disp_y = world_to_local(end_x - start_x, end_y - start_y, start_th)
        expected_x = 0.80
        err_x = abs(disp_x - expected_x)
        pct_err_x = (err_x / expected_x) * 100.0
        drift_y = abs(disp_y)
        drift_th = abs(end_th - start_th)
        click.secho(
            f"Omnibase Vel X (0.20 m/s x 4s): disp_x={disp_x:.4f}m (exp {expected_x:.2f}m), "
            f"err={err_x:.4f}m ({pct_err_x:.2f}%), drift_y={drift_y:.4f}m, drift_th={drift_th:.4f}rad [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_x, expected_x, atol=0.05), f"Base X velocity failed: disp_x={disp_x:.4f}m"
        assert drift_y < 0.03, f"Base X velocity transverse drift Y too high: drift_y={drift_y:.4f}m"
        assert drift_th < 0.05, f"Base X velocity heading drift too high: drift_th={drift_th:.4f}rad"

    # 1d. Base set_velocity Y (Lateral 0.20 m/s for 3.0s -> Expected 0.60m)
    with recorder.segment("Base set_velocity vy=0.20 m/s for 3 s", "base_x", "base_y"):
        start_x = sim.base.status.x
        start_y = sim.base.status.y
        start_th = sim.base.status.theta
        sim.base.set_velocity(vx_m=0.0, vy_m=0.20, w_r=0.0)
        wait_for_sim_time(sim, 3.0)
        sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.0)
        time.sleep(0.5)
        end_x = sim.base.status.x
        end_y = sim.base.status.y
        end_th = sim.base.status.theta

        disp_x, disp_y = world_to_local(end_x - start_x, end_y - start_y, start_th)
        expected_y = 0.60
        err_y = abs(disp_y - expected_y)
        pct_err_y = (err_y / expected_y) * 100.0
        drift_x = abs(disp_x)
        drift_th = abs(end_th - start_th)
        click.secho(
            f"Omnibase Vel Y (0.20 m/s x 3s): disp_y={disp_y:.4f}m (exp {expected_y:.2f}m), "
            f"err={err_y:.4f}m ({pct_err_y:.2f}%), drift_x={drift_x:.4f}m, drift_th={drift_th:.4f}rad [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_y, expected_y, atol=0.05), f"Base Y velocity failed: disp_y={disp_y:.4f}m"
        assert drift_x < 0.03, f"Base Y velocity transverse drift X too high: drift_x={drift_x:.4f}m"
        assert drift_th < 0.05, f"Base Y velocity heading drift too high: drift_th={drift_th:.4f}rad"

    # 1e. Base set_velocity Omnidirectional (vx=0.15, vy=0.15 m/s for 3.0s -> Expected dx=0.45m, dy=0.45m, dist=0.6364m)
    with recorder.segment("Base set_velocity vx=vy=0.15 m/s for 3 s", "base_x", "base_y"):
        start_x = sim.base.status.x
        start_y = sim.base.status.y
        start_th = sim.base.status.theta
        sim.base.set_velocity(vx_m=0.15, vy_m=0.15, w_r=0.0)
        wait_for_sim_time(sim, 3.0)
        sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.0)
        time.sleep(0.5)
        end_x = sim.base.status.x
        end_y = sim.base.status.y
        end_th = sim.base.status.theta

        dx, dy = world_to_local(end_x - start_x, end_y - start_y, start_th)
        disp_dist = np.hypot(dx, dy)
        expected_dist = float(np.hypot(0.45, 0.45))
        err_dist = abs(disp_dist - expected_dist)
        pct_err_dist = (err_dist / expected_dist) * 100.0
        drift_th = abs(end_th - start_th)
        click.secho(
            f"Omnibase Vel Omni (0.15, 0.15 m/s x 3s): disp_dist={disp_dist:.4f}m (exp {expected_dist:.4f}m, dx={dx:.4f}m, dy={dy:.4f}m), "
            f"err={err_dist:.4f}m ({pct_err_dist:.2f}%), drift_th={drift_th:.4f}rad [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(dx, 0.45, atol=0.05), f"Base Omni velocity dx failed: dx={dx:.4f}m"
        assert np.isclose(dy, 0.45, atol=0.05), f"Base Omni velocity dy failed: dy={dy:.4f}m"
        assert np.isclose(disp_dist, expected_dist, atol=0.05), f"Base Omni velocity total dist failed: disp_dist={disp_dist:.4f}m"
        assert drift_th < 0.05, f"Base Omni velocity heading drift too high: drift_th={drift_th:.4f}rad"

    # 1f. Base set_velocity Angular (w_r=0.50 rad/s for 3.0s -> Expected ~1.32-1.50 rad with accel ramp)
    with recorder.segment("Base set_velocity w=0.50 rad/s for 3 s", "base_theta"):
        start_x = sim.base.status.x
        start_y = sim.base.status.y
        start_th = sim.base.status.theta
        sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.50)
        wait_for_sim_time(sim, 3.0)
        sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.0)
        time.sleep(0.5)
        end_x = sim.base.status.x
        end_y = sim.base.status.y
        end_th = sim.base.status.theta
        disp_th = end_th - start_th
        expected_th = 1.50
        err_th = abs(disp_th - expected_th)
        pct_err_th = (err_th / expected_th) * 100.0
        drift_pos = np.hypot(end_x - start_x, end_y - start_y)
        click.secho(
            f"Omnibase Vel Angular (0.50 rad/s x 3s): disp_th={disp_th:.4f}rad (exp {expected_th:.2f}rad), "
            f"err={err_th:.4f}rad ({pct_err_th:.2f}%), drift_pos={drift_pos:.4f}m [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_th, expected_th, atol=0.22), f"Base Angular velocity failed: disp_th={disp_th:.4f}rad"
        assert drift_pos < 0.04, f"Base Angular velocity position drift too high: drift_pos={drift_pos:.4f}m"

    # ==========================================
    # 2. Arm Tests
    # ==========================================
    click.secho("\n--- Testing Arm Movement ---", fg="yellow")

    # 2a. Arm move_by
    with recorder.segment("Arm move_by(0.10 m)", "arm"):
        start_arm = sim.arm.status.pos
        sim.arm.move_by(0.10)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_arm = sim.arm.status.pos
        disp_arm = end_arm - start_arm
        click.secho(
            f"Arm move_by(0.10): start={start_arm:.4f}, end={end_arm:.4f}, disp={disp_arm:.4f}m [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_arm, 0.10, atol=0.03), f"Arm move_by failed: disp={disp_arm:.4f}m"

    # 2b. Arm set_velocity
    with recorder.segment("Arm set_velocity(0.08 m/s for 2 s)", "arm"):
        start_arm = sim.arm.status.pos
        v_arm = 0.08
        sim.arm.set_velocity(v_arm)
        wait_for_sim_time(sim, 2.0)
        sim.arm.set_velocity(0.0)
        time.sleep(0.5)
        end_arm = sim.arm.status.pos
        disp_arm = end_arm - start_arm
        click.secho(
            f"Arm set_velocity(0.08 m/s for 2s): start={start_arm:.4f}, end={end_arm:.4f}, disp={disp_arm:.4f}m (Expected ~0.16m) [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_arm, 0.16, atol=0.03), f"Arm set_velocity failed: disp={disp_arm:.4f}m"

    # ==========================================
    # 3. Lift Tests
    # ==========================================
    click.secho("\n--- Testing Lift Movement ---", fg="yellow")

    # 3a. Lift move_by
    with recorder.segment("Lift move_by(0.10 m)", "lift"):
        start_lift = sim.lift.status.pos
        sim.lift.move_by(0.10)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_lift = sim.lift.status.pos
        disp_lift = end_lift - start_lift
        click.secho(
            f"Lift move_by(0.10): start={start_lift:.4f}, end={end_lift:.4f}, disp={disp_lift:.4f}m [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_lift, 0.10, atol=0.03), f"Lift move_by failed: disp={disp_lift:.4f}m"

    # 3b. Lift set_velocity
    with recorder.segment("Lift set_velocity(0.08 m/s for 1.5 s)", "lift"):
        start_lift = sim.lift.status.pos
        v_lift = 0.08
        sim.lift.set_velocity(v_lift)
        wait_for_sim_time(sim, 1.5)
        sim.lift.set_velocity(0.0)
        time.sleep(0.5)
        end_lift = sim.lift.status.pos
        disp_lift = end_lift - start_lift
        click.secho(
            f"Lift set_velocity(0.08 m/s for 1.5s): start={start_lift:.4f}, end={end_lift:.4f}, disp={disp_lift:.4f}m (Expected ~0.12m) [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_lift, 0.12, atol=0.03), f"Lift set_velocity failed: disp={disp_lift:.4f}m"

    # ==========================================
    # 4. Wrist Yaw Tests
    # ==========================================
    click.secho("\n--- Testing Wrist Yaw Movement ---", fg="yellow")

    # 4a. Wrist Yaw move_by
    with recorder.segment("Wrist yaw move_by(0.20 rad)", "wrist_yaw"):
        start_yaw = sim.end_of_arm.wrist_yaw.status.pos
        sim.end_of_arm.wrist_yaw.move_by(0.20)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_yaw = sim.end_of_arm.wrist_yaw.status.pos
        disp_yaw = end_yaw - start_yaw
        click.secho(
            f"Wrist Yaw move_by(0.20): start={start_yaw:.4f}, end={end_yaw:.4f}, disp={disp_yaw:.4f}rad [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_yaw, 0.20, atol=0.04), f"Wrist Yaw move_by failed: disp={disp_yaw:.4f}rad"

    # 4b. Wrist Yaw set_velocity
    with recorder.segment("Wrist yaw set_velocity(-0.20 rad/s for 1 s)", "wrist_yaw"):
        start_yaw = sim.end_of_arm.wrist_yaw.status.pos
        v_yaw = -0.20
        sim.end_of_arm.wrist_yaw.set_velocity(v_yaw)
        wait_for_sim_time(sim, 1.0)
        sim.end_of_arm.wrist_yaw.set_velocity(0.0)
        time.sleep(0.5)
        end_yaw = sim.end_of_arm.wrist_yaw.status.pos
        disp_yaw = end_yaw - start_yaw
        click.secho(
            f"Wrist Yaw set_velocity(-0.20 rad/s for 1.0s): start={start_yaw:.4f}, end={end_yaw:.4f}, disp={disp_yaw:.4f}rad (Expected ~-0.20rad) [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_yaw, -0.20, atol=0.04), f"Wrist Yaw set_velocity failed: disp={disp_yaw:.4f}rad"

    # ==========================================
    # 5. Wrist Pitch Tests
    # ==========================================
    click.secho("\n--- Testing Wrist Pitch Movement ---", fg="yellow")

    # 5a. Wrist Pitch move_by
    with recorder.segment("Wrist pitch move_by(0.20 rad)", "wrist_pitch"):
        start_pitch = sim.end_of_arm.wrist_pitch.status.pos
        sim.end_of_arm.wrist_pitch.move_by(0.20)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_pitch = sim.end_of_arm.wrist_pitch.status.pos
        disp_pitch = end_pitch - start_pitch
        click.secho(
            f"Wrist Pitch move_by(0.20): start={start_pitch:.4f}, end={end_pitch:.4f}, disp={disp_pitch:.4f}rad [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_pitch, 0.20, atol=0.04), f"Wrist Pitch move_by failed: disp={disp_pitch:.4f}rad"

    # 5b. Wrist Pitch set_velocity
    with recorder.segment("Wrist pitch set_velocity(-0.20 rad/s for 1 s)", "wrist_pitch"):
        start_pitch = sim.end_of_arm.wrist_pitch.status.pos
        v_pitch = -0.20
        sim.end_of_arm.wrist_pitch.set_velocity(v_pitch)
        wait_for_sim_time(sim, 1.0)
        sim.end_of_arm.wrist_pitch.set_velocity(0.0)
        time.sleep(0.5)
        end_pitch = sim.end_of_arm.wrist_pitch.status.pos
        disp_pitch = end_pitch - start_pitch
        click.secho(
            f"Wrist Pitch set_velocity(-0.20 rad/s for 1.0s): start={start_pitch:.4f}, end={end_pitch:.4f}, disp={disp_pitch:.4f}rad (Expected ~-0.20rad) [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_pitch, -0.20, atol=0.04), f"Wrist Pitch set_velocity failed: disp={disp_pitch:.4f}rad"

    # ==========================================
    # 6. Wrist Roll Tests
    # ==========================================
    click.secho("\n--- Testing Wrist Roll Movement ---", fg="yellow")

    # 6a. Wrist Roll move_by
    with recorder.segment("Wrist roll move_by(0.20 rad)", "wrist_roll"):
        start_roll = sim.end_of_arm.wrist_roll.status.pos
        sim.end_of_arm.wrist_roll.move_by(0.20)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_roll = sim.end_of_arm.wrist_roll.status.pos
        disp_roll = end_roll - start_roll
        click.secho(
            f"Wrist Roll move_by(0.20): start={start_roll:.4f}, end={end_roll:.4f}, disp={disp_roll:.4f}rad [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_roll, 0.20, atol=0.04), f"Wrist Roll move_by failed: disp={disp_roll:.4f}rad"

    # 6b. Wrist Roll set_velocity
    with recorder.segment("Wrist roll set_velocity(0.20 rad/s for 1 s)", "wrist_roll"):
        start_roll = sim.end_of_arm.wrist_roll.status.pos
        v_roll = 0.20
        sim.end_of_arm.wrist_roll.set_velocity(v_roll)
        wait_for_sim_time(sim, 1.0)
        sim.end_of_arm.wrist_roll.set_velocity(0.0)
        time.sleep(0.5)
        end_roll = sim.end_of_arm.wrist_roll.status.pos
        disp_roll = end_roll - start_roll
        click.secho(
            f"Wrist Roll set_velocity(0.20 rad/s for 1.0s): start={start_roll:.4f}, end={end_roll:.4f}, disp={disp_roll:.4f}rad (Expected ~0.20rad) [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_roll, 0.20, atol=0.04), f"Wrist Roll set_velocity failed: disp={disp_roll:.4f}rad"

    # ==========================================
    # 7. Gripper Tests
    # ==========================================
    click.secho("\n--- Testing Gripper Movement ---", fg="yellow")

    # 7a. Gripper move_by (open)
    with recorder.segment("Gripper move_by(0.10)", "gripper"):
        start_grip = sim.end_of_arm.stretch_gripper.status.pos
        sim.end_of_arm.stretch_gripper.move_by(0.10)
        time.sleep(0.3)
        sim.wait_command(timeout=5.0, position_tolerance=0.0001)
        end_grip = sim.end_of_arm.stretch_gripper.status.pos
        disp_grip = end_grip - start_grip
        click.secho(
            f"Gripper move_by(0.10): start={start_grip:.4f}, end={end_grip:.4f}, disp={disp_grip:.4f} [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_grip, 0.10, atol=0.03), f"Gripper move_by failed: disp={disp_grip:.4f}"

    # 7b. Gripper set_velocity (close)
    with recorder.segment("Gripper set_velocity(-0.10 for 1 s)", "gripper"):
        start_grip = sim.end_of_arm.stretch_gripper.status.pos
        sim.end_of_arm.stretch_gripper.set_velocity(-0.10)
        wait_for_sim_time(sim, 1.0)
        sim.end_of_arm.stretch_gripper.set_velocity(0.0)
        time.sleep(0.5)
        end_grip = sim.end_of_arm.stretch_gripper.status.pos
        disp_grip = end_grip - start_grip
        click.secho(
            f"Gripper set_velocity(-0.10 for 1.0s): start={start_grip:.4f}, end={end_grip:.4f}, disp={disp_grip:.4f} [{get_perf_str(sim)}]",
            fg="green",
        )
        assert np.isclose(disp_grip, -0.10, atol=0.03), f"Gripper set_velocity failed: disp={disp_grip:.4f}"

    # Optional camera check
    if test_cameras:
        click.secho("\n--- Verifying Camera Feeds ---", fg="yellow")
        cam_data = sim.pull_camera_data()
        assert cam_data is not None, "Failed to pull camera data from active simulator."
        click.secho(f"Camera feed active at sim_time={cam_data.time:.2f}s [{get_perf_str(sim)}]", fg="green")

    click.secho("\nTESTS PASSED!", fg="green", bold=True)


def run_suite_with_plots(sim, variant: str, headless: bool = True, test_cameras: bool = False):
    """Runs the suite against `sim`, writing this variant's plots either way.

    The plots are saved from a `finally`, so a suite that fails part-way still
    leaves the profiles recorded up to the failure.
    """
    sim.start(headless=headless)
    time.sleep(1.0)
    recorder = MotionProfileRecorder(sim, PLOT_ROOT / variant)
    recorder.start()
    try:
        run_movement_tests(sim, test_cameras=test_cameras, recorder=recorder)
    finally:
        recorder.stop()
        recorder.save_plots()
        sim.stop()


def test_joint_and_base_movement():
    """Headless movement test suite."""
    click.secho("\n=== Starting Headless Movement Test Suite ===", fg="cyan", bold=True)
    run_suite_with_plots(Stretch4MujocoSimulator(), variant="headless", headless=True)


def test_joint_and_base_movement_with_viewer():
    """Movement test suite with passive viewer enabled."""
    click.secho("\n=== Starting Movement Test Suite (With Passive Viewer) ===", fg="cyan", bold=True)
    run_suite_with_plots(Stretch4MujocoSimulator(), variant="viewer", headless=False)


def test_joint_and_base_movement_with_cameras():
    """Movement test suite with all Stretch 4 cameras running."""
    click.secho("\n=== Starting Movement Test Suite (With All Cameras Enabled) ===", fg="cyan", bold=True)
    all_cameras = Stretch4MujocoSimulator.get_all_cameras()
    run_suite_with_plots(
        Stretch4MujocoSimulator(cameras_to_use=all_cameras),
        variant="cameras",
        headless=True,
        test_cameras=True,
    )


if __name__ == "__main__":
    click.secho("===================================================", fg="blue", bold=True)
    click.secho("RUNNING STRETCH 4 MOVEMENT TEST SUITE VARIATIONS", fg="blue", bold=True)
    click.secho("===================================================", fg="blue", bold=True)

    click.secho("\n[1/3] Headless Test Routine", fg="cyan", bold=True)
    test_joint_and_base_movement()

    click.secho("\n[2/3] Viewer Test Routine", fg="cyan", bold=True)
    test_joint_and_base_movement_with_viewer()

    click.secho("\n[3/3] All Cameras Test Routine", fg="cyan", bold=True)
    test_joint_and_base_movement_with_cameras()

    click.secho("\n===================================================", fg="green", bold=True)
    click.secho("ALL 3 MOVEMENT TEST SUITES PASSED SUCCESSFULLY!", fg="green", bold=True)
    click.secho("===================================================", fg="green", bold=True)
