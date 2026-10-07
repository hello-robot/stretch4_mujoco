"""Tests for `SafeMotionOvertiltAvoid`.

These drive the server in-process so the base can be tilted directly, rather
than waiting for the physics to tip a robot over.
"""

import math
import multiprocessing
import threading

import mujoco
import numpy as np
import pytest

from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.mujoco_server import MujocoServer, MujocoServerProxies
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
from stretch4_mujoco.utils import (
    gravity_tilt_from_quaternion,
    gravity_tilt_from_z_axis,
)

WHEELS = ("left_wheel_vel", "right_wheel_vel", "back_wheel_vel")


def roll_quaternion(degrees: float) -> np.ndarray:
    """A `(w, x, y, z)` rotation of `degrees` about +X."""
    half = math.radians(degrees) / 2
    return np.array([math.cos(half), math.sin(half), 0.0, 0.0])


@pytest.fixture(scope="module")
def server():
    """A headless `MujocoServer` we can step and poke by hand."""
    manager = multiprocessing.Manager()
    server = MujocoServer(
        scene_xml_path=Stretch4MujocoSimulator.get_scene_xml_path(),
        model=None,
        stop_mujoco_process_event=threading.Event(),
        data_proxies=MujocoServerProxies.default(manager),
        start_translation=None,
        start_rotation_quat=None,
    )
    yield server
    server.request_to_stop()
    manager.shutdown()


def tilt_base(server: MujocoServer, degrees: float) -> None:
    """Roll the whole robot by `degrees` and refresh the kinematics."""
    # qpos[0:3] is the floating base's position, qpos[3:7] its quaternion.
    server.mjdata.qpos[3:7] = roll_quaternion(degrees)
    mujoco.mj_forward(server.mjmodel, server.mjdata)


def wheel_forceranges(server: MujocoServer) -> np.ndarray:
    ids = [
        mujoco.mj_name2id(server.mjmodel, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in WHEELS
    ]
    return server.mjmodel.actuator_forcerange[ids].copy()


# ---------------------------------------------------------------------------
# The tilt measurement itself
# ---------------------------------------------------------------------------


def test_gravity_tilt_matches_the_robots_imu_math():
    """`gravity_tilt_from_quaternion` reproduces `IMUBase.calculate_tilt_angle`."""
    assert gravity_tilt_from_quaternion([1, 0, 0, 0]) == 0.0
    assert math.degrees(gravity_tilt_from_quaternion(roll_quaternion(6))) == pytest.approx(6)
    assert math.degrees(gravity_tilt_from_quaternion([0, 1, 0, 0])) == pytest.approx(180)
    # An all-zero quaternion is what the robot's IMU reports while it warms up.
    # stretch4_body reads that as upright rather than raising; so do we.
    assert gravity_tilt_from_quaternion([0, 0, 0, 0]) == 0.0

    assert math.degrees(gravity_tilt_from_z_axis([0, 0, 1])) == pytest.approx(0)
    assert math.degrees(gravity_tilt_from_z_axis([1, 0, 0])) == pytest.approx(90)
    assert gravity_tilt_from_z_axis([0, 0, 0]) == 0.0


def test_base_quat_sensor_is_in_the_model(server):
    """The sim publishes an IMU orientation, the robot's `qw`..`qz`."""
    quat = server.mjdata.sensor("base_quat").data
    assert len(quat) == 4
    assert math.degrees(gravity_tilt_from_quaternion(quat)) == pytest.approx(0, abs=1.0)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_upright_robot_does_not_trip(server):
    tilt_base(server, 0.0)
    overtilt = server.safe_motion_manager.controllers["safe_motion_overtilt_avoid"]

    assert overtilt.step() is False
    assert overtilt.status["in_overtilt"] is False
    assert math.degrees(overtilt.status["gravity_tilt"]) == pytest.approx(0, abs=0.5)


def test_tilt_past_the_threshold_trips(server):
    overtilt = server.safe_motion_manager.controllers["safe_motion_overtilt_avoid"]
    assert overtilt.threshold_deg == 6.0, "should match the robot's 6 deg threshold"

    tilt_base(server, overtilt.threshold_deg - 1.0)
    assert overtilt.step() is False

    tilt_base(server, overtilt.threshold_deg + 1.0)
    assert overtilt.step() is True
    assert math.degrees(overtilt.status["gravity_tilt"]) == pytest.approx(7.0, abs=0.5)

    tilt_base(server, 0.0)
    overtilt.step()


def test_release_needs_the_robot_back_under_the_release_angle(server):
    """Hysteresis: once tripped, 5 deg holds the override; from level it does not."""
    overtilt = server.safe_motion_manager.controllers["safe_motion_overtilt_avoid"]
    assert overtilt.release_deg < overtilt.threshold_deg

    tilt_base(server, 0.0)
    overtilt.step()
    tilt_base(server, 5.0)
    assert overtilt.step() is False, "5 deg is under the 6 deg trip angle"

    tilt_base(server, 10.0)
    assert overtilt.step() is True
    tilt_base(server, 5.0)
    assert overtilt.step() is True, "5 deg is over the 4 deg release angle"
    tilt_base(server, 3.0)
    assert overtilt.step() is False

    tilt_base(server, 0.0)
    overtilt.step()


# ---------------------------------------------------------------------------
# The override
# ---------------------------------------------------------------------------


def test_wheels_freewheel_while_overtilted_and_recover_after(server):
    overtilt = server.safe_motion_manager.controllers["safe_motion_overtilt_avoid"]
    tilt_base(server, 0.0)
    overtilt.step()
    driving_forceranges = wheel_forceranges(server)
    assert np.any(driving_forceranges != 0.0), "wheels should have torque when level"

    tilt_base(server, 10.0)
    overtilt.step()
    assert np.all(wheel_forceranges(server) == 0.0), "wheels should make no torque"
    # Stepping again while still tilted must not overwrite the saved limits.
    overtilt.step()
    assert np.all(wheel_forceranges(server) == 0.0)

    tilt_base(server, 0.0)
    overtilt.step()
    assert np.array_equal(wheel_forceranges(server), driving_forceranges)


def test_base_velocity_command_is_dropped_while_overtilted(server):
    overtilt = server.safe_motion_manager.controllers["safe_motion_overtilt_avoid"]
    base_controller = server.base_controller
    tilt_base(server, 0.0)
    overtilt.step()

    base_controller._set_base_velocity_omni_drive(0.2, 0.0, 0.0)
    tilt_base(server, 10.0)
    overtilt.step()

    assert base_controller.active_velocity is None
    assert base_controller.left_wheel_profile.target_vel == 0.0
    assert base_controller.right_wheel_profile.target_vel == 0.0
    assert base_controller.back_wheel_profile.target_vel == 0.0

    tilt_base(server, 0.0)
    overtilt.step()


@pytest.mark.parametrize("actuator", [Actuators.lift.name, Actuators.arm.name])
def test_lift_and_arm_stop_while_overtilted(server, actuator):
    """A move in flight is decelerated to a stop instead of running to its goal."""
    overtilt = server.safe_motion_manager.controllers["safe_motion_overtilt_avoid"]
    tilt_base(server, 0.0)
    overtilt.step()

    profile = server.joint_profiles[actuator]
    start = server._measured_position(actuator)
    profile.set_position(start)

    # Get the joint moving, then tilt.
    server._set_actuator_position(actuator, start + 0.3)
    for _ in range(10):
        server._update_joint_profiles(server.control_rate_hz**-1)
    assert profile.is_moving(), "the joint should be under way before we tilt"

    tilt_base(server, 10.0)
    settled_at = None
    for _ in range(200):
        # What a control cycle does: the command is applied, then the safe
        # motion gets the last word.
        server._update_joint_profiles(server.control_rate_hz**-1)
        overtilt.step()
        if not profile.is_moving():
            settled_at = profile.current_pos
            break

    assert settled_at is not None, "the joint never came to a stop"
    assert abs(settled_at - start) < 0.3, "the joint should stop short of its goal"

    # And it stays put, even if the caller keeps commanding the original goal.
    for _ in range(50):
        server._set_actuator_position(actuator, start + 0.3)
        server._update_joint_profiles(server.control_rate_hz**-1)
        overtilt.step()
    assert profile.current_pos == pytest.approx(settled_at, abs=1e-6)

    tilt_base(server, 0.0)
    overtilt.step()


def test_status_reports_the_tilt_to_clients(server):
    overtilt = server.safe_motion_manager.controllers["safe_motion_overtilt_avoid"]

    tilt_base(server, 10.0)
    overtilt.step()
    server.pull_status()
    status = server.data_proxies.get_status()
    assert status.in_overtilt is True
    assert math.degrees(status.gravity_tilt) == pytest.approx(10, abs=0.5)

    tilt_base(server, 0.0)
    overtilt.step()
    server.pull_status()
    status = server.data_proxies.get_status()
    assert status.in_overtilt is False
    assert math.degrees(status.gravity_tilt) == pytest.approx(0, abs=0.5)
