"""Tests for `SafeMotionGuardedContact`.

These drive a server in-process so the physics actually runs and joints can be
pushed into the scene's table, which is a static body and so stands in for any
immovable obstacle.
"""

import math
import multiprocessing
import threading

import mujoco
import pytest

from stretch4_mujoco.mujoco_server import MujocoServer, MujocoServerProxies
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

GUARDED = "safe_motion_guarded_contact"
OVERTILT = "safe_motion_overtilt_avoid"


class Harness:
    """A headless server we can step by hand, with the arm aimed at the table."""

    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.server = MujocoServer(
            scene_xml_path=Stretch4MujocoSimulator.get_scene_xml_path(),
            model=None,
            stop_mujoco_process_event=threading.Event(),
            data_proxies=MujocoServerProxies.default(self.manager),
            start_translation=None,
            start_rotation_quat=None,
        )
        # The lidar sensor thread copies mjData; share its lock or the copy
        # lands mid-step and MuJoCo aborts the process.
        self.lock = self.server.sensor_manager.sensor_lock

    @property
    def guarded(self):
        return self.server.safe_motion_manager.controllers[GUARDED]

    def face_table(self):
        """Yaw the base so the arm (its +x) points at the table at y=-1."""
        model, data = self.server.mjmodel, self.server.mjdata
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
        adr = model.jnt_qposadr[joint_id]
        with self.lock:
            half = math.radians(-90) / 2
            data.qpos[adr + 3 : adr + 7] = [math.cos(half), 0, 0, math.sin(half)]
            mujoco.mj_forward(model, data)

    def cycles(self, count: int) -> float:
        """Run `count` control cycles, returning the peak tilt seen, in degrees."""
        from stretch4_mujoco.utils import site_gravity_tilt

        peak = 0.0
        for _ in range(count):
            self.server._physics_step(self.lock)
            peak = max(
                peak,
                math.degrees(site_gravity_tilt(self.server.mjmodel, self.server.mjdata)),
            )
        return peak

    def close(self):
        self.server.request_to_stop()
        # Let the sensor thread notice before the manager's proxies go away,
        # or its next write hits a closed pipe.
        self.server.sensor_manager.sensors_thread.join(timeout=5)
        self.manager.shutdown()

    def jam_arm(self):
        """Drive the arm into the table until the guard trips. Returns the controller."""
        self.face_table()
        self.cycles(200)
        self.server._set_actuator_position("arm", 0.5)
        self.server._set_actuator_position("lift", 0.62)
        self.cycles(1200)
        return self.guarded


@pytest.fixture
def harness():
    h = Harness()
    h.cycles(200)  # let the robot settle onto its wheels
    yield h
    h.close()


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def test_thresholds_mirror_the_robots_stepper_gains(harness):
    """Trip points come from `i_contact_a / i_max_a`, as on the robot."""
    guarded = harness.guarded
    # robot_params_SE4.py: arm iMax 7.7 A / i_contact 2.0 A, lift 6.7 A / 2.0 A.
    assert guarded.threshold_pct("arm") == pytest.approx(100 * 2.0 / 7.7, abs=0.1)
    assert guarded.threshold_pct("lift") == pytest.approx(100 * 2.0 / 6.7, abs=0.1)
    assert guarded.sensitivity == 1.0


def test_sensitivity_profile_scales_the_threshold(harness):
    guarded = harness.guarded
    at_default = guarded.threshold_pct("arm")
    guarded.params["sensitivity_profile"] = "high"
    assert guarded.threshold_pct("arm") < at_default, "high sensitivity trips sooner"
    guarded.params["sensitivity_profile"] = "low"
    assert guarded.threshold_pct("arm") > at_default
    guarded.params["sensitivity_profile"] = "default"


# ---------------------------------------------------------------------------
# No false positives
# ---------------------------------------------------------------------------


def test_ordinary_motion_does_not_trip(harness):
    """Free lift and arm moves stay well under the contact thresholds."""
    server, guarded = harness.server, harness.guarded
    for lift, arm in ((0.9, 0.4), (0.1, 0.0), (0.5, 0.25)):
        server._set_actuator_position("lift", lift)
        server._set_actuator_position("arm", arm)
        harness.cycles(500)

    assert guarded.status["guarded_events"] == 0
    assert not any(guarded.status["in_guarded_event"].values())
    for joint in ("lift", "arm"):
        assert abs(guarded.status["effort_pct"][joint]) < guarded.threshold_pct(joint)


def test_base_driving_in_the_open_does_not_trip(harness):
    """Accelerating the base is nowhere near the wheels' contact threshold."""
    server, guarded = harness.server, harness.guarded
    server.base_controller._set_base_velocity_omni_drive(0.3, 0.0, 0.0)
    harness.cycles(300)
    server.base_controller._set_base_velocity_omni_drive(0.0, 0.0, 2.0)
    harness.cycles(300)
    server.base_controller._clear_command(is_stop_motion=True)
    harness.cycles(200)

    assert guarded.status["guarded_events"] == 0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_arm_driven_into_the_table_trips_and_does_not_topple(harness):
    """The scenario this exists for: pressing the gripper onto a table."""
    server, guarded = harness.server, harness.guarded
    harness.face_table()
    harness.cycles(200)

    server._set_actuator_position("arm", 0.5)
    server._set_actuator_position("lift", 0.62)
    peak_tilt = harness.cycles(1500)

    assert guarded.status["guarded_events"] > 0, "contact was never detected"
    assert guarded.status["in_guarded_event"]["arm"], "the arm should be guarded"
    assert peak_tilt < 5.0, f"robot should stay level, peaked at {peak_tilt:.1f} deg"


def test_contact_takes_more_than_one_cycle_to_declare(harness):
    """The effort low-pass means a single-cycle solver spike is not a contact."""
    server, guarded = harness.server, harness.guarded
    harness.face_table()
    harness.cycles(200)

    server._set_actuator_position("arm", 0.5)
    server._set_actuator_position("lift", 0.62)
    cycles_to_trip = 0
    for cycles_to_trip in range(1, 1500):
        harness.cycles(1)
        if guarded.status["guarded_events"]:
            break
    assert cycles_to_trip > 1, "tripped on a single cycle; the filter is not working"


# ---------------------------------------------------------------------------
# Latch and release
# ---------------------------------------------------------------------------


def test_guard_holds_while_still_commanded_into_the_obstacle(harness):
    server, guarded = harness.server, harness.jam_arm()
    assert guarded.status["in_guarded_event"]["arm"]
    into_it = guarded.status["trip_direction"]["arm"]

    held = server.joint_profiles["arm"].current_pos
    # Keep asking to go further the way that jammed; it must not creep.
    for _ in range(200):
        server._set_actuator_position("arm", held + into_it * 0.1)
        harness.cycles(1)
    assert guarded.status["in_guarded_event"]["arm"], "guard released while pushed"
    assert server.joint_profiles["arm"].current_pos == pytest.approx(held, abs=1e-6)


def test_commanding_the_other_way_releases_the_guard(harness):
    server, guarded = harness.server, harness.jam_arm()
    assert guarded.status["in_guarded_event"]["arm"]
    into_it = guarded.status["trip_direction"]["arm"]

    held = server.joint_profiles["arm"].current_pos
    server._set_actuator_position("arm", held - into_it * 0.2)
    harness.cycles(5)
    assert not guarded.status["in_guarded_event"]["arm"]
    assert not server.safe_motion_manager.overrides.is_holding("arm")

    # The arm trips early -- the gripper is already on the table -- so backing
    # off 0.2 m runs into the joint's own travel limit. Check it reaches where
    # it is allowed to go, not an arbitrary distance.
    low, high = server.joint_profile_travel["arm"]
    reachable = min(max(held - into_it * 0.2, low), high)
    assert reachable != held, "test needs room to back off"
    harness.cycles(500)
    assert server.joint_profiles["arm"].current_pos == pytest.approx(reachable, abs=0.01)


def test_releasing_the_command_entirely_releases_the_guard(harness):
    server, guarded = harness.server, harness.jam_arm()
    assert guarded.status["in_guarded_event"]["arm"]

    # A jog that stops is the teleop case: let go of the stick, guard lifts.
    server._set_actuator_velocity("arm", 0.0, harness.server.control_rate_hz**-1)
    harness.cycles(5)
    assert not guarded.status["in_guarded_event"]["arm"]


# ---------------------------------------------------------------------------
# Shared overrides
# ---------------------------------------------------------------------------


def test_two_safe_motions_holding_one_joint_do_not_release_each_other(harness):
    """An overtilt release must not free a joint guarded contact still holds."""
    overrides = harness.server.safe_motion_manager.overrides
    overrides.hold_joint("lift", OVERTILT)
    overrides.hold_joint("lift", GUARDED)
    assert overrides.is_holding("lift")

    overrides.release_joint("lift", OVERTILT)
    assert overrides.is_holding("lift"), "still held by guarded contact"

    overrides.release_joint("lift", GUARDED)
    assert not overrides.is_holding("lift")


def test_freewheel_is_shared_the_same_way(harness):
    overrides = harness.server.safe_motion_manager.overrides
    overrides.freewheel_base(OVERTILT)
    overrides.freewheel_base(GUARDED, clear_commands=False)
    assert overrides.is_freewheeling

    overrides.release_base(OVERTILT)
    assert overrides.is_freewheeling

    overrides.release_base(GUARDED)
    assert not overrides.is_freewheeling


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_status_reports_guarded_state_to_clients(harness):
    server, guarded = harness.server, harness.guarded
    harness.face_table()
    harness.cycles(200)
    server._set_actuator_position("arm", 0.5)
    server._set_actuator_position("lift", 0.62)
    harness.cycles(1200)

    status = server.data_proxies.get_status()
    assert status.guarded_events == guarded.status["guarded_events"]
    assert status.in_guarded_event["arm"] is True


# ---------------------------------------------------------------------------
# Servo joints (wrist, gripper) -- the stall guard
# ---------------------------------------------------------------------------


def test_servo_joints_are_watched(harness):
    """The wrist and gripper are guarded too, by the servos' stall model."""
    guarded = harness.guarded
    for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
        assert joint in guarded.status["in_guarded_event"], f"{joint} is unguarded"
        # SE4_wrist_*_DW4: stall_max_effort 20.0.
        assert guarded.threshold_pct(joint) == pytest.approx(20.0)


def test_wrist_actuators_carry_servo_torque_not_a_shaft_force(harness):
    """Regression on the units bug that let the wrist flip the robot.

    The wrist actuators drive hinge joints at gear 1, so MuJoCo reads their
    `forcerange` as a torque. It was set to 400, which is the linear force a
    2 N.m servo makes at the rim of a 25 mm shaft -- a real number, in the slot
    that wants N.m. The robot then had a wrist that could apply 400 N.m and
    flip itself. The right figure is the joint torque: 2 N.m rated through the
    2:1 reduction. `arm` and `lift` are linear transmissions and do carry
    newtons, which is why 175/200 are right for them.
    """
    model, data = harness.server.mjmodel, harness.server.mjdata
    for joint in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
        actuator_id = data.actuator(joint).id
        transmission = model.actuator_trnid[actuator_id][0]
        assert model.jnt_type[transmission] == mujoco.mjtJoint.mjJNT_HINGE
        assert model.actuator_gear[actuator_id][0] == 1.0
        assert model.actuator_forcerange[actuator_id][1] == pytest.approx(4.0), (
            "wrist forcerange is a joint torque in N.m: 2 N.m rated x 2:1"
        )


def test_wrist_guard_trips_below_the_tipping_moment(harness):
    """The wrist can no longer out-muscle the robot's own weight.

    Measured: the robot's weight holds it down with about 50 N.m about the
    nearest tipping edge. The wrist's whole range is now 4 N.m and its guard
    trips at a fifth of that, so neither can get near tipping it.
    """
    guarded = harness.guarded
    actuator_id = harness.server.mjdata.actuator("wrist_pitch").id
    force_range = float(harness.server.mjmodel.actuator_forcerange[actuator_id][1])
    trip_at_nm = force_range * guarded.threshold_pct("wrist_pitch") / 100.0

    tipping_moment_nm = 50.0
    assert force_range < tipping_moment_nm, "the wrist alone could tip the robot"
    assert trip_at_nm < tipping_moment_nm


def test_wrist_jammed_against_the_table_stalls_out(harness):
    """A wrist driven into the table stalls out instead of levering the robot."""
    server, guarded = harness.server, harness.guarded
    harness.face_table()
    harness.cycles(200)
    server._set_actuator_position("arm", 0.45)
    server._set_actuator_position("lift", 0.55)
    harness.cycles(900)
    server._set_actuator_position("wrist_pitch", 1.5)
    peak_tilt = harness.cycles(1500)

    assert peak_tilt < 5.0, f"robot tipped to {peak_tilt:.1f} deg"


def test_full_range_wrist_motion_does_not_false_trip(harness):
    """Big wrist sweeps at 4 N.m still track, and the stall guard stays out.

    Peak effort touches the limit on fast reversals, which is why the servo
    guard is gated on velocity as well: a joint that is moving is not stalled,
    however hard it is working.
    """
    server, guarded = harness.server, harness.guarded
    server._set_actuator_position("arm", 0.5)
    server._set_actuator_position("lift", 0.9)
    harness.cycles(800)

    joints = ("wrist_yaw", "wrist_pitch", "wrist_roll")
    # Inside the joint ranges: yaw/pitch -1.135..4.276, roll -4.276..1.135.
    for targets in ((1.5, 1.0, -1.5), (-1.0, -1.0, 1.0), (0.0, 0.0, 0.0)):
        for joint, target in zip(joints, targets):
            server._set_actuator_position(joint, target)
        harness.cycles(900)
        for joint, target in zip(joints, targets):
            reached = float(server.mjdata.actuator(joint).length[0])
            assert reached == pytest.approx(target, abs=0.05), f"{joint} did not track"

    assert guarded.status["guarded_events"] == 0


def test_lift_pressed_down_onto_a_table_does_not_topple(harness):
    """Arm out over the table, lift driven down onto the surface.

    The case that exposed the hold bug. Pressing down at a gripper 0.74 m out
    is the worst lever the robot has on itself: the restoring moment is about
    50 N.m and the lift's 200 N acting that far out dwarfs it, so the guard has
    to actually take the force out, not merely report the joint stopped.
    """
    server, guarded = harness.server, harness.guarded
    harness.face_table()
    harness.cycles(200)

    server._set_actuator_position("lift", 0.75)   # clear of the table top
    harness.cycles(700)
    server._set_actuator_position("arm", 0.5)     # out over it
    harness.cycles(900)
    server._set_actuator_position("lift", 0.0)    # press down onto it
    peak_tilt = harness.cycles(2500)

    assert guarded.status["guarded_events"] > 0
    assert peak_tilt < 5.0, f"robot tipped to {peak_tilt:.1f} deg"


def test_hold_parks_at_the_measured_position_not_the_setpoint(harness):
    """A held joint must stop pushing, which means holding where it actually is.

    A blocked joint's setpoint keeps advancing past the joint. Holding that
    stale setpoint leaves the full position error -- and so nearly full motor
    force -- applied to whatever the joint is jammed against, while every
    status field says the joint is stopped.
    """
    server, guarded = harness.server, harness.guarded
    harness.face_table()
    harness.cycles(200)
    server._set_actuator_position("arm", 0.5)
    server._set_actuator_position("lift", 0.62)
    harness.cycles(1200)
    assert guarded.status["in_guarded_event"]["arm"]

    held = server.joint_profiles["arm"].current_pos
    measured = server._measured_position("arm")
    # Tolerance is physical, not numerical: the joint settles a little once the
    # contact unloads. What matters is that the residual error is far too small
    # to be a force -- a tenth of a millimetre against the arm's kp of 2500 is
    # a fraction of a newton, where the stale setpoint was worth tens.
    assert held == pytest.approx(measured, abs=1e-3), "hold left a standing error"

    force_range = float(server.mjmodel.actuator_forcerange[server.mjdata.actuator("arm").id][1])
    effort_pct = abs(100.0 * float(server.mjdata.actuator("arm").force[0]) / force_range)
    assert effort_pct < guarded.threshold_pct("arm"), (
        f"held arm is still pushing at {effort_pct:.0f}% effort"
    )
