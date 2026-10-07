"""
Does the real robot's gripper open and close when `run_on_real_stretch.py` tells it to?

**This moves a real robot's gripper.** It only runs when `STRETCH_ROBOT_IP` names
a robot, and is skipped otherwise:

    STRETCH_ROBOT_IP=<robot ip> pytest \\
        examples/machine_learning/molmospaces/retargetting/tests/test_real_gripper.py -v

Only the gripper moves. Keep its jaws clear: the close is checked by the jaws
getting most of the way shut, so anything between them fails the test.

Everything here goes through the module's own code rather than through a
`RobotClient` built by hand, because the code is what is being checked:
`connect_robot` (the robot's status read before `stretch4_body`'s parameters are
built for that tool), `GripperUnits` (the robot's units and range), and
`RobotCommander.send` (a policy step's finger targets turned into a `move_to`).
The first tests go straight to `end_of_arm.move_to` and the last ones through
`RobotCommander`, so that a failure says which half of the chain to look at.

What is measured is the robot's own status, read over the same connection, not
the camera stream: whether the hand moved is the question, not whether the
sender reports it.
"""

import os
import time

import pytest

from examples.machine_learning.molmospaces.retargetting.run_on_real_stretch import (
    GRIPPER_JOINT,
    PARALLEL_GRIPPER_JOINT,
    GripperUnits,
    RobotCommander,
    connect_robot,
)
from examples.machine_learning.molmospaces.stretch.robot_view import PG4_GRIPPER, SG4_GRIPPER

ROBOT_IP = os.environ.get("STRETCH_ROBOT_IP")

pytestmark = pytest.mark.skipif(
    not ROBOT_IP, reason="set STRETCH_ROBOT_IP to the robot's address to run against it"
)

GRIPPER_KIND_FOR_JOINT = {GRIPPER_JOINT: SG4_GRIPPER, PARALLEL_GRIPPER_JOINT: PG4_GRIPPER}

ARRIVAL_TIMEOUT_S = 5.0
"""How long the hand gets to travel its whole range. It takes about one."""

POLL_S = 0.05

OPEN_ENOUGH = 0.9
SHUT_ENOUGH = 0.1
"""Fractions of the hand's travel, 0 shut to 1 open, that count as arrived. Not
exact: the servo stops a little short of a target, and the SG4 shut is a stall."""


@pytest.fixture(scope="module")
def connection():
    """One connection for the module, stopped however the tests end."""
    robot, gripper_joint = connect_robot(ROBOT_IP)
    yield robot, gripper_joint
    robot.stop()


@pytest.fixture(scope="module")
def units(connection) -> GripperUnits:
    robot, gripper_joint = connection
    return GripperUnits.for_joint(gripper_joint, GRIPPER_KIND_FOR_JOINT[gripper_joint], robot)


def measured(robot, units: GripperUnits) -> float:
    """The gripper's position in robot units, from the robot's own status."""
    robot.pull_status(blocking=True)
    status = robot.end_of_arm.status[units.joint]
    if units.joint == PARALLEL_GRIPPER_JOINT:
        return float(status["pos_mm"]) / 1000.0
    return float(status["pos_pct"])


def is_moving(robot, units: GripperUnits) -> bool:
    return bool(robot.end_of_arm.status[units.joint].get("is_moving", False))


def wait_for_fraction(robot, units: GripperUnits, arrived) -> list[float]:
    """Poll until `arrived(fraction)` and the hand has stopped, or the timeout.
    Returns every fraction seen.

    Stopped as well as arrived, so that the next test's command goes to a hand at
    rest: a command sent while the last one is still running is its own case,
    `test_close_sent_while_opening`.
    """
    seen = []
    deadline = time.monotonic() + ARRIVAL_TIMEOUT_S
    while time.monotonic() < deadline:
        seen.append(units.fraction(measured(robot, units)))
        if arrived(seen[-1]) and not is_moving(robot, units):
            break
        time.sleep(POLL_S)
    return seen


def wait_until_moving(robot, units: GripperUnits) -> list[float]:
    """Poll until the hand reports moving, or the timeout. Returns every fraction seen."""
    seen = []
    deadline = time.monotonic() + ARRIVAL_TIMEOUT_S
    while time.monotonic() < deadline:
        seen.append(units.fraction(measured(robot, units)))
        if is_moving(robot, units):
            break
        time.sleep(POLL_S)
    return seen


def move_to(robot, units: GripperUnits, value: float) -> None:
    """`end_of_arm.move_to` at the limits `RobotCommander` sends the gripper."""
    from examples.digital_twin import EOA_ACCELERATION_R, EOA_VELOCITY_R

    accepted = robot.end_of_arm.move_to(units.joint, value, EOA_VELOCITY_R, EOA_ACCELERATION_R)
    assert accepted is not False, f"the client refused move_to({units.joint!r}, {value})"
    robot.push_command()


def send_fraction(robot, units: GripperUnits, fraction: float) -> dict[str, float]:
    """One policy step that only has a gripper target, through `RobotCommander.send`."""
    commander = RobotCommander(robot, units)
    finger = float(units.kind.joint_pos_for_fraction(fraction))
    return commander.send(
        {"gripper": [finger, finger]}, {"gripper_pos": measured(robot, units)}
    )


STATUS_FLAGS = (
    "temp",
    "effort",
    "in_collision_stop",
    "torque_enabled",
    "is_moving",
    "overload_error",
    "overtemp_error",
    "overcurrent_error",
    "hardware_error",
    "stalled",
    "stall_overload",
)
"""The servo status that explains a command the server dropped without a word:
`FeetechSMHello.move_to` returns early, logging only on the robot, while a
direction is in collision stop or after a runstop."""


def describe(robot, units: GripperUnits, seen: list[float]) -> str:
    status = robot.end_of_arm.status[units.joint]
    flags = {key: status[key] for key in STATUS_FLAGS if key in status}
    flags["runstop_event"] = robot.status.get("power_periph", {}).get("runstop_event")
    return (
        f"{units.describe()}; fraction went {seen[0]:.2f} -> {seen[-1]:.2f} "
        f"over {len(seen)} reads; status {flags}"
    )


# -- the connection -----------------------------------------------------------


def test_connect_builds_the_tool_the_robot_has_on(connection):
    """The parameters `connect_robot` built name the gripper the robot reports."""
    robot, gripper_joint = connection
    assert gripper_joint in robot.end_of_arm.status, robot.end_of_arm.status.keys()
    assert gripper_joint in robot.end_of_arm.joints, (
        f"the client's end of arm is {robot.end_of_arm.joints}, but the robot has "
        f"{gripper_joint} on"
    )


def test_gripper_is_homed(connection):
    robot, gripper_joint = connection
    assert robot.end_of_arm.is_homed(gripper_joint)


# -- straight to the client ---------------------------------------------------


def test_client_opens(connection, units):
    robot, _ = connection
    move_to(robot, units, units.robot_open)
    seen = wait_for_fraction(robot, units, lambda f: f >= OPEN_ENOUGH)
    assert seen[-1] >= OPEN_ENOUGH, describe(robot, units, seen)


def test_client_closes(connection, units):
    robot, _ = connection
    move_to(robot, units, units.robot_closed)
    seen = wait_for_fraction(robot, units, lambda f: f <= SHUT_ENOUGH)
    assert seen[-1] <= SHUT_ENOUGH, describe(robot, units, seen)


# -- through RobotCommander, as a policy step ---------------------------------


def test_commander_opens(connection, units):
    robot, _ = connection
    commanded = send_fraction(robot, units, 1.0)
    assert commanded["gripper_pos"] == pytest.approx(units.robot_open)
    seen = wait_for_fraction(robot, units, lambda f: f >= OPEN_ENOUGH)
    assert seen[-1] >= OPEN_ENOUGH, describe(robot, units, seen)


def test_commander_closes(connection, units):
    robot, _ = connection
    commanded = send_fraction(robot, units, 0.0)
    assert commanded["gripper_pos"] == pytest.approx(units.robot_closed)
    seen = wait_for_fraction(robot, units, lambda f: f <= SHUT_ENOUGH)
    assert seen[-1] <= SHUT_ENOUGH, describe(robot, units, seen)


def test_close_sent_while_opening(connection, units):
    """A close that arrives while an open is still running, as a 15Hz policy sends them.

    Opened from shut, so that the open has its whole travel to run, and the
    close sent as soon as the hand reports moving.
    """
    robot, _ = connection
    move_to(robot, units, units.robot_closed)
    seen = wait_for_fraction(robot, units, lambda f: f <= SHUT_ENOUGH)
    assert seen[-1] <= SHUT_ENOUGH, "did not start shut: " + describe(robot, units, seen)

    move_to(robot, units, units.robot_open)
    seen = wait_until_moving(robot, units)
    assert is_moving(robot, units), "the open never started: " + describe(robot, units, seen)

    move_to(robot, units, units.robot_closed)
    seen = wait_for_fraction(robot, units, lambda f: f <= SHUT_ENOUGH)
    assert seen[-1] <= SHUT_ENOUGH, describe(robot, units, seen)


def test_commander_leaves_it_open(connection, units):
    """Last, so that the gripper is left open whatever happened above."""
    robot, _ = connection
    send_fraction(robot, units, 1.0)
    seen = wait_for_fraction(robot, units, lambda f: f >= OPEN_ENOUGH)
    assert seen[-1] >= OPEN_ENOUGH, describe(robot, units, seen)

if __name__ == "__main__":
    pytest.main([__file__])