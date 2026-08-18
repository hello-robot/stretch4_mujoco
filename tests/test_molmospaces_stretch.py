"""
Fast checks for the MolmoSpaces Stretch integration in
`examples/machine_learning/molmospaces/`.

These cover the parts that can be verified without loading a house or rendering:
the robot attaches and compiles, the move groups resolve and track, reaching
solves, an episode retargets, and the behaviour-cloning encoding round-trips.
Actually scoring a benchmark needs assets and a GPU, and lives in
`run_benchmarks.py`.

Requires the optional MolmoSpaces dependency:  pip install -e ".[molmo]"
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("molmo_spaces")

from mujoco import MjSpec  # noqa: E402

from examples.machine_learning.molmospaces.policies.kinematics import (  # noqa: E402
    PITCH_HORIZONTAL,
    PITCH_TOP_DOWN,
    StretchReachSolver,
    planar_pose,
)
from examples.machine_learning.molmospaces.policies.networks import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    decode_action,
    encode_action,
    encode_state,
)
from examples.machine_learning.molmospaces.stretch.config import (  # noqa: E402
    Stretch4RobotConfig,
)
from examples.machine_learning.molmospaces.stretch.episode_overrides import (  # noqa: E402
    REACH_BAND_M,
    retarget_base_pose,
    stretch_home_init_qpos,
)
from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot  # noqa: E402
from examples.machine_learning.molmospaces.stretch.robot_view import (  # noqa: E402
    Stretch4RobotView,
)

NAMESPACE = "robot_0/"


@pytest.fixture(scope="module")
def robot_config():
    return Stretch4RobotConfig()


@pytest.fixture(scope="module")
def compiled_robot(robot_config):
    """Stretch attached to a bare floor, compiled, with a robot view over it."""
    spec = MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, name="floor", size=[10.0, 10.0, 0.1])
    Stretch4Robot.add_robot_to_scene(
        robot_config, spec, prefix=NAMESPACE, pos=[0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    Stretch4Robot.apply_control_overrides(spec, robot_config)
    model = spec.compile()
    data = mujoco.MjData(model)
    view = Stretch4RobotView(data, NAMESPACE)
    mujoco.mj_forward(model, data)
    return model, data, view


def test_attached_model_has_holonomic_base_actuators(compiled_robot):
    model, _, _ = compiled_robot
    names = {model.actuator(i).name for i in range(model.nu)}
    assert {
        f"{NAMESPACE}base_x_act",
        f"{NAMESPACE}base_y_act",
        f"{NAMESPACE}base_theta_act",
    } <= names
    # The freejoint on the root body must be gone, or the base could not be
    # driven by the holonomic joints.
    joint_types = {int(model.jnt_type[i]) for i in range(model.njnt)}
    assert int(mujoco.mjtJoint.mjJNT_FREE) not in joint_types
    # Keyframes are sized for the standalone model's actuator count.
    assert model.nkey == 0


def test_move_groups_are_simply_actuated(compiled_robot):
    _, _, view = compiled_robot
    for group_id in view.MOVE_GROUP_ORDER:
        group = view.get_move_group(group_id)
        # Every MolmoSpaces position controller clips its target against the
        # actuator control range, so these two widths have to agree.
        assert group.pos_dim == group.n_actuators, group_id
        assert group.ctrl_limits.shape == (group.n_actuators, 2), group_id

    # A [0, 0] control range would silently pin the base yaw at zero.
    yaw_limits = view.base.ctrl_limits[2]
    assert yaw_limits[1] - yaw_limits[0] > 2.0 * np.pi


def test_base_waypoint_distance_wraps_yaw(compiled_robot):
    """`AStarPlannerPolicy` needs these; `RobotView` does not provide them."""
    model, data, view = compiled_robot
    view.base.joint_pos = np.array([1.0, 2.0, np.pi - 0.05])
    mujoco.mj_kinematics(model, data)

    assert view.distance_to(["base"], [1.0, 2.0, np.pi - 0.05]) == pytest.approx(0.0, abs=1e-9)
    assert view.is_close_to(["base"], [1.0, 2.0, np.pi - 0.05])
    # A target just the other side of +-pi is 0.1rad away, not ~2pi away.
    assert view.distance_to(["base"], [1.0, 2.0, -np.pi + 0.05]) == pytest.approx(0.1, abs=1e-6)
    assert not view.is_close_to(["base"], [1.5, 2.0, np.pi - 0.05])


def test_telescoping_arm_reports_total_extension(compiled_robot):
    model, data, view = compiled_robot
    arm = view.get_move_group("arm")

    arm.joint_pos = np.array([0.4])
    mujoco.mj_kinematics(model, data)
    assert arm.joint_pos[0] == pytest.approx(0.4)
    # Split evenly across the four equality-constrained segments.
    assert np.allclose(data.qpos[arm._joint_posadr], 0.1)

    # The Jacobian column must be the mean of the four segment columns, not
    # their sum: commanding an extension of `a` moves each segment by `a / 4`.
    jacobian = view.get_jacobian("gripper", ["lift", "arm", "wrist"])
    assert jacobian.shape == (6, 5)
    assert np.linalg.norm(jacobian[:3, 1]) == pytest.approx(1.0, abs=0.05)


def test_gripper_open_and_closed(compiled_robot):
    model, data, view = compiled_robot
    gripper = view.get_gripper("gripper")

    gripper.joint_pos = np.array([gripper.OPEN_JOINT_POS] * 2)
    mujoco.mj_kinematics(model, data)
    assert gripper.is_open
    assert gripper.inter_finger_dist == pytest.approx(gripper.INTER_FINGER_DIST_RANGE[1], abs=0.02)

    gripper.joint_pos = np.array([gripper.CLOSED_JOINT_POS] * 2)
    mujoco.mj_kinematics(model, data)
    assert not gripper.is_open


def test_position_controllers_track_their_targets(compiled_robot, robot_config):
    """The whole robot has to reach a commanded configuration under gravity."""
    model, data, view = compiled_robot
    mujoco.mj_resetData(model, data)
    targets = {
        "base": np.array([1.0, 0.5, 0.7]),
        "lift": np.array([0.9]),
        "arm": np.array([0.4]),
        "wrist": np.array([0.5, 0.0, 0.0]),
        "gripper": np.array([0.0, 0.0]),
    }
    for group_id, target in targets.items():
        view.get_move_group(group_id).ctrl = target
    for _ in range(3000):
        mujoco.mj_step(model, data)

    for group_id, target in targets.items():
        assert np.allclose(view.get_move_group(group_id).joint_pos, target, atol=2e-2), group_id


def test_horizontal_reach_solves_without_moving_the_base(robot_config):
    solver = StretchReachSolver(robot_config)
    base_pose = planar_pose(1.0, 2.0, 0.0)

    solved = 0
    for forward in (0.55, 0.70, 0.85):
        for lateral in (-0.1, 0.0, 0.1):
            for height in (0.4, 0.75, 1.1):
                target = np.array([1.0 + forward, 2.0 + lateral, height])
                solution = solver.solve(base_pose, target, wrist_pitch=PITCH_HORIZONTAL)
                if solution is None:
                    continue
                assert np.linalg.norm(solver.forward(solution)[:3, 3] - target) < 6e-3
                # Horizontal reaching is the case Stretch's wrist yaw covers on
                # its own; the base must not have to creep.
                assert np.allclose(solution["base"][:2], [1.0, 2.0], atol=1e-6)
                solved += 1
    assert solved >= 24, f"only {solved}/27 in-band targets solved"


def test_top_down_reach_recruits_the_base(robot_config):
    """Reaching straight down has no wrist-driven lateral freedom."""
    solver = StretchReachSolver(robot_config)
    base_pose = planar_pose(0.0, 0.0, 0.0)
    target = np.array([0.65, 0.25, 0.8])

    without_base = solver.solve(
        base_pose, target, wrist_pitch=PITCH_TOP_DOWN, dofs=("lift", "arm", "wrist_yaw")
    )
    assert without_base is None

    with_base = solver.solve(base_pose, target, wrist_pitch=PITCH_TOP_DOWN)
    assert with_base is not None
    assert np.linalg.norm(solver.forward(with_base)[:3, 3] - target) < 6e-3
    assert np.linalg.norm(with_base["base"][:2]) > 1e-3


def test_home_init_qpos_omits_the_base():
    """The base pose belongs to the episode, not to the robot config default."""
    init_qpos = stretch_home_init_qpos()
    assert "base" not in init_qpos
    assert set(init_qpos) == {"lift", "arm", "wrist", "gripper"}


def test_retarget_aims_the_base_at_the_target_and_keeps_the_approach():
    # A base 2m from the pickup object, facing away from it: too far to reach,
    # and pointing the wrong way.
    task = {
        "task_cls": "molmo_spaces.tasks.pick_task.PickTask",
        "robot_base_pose": [2.0, 0.0, 0.58, 1.0, 0.0, 0.0, 0.0],
        "pickup_obj_start_pose": [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
    }
    retarget_base_pose(task)
    pose = task["robot_base_pose"]

    distance = np.linalg.norm(np.array(pose[:2]))
    assert REACH_BAND_M[0] <= distance <= REACH_BAND_M[1]
    # Pulled straight back along the original approach direction (+x from the
    # object), not moved around it.
    assert pose[1] == pytest.approx(0.0, abs=1e-9)
    assert pose[0] > 0.0
    # Base +x now points at the object, i.e. yaw is pi.
    yaw = 2.0 * np.arctan2(pose[6], pose[3])
    assert np.cos(yaw) == pytest.approx(-1.0, abs=1e-6)
    # Stretch stands on the floor, not on the Franka's plinth.
    assert pose[2] == 0.0


def test_retarget_leaves_navigation_spawns_alone():
    task = {
        "task_cls": "molmo_spaces.tasks.nav_task.NavToObjTask",
        "robot_base_pose": [4.0, -1.0, 0.58, 1.0, 0.0, 0.0, 0.0],
        "pickup_obj_name": "television_1",
    }
    retarget_base_pose(task)
    assert task["robot_base_pose"][:2] == [4.0, -1.0]
    assert task["robot_base_pose"][3:] == [1.0, 0.0, 0.0, 0.0]


def test_retarget_keeps_an_already_reachable_standoff():
    task = {
        "task_cls": "molmo_spaces.tasks.pick_task.PickTask",
        "robot_base_pose": [0.7, 0.0, 0.58, 1.0, 0.0, 0.0, 0.0],
        "pickup_obj_start_pose": [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
    }
    retarget_base_pose(task)
    assert task["robot_base_pose"][:2] == [0.7, 0.0]


def test_action_encoding_round_trips():
    """The base is encoded relative and everything else absolute."""
    base_xytheta = np.array([3.0, -2.0, 0.9], dtype=np.float32)
    commanded = {
        "base": np.array([3.2, -1.9, 1.1]),
        "lift": np.array([0.7]),
        "arm": np.array([0.25]),
        "wrist": np.array([0.3, 1.57, 0.0]),
        "gripper": np.array([0.5, 0.5]),
    }
    encoded = encode_action(commanded, base_xytheta)
    assert encoded.shape == (ACTION_DIM,)
    # A relative base step, not the world pose.
    assert not np.allclose(encoded[:3], commanded["base"])

    decoded = decode_action(encoded, base_xytheta)
    for group, value in commanded.items():
        assert np.allclose(decoded[group], value, atol=1e-5), group


def test_state_encoding_matches_the_move_groups(compiled_robot):
    _, _, view = compiled_robot
    state = encode_state(view.get_qpos_dict())
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32
