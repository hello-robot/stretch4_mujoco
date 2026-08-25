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
from examples.machine_learning.molmospaces.franka_remapping.episode_overrides import (  # noqa: E402
    REACH_BAND_M,
    TCP_MIN_REACH_M,
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


# =============================================================================
# Live-sim adapter: the contract between the benchmark robot and the simulator
# =============================================================================


def test_live_camera_mapping_matches_the_trained_camera_system():
    """The live sim must render the same MJCF cameras the policy trained on."""
    from examples.machine_learning.molmospaces.live_policy import CAMERA_FOR_TRAINED_NAME
    from examples.machine_learning.molmospaces.stretch.config import (
        HEAD_CAMERA,
        HEAD_CAMERA_MJCF_NAME,
        WRIST_CAMERA,
        WRIST_CAMERA_MJCF_NAME,
    )

    assert CAMERA_FOR_TRAINED_NAME[HEAD_CAMERA].camera_name_in_mjcf == HEAD_CAMERA_MJCF_NAME
    assert CAMERA_FOR_TRAINED_NAME[WRIST_CAMERA].camera_name_in_mjcf == WRIST_CAMERA_MJCF_NAME


def test_simulator_status_encodes_the_same_state_as_the_robot_view():
    """`read_state` and `encode_state` must agree, or a checkpoint silently degrades.

    They read the same seven numbers out of two unrelated data structures --
    `StatusStretchJoints` in the live sim, `get_qpos_dict()` in the benchmark --
    and nothing at runtime would notice if their orders diverged.
    """
    from types import SimpleNamespace

    from examples.machine_learning.molmospaces.live_policy import read_state

    def joint(position: float):
        return SimpleNamespace(pos=position, vel=0.0, effort=0.0)

    status = SimpleNamespace(
        lift=joint(0.61),
        arm=joint(0.23),
        wrist_yaw=joint(0.31),
        wrist_pitch=joint(1.57),
        wrist_roll=joint(-0.2),
        gripper_right_finger=joint(0.44),
        gripper_left_finger=joint(0.42),
    )
    equivalent_qpos = {
        "base": [1.0, 2.0, 0.5],  # present in the robot view, absent from the state
        "lift": [0.61],
        "arm": [0.23],
        "wrist": [0.31, 1.57, -0.2],
        "gripper": [0.44, 0.42],
    }
    assert np.allclose(read_state(status), encode_state(equivalent_qpos))


def test_absolute_base_target_becomes_a_holonomic_velocity():
    """The benchmark base takes a pose; the omniwheel base takes a body-frame twist."""
    from types import SimpleNamespace

    from examples.machine_learning.molmospaces.live_policy import (
        MAX_BASE_SPEED_MPS,
        MAX_BASE_TURN_RADPS,
        apply_action,
    )

    commands: dict[str, object] = {"joints": {}, "base": None}

    sim = SimpleNamespace(
        _move_to=lambda actuator, value: commands["joints"].__setitem__(actuator.name, value),
        base=SimpleNamespace(
            set_velocity=lambda vx, vy, w: commands.__setitem__("base", (vx, vy, w))
        ),
    )
    targets = {
        "base": np.array([0.0, 0.0, 0.0]),
        "lift": np.array([0.8]),
        "arm": np.array([0.3]),
        "wrist": np.array([0.1, 0.2, 0.3]),
        "gripper": np.array([0.5, 0.5]),
    }

    # Facing +y, asked to return to the origin from 0.1m ahead of it: that is
    # 0.1m *backwards* in the base's own frame, and nothing sideways. The yaw
    # target is a quarter turn away, which over one second is faster than the
    # base is allowed to spin, so it comes back clipped.
    commanded = apply_action(
        sim, targets, base_xytheta=np.array([0.0, 0.1, np.pi / 2]), control_period_s=1.0
    )
    forward, left, turn = commands["base"]
    assert forward == pytest.approx(-0.1, abs=1e-6)
    assert left == pytest.approx(0.0, abs=1e-6)
    assert turn == pytest.approx(-MAX_BASE_TURN_RADPS)
    assert commands["joints"]["lift"] == pytest.approx(0.8)
    assert commanded["base_forward"] == pytest.approx(-0.1, abs=1e-6)

    # Sideways, and slow enough to pass through unclipped: facing +x and asked
    # to move to +0.2m in world y is 0.2m to the left over two seconds.
    apply_action(
        sim,
        {**targets, "base": np.array([0.0, 0.2, 0.1])},
        base_xytheta=np.array([0.0, 0.0, 0.0]),
        control_period_s=2.0,
    )
    forward, left, turn = commands["base"]
    assert forward == pytest.approx(0.0, abs=1e-6)
    assert left == pytest.approx(0.1, abs=1e-6)
    assert turn == pytest.approx(0.05, abs=1e-6)

    # A target far enough away to imply an absurd speed is clipped, not obeyed.
    apply_action(
        sim,
        {**targets, "base": np.array([50.0, 0.0, 0.0])},
        base_xytheta=np.array([0.0, 0.0, 0.0]),
        control_period_s=0.066,
    )
    assert commands["base"][0] == pytest.approx(MAX_BASE_SPEED_MPS)


def test_trained_policy_serves_a_chunk_before_re_querying(tmp_path):
    """One network call should cover `execute_chunk_steps` control steps."""
    import torch

    from examples.machine_learning.molmospaces.policies.checkpoint import TrainedPolicy
    from examples.machine_learning.molmospaces.policies.networks import (
        ACTION_DIM,
        StretchBCNet,
    )

    chunk_size = 4
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": StretchBCNet(num_cameras=2, chunk_size=chunk_size).state_dict(),
            "camera_names": ["head_camera", "wrist_camera"],
            "chunk_size": chunk_size,
            "normalisation": {
                "state_mean": np.zeros(STATE_DIM).tolist(),
                "state_std": np.ones(STATE_DIM).tolist(),
                "action_mean": np.zeros(ACTION_DIM).tolist(),
                "action_std": np.ones(ACTION_DIM).tolist(),
            },
        },
        checkpoint,
    )

    policy = TrainedPolicy.load(checkpoint, device="cpu")
    images = {name: np.zeros((48, 64, 3), np.uint8) for name in policy.camera_names}
    state = np.zeros(STATE_DIM, np.float32)
    base = np.zeros(3)

    calls = 0
    original_predict = policy.predict_chunk

    def counting_predict(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_predict(*args, **kwargs)

    policy.predict_chunk = counting_predict
    for _ in range(chunk_size * 2):
        targets = policy.act(images, state, base)
        assert set(targets) == {"base", "lift", "arm", "wrist", "gripper"}
    assert calls == 2, f"expected one network call per chunk of {chunk_size}, got {calls}"

    # A missing camera is an error, not a silently zero-filled input.
    policy.reset()
    with pytest.raises(KeyError):
        policy.act({"head_camera": images["head_camera"]}, state, base)


# =============================================================================
# Viewer camera
# =============================================================================


class _FakeViewerCamera:
    """Stands in for `viewer.cam`, which only exists once a window is open."""

    type = None
    trackbodyid = None
    distance = None
    azimuth = None
    elevation = None


class _FakeViewer:
    def __init__(self) -> None:
        self.cam = _FakeViewerCamera()


def test_viewer_camera_follows_the_robot(compiled_robot):
    """A large scene needs the camera on the robot, not on the whole model."""
    from stretch4_mujoco.mujoco_server_passive import MujocoServerPassive

    model, _, _ = compiled_robot
    server = MujocoServerPassive.__new__(MujocoServerPassive)
    server.mjmodel = model

    viewer = _FakeViewer()
    MujocoServerPassive._track_body_with_viewer_camera(server, viewer, f"{NAMESPACE}base")

    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert viewer.cam.trackbodyid == model.body(f"{NAMESPACE}base").id
    assert viewer.cam.distance > 0
    assert viewer.cam.elevation < 0, "the camera should look down at the robot"


def test_viewer_camera_survives_an_unknown_body(compiled_robot):
    """A bad body name must not stop the simulator from starting."""
    from stretch4_mujoco.mujoco_server_passive import MujocoServerPassive

    model, _, _ = compiled_robot
    server = MujocoServerPassive.__new__(MujocoServerPassive)
    server.mjmodel = model

    viewer = _FakeViewer()
    MujocoServerPassive._track_body_with_viewer_camera(server, viewer, "no_such_body")
    assert viewer.cam.type is None, "the camera should be left at Mujoco's default"


def test_robot_carries_a_chase_camera_for_the_benchmark_viewer(compiled_robot):
    """MolmoSpaces' --viewer can only be aimed via a fixed MJCF camera.

    Without one it keeps Mujoco's whole-model framing, which on a benchmark
    house -- loaded in its "ceiling" variant -- is a sealed building seen from
    tens of metres away with the robot invisible inside it.
    """
    # A concrete subclass: the shared base declares no policy_config, so it is
    # not instantiable on its own.
    from examples.machine_learning.molmospaces.configs import StretchScriptedEvalConfig
    from examples.machine_learning.molmospaces.stretch.robot import CHASE_CAMERA

    model, _, _ = compiled_robot
    camera = model.camera(f"{NAMESPACE}{CHASE_CAMERA}")

    # Mounted on the holonomic base, so it follows the robot's yaw as well as
    # its position -- which a tracking free camera would not do.
    assert model.body(camera.bodyid.item()).name == f"{NAMESPACE}base"
    # Aimed at the robot rather than by a hand-computed quaternion.
    assert camera.mode == mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM
    assert model.body(camera.targetbodyid.item()).name == f"{NAMESPACE}stretch4"
    # Close enough to stay out of the walls of the room the robot works in.
    assert np.linalg.norm(camera.pos[:2]) < 1.2

    assert StretchScriptedEvalConfig().viewer_cam_dict == {"camera": f"{NAMESPACE}{CHASE_CAMERA}"}


def test_episode_starts_stowed():
    """An unstowed spawn puts the gripper inside whatever holds the target.

    The base pose is a 0.55-0.90m standoff and an unstowed tool sits 0.567m in
    front of the base, so the two coincide. Measured over eight MB-Pick episodes
    that spawned the robot interpenetrating the scene five times, by up to 19cm.
    """
    from examples.machine_learning.molmospaces.franka_remapping.episode_overrides import (
        stretch_home_init_qpos,
    )

    init_qpos = stretch_home_init_qpos()
    assert init_qpos["arm"] == [0.0], "the arm must start retracted"
    # Turned away from straight ahead, so the tool is not out in front at all.
    assert abs(init_qpos["wrist"][0]) > 1.5
    # And the stowed tool has to clear the nearest standoff the retarget allows.
    assert TCP_MIN_REACH_M < REACH_BAND_M[0]


def test_reach_solver_unwinds_a_stowed_wrist(robot_config):
    """Damped least squares alone cannot turn the wrist back through zero.

    A wrist that starts turned away from the target sits in a local minimum: the
    yaw column points the wrong way, so the step drives it further round into
    its joint limit. Since Stretch now spawns stowed, every episode's first
    reach starts there, and the retry from a straightened wrist is what makes it
    solvable at all.
    """
    from examples.machine_learning.molmospaces.policies.kinematics import PITCH_HORIZONTAL

    solver = StretchReachSolver(robot_config)
    base_pose = planar_pose(0.0, 0.0, 0.0)
    target = np.array([0.7, 0.0, 0.8])
    stowed = {
        "lift": np.array([0.35]),
        "arm": np.array([0.0]),
        "wrist": np.array([3.14, -0.4, 0.0]),
    }

    # Descending from the stowed wrist alone gets nowhere...
    assert (
        solver._solve_from(
            base_pose, target, PITCH_HORIZONTAL, 0.0, stowed, None, 0.0, 5e-3, 80, 1e-5
        )
        is None
    )
    # ...but `solve` retries from a straightened wrist and reaches the target.
    solution = solver.solve(base_pose, target, wrist_pitch=PITCH_HORIZONTAL, seed=stowed)
    assert solution is not None
    assert np.linalg.norm(solver.forward(solution)[:3, 3] - target) < 6e-3
