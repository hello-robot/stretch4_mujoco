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
from scipy.spatial.transform import Rotation as R  # noqa: E402

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
    TCP_MIN_REACH_M,
    retarget_base_pose,
    stretch_home_init_qpos,
)
from examples.machine_learning.molmospaces.stretch.robot import Stretch4Robot  # noqa: E402
from examples.machine_learning.molmospaces.stretch.robot_view import (  # noqa: E402
    JointTargetClipper,
    Stretch4RobotView,
    commandable_limits,
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


def test_reach_solver_turns_the_base_but_never_drives_it(robot_config):
    """Base yaw is the lateral freedom Stretch has; base translation is not.

    The arm extends along the base's +x axis and the wrist yaw sweeps a narrow
    band either side of it, so a target more than about a quarter radian off that
    axis is out of reach unless the robot turns to face it. The library's local
    IK turns the base and cannot translate it, which is exactly the right pair of
    permissions: re-aiming is free, wandering across the room is not.
    """
    solver = StretchReachSolver(robot_config)
    base_pose = planar_pose(1.0, 2.0, 0.4)
    # Well off the base's axis -- the old position-only solver reached 6.5% of
    # these; it is the base rotation that makes them solvable.
    bearing = 0.4 + 0.65
    target = np.array([1.0 + 0.75 * np.cos(bearing), 2.0 + 0.75 * np.sin(bearing), 0.8])

    solution = solver.solve(base_pose, target, wrist_pitch=PITCH_HORIZONTAL)
    assert solution is not None
    assert np.linalg.norm(solver.forward(solution)[:3, 3] - target) < 6e-3
    assert np.allclose(solution["base"][:2], [1.0, 2.0], atol=1e-9), "the base must not translate"
    assert abs(solution["base"][2] - 0.4) > 1e-2, "the base should have turned to face the target"


def test_top_down_reach_is_solvable(robot_config):
    """Reaching straight down has no wrist-driven lateral freedom of its own."""
    solver = StretchReachSolver(robot_config)
    base_pose = planar_pose(0.0, 0.0, 0.0)
    target = np.array([0.65, 0.25, 0.8])

    solution = solver.solve(base_pose, target, wrist_pitch=PITCH_TOP_DOWN)
    assert solution is not None
    assert np.linalg.norm(solver.forward(solution)[:3, 3] - target) < 6e-3


def test_stretch_wrist_is_a_zyx_euler_triple(robot_config):
    """`R_tcp = Rz(base_yaw + wrist_yaw) Ry(pitch) Rx(roll)` in the base frame, exactly.

    The measurement `policies/kinematics.py` rests on: it is what lets
    `grasp_orientation()` build a target orientation from a grasp style by
    writing the Euler angles down instead of solving for them, and what makes the
    solver's split of a heading between base yaw and wrist yaw orientation-
    preserving. If the wrist's MJCF axes or joint order ever change, this is the
    test that should fail first.
    """
    solver = StretchReachSolver(robot_config)
    rng = np.random.default_rng(0)
    wrist_limits = solver.joint_limits["wrist"]
    base_yaw = 0.7

    worst = 0.0
    for _ in range(50):
        wrist = rng.uniform(wrist_limits[:, 0], wrist_limits[:, 1])
        pose = solver.forward(
            {
                "base": np.array([0.3, -0.2, base_yaw]),
                "lift": np.array([0.5]),
                "arm": np.array([0.2]),
                "wrist": wrist,
            }
        )
        in_base_frame = R.from_euler("z", base_yaw).as_matrix().T @ pose[:3, :3]
        predicted = R.from_euler("ZYX", wrist).as_matrix()
        worst = max(worst, float(np.abs(in_base_frame - predicted).max()))

    assert worst < 1e-9, f"wrist is not a ZYX triple; worst deviation {worst}"


def test_forward_kinematics_matches_the_mujoco_model(robot_config):
    """The Pinocchio FK must agree with the model MolmoSpaces actually steps.

    The two disagree by a fixed ~7cm translation unless the base-frame offset is
    applied (MuJoCo poses the robot by a virtual holonomic base body wrapped
    around the URDF root). A silent regression here would put that error into
    every grasp, so it is asserted rather than assumed.
    """
    import mujoco
    from mujoco import MjSpec

    from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView

    solver = StretchReachSolver(robot_config)

    spec = MjSpec()
    robot_config.robot_cls.add_robot_to_scene(
        robot_config,
        spec,
        prefix=robot_config.robot_namespace,
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        strip_meshes=True,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    view = Stretch4RobotView(data, robot_config.robot_namespace)

    rng = np.random.default_rng(0)
    for _ in range(50):
        configuration = {
            "base": np.array(
                [rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-np.pi, np.pi)]
            ),
            "lift": np.array([rng.uniform(0.0, 1.2)]),
            "arm": np.array([rng.uniform(0.0, 0.52)]),
            "wrist": np.array(
                [rng.uniform(-1.1, 4.2), rng.uniform(-1.1, 1.5), rng.uniform(-1.4, 1.1)]
            ),
        }
        view.base.pose = planar_pose(*configuration["base"])
        for group in ("lift", "arm", "wrist"):
            view.get_move_group(group).joint_pos = configuration[group]
        mujoco.mj_kinematics(model, data)

        expected = np.asarray(view.get_move_group("gripper").leaf_frame_to_world[:3, 3])
        assert np.linalg.norm(solver.forward(configuration)[:3, 3] - expected) < 1e-9


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


def test_the_override_rewrites_the_robot_and_leaves_the_task_alone():
    """The registered hook, end to end: robot, cameras and spawn, nothing else.

    What is being scored has to survive the retarget untouched -- the instruction
    and the object poses are the benchmark -- while everything keyed to the
    authoring robot has to be gone by the time the task sampler is built.
    """
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec

    from examples.machine_learning.molmospaces.configs import StretchDummyEvalConfig
    from examples.machine_learning.molmospaces.stretch.config import HEAD_CAMERA, WRIST_CAMERA_LEFT
    from examples.machine_learning.molmospaces.stretch.episode_overrides import (
        stretch_episode_override,
    )

    spec = EpisodeSpec(
        house_index=101,
        scene_dataset="procthor-objaverse",
        data_split="val",
        robot={
            "robot_name": "franka_droid",
            "init_qpos": {"arm": [0.07, -0.92, -0.13, -2.50, -0.21, 1.41, 0.83]},
        },
        img_resolution=[624, 352],
        cameras=[],
        language={"task_description": "pick up the mug"},
        task={
            "task_cls": "molmo_spaces.tasks.pick_task.PickTask",
            # 0.2m from the target: far too close for Stretch, so the retarget
            # has to move it.
            "robot_base_pose": [12.0, 22.5, 0.17, 1.0, 0.0, 0.0, 0.0],
            "pickup_obj_start_pose": [12.2, 22.5, 1.12, 1.0, 0.0, 0.0, 0.0],
        },
    )
    # Any concrete eval config: the override only touches the camera system.
    exp_config = StretchDummyEvalConfig()

    stretch_episode_override(spec, exp_config)

    assert spec.robot.robot_name == "stretch4"
    assert "arm" in spec.robot.init_qpos and "base" not in spec.robot.init_qpos
    assert spec.robot.init_qpos["arm"] == [0.0], "the arm must start stowed"
    from examples.machine_learning.molmospaces.stretch.config import (
        HEAD_CAMERA,
        HEAD_CAMERA_LEFT,
        HEAD_CAMERA_RIGHT,
        WRIST_CAMERA_LEFT,
        WRIST_CAMERA_RIGHT,
        WRIST_CAMERA_STEREO,
    )
    # Stretch's own cameras, at the episode's resolution.
    assert [camera.name for camera in exp_config.camera_config.cameras] == [
        HEAD_CAMERA,
        WRIST_CAMERA_LEFT,
        WRIST_CAMERA_RIGHT,
        WRIST_CAMERA_STEREO,
        HEAD_CAMERA_LEFT,
        HEAD_CAMERA_RIGHT,
    ]
    assert tuple(exp_config.camera_config.img_resolution) == (624, 352)
    # The spawn moved into the reach band, and nothing else about the task did.
    base_xy = np.array(spec.task["robot_base_pose"][:2])
    target_xy = np.array(spec.task["pickup_obj_start_pose"][:2])
    assert REACH_BAND_M[0] <= np.linalg.norm(base_xy - target_xy) <= REACH_BAND_M[1]
    assert spec.task["robot_base_pose"][2] == 0.0
    assert spec.task["pickup_obj_start_pose"] == [12.2, 22.5, 1.12, 1.0, 0.0, 0.0, 0.0]
    assert spec.language.task_description == "pick up the mug"


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
        WRIST_CAMERA_LEFT,
        WRIST_LEFT_CAMERA_MJCF_NAME,
    )

    assert CAMERA_FOR_TRAINED_NAME[HEAD_CAMERA].camera_name_in_mjcf == HEAD_CAMERA_MJCF_NAME
    assert CAMERA_FOR_TRAINED_NAME[WRIST_CAMERA_LEFT].camera_name_in_mjcf == WRIST_LEFT_CAMERA_MJCF_NAME


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
    from examples.machine_learning.molmospaces.configs import StretchSimpleIKEvalConfig
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

    assert StretchSimpleIKEvalConfig().viewer_cam_dict == {"camera": f"{NAMESPACE}{CHASE_CAMERA}"}


def test_episode_starts_stowed():
    """An unstowed spawn puts the gripper inside whatever holds the target.

    The base pose is a 0.55-0.90m standoff and an unstowed tool sits 0.567m in
    front of the base, so the two coincide. Measured over eight MB-Pick episodes
    that spawned the robot interpenetrating the scene five times, by up to 19cm.
    """
    init_qpos = stretch_home_init_qpos()
    assert init_qpos["arm"] == [0.0], "the arm must start retracted"
    # Turned away from straight ahead, so the tool is not out in front at all.
    assert abs(init_qpos["wrist"][0]) > 1.5
    # And the stowed tool has to clear the nearest standoff the retarget allows.
    assert TCP_MIN_REACH_M < REACH_BAND_M[0]


def test_reach_solver_unwinds_a_stowed_wrist(robot_config):
    """A solve seeded from the stow pose still has to reach forwards.

    Stretch spawns stowed -- wrist yawed right round to 3.14 -- so every
    episode's first reach starts from a configuration pointing away from the
    target. A gradient solver seeded there walks into its joint limit instead of
    back through zero, which is why the library retries from a neutral guess.
    """
    solver = StretchReachSolver(robot_config)
    base_pose = planar_pose(0.0, 0.0, 0.0)
    target = np.array([0.7, 0.0, 0.8])
    stowed = {
        "lift": np.array([0.35]),
        "arm": np.array([0.0]),
        "wrist": np.array([3.14, -0.4, 0.0]),
    }

    solution = solver.solve(base_pose, target, wrist_pitch=PITCH_HORIZONTAL, seed=stowed)
    assert solution is not None
    assert np.linalg.norm(solver.forward(solution)[:3, 3] - target) < 6e-3


def test_grasp_style_fixes_the_approach_axis(robot_config):
    """Grasp style fixes how the tool is tilted, which is what a grasp cares about.

    `R_tcp = Rz(base_yaw + wrist_yaw) @ Ry(pitch) @ Rx(roll)` holds exactly for
    Stretch, so the pitch a solve comes back with is the pitch it was asked for.
    Roll is only pinned away from `PITCH_TOP_DOWN`: at pitch pi/2 the yaw and roll
    axes line up and the pair is degenerate, so the solver may trade one against
    the other and still produce exactly the requested orientation. The invariant
    that survives both cases is the direction the tool closes along.
    """
    solver = StretchReachSolver(robot_config)
    base_pose = planar_pose(0.0, 0.0, 0.0)
    target = np.array([0.72, 0.1, 0.85])

    for pitch in (PITCH_HORIZONTAL, PITCH_TOP_DOWN):
        for roll in (0.0, 0.4):
            solution = solver.solve(base_pose, target, wrist_pitch=pitch, wrist_roll=roll)
            assert solution is not None, f"no solution for pitch={pitch} roll={roll}"
            assert abs(solution["wrist"][1] - pitch) < 1e-3, f"pitch drifted for {pitch}/{roll}"

            approach = solver.forward(solution)[:3, 0]
            if pitch == PITCH_TOP_DOWN:
                assert np.allclose(approach, [0.0, 0.0, -1.0], atol=1e-3)
            else:
                assert abs(approach[2]) < 1e-3, "a horizontal grasp must close level"
                assert abs(solution["wrist"][2] - roll) < 1e-3, "roll is pinned away from lock"


def _library_grasp(approach, closing):
    """A 4x4 grasp pose in the MolmoSpaces grasp-library convention.

    That convention is +z along the approach and +-y along the fingers, taken from
    the gripper markers `molmo_spaces/utils/grasp_sample.py` draws in this frame.
    `closing` only has to be roughly perpendicular to `approach`; it is squared up
    here so the caller can write readable axes and still get a proper rotation.
    """
    approach = np.asarray(approach, float)
    approach = approach / np.linalg.norm(approach)
    closing = np.asarray(closing, float)
    closing = closing - approach * float(closing @ approach)
    closing = closing / np.linalg.norm(closing)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([np.cross(closing, approach), closing, approach])
    pose[:3, 3] = [0.7, 0.05, 0.9]
    return pose


def test_authored_grasp_conversion_reproduces_the_pose_exactly():
    """The library-to-tool conversion is a change of coordinates, not a fit.

    `tcp_orientation_from_grasp` re-expresses an authored grasp in the three
    angles Stretch's kinematics factor into, so feeding the result back through
    `grasp_orientation` has to return the orientation it started from -- including
    at pitch = pi/2, where the yaw/roll split is degenerate but any split of it
    still reconstructs the same rotation.
    """
    from examples.machine_learning.molmospaces.policies.kinematics import (
        GRASP_LIBRARY_TO_TCP,
        grasp_orientation,
        tcp_orientation_from_grasp,
    )

    for approach, closing in (
        ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),  # straight down, the degenerate case
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),  # level, along +x
        ([0.4, -0.8, -0.45], [0.9, 0.44, 0.0]),  # a diagonal with real roll in it
    ):
        pose = _library_grasp(approach, closing)
        angles = tcp_orientation_from_grasp(pose)
        expected = pose[:3, :3] @ GRASP_LIBRARY_TO_TCP
        assert np.allclose(grasp_orientation(*angles), expected, atol=1e-10)


def test_authored_grasp_axes_land_on_the_tool_axes():
    """Approach maps to approach and the closing axis is preserved.

    Stretch's tool frame approaches along +x and separates its fingers along +-y
    (measured on the compiled model: the fingertips sit at (-0.019, +-0.094, 0)).
    A conversion that got either axis wrong would still round-trip through the
    previous test, so this pins the geometry itself.
    """
    from examples.machine_learning.molmospaces.policies.kinematics import (
        grasp_orientation,
        tcp_orientation_from_grasp,
    )

    approach, closing = [0.3, -0.9, -0.31], [0.94, 0.31, 0.0]
    pose = _library_grasp(approach, closing)
    rotation = grasp_orientation(*tcp_orientation_from_grasp(pose))

    assert np.allclose(rotation[:, 0], pose[:3, 2], atol=1e-10), "tool +x must be the approach"
    assert np.allclose(rotation[:, 1], pose[:3, 1], atol=1e-10), "tool +y must be the closing axis"


def test_the_hand_written_styles_are_special_cases_of_authored_grasps():
    """The two styles this policy used to hard-code fall out of the conversion.

    This is the check that the frame transform is the *right* one rather than
    merely a plausible one: a library grasp approaching straight down has to come
    back as exactly `PITCH_TOP_DOWN`, and one approaching along +x as exactly
    `PITCH_HORIZONTAL` with no yaw and no roll.
    """
    from examples.machine_learning.molmospaces.policies.kinematics import (
        tcp_orientation_from_grasp,
    )

    _, pitch, _ = tcp_orientation_from_grasp(_library_grasp([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]))
    assert abs(pitch - PITCH_TOP_DOWN) < 1e-12

    yaw, pitch, roll = tcp_orientation_from_grasp(_library_grasp([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    assert abs(pitch - PITCH_HORIZONTAL) < 1e-12
    assert abs(yaw) < 1e-12 and abs(roll) < 1e-12


def test_authored_grasps_are_reachable_at_the_orientation_they_were_authored_at(robot_config):
    """An authored grasp is only usable if Stretch can hold its whole pose.

    The solve has to be asked for with the yaw pinned, because that is what
    `_command_for` will ask for at execution time -- a candidate accepted with a
    free yaw is one the executor cannot reproduce. Sweeping level approaches
    around a target inside the workspace, enough of them have to solve for the
    per-object search to find one.
    """
    from examples.machine_learning.molmospaces.policies.kinematics import (
        StretchReachSolver,
        tcp_orientation_from_grasp,
    )

    solver = StretchReachSolver(robot_config)
    base_pose = planar_pose(0.0, 0.0, 0.0)

    solved = 0
    bearings = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    for bearing in bearings:
        approach = [np.cos(bearing), np.sin(bearing), 0.0]
        closing = [-np.sin(bearing), np.cos(bearing), 0.0]
        pose = _library_grasp(approach, closing)
        pose[:3, 3] = [0.75, 0.0, 0.95]
        yaw, pitch, roll = tcp_orientation_from_grasp(pose)
        solution = solver.solve(
            base_pose,
            pose[:3, 3],
            wrist_pitch=pitch,
            wrist_roll=roll,
            approach_yaw=yaw,
            yaw_spread=0.0,
        )
        if solution is None:
            continue
        solved += 1
        # The solve honoured the authored orientation rather than substituting a
        # more convenient one, which is the entire point of pinning the yaw.
        achieved = solver.forward(solution)
        assert np.allclose(achieved[:3, 0], pose[:3, 2], atol=5e-3), "approach axis drifted"
        assert np.allclose(achieved[:3, 1], pose[:3, 1], atol=5e-3), "closing axis drifted"

    assert solved >= len(bearings) // 3, f"only {solved}/{len(bearings)} level approaches solved"


def _grasp_check_policy(monkeypatch, drift, tolerance=0.05):
    """A `StretchSimpleIKPolicy` stub wired only for the grasp-follow check.

    Builds nothing and simulates nothing: the check is a comparison between two
    tool-to-object offsets, so the test supplies those directly and asserts on
    the verdict rather than on a whole rollout.
    """
    from examples.machine_learning.molmospaces.policies.simple_ik_policy import StretchSimpleIKPolicy

    policy = StretchSimpleIKPolicy.__new__(StretchSimpleIKPolicy)
    policy._grasp_offset = np.array([0.0, 0.0, -0.02])
    policy._grasp_lost = False

    class _PolicyConfig:
        grasp_slip_tolerance_m = tolerance

    class _Config:
        policy_config = _PolicyConfig()

    policy.config = _Config()
    monkeypatch.setattr(
        StretchSimpleIKPolicy,
        "_tool_to_object_offset",
        lambda self, tool: np.array([0.0, 0.0, -0.02]) + drift,
    )
    return policy


def test_grasp_is_held_when_the_object_rides_with_the_tool(monkeypatch):
    policy = _grasp_check_policy(monkeypatch, drift=np.array([0.002, 0.0, 0.003]))
    assert policy._grasp_still_held(np.zeros(3)) is True


def test_grasp_is_lost_when_the_object_stays_behind(monkeypatch):
    """The failure this exists to catch: the arm lifts, the object does not."""
    policy = _grasp_check_policy(monkeypatch, drift=np.array([0.0, 0.0, -0.18]))
    assert policy._grasp_still_held(np.zeros(3)) is False


def test_grasp_check_passes_when_there_is_nothing_to_check(monkeypatch):
    """A task with no pickup object must not be failed by a pick-specific check."""
    from examples.machine_learning.molmospaces.policies.simple_ik_policy import StretchSimpleIKPolicy

    policy = _grasp_check_policy(monkeypatch, drift=np.array([0.0, 0.0, -0.18]))
    policy._grasp_offset = None
    assert policy._grasp_still_held(np.zeros(3)) is True

    policy._grasp_offset = np.array([0.0, 0.0, -0.02])
    monkeypatch.setattr(
        StretchSimpleIKPolicy, "_tool_to_object_offset", lambda self, tool: None
    )
    assert policy._grasp_still_held(np.zeros(3)) is True


def test_grasp_offset_in_tool_frame_is_invariant_to_pitch_rotation():
    """Rotating the gripper (e.g. pitching up 90 deg) must not trigger false grasp slip."""
    from scipy.spatial.transform import Rotation as R
    from examples.machine_learning.molmospaces.policies.simple_ik_policy import StretchSimpleIKPolicy

    class MockMoveGroup:
        def __init__(self, pose):
            self.leaf_frame_to_world = pose

    class MockRobotView:
        def __init__(self, pose):
            self._mg = MockMoveGroup(pose)
        def get_move_group(self, name):
            return self._mg

    class MockTaskConfig:
        pickup_obj_name = "test_obj"

    class MockConfig:
        task_config = MockTaskConfig()
        class policy_config:
            grasp_slip_tolerance_m = 0.05

    policy = StretchSimpleIKPolicy.__new__(StretchSimpleIKPolicy)
    policy.config = MockConfig()
    policy._grasp_lost = False

    # Grasp established with tool pointing down (pitch = 90 deg)
    r0 = R.from_euler("y", 90, degrees=True).as_matrix()
    p0 = np.array([0.5, 0.0, 0.8])
    pose0 = np.eye(4)
    pose0[:3, :3] = r0
    pose0[:3, 3] = p0

    # Object is 3cm ahead of TCP in local tool coordinates: p_obj = p + R @ [0, 0, 0.03]
    obj_local = np.array([0.0, 0.0, 0.03])
    obj_pos_0 = p0 + r0 @ obj_local
    policy._object_grasp_point = lambda name: obj_pos_0

    policy._grasp_offset = policy._tool_to_object_offset(MockRobotView(pose0))
    np.testing.assert_allclose(policy._grasp_offset, obj_local, atol=1e-5)

    # Now tool pitches up to horizontal (pitch = 0 deg) and lifts
    r1 = R.from_euler("y", 0, degrees=True).as_matrix()
    p1 = np.array([0.4, 0.0, 1.1])
    pose1 = np.eye(4)
    pose1[:3, :3] = r1
    pose1[:3, 3] = p1

    # Object moved rigidly with the gripper
    obj_pos_1 = p1 + r1 @ obj_local
    policy._object_grasp_point = lambda name: obj_pos_1

    assert policy._grasp_still_held(MockRobotView(pose1)) is True


def test_lift_waypoint_is_the_one_that_verifies_the_grasp():
    """The plan must actually carry the flags the check keys off."""
    from examples.machine_learning.molmospaces.policies.simple_ik_policy import Waypoint

    closing = Waypoint(
        position=np.zeros(3),
        wrist_pitch=0.0,
        gripper_open=False,
        label="close",
        establishes_grasp=True,
    )
    lifting = Waypoint(
        position=np.zeros(3),
        wrist_pitch=0.0,
        gripper_open=False,
        label="lift",
        verify_grasp=True,
    )
    assert closing.establishes_grasp and not closing.verify_grasp
    assert lifting.verify_grasp and not lifting.establishes_grasp
    # Everything else must default to not participating.
    plain = Waypoint(position=np.zeros(3), wrist_pitch=0.0, gripper_open=True, label="reach")
    assert not plain.establishes_grasp and not plain.verify_grasp


def test_head_camera_renders_upright(robot_config):
    """The head camera must match the orientation hardware delivers.

    Stretch 4's head cameras are mounted rotated and the robot's own driver
    undoes it (`StatusStretchCamera.get_camera_data` applies
    `np.rot90(data, rotate_number_of_times)` by default). Simulation has to agree,
    or a policy trained here meets a quarter-turned world on the real robot.

    Asserted on a rendered image rather than on the quaternion, because the
    quaternion is the thing that could be right in the wrong direction.
    """
    import mujoco
    from mujoco import MjSpec

    from examples.machine_learning.molmospaces.stretch.robot_view import Stretch4RobotView
    from stretch4_mujoco.enums.stretch_cameras import StretchCameras

    spec = MjSpec()
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.1], rgba=[0.5, 0.5, 0.5, 1]
    )
    # A post that is unambiguously vertical in the world.
    post = spec.worldbody.add_body(pos=[1.4, 0.0, 0.9])
    post.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.06, 0.9], rgba=[1, 0, 0, 1])

    robot_config.robot_cls.add_robot_to_scene(
        robot_config,
        spec,
        prefix=robot_config.robot_namespace,
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        strip_meshes=False,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    view = Stretch4RobotView(data, robot_config.robot_namespace)
    view.get_move_group("lift").joint_pos = np.array([0.7])
    view.get_move_group("arm").joint_pos = np.array([0.3])
    view.get_move_group("wrist").joint_pos = np.zeros(3)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, 368, 640)
    renderer.update_scene(data, camera=f"{robot_config.robot_namespace}camera_center_link")
    raw = renderer.render()
    rot = StretchCameras.cam_nav_rgb_se4_center.initial_camera_settings.rotate_number_of_times
    image = np.rot90(raw, rot)

    assert image.shape == (640, 368, 3), "rotated head camera frame should be portrait (640x368)"

    red = (
        (image[:, :, 0].astype(int) > 90)
        & (image[:, :, 1].astype(int) < 60)
        & (image[:, :, 2].astype(int) < 60)
    )
    assert red.sum() > 20, "the post should be in view"
    rows = np.where(red.any(axis=1))[0]
    columns = np.where(red.any(axis=0))[0]
    height = rows.max() - rows.min() + 1
    width = columns.max() - columns.min() + 1
    assert height > width, (
        f"a vertical post renders {height}px tall x {width}px wide -- the head camera is still "
        "rotated relative to what the hardware driver produces"
    )


def test_gripper_cameras_are_left_alone(robot_config):
    """Only cameras the hardware rotates get rotated; the wrist ones do not."""
    from stretch4_mujoco.enums.stretch_cameras import StretchCameras

    from examples.machine_learning.molmospaces.stretch.robot import HARDWARE_CAMERA_EQUIVALENTS

    assert "gripper_camera_left_rgb" not in HARDWARE_CAMERA_EQUIVALENTS
    for hardware_name in HARDWARE_CAMERA_EQUIVALENTS.values():
        settings = StretchCameras[hardware_name].initial_camera_settings
        assert settings.rotate_number_of_times != 0, (
            f"{hardware_name} needs no correction and should not be listed"
        )


def test_thin_object_grasp_offset():
    """Thin objects like a fork get a vertical offset so fingertips grasp at their tips without table collision."""
    from unittest.mock import MagicMock
    from examples.machine_learning.molmospaces.policies.simple_ik_policy import (
        StretchSimpleIKPolicy,
        TCP_TO_FINGERTIP_EDGE_M,
    )

    policy = StretchSimpleIKPolicy.__new__(StretchSimpleIKPolicy)
    policy.config = MagicMock()
    policy.config.task_config.pickup_obj_name = "fork"

    mock_scene_obj = MagicMock()
    mock_scene_obj.position = np.array([0.5, 0.2, 0.75])
    mock_scene_obj.body_id = 1
    mock_scene_obj.aabb_size = np.array([0.02, 0.15, 0.01])  # 1cm tall

    mock_env = MagicMock()
    mock_env.current_batch_index = 0
    mock_obj_mgr = MagicMock()
    mock_obj_mgr.get_object_by_name.return_value = mock_scene_obj
    mock_env.object_managers = [mock_obj_mgr]
    mock_env.current_model.ngeom = 0
    mock_env.current_data.geom_xpos = []

    mock_task = MagicMock()
    mock_task.env = mock_env
    policy.task = mock_task

    grasp_pt = policy._object_grasp_point("fork")
    expected_offset = TCP_TO_FINGERTIP_EDGE_M
    assert grasp_pt[2] == pytest.approx(0.75 + expected_offset)


def test_grasp_target_bounds_validation():
    """Grasp targets must be rejected if underneath the object/table or out of bounds."""
    from unittest.mock import MagicMock
    from examples.machine_learning.molmospaces.policies.simple_ik_policy import StretchSimpleIKPolicy

    policy = StretchSimpleIKPolicy.__new__(StretchSimpleIKPolicy)
    mock_scene_obj = MagicMock()
    mock_scene_obj.position = np.array([0.5, 0.2, 0.75])
    mock_scene_obj.aabb_size = np.array([0.02, 0.20, 0.02])  # pencil: x=[-0.01, 0.01], y=[-0.1, 0.1], z=[-0.01, 0.01]
    # world bounds: x=[0.49, 0.51], y=[0.10, 0.30], z=[0.74, 0.76]

    mock_env = MagicMock()
    mock_env.current_batch_index = 0
    mock_obj_mgr = MagicMock()
    mock_obj_mgr.get_object_by_name.return_value = mock_scene_obj
    mock_env.object_managers = [mock_obj_mgr]
    mock_env.current_model = None
    mock_env.current_data = None

    mock_task = MagicMock()
    mock_task.env = mock_env
    policy.task = mock_task

    # 1. Valid grasp in the middle of the pencil
    assert policy._is_grasp_within_bounds("pencil", np.array([0.50, 0.20, 0.75])) is True

    # 2. Grasp underneath the pencil / table (z = 0.70 < 0.74 - 0.005) -> False
    assert policy._is_grasp_within_bounds("pencil", np.array([0.50, 0.20, 0.70])) is False

    # 3. Grasp far away laterally (x = 0.70 > 0.51 + 0.04) -> False
    assert policy._is_grasp_within_bounds("pencil", np.array([0.70, 0.20, 0.75])) is False

    # 4. Grasp far away laterally along y (y = 0.40 > 0.30 + 0.04) -> False
    assert policy._is_grasp_within_bounds("pencil", np.array([0.50, 0.40, 0.75])) is False

    # 5. Grasp floating way above the pencil (z = 0.85 > 0.76 + 0.04) -> False
    assert policy._is_grasp_within_bounds("pencil", np.array([0.50, 0.20, 0.85])) is False


def test_tall_object_pitch_search_order(robot_config):
    """For objects taller than 100mm, pitch angle search starts at 45 deg down (45->0 then 90->45)."""
    from unittest.mock import MagicMock
    from examples.machine_learning.molmospaces.policies.simple_ik_policy import StretchSimpleIKPolicy
    from scipy.spatial.transform import Rotation as R

    policy = StretchSimpleIKPolicy.__new__(StretchSimpleIKPolicy)
    mock_config = MagicMock()
    mock_config.policy_config.grasp_style = "top_down"
    policy.config = mock_config
    # A real solver rather than a mock: candidate construction asks it for the
    # wrist's limits, and the point of reading those off the compiled model is
    # lost if the test supplies its own numbers.
    policy._solver = StretchReachSolver(robot_config)

    mock_gripper = MagicMock()
    mock_gripper.inter_finger_dist_range = (0.0, 0.1885)
    policy._gripper_group = MagicMock(return_value=mock_gripper)

    mock_robot_view = MagicMock()
    mock_robot_view.base.pose = np.eye(4)
    mock_task = MagicMock()
    mock_task.env.current_robot.robot_view = mock_robot_view
    policy.task = mock_task

    # Mock object with height = 0.25m (> 100mm)
    policy._object_bounds = MagicMock(return_value=(np.array([0, 0, 0]), np.array([0.1, 0.1, 0.25])))
    policy._object_grasp_width = MagicMock(return_value=0.03)

    # Test pitches: 90 deg, 60 deg, 45 deg, 30 deg, 0 deg
    from examples.machine_learning.molmospaces.policies.kinematics import GRASP_LIBRARY_TO_TCP, grasp_orientation

    test_pitches = [np.pi / 2, np.pi / 3, np.pi / 4, np.pi / 6, 0.0]
    poses = []
    for pitch in test_pitches:
        mat = np.eye(4)
        mat[:3, :3] = grasp_orientation(0.0, pitch, 0.0) @ GRASP_LIBRARY_TO_TCP.T
        poses.append(mat)

    policy._library_grasp_poses = MagicMock(return_value=np.array(poses))

    solved_pitches = []
    def mock_solve_at(base_pose, pos, candidate):
        solved_pitches.append(candidate.wrist_pitch)
        return None  # return None so it evaluates all candidates in order

    policy._solve_at = mock_solve_at

    policy._authored_grasp("tall_bottle", np.zeros(3))

    # Expected order: 45 -> 0 (pi/4, pi/6, 0.0) then 90 -> 45 (pi/2, pi/3)
    expected_order = [np.pi / 4, np.pi / 6, 0.0, np.pi / 2, np.pi / 3]
    assert len(solved_pitches) == len(expected_order)
    for actual, expected in zip(solved_pitches, expected_order):
        assert actual == pytest.approx(expected, abs=1e-4)

    # For short objects (< 100mm), expected order is standard descending: 90 -> 0
    policy._object_bounds = MagicMock(return_value=(np.array([0, 0, 0]), np.array([0.1, 0.1, 0.05])))
    solved_pitches.clear()
    policy._authored_grasp("short_mug", np.zeros(3))
    expected_short_order = [np.pi / 2, np.pi / 3, np.pi / 4, np.pi / 6, 0.0]
    assert len(solved_pitches) == len(expected_short_order)
    for actual, expected in zip(solved_pitches, expected_short_order):
        assert actual == pytest.approx(expected, abs=1e-4)


def test_object_local_target_tracking_verification():
    """Verify that target grasp point is tracked relative to object origin and verified after lift."""
    from unittest.mock import MagicMock
    from scipy.spatial.transform import Rotation as R
    from examples.machine_learning.molmospaces.policies.simple_ik_policy import StretchSimpleIKPolicy, Waypoint, ToolGrasp

    class MockMoveGroup:
        def __init__(self, pose):
            self.leaf_frame_to_world = pose

    class MockRobotView:
        def __init__(self, pose):
            self._mg = MockMoveGroup(pose)
        def get_move_group(self, name):
            return self._mg

    policy = StretchSimpleIKPolicy.__new__(StretchSimpleIKPolicy)
    mock_config = MagicMock()
    mock_config.task_config.pickup_obj_name = "can"
    mock_config.policy_config.grasp_slip_tolerance_m = 0.05
    policy.config = mock_config

    # Object starts at origin [0.5, 0.2, 0.8] with no rotation
    obj_pos_0 = np.array([0.5, 0.2, 0.8])
    obj_rot_0 = np.eye(3)
    current_obj_pos = obj_pos_0.copy()
    current_obj_rot = obj_rot_0.copy()

    policy._object_pose = lambda name: (current_obj_pos, current_obj_rot)

    # Grasp target is on the handle: offset by [0.02, 0.0, 0.05] relative to object origin
    target_pos = obj_pos_0 + np.array([0.02, 0.0, 0.05])
    policy._grasp = ToolGrasp(
        position=target_pos,
        approach_yaw=0.0,
        wrist_pitch=0.0,
        wrist_roll=0.0,
        authored=True,
    )

    # TCP reaches the target
    tcp_pose_0 = np.eye(4)
    tcp_pose_0[:3, 3] = target_pos
    robot_view_0 = MockRobotView(tcp_pose_0)

    # Record grasp state
    close_waypoint = Waypoint(
        position=target_pos,
        wrist_pitch=0.0,
        label="close",
        gripper_open=False,
        establishes_grasp=True,
    )
    policy._record_grasp_state(robot_view_0, close_waypoint)

    # Verify that target was stored relative to object frame
    assert policy._grasp_target_rel_obj is not None
    np.testing.assert_allclose(policy._grasp_target_rel_obj, np.array([0.02, 0.0, 0.05]))
    assert policy._grasp_init_tcp_dist == pytest.approx(0.0, abs=1e-5)

    # Scenario 1: Object is successfully lifted (lifted by +0.2m and rotated 30 deg around z)
    rot_lift = R.from_euler("z", 30, degrees=True).as_matrix()
    current_obj_rot = rot_lift
    current_obj_pos = obj_pos_0 + np.array([0.0, 0.0, 0.2])

    # Gripper moves with the object
    tcp_pose_lifted = np.eye(4)
    tcp_pose_lifted[:3, :3] = rot_lift
    # TCP should be at new target position in world
    tcp_pose_lifted[:3, 3] = current_obj_pos + rot_lift @ np.array([0.02, 0.0, 0.05])
    robot_view_lifted = MockRobotView(tcp_pose_lifted)

    assert policy._grasp_still_held(robot_view_lifted) is True

    # Scenario 2: Object dropped / stayed on table while TCP lifted
    current_obj_pos = obj_pos_0.copy()  # back on table
    current_obj_rot = np.eye(3)
    assert policy._grasp_still_held(robot_view_lifted) is False


def test_unstow_joint_group_reached():
    """Verify that joint groups reach unstow targets without timing out on small steady-state errors."""
    from examples.machine_learning.molmospaces.policies.simple_ik_policy import StretchSimpleIKPolicy

    policy = StretchSimpleIKPolicy.__new__(StretchSimpleIKPolicy)

    class MockMoveGroup:
        def __init__(self, pos, vel=None):
            self.joint_pos = np.array(pos)
            if vel is not None:
                self.joint_vel = np.array(vel)

    # 1. Wrist yaw unwinding from 3.14 to near 0.0 (e.g. at 0.12 rad, pitch=0.01, roll=0.01)
    wrist_mg = MockMoveGroup([0.12, 0.01, 0.01])
    assert policy._joint_group_reached("wrist", wrist_mg, np.array([0.0, 0.0, 0.0]), 0.05) is True

    # 2. Wrist yaw settled at 0.30 rad with near-zero velocity
    wrist_settled = MockMoveGroup([0.30, 0.05, 0.02], vel=[0.001, 0.001, 0.001])
    assert policy._joint_group_reached("wrist", wrist_settled, np.array([0.0, 0.0, 0.0]), 0.05) is True

    # 3. Wrist still at stowed yaw (3.14 rad) -> not reached
    wrist_stowed = MockMoveGroup([3.14, 0.0, 0.0])
    assert policy._joint_group_reached("wrist", wrist_stowed, np.array([0.0, 0.0, 0.0]), 0.05) is False

    # 4. Lift reaching target within tolerance
    lift_mg = MockMoveGroup([0.72])
    assert policy._joint_group_reached("lift", lift_mg, np.array([0.70]), 0.05) is True


def test_joint_targets_are_clipped_to_the_models_limits(compiled_robot):
    """A joint-space waypoint cannot ask for a position the model forbids.

    The failure this pins down: a top-down authored grasp decomposes to a wrist
    roll outside the joint's asymmetric range, `unstow` commanded it verbatim,
    and `JointPosController` clipped it -- leaving the joint parked at its limit
    with zero velocity while the policy waited out its whole step budget for an
    arrival that could not happen. Nothing was colliding; the target simply did
    not exist.
    """
    from examples.machine_learning.molmospaces.policies.simple_ik_policy import (
        StretchSimpleIKPolicy,
    )

    _, _, robot_view = compiled_robot
    clipper = JointTargetClipper(robot_view)

    roll_limits = commandable_limits(robot_view.get_move_group("wrist"))[2]
    assert roll_limits[1] < np.pi, (
        "this test is only meaningful while the wrist roll range is asymmetric and "
        f"narrower than a half turn; the model now says {roll_limits}"
    )

    out_of_range = float(roll_limits[1]) + 0.9
    commanded = clipper.clip_group("wrist", [0.0, 0.5, out_of_range])
    assert commanded[2] == pytest.approx(roll_limits[1])

    # The joint can sit at its limit, so measuring against the commanded target
    # is what lets the waypoint finish. Against the raw target it never would.
    wrist = robot_view.get_move_group("wrist")
    restore = np.asarray(wrist.joint_pos, dtype=float).copy()
    try:
        wrist.joint_pos = np.array([0.0, 0.5, float(roll_limits[1])])
        policy = StretchSimpleIKPolicy.__new__(StretchSimpleIKPolicy)
        assert policy._joint_group_reached("wrist", wrist, commanded, 0.05) is True
        assert (
            policy._joint_group_reached("wrist", wrist, np.array([0.0, 0.5, out_of_range]), 0.05)
            is False
        )
    finally:
        # `compiled_robot` is module-scoped, so anything set here outlives the test.
        wrist.joint_pos = restore


def test_unstow_finishes_instead_of_waiting_out_its_budget(compiled_robot):
    """The observed stall, end to end: `unstow` advances, it does not time out.

    Every number here was read off a stalled episode. The mast was asked for
    1.23m and sat at 1.199 (the MJCF stops at 1.2); the wrist was asked for a
    roll of 2.020 and sat at 1.130 (the MJCF stops at 1.135), velocity zero on
    every joint, nothing in contact. That configuration is *arrived*, and the
    waypoint has to be able to say so.
    """
    from unittest.mock import MagicMock

    from examples.machine_learning.molmospaces.policies.simple_ik_policy import (
        StretchSimpleIKPolicy,
        Waypoint,
    )

    _, _, robot_view = compiled_robot
    stalled = {
        "lift": np.array([1.199]),
        "arm": np.array([0.001]),
        "wrist": np.array([0.005, 1.547, 1.130]),
    }
    # `compiled_robot` is module-scoped, so put back whatever was there.
    restore = {
        group: np.asarray(robot_view.get_move_group(group).joint_pos, dtype=float).copy()
        for group in stalled
    }
    try:
        for group, value in stalled.items():
            robot_view.get_move_group(group).joint_pos = value

        policy = StretchSimpleIKPolicy.__new__(StretchSimpleIKPolicy)
        policy._clipper_cache = None
        policy._waypoint_index = 0
        policy._steps_in_waypoint = 0
        policy._settled_steps = 0
        policy.config = MagicMock()
        policy.config.policy_config.max_steps_per_waypoint = 120
        policy.task = MagicMock()
        policy.task.env.current_robot.robot_view = robot_view

        waypoint = Waypoint(
            position=np.array([0.4, 0.0, 1.1]),
            wrist_pitch=1.552,
            wrist_roll=2.020,
            gripper_open=True,
            label="unstow",
            joint_targets={
                "lift": np.array([1.23]),
                "arm": np.array([0.0]),
                "wrist": np.array([0.0, 1.552, 2.020]),
            },
        )

        for step in range(3):
            policy._advance(waypoint, robot_view)
            if policy._waypoint_index == 1:
                break
        assert policy._waypoint_index == 1, (
            "unstow did not advance from a settled, at-the-limit configuration -- it "
            f"would burn all {policy.config.policy_config.max_steps_per_waypoint} steps"
        )
        assert step < 2
    finally:
        for group, value in restore.items():
            robot_view.get_move_group(group).joint_pos = value


def test_solver_limits_come_from_the_model(robot_config, compiled_robot):
    """The IK's joint limits are the compiled model's, not numbers written in Python.

    A solver limit wider than the model's produces solutions the controller
    clips: the arm holds its commanded configuration, the tool sits short of the
    waypoint, and no amount of settling closes the gap. The mast is the one that
    bit -- the old override said 1.23m where both the URDF and the MJCF say 1.2.
    """
    _, _, robot_view = compiled_robot
    solver = StretchReachSolver(robot_config)
    for group in ("lift", "arm", "wrist"):
        assert solver.joint_limits[group] == pytest.approx(
            commandable_limits(robot_view.get_move_group(group))
        ), f"{group} limits have drifted from the model"



