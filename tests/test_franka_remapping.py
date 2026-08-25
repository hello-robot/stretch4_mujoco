"""
Fast checks for the Franka -> Stretch 4 remapping in
`examples/machine_learning/molmospaces/franka_remapping/` and the fine-tuning
export in `.../finetuning/`.

These cover the claims the rest of the code is built on -- Stretch's wrist really
is a ZYX Euler triple, the Franka FK/IK round-trips, a retargeted pose comes back
where it was asked for, and the two directions of the action remap are inverses --
plus the two encodings the fine-tuning export writes. Scoring an actual benchmark
needs assets, a GPU and an inference server, and lives in `run_benchmarks.py`.

Requires the optional MolmoSpaces dependency:  pip install -e ".[molmo]"
"""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("molmo_spaces")

from scipy.spatial.transform import Rotation as R  # noqa: E402

from examples.machine_learning.molmospaces.franka_remapping import episode_frame  # noqa: E402
from examples.machine_learning.molmospaces.franka_remapping.action_remap import (  # noqa: E402
    FrankaActionRemapper,
)
from examples.machine_learning.molmospaces.franka_remapping.franka_arm import (  # noqa: E402
    HOME_QPOS,
    FrankaArm,
    franka_model_path,
)
from examples.machine_learning.molmospaces.franka_remapping.pose_solver import (  # noqa: E402
    EXACT_ORIENTATION_DOFS,
    FREE_AZIMUTH_DOFS,
    StretchPoseSolver,
    fit_wrist,
)
from examples.machine_learning.molmospaces.stretch.config import (  # noqa: E402
    Stretch4RobotConfig,
)

pytestmark = pytest.mark.skipif(
    not franka_model_path().exists(),
    reason="the franka_droid MJCF is part of the MolmoSpaces resource bundle",
)


@pytest.fixture(scope="module")
def robot_config():
    return Stretch4RobotConfig()


@pytest.fixture(scope="module")
def solver(robot_config):
    return StretchPoseSolver(robot_config)


@pytest.fixture(scope="module")
def franka():
    return FrankaArm()


def _pose(position, rotation=None) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = position
    if rotation is not None:
        pose[:3, :3] = rotation
    return pose


# =============================================================================
# The kinematic claim the pose solve rests on
# =============================================================================


def test_stretch_wrist_is_a_zyx_euler_triple(solver):
    """`R_tool = Rz(yaw) Ry(pitch) Rx(roll)` in the base frame, exactly.

    This is the load-bearing measurement of the whole module: it is what lets
    `fit_wrist` read a requested orientation straight off as Euler angles instead
    of solving for it, and what makes the yaw split orientation-preserving. If
    the wrist's MJCF axes or joint order ever change, this is the test that
    should fail first.
    """
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


def test_fit_wrist_reproduces_the_requested_orientation(solver):
    """Every candidate representation is exact, so the chosen one has to be too."""
    rng = np.random.default_rng(1)
    wrist_limits = solver.joint_limits["wrist"]

    for _ in range(50):
        base_yaw = float(rng.uniform(-np.pi, np.pi))
        requested = R.from_euler(
            "ZYX",
            [rng.uniform(-1.0, 1.0) + base_yaw, rng.uniform(-0.4, 1.4), rng.uniform(-1.0, 1.0)],
        ).as_matrix()
        wrist = fit_wrist(requested, base_yaw, wrist_limits)
        achieved = R.from_euler("ZYX", [base_yaw + wrist[0], wrist[1], wrist[2]]).as_matrix()
        assert np.allclose(achieved, requested, atol=1e-6)


def test_fit_wrist_prefers_a_wrist_the_arm_can_reach_out_of(solver):
    """The mirror branch is legal on this wrist but folds the gripper backwards.

    A pitch past `NATURAL_PITCH_RANGE` costs about 0.3m of forward reach, which
    is enough to turn a reachable grasp into a saturated arm, so the canonical
    branch has to win when both are in limits.
    """
    wrist_limits = solver.joint_limits["wrist"]
    # A gentle forward-and-down approach: nothing here needs a folded wrist.
    requested = R.from_euler("ZYX", [0.0, 0.5, 0.0]).as_matrix()
    wrist = fit_wrist(requested, 0.0, wrist_limits)
    assert wrist[1] == pytest.approx(0.5, abs=1e-6)


# =============================================================================
# The Franka arm
# =============================================================================


def test_franka_forward_kinematics_matches_the_authored_home_pose(franka):
    """FK at `FrankaRobotConfig.init_qpos` puts the tool where the MJCF does.

    0.307m forward and 0.435m up of `fr3_link0`, gripper pointing down. Pinned
    because every absolute pose the remap produces is measured from this frame.
    """
    pose = franka.forward(HOME_QPOS)
    assert pose[0, 3] == pytest.approx(0.3069, abs=1e-3)
    assert pose[1, 3] == pytest.approx(0.0, abs=1e-3)
    assert pose[2, 3] == pytest.approx(0.4353, abs=1e-3)
    # The tool's approach axis is its own +z, pointing down at home.
    assert pose[2, 2] == pytest.approx(-1.0, abs=1e-3)


def test_franka_inverse_kinematics_round_trips(franka):
    """FK then IK returns the configuration it started from."""
    qpos = HOME_QPOS + np.array([0.1, -0.1, 0.05, 0.2, -0.05, 0.1, 0.3])
    solution = franka.inverse(franka.forward(qpos), seed=qpos)
    assert solution.converged
    assert solution.position_error < 1e-3
    assert np.abs(solution.qpos - qpos).max() < 1e-3


def test_franka_inverse_kinematics_retries_from_home(franka):
    """A warm start on the wrong side of the self-motion manifold must not stick.

    Damped least squares only walks downhill, so a bad seed sits in a local
    minimum forever; the retry from home is what keeps the occasional stuck step
    from dominating a dataset's statistics.
    """
    target = franka.forward(np.array([1.2, -0.4, 0.3, -1.8, 0.5, 1.2, -0.6]))
    adversarial_seed = np.array([-1.2, 0.4, -0.3, -2.9, -0.5, 3.0, 0.6])
    solution = franka.inverse(target, seed=adversarial_seed)
    assert solution.position_error < 5e-3


def test_franka_gripper_conventions_round_trip(franka):
    """0 is open and 1 is closed, on the scale the reference clients use."""
    assert franka.gripper_closedness_from_qpos(0.0) == pytest.approx(0.0)
    assert franka.gripper_closedness_from_qpos(0.824033) == pytest.approx(1.0)
    for closedness in (0.0, 0.25, 0.5, 1.0):
        recovered = franka.gripper_closedness_from_qpos(
            franka.gripper_qpos_from_closedness(closedness)
        )
        assert recovered == pytest.approx(closedness, abs=1e-6)
    # Aperture shrinks as the gripper closes.
    assert franka.gripper_aperture_m(0.0) > franka.gripper_aperture_m(1.0)


# =============================================================================
# The pose solve
# =============================================================================


def test_pose_solve_recovers_a_reachable_pose(solver):
    """Poses generated by forward kinematics come back exactly."""
    rng = np.random.default_rng(2)
    base = _pose([1.5, -2.0, 0.0], R.from_euler("z", 0.8).as_matrix())

    for _ in range(25):
        configuration = {
            "base": np.array([1.5, -2.0, 0.8]),
            "lift": np.array([rng.uniform(0.25, 1.1)]),
            "arm": np.array([rng.uniform(0.05, 0.45)]),
            "wrist": np.array(
                [rng.uniform(-0.7, 0.7), rng.uniform(-0.3, 1.4), rng.uniform(-0.8, 0.8)]
            ),
        }
        target = solver.forward(configuration).copy()
        solution = solver.solve(base, target, dofs=EXACT_ORIENTATION_DOFS)
        assert solution.converged
        assert solution.position_error < 5e-3
        assert solution.orientation_error < 1e-3


def test_pose_solve_always_returns_something(solver):
    """An unreachable pose gets the nearest reachable one, not a None.

    A pose solve feeds controllers that have to be given a target every step, so
    the useful answer to "that is out of reach" is the closest pose plus a
    residual the caller can log.
    """
    base = _pose([0.0, 0.0, 0.0])
    # 4m up is not reachable by any configuration.
    solution = solver.solve(base, _pose([0.6, 0.0, 4.0]), dofs=EXACT_ORIENTATION_DOFS)
    assert not solution.converged
    assert solution.position_error > 1.0
    assert set(solution.configuration) == {"base", "lift", "arm", "wrist"}
    assert np.all(np.isfinite(solution.configuration["lift"]))


# A pose Stretch can hold with its base yawed 0.3 rad and its wrist straight. Asked
# for from a base at yaw 0, the only way to match both the position and the
# azimuth is the split: turn the base by 0.3 and take it back off the wrist.
_SPLIT_CONFIGURATION = {
    "base": np.array([0.0, 0.0, 0.3]),
    "lift": np.array([0.7]),
    "arm": np.array([0.2]),
    "wrist": np.array([0.0, 0.6, 0.0]),
}


def test_yaw_split_preserves_orientation_while_reaching(solver):
    """The exact-orientation mode may turn the base but must not turn the tool."""
    target = solver.forward(_SPLIT_CONFIGURATION).copy()
    solution = solver.solve(_pose([0.0, 0.0, 0.0]), target, dofs=EXACT_ORIENTATION_DOFS)

    assert solution.converged
    assert solution.orientation_error < 1e-3
    assert solution.base_rotation == pytest.approx(0.3, abs=1e-2), "expected the split"
    # And the split really did come back off the wrist rather than being added to it.
    assert solution.configuration["wrist"][0] == pytest.approx(0.0, abs=1e-2)


def test_free_azimuth_mode_does_not_turn_the_base(solver):
    """The alternative mode trades orientation for a still base, by construction."""
    target = solver.forward(_SPLIT_CONFIGURATION).copy()
    solution = solver.solve(_pose([0.0, 0.0, 0.0]), target, dofs=FREE_AZIMUTH_DOFS)
    assert solution.base_rotation == pytest.approx(0.0, abs=1e-9)
    # It reaches the position with the wrist instead, and pays in orientation.
    assert solution.position_error < 5e-3


def test_pose_solve_holds_the_base_position_by_default(solver):
    """`max_base_translation` defaults to zero, so the standoff is preserved."""
    base = _pose([1.0, 2.0, 0.0])
    solution = solver.solve(base, _pose([1.6, 2.0, 0.85]), dofs=EXACT_ORIENTATION_DOFS)
    assert solution.base_translation == pytest.approx(0.0, abs=1e-9)


# =============================================================================
# The action remap, both directions
# =============================================================================


@pytest.fixture
def remapper(robot_config):
    episode_frame.clear()
    return FrankaActionRemapper(robot_config)


def test_remap_is_its_own_inverse_through_the_tool_pose(remapper, franka):
    """A VLA action, executed, reads back as (nearly) the same VLA action.

    This is the property the fine-tuning export depends on: `lerobot_export.py`
    encodes Stretch motions as Franka joints with the same code the policy path
    decodes them with, so a round trip has to close.
    """
    base_pose = _pose([2.0, 1.0, 0.0], R.from_euler("z", 0.4).as_matrix())
    remapper.reset(base_pose)

    qpos = {
        "base": np.array([2.0, 1.0, 0.4]),
        "lift": np.array([0.6]),
        "arm": np.array([0.15]),
        "wrist": np.zeros(3),
        "gripper": np.array([0.5, 0.5]),
    }
    # A pose inside both robots' workspaces: the frame here is the mast mount, so
    # start from the virtual arm's own home and reach a little further out.
    commanded_qpos = franka.clip(HOME_QPOS + np.array([0.0, -0.2, 0.0, 0.3, 0.0, -0.1, 0.0]))
    action = remapper.action(np.append(commanded_qpos, 0.0), base_pose, qpos)

    assert remapper.telemetry.position_error < 0.02

    achieved = remapper.solver.forward(
        {key: action[key] for key in ("base", "lift", "arm", "wrist")}
    )
    observation = remapper.observation(achieved, gripper_closedness=0.0)
    assert observation["joint_position"].shape == (7,)
    assert remapper.telemetry.shadow_position_error < 0.02

    # The recovered joints put the tool in the same place, which is the invariant
    # -- not the joints themselves, since a 7-DOF arm has a null space.
    assert np.allclose(
        franka.forward(observation["joint_position"])[:3, 3],
        franka.forward(commanded_qpos)[:3, 3],
        atol=0.02,
    )


def test_remap_rejects_a_short_action(remapper):
    remapper.reset(np.eye(4))
    with pytest.raises(ValueError, match="7 joints plus a gripper"):
        remapper.action(np.zeros(5), np.eye(4), {"lift": np.array([0.6])})


def test_gripper_action_is_one_command_repeated(remapper):
    """Stretch has one gripper DOF; the pair is the MJCF's mirrored fingers."""
    closed = remapper.gripper_action(1.0)
    opened = remapper.gripper_action(0.0)
    assert closed[0] == closed[1]
    assert opened[0] == opened[1]
    assert closed[0] < opened[0]
    assert FrankaActionRemapper.stretch_gripper_closedness(closed) == pytest.approx(1.0, abs=1e-6)
    assert FrankaActionRemapper.stretch_gripper_closedness(opened) == pytest.approx(0.0, abs=1e-6)


def test_velocity_action_space_integrates_onto_the_shadow_arm(robot_config):
    """`joint_velocity` accumulates; `joint_position` does not."""
    episode_frame.clear()
    remapper = FrankaActionRemapper(robot_config, action_space="joint_velocity", velocity_dt=0.1)
    base_pose = np.eye(4)
    remapper.reset(base_pose)
    qpos = {
        "base": np.zeros(3),
        "lift": np.array([0.6]),
        "arm": np.array([0.15]),
        "wrist": np.zeros(3),
        "gripper": np.array([0.5, 0.5]),
    }
    velocity = np.zeros(8)
    velocity[1] = 0.5  # rad/s on joint 2

    before = remapper._shadow_qpos.copy()
    remapper.action(velocity, base_pose, qpos)
    after = remapper._shadow_qpos.copy()
    assert after[1] - before[1] == pytest.approx(0.05, abs=1e-6)


def test_base_budgets_are_measured_from_where_the_episode_started(robot_config):
    """A per-step cap would let the base creep a full allowance every step.

    The same mistake `policies/scripted.py` documents in
    `_remaining_base_budget`, and the reason `reset()` takes the base pose.
    """
    episode_frame.clear()
    remapper = FrankaActionRemapper(robot_config, max_base_rotation=0.5)
    start = _pose([0.0, 0.0, 0.0])
    remapper.reset(start)
    assert remapper._remaining_rotation(start) == pytest.approx(0.5)

    turned = _pose([0.0, 0.0, 0.0], R.from_euler("z", 0.4).as_matrix())
    assert remapper._remaining_rotation(turned) == pytest.approx(0.1, abs=1e-6)

    spun = _pose([0.0, 0.0, 0.0], R.from_euler("z", 1.2).as_matrix())
    assert remapper._remaining_rotation(spun) == 0.0


# =============================================================================
# Episode retargeting, with the frame recorded
# =============================================================================


def test_the_override_records_the_authoring_arm_before_overwriting_it():
    """`stretch_episode_override` must leave the pre-retarget frame recoverable."""
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec

    from examples.machine_learning.molmospaces.configs import StretchDummyEvalConfig
    from examples.machine_learning.molmospaces.franka_remapping.episode_overrides import (
        stretch_episode_override,
    )

    episode_frame.clear()
    authoring_qpos = [0.07, -0.92, -0.13, -2.50, -0.21, 1.41, 0.83]
    spec = EpisodeSpec(
        house_index=101,
        scene_dataset="procthor-objaverse",
        data_split="val",
        robot={"robot_name": "franka_droid", "init_qpos": {"arm": authoring_qpos}},
        img_resolution=[624, 352],
        cameras=[],
        language={"task_description": "pick up the mug"},
        task={
            "task_cls": "molmo_spaces.tasks.pick_task.PickTask",
            # 0.2m from the target: far too close for Stretch, so the retarget
            # has to move it and the recorded frame has to keep the original.
            "robot_base_pose": [12.0, 22.5, 0.17, 1.0, 0.0, 0.0, 0.0],
            "pickup_obj_start_pose": [12.2, 22.5, 1.12, 1.0, 0.0, 0.0, 0.0],
        },
    )
    # Any concrete eval config: the override only touches the camera system.
    exp_config = StretchDummyEvalConfig()

    stretch_episode_override(spec, exp_config)

    frame = episode_frame.current()
    assert frame is not None
    assert frame.base_pose[:3, 3] == pytest.approx([12.0, 22.5, 0.17])
    assert frame.init_qpos == pytest.approx(authoring_qpos)
    assert frame.robot_name == "franka_droid"
    # And the episode itself really was retargeted.
    assert spec.task["robot_base_pose"][2] == 0.0
    assert spec.task["robot_base_pose"][:2] != [12.0, 22.5]
    assert "arm" in spec.robot.init_qpos and "base" not in spec.robot.init_qpos


def test_a_non_franka_episode_falls_back_to_the_franka_home_pose():
    """An RBY1-authored episode has no seven-vector to read."""
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec

    from examples.machine_learning.molmospaces.franka_remapping.episode_overrides import (
        _authoring_arm_qpos,
    )

    spec = EpisodeSpec(
        house_index=1,
        scene_dataset="procthor-10k",
        data_split="val",
        robot={"robot_name": "rby1", "init_qpos": {"right_arm": [0.0] * 7}},
        img_resolution=[224, 224],
        cameras=[],
        language={"task_description": "pick up the mug"},
        task={"task_cls": "molmo_spaces.tasks.pick_task.PickTask"},
    )
    assert _authoring_arm_qpos(spec) == pytest.approx(HOME_QPOS)


def test_the_mast_mount_frame_sits_on_the_base(robot_config):
    """`default_frame_for` offsets along the base's own +x, not the world's."""
    base = _pose([3.0, 4.0, 0.0], R.from_euler("z", np.pi / 2).as_matrix())
    frame = episode_frame.default_frame_for(base)
    # +x of a base yawed 90 degrees points along world +y.
    assert frame.base_pose[:3, 3] == pytest.approx(
        [3.0, 4.0 + episode_frame.MAST_MOUNT_FORWARD_M, episode_frame.MAST_MOUNT_HEIGHT_M],
        abs=1e-6,
    )


# =============================================================================
# The fine-tuning export
# =============================================================================


def test_datagen_configs_substitute_stretch_and_widen_the_standoff():
    """Every registered datagen config must place the robot where Stretch works."""
    from examples.machine_learning.molmospaces.finetuning import datagen_configs as configs
    from examples.machine_learning.molmospaces.franka_remapping.episode_overrides import (
        REACH_BAND_M,
    )

    for task, class_name in configs.DATAGEN_CONFIGS.items():
        config = getattr(configs, class_name)()
        assert isinstance(config.robot_config, Stretch4RobotConfig), task
        assert type(config.camera_config).__name__ == "Stretch4CameraSystem", task
        assert type(config.policy_config).__name__ == "StretchScriptedPolicyConfig", task
        sampler = config.task_sampler_config
        assert tuple(sampler.base_pose_sampling_radius_range) == tuple(REACH_BAND_M), task
        assert sampler.robot_safety_radius == configs.STRETCH_BASE_SAFETY_RADIUS_M, task


def test_export_feature_names_describe_one_gripper():
    """Stretch has one commanded gripper DOF; the names must not imply two."""
    from examples.machine_learning.molmospaces.finetuning.lerobot_export import (
        FRANKA_STATE_NAMES,
        GRIPPER_CHANNEL_NAMES,
        STRETCH_ACTION_NAMES,
        STRETCH_STATE_NAMES,
    )
    from examples.machine_learning.molmospaces.policies.networks import ACTION_DIM, STATE_DIM

    assert GRIPPER_CHANNEL_NAMES[0] == "stretch_gripper"
    assert len(FRANKA_STATE_NAMES) == 8
    assert FRANKA_STATE_NAMES[-1] == "stretch_gripper"
    # The `stretch` space is `policies/networks.py`'s encoding, so the names have
    # to be exactly as wide as it is or the dataset metadata lies.
    assert len(STRETCH_STATE_NAMES) == STATE_DIM
    assert len(STRETCH_ACTION_NAMES) == ACTION_DIM


def test_implausible_base_commands_are_replaced_not_encoded():
    """Step 0's recorded no-op base command is the world origin, not "stay put".

    Left alone it encodes a 25-metre tool displacement into the first frame of
    every episode. The threshold has to be well clear of any real command.
    """
    from examples.machine_learning.molmospaces.finetuning.lerobot_export import (
        IMPLAUSIBLE_BASE_COMMAND_M,
    )

    assert IMPLAUSIBLE_BASE_COMMAND_M > 0.35, "must not fire on a real base solve"
    assert IMPLAUSIBLE_BASE_COMMAND_M < 5.0, "must still catch a zeroed no-op"


def test_stretch_space_encoder_matches_the_bc_encoding():
    """The `stretch` export must not fork the encoding the BC trainer uses."""
    from examples.machine_learning.molmospaces.finetuning.lerobot_export import (
        _StretchSpaceEncoder,
    )
    from examples.machine_learning.molmospaces.policies.networks import (
        ACTION_DIM,
        STATE_DIM,
        encode_state,
    )

    encoder = _StretchSpaceEncoder()
    qpos = {
        "base": np.array([1.0, 2.0, 0.3]),
        "lift": np.array([0.6]),
        "arm": np.array([0.2]),
        "wrist": np.array([0.1, 0.2, 0.3]),
        "gripper": np.array([0.4, 0.4]),
    }
    assert encoder.state_dim == STATE_DIM
    assert encoder.action_dim == ACTION_DIM
    assert np.allclose(encoder.state(qpos, np.eye(4)), encode_state(qpos))
