"""
Tests for examples/vla/molmobot_droid: the geometry the Franka -> Stretch 4 retargeting rests on,
image preparation, and the policy's observation history. None of them load the model.

Run from the repo root with `python -m pytest tests/test_molmobot_droid.py`; they are skipped
without the `molmobot-droid` extra.
"""

import math

import mujoco
import numpy as np
import pytest

pytest.importorskip("molmo_spaces")
pytest.importorskip("stretch4_kinematics")

from examples.vla.molmobot_droid import checkpoint  # noqa: E402
from examples.vla.molmobot_droid.droid import (  # noqa: E402
    FRANKA_PEDESTAL_HEIGHT,
    FrankaKinematics,
    FrankaSpawn,
    RobotPose,
    body_transform,
    franka_link0_height_for_object,
    make_transform,
    spawn_franka_droid,
    stretch4_camera_poses,
)
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (  # noqa: E402
    STRETCH_TCP,
    TCP_ALIGN,
    FrankaStretchRetargeter,
    RetargetParams,
    StretchJoints,
    crop_to_aspect,
    measure_arm_offset,
    planar_transform,
    stretch_kinematics,
    to_droid_frame,
)
from examples.vla.molmobot_droid.molmospaces.benchmark import run_name  # noqa: E402
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator  # noqa: E402


@pytest.fixture(scope="module")
def stretch_mujoco():
    Stretch4MujocoSimulator.get_robot_xml_path()
    model = mujoco.MjModel.from_xml_path(Stretch4MujocoSimulator.get_scene_xml_path())
    return model, mujoco.MjData(model)


def pose_stretch(model, data, lift, arm, yaw, pitch, roll, gripper=0.0):
    data.joint("floating_base").qpos = [0, 0, 0, 1, 0, 0, 0]
    data.joint("lift_joint").qpos = lift
    for i in range(1, 5):
        data.joint(f"arm_l{i}_joint").qpos = arm / 4
    data.joint("wrist_yaw_joint").qpos = yaw
    data.joint("wrist_pitch_joint").qpos = pitch
    data.joint("wrist_roll_joint").qpos = roll
    if gripper is not None:
        for side in ("left", "right"):
            data.joint(f"gripper_finger_{side}_joint").qpos = gripper
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Stretch 4 kinematics vs. the simulated model
# ---------------------------------------------------------------------------


def test_stretch_kinematics_with_arm_offset_matches_mujoco(stretch_mujoco):
    """stretch4_kinematics FK, corrected by `measure_arm_offset()`, is the simulator's TCP."""
    from stretch4_kinematics import StretchJointPositions

    model, data = stretch_mujoco
    kinematics = stretch_kinematics()
    offset = measure_arm_offset()
    rng = np.random.default_rng(0)
    for _ in range(10):
        lift, arm = rng.uniform(0.2, 1.1), rng.uniform(0.0, 0.5)
        yaw, pitch, roll = rng.uniform(-1, 3), rng.uniform(-1, 1.5), rng.uniform(-3, 1)
        pose_stretch(model, data, lift, arm, yaw, pitch, roll)
        expected = body_transform(data, STRETCH_TCP)
        q = StretchJointPositions(0, 0, 0, lift, arm - offset, yaw, pitch, roll)
        actual = kinematics.forward(q, STRETCH_TCP).homogeneous
        np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=1e-4)
        np.testing.assert_allclose(actual[:3, :3], expected[:3, :3], atol=1e-4)


def test_head_cameras_match_stretch_kinematics():
    """The camera poses transplanted onto the Franka agree with stretch4_kinematics' URDF."""
    from stretch4_kinematics import StretchJointPositions

    poses = stretch4_camera_poses()
    kinematics = stretch_kinematics()
    q = StretchJointPositions(0, 0, 0, 0.6, 0.1, 0, 0, 0)
    for camera, frame in [("camera_left_link", "camera_left_optical_link"),
                          ("camera_right_link", "camera_right_optical_link")]:
        urdf = kinematics.forward(q, frame).translation
        np.testing.assert_allclose(poses[camera][:3, 3], urdf, atol=2e-3)
    assert 1.4 < poses["camera_center_link"][2, 3] < 1.7, "head cameras are ~1.5 m off the floor"


# ---------------------------------------------------------------------------
# TCP alignment
# ---------------------------------------------------------------------------


def test_tcp_align_maps_approach_and_finger_axes(stretch_mujoco):
    model, data = stretch_mujoco
    pose_stretch(model, data, 0.8, 0.1, 0, 0, 0, gripper=0.4)
    tcp = body_transform(data, STRETCH_TCP)
    left = np.linalg.inv(tcp) @ np.append(data.body("gripper_fingertip_left_link").xpos, 1)
    right = np.linalg.inv(tcp) @ np.append(data.body("gripper_fingertip_right_link").xpos, 1)
    wrist = np.linalg.inv(tcp) @ np.append(data.body("wrist_roll_link").xpos, 1)
    # Stretch: approach is +x (the wrist is behind), fingers open along y.
    assert wrist[0] < -0.2
    assert left[1] > 0.05 and right[1] < -0.05

    # Franka grasp_site: approach +z, fingers on y. TCP_ALIGN takes Stretch's axes to those.
    stretch_in_franka = TCP_ALIGN[:3, :3]
    np.testing.assert_allclose(stretch_in_franka @ [1, 0, 0], [0, 0, 1], atol=1e-9)
    assert abs(stretch_in_franka @ [0, 1, 0] @ [0, 1, 0]) == pytest.approx(1)
    assert np.linalg.det(stretch_in_franka) == pytest.approx(1)


def test_tcp_align_puts_wrist_cameras_on_the_same_side(stretch_mujoco):
    model, data = stretch_mujoco
    pose_stretch(model, data, 0.8, 0.1, 0, 0, 0)
    tcp = body_transform(data, STRETCH_TCP)
    stretch_camera = (np.linalg.inv(tcp) @ np.append(data.cam("gripper_camera_right_rgb").xpos, 1))[:3]

    franka = FrankaKinematics()
    franka._set(checkpoint.FRANKA_HOME_QPOS)
    mujoco.mj_camlight(franka.model, franka.data)
    site = franka.data.site("robot_0/gripper/grasp_site")
    franka_tcp = make_transform(site.xmat.reshape(3, 3), site.xpos)
    franka_camera = (np.linalg.inv(franka_tcp) @ np.append(franka.data.cam("robot_0/gripper/wrist_camera").xpos, 1))[:3]
    franka_camera_in_stretch_axes = TCP_ALIGN[:3, :3].T @ franka_camera
    # Both sit behind the TCP and on its +z side.
    assert stretch_camera[0] < 0 and franka_camera_in_stretch_axes[0] < 0
    assert stretch_camera[2] > 0 and franka_camera_in_stretch_axes[2] > 0


# ---------------------------------------------------------------------------
# Franka
# ---------------------------------------------------------------------------


def test_franka_ik_recovers_fk():
    franka = FrankaKinematics()
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = np.clip(np.array(checkpoint.FRANKA_HOME_QPOS) + rng.uniform(-0.4, 0.4, 7), franka.lower, franka.upper)
        target = franka.fk(q)
        solved, converged = franka.ik(target, q + rng.uniform(-0.05, 0.05, 7))
        assert converged
        np.testing.assert_allclose(franka.fk(solved)[:3, 3], target[:3, 3], atol=2e-4)


def test_franka_link0_at_training_height():
    """molmospaces: mocap base at object z - 0.75 under a 0.58 m pedestal."""
    assert franka_link0_height_for_object(0.9) == pytest.approx(0.9 - 0.75 + FRANKA_PEDESTAL_HEIGHT)


@pytest.mark.parametrize("standing", [True, False])
def test_spawned_franka_link0_and_head_camera(standing):
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    pose = RobotPose(1.0, -2.0, 0.7)
    spawn = spawn_franka_droid(spec, pose, link0_height=0.7, exo_camera="left", standing_on_floor=standing)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(body_transform(data, spawn.link0_name), spawn.world_from_link0, atol=1e-6)
    np.testing.assert_allclose(data.body(spawn.link0_name).xpos[2], 0.7, atol=1e-6)

    camera = make_transform(data.cam(spawn.exo_camera_name).xmat.reshape(3, 3), data.cam(spawn.exo_camera_name).xpos)
    expected = pose.matrix(0.0) @ stretch4_camera_poses()["camera_left_link"]
    np.testing.assert_allclose(camera, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Retargeting
# ---------------------------------------------------------------------------


def make_retargeter(params=None, link0_height=0.5):
    pose = RobotPose(1.0, 2.0, 0.5)
    franka = FrankaSpawn("x/", pose, 0.0, 0.0, link0_height, None, None, "")
    return pose, FrankaStretchRetargeter(franka, params or RetargetParams())


def test_franka_to_stretch_and_back():
    pose, retargeter = make_retargeter()
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(60):
        q = np.array(checkpoint.FRANKA_HOME_QPOS) + rng.uniform(-0.3, 0.3, 7)
        retargeter.at_start = True  # each pose is a new target, not a step of a motion
        targets = retargeter.franka_to_stretch(np.append(q, 0), pose.matrix(), StretchJoints(0.6, 0.1, 0, 0, 0, 1))
        if targets is None or targets.clamped:
            continue
        footprint = pose.matrix() @ planar_transform(0, 0, targets.base_rotate_by)
        joints = StretchJoints(targets.lift, targets.arm, targets.wrist_yaw, targets.wrist_pitch, targets.wrist_roll, 1)
        tool = retargeter.stretch_tool_world(footprint, joints)
        expected = retargeter.stretch_tool_target_world(q)
        np.testing.assert_allclose(tool, expected, atol=1e-3)
        retargeter.franka_seed = q
        state8, converged = retargeter.stretch_to_franka(footprint, joints)
        assert converged
        np.testing.assert_allclose(retargeter.franka_kinematics.fk(state8[:7]), retargeter.franka_kinematics.fk(q), atol=1e-3)
        checked += 1
    assert checked > 20


def test_wrist_never_swings_round_while_following_a_policy():
    """A policy's small steps never make Stretch's base or wrist jump more than MAX_STEP_JUMP."""
    from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import MAX_STEP_JUMP

    pose, retargeter = make_retargeter(link0_height=0.6)
    footprint = pose.matrix()
    joints = StretchJoints(0.6, 0.1, 0, 0, 0, 1)
    q = np.array(checkpoint.FRANKA_HOME_QPOS)
    rng = np.random.default_rng(11)
    for step in range(150):
        if step:
            q = np.clip(q + rng.uniform(-0.08, 0.08, 7), retargeter.franka_kinematics.lower, retargeter.franka_kinematics.upper)
        targets = retargeter.franka_to_stretch(np.append(q, 0), footprint, joints)
        if targets is None:
            continue
        if step:
            moves = [targets.base_rotate_by, targets.wrist_yaw - joints.wrist_yaw,
                     targets.wrist_pitch - joints.wrist_pitch, targets.wrist_roll - joints.wrist_roll]
            assert max(abs(m) for m in moves) <= MAX_STEP_JUMP + 1e-6, f"step {step}: {np.degrees(moves).round(0)}"
        footprint = footprint @ planar_transform(0, 0, targets.base_rotate_by)
        joints = StretchJoints(targets.lift, targets.arm, targets.wrist_yaw, targets.wrist_pitch, targets.wrist_roll, 1)


def test_grasp_offset_moves_stretch_but_not_the_franka_state():
    offset = RetargetParams(grasp_offset_mm=(20.0, 0.0, -10.0), grasp_offset_deg=(10.0, 0.0, 0.0))
    pose, plain = make_retargeter(RetargetParams(grasp_offset_mm=(0.0, 0.0, 0.0)), link0_height=0.45)
    _, shifted = make_retargeter(offset, link0_height=0.45)
    q = np.array(checkpoint.FRANKA_HOME_QPOS)
    a, b = plain.stretch_tool_target_world(q), shifted.stretch_tool_target_world(q)
    # 20 mm along the approach, -10 mm along z, in the TCP frame.
    np.testing.assert_allclose(np.linalg.inv(a) @ b[:, 3], [0.02, 0, -0.01, 1], atol=1e-9)

    targets = shifted.franka_to_stretch(np.append(q, 0), pose.matrix(), StretchJoints(0.6, 0.1, 0, 0, 0, 1))
    assert targets is not None and not targets.clamped
    footprint = pose.matrix() @ planar_transform(0, 0, targets.base_rotate_by)
    joints = StretchJoints(targets.lift, targets.arm, targets.wrist_yaw, targets.wrist_pitch, targets.wrist_roll, 1)
    state8, _ = shifted.stretch_to_franka(footprint, joints)
    np.testing.assert_allclose(state8[:7], q, atol=2e-3)


def test_unreachable_target_is_clamped_or_refused():
    """The Franka's home pose over a tall counter is above Stretch's lift; it goes as close as it can."""
    # fr3_link0 at 0.75 m: the Franka's downward TCP at 1.185 m, just over what Stretch reaches.
    pose, retargeter = make_retargeter(link0_height=0.75)
    targets = retargeter.franka_to_stretch(
        np.append(checkpoint.FRANKA_HOME_QPOS, 0), pose.matrix(), StretchJoints(0.6, 0.1, 0, 0, 0, 1)
    )
    assert targets is not None and targets.clamped
    assert targets.lift == pytest.approx(1.2 - 0.005, abs=1e-3), "at the top of its travel, short of the end stop"
    assert 0 < targets.tool_error_m < 0.05

    far = np.array(checkpoint.FRANKA_HOME_QPOS)
    far[3] = -0.2  # arm stretched straight up and out
    assert retargeter.franka_to_stretch(np.append(far, 0), pose.matrix(), StretchJoints(0.6, 0.1, 0, 0, 0, 1)) is None
    assert retargeter.ik_failures == 1


def test_ik_respects_the_robots_wrist_roll_range(stretch_mujoco):
    """stretch4_kinematics' URDF mirrors the roll range; the solutions must fit the real one."""
    model, _ = stretch_mujoco
    lower, upper = model.joint("wrist_roll_joint").range
    pose, retargeter = make_retargeter()
    rng = np.random.default_rng(3)
    for _ in range(40):
        q = np.array(checkpoint.FRANKA_HOME_QPOS) + rng.uniform(-0.4, 0.4, 7)
        targets = retargeter.franka_to_stretch(np.append(q, 0), pose.matrix(), StretchJoints(0.6, 0.1, 0, 0, 0, 1))
        if targets is not None:
            assert lower - 1e-6 <= targets.wrist_roll <= upper + 1e-6


def test_wrist_view_is_the_gripper_camera_droid_framed():
    from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import wrist_view

    image = np.zeros((270, 480, 3), np.uint8)
    image[:10, :10] = 255
    view = wrist_view(image)
    assert view.shape == (368, 640, 3)
    assert view[0, 0].all() and not view[-1, -1].any(), "never turned round"


def test_params_validation():
    with pytest.raises(ValueError):
        RetargetParams(execute_horizon=4, execute_first_n=5)
    with pytest.raises(ValueError):
        RetargetParams(execute_horizon=17)
    with pytest.raises(ValueError):
        RetargetParams(exo_camera="top")


# ---------------------------------------------------------------------------
# Images and names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1920, 1200, 3), (270, 480, 3), (1280, 965, 3), (368, 640, 3)])
def test_droid_frame_size(shape):
    frame = to_droid_frame(np.zeros(shape, np.uint8))
    assert frame.shape == (368, 640, 3)


def test_crop_to_aspect_is_centered():
    image = np.zeros((100, 300), np.uint8)
    image[:, 140:160] = 1
    cropped = crop_to_aspect(image, 1.0)
    assert cropped.shape == (100, 100)
    assert cropped[:, 40:60].all()


def test_run_name_joins_flags_with_underscores():
    flags = RetargetParams(grasp_offset_mm=(0, 5, 0), slow=True).flags()
    name = run_name("stretch4", {**flags, "include_franka": False})
    assert name == (
        "stretch4_exo-left_grip-left_eh-8_n-2_crop-droid_slow-1_wait-1_pg-0_offmm-0,5,0_offdeg-0,0,0_ghost-0"
    )


@pytest.mark.parametrize("use_parallel_gripper", [False, True])
def test_default_grasp_offset_per_tool(use_parallel_gripper):
    from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import DEFAULT_GRASP_OFFSET_MM

    params = RetargetParams(use_parallel_gripper=use_parallel_gripper)
    assert params.effective_grasp_offset_mm == DEFAULT_GRASP_OFFSET_MM[params.tool_name]
    assert RetargetParams(use_parallel_gripper=use_parallel_gripper, grasp_offset_mm=(1, 2, 3)).effective_grasp_offset_mm == (1, 2, 3)


def test_parallel_gripper_tcp_correction_matches_its_model():
    """IK runs on the Stretch gripper model; the correction must land the PG4's own TCP."""
    from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import PARALLEL_GRIPPER_TOOL

    spec = mujoco.MjSpec.from_file(Stretch4MujocoSimulator.get_robot_xml_path(PARALLEL_GRIPPER_TOOL))
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
    model = spec.compile()
    data = mujoco.MjData(model)
    pose, retargeter = make_retargeter(RetargetParams(use_parallel_gripper=True))
    joints = StretchJoints(0.8, 0.2, 0.3, 0.9, -0.4, 1)
    pose_stretch(model, data, joints.lift, joints.arm, joints.wrist_yaw, joints.wrist_pitch, joints.wrist_roll, gripper=None)
    expected = body_transform(data, STRETCH_TCP)
    np.testing.assert_allclose(retargeter.stretch_tool_world(np.eye(4), joints), expected, atol=1e-4)


def test_instructions():
    assert checkpoint.build_instruction("pick", "red mug") == "Pick up the red mug"
    with pytest.raises(ValueError):
        checkpoint.build_instruction("pick_and_place", "mug")


# ---------------------------------------------------------------------------
# Policy observation history (no model)
# ---------------------------------------------------------------------------


class FakeWrapper:
    action_horizon = 16

    def __init__(self):
        self.calls = []

    def get_action_chunk(self, images, task_description, state):
        self.calls.append([int(image[0, 0, 0]) for image in images])
        chunk = np.zeros((16, 8))
        chunk[:, 7] = 200  # closes
        return chunk


def fake_policy():
    from collections import deque

    policy = object.__new__(checkpoint.MolmoBotDroidPolicy)
    policy._wrapper = FakeWrapper()
    policy.n_obs_steps, policy.obs_step_delta, policy.action_horizon = 2, 8, 16
    policy._history = deque(maxlen=9)
    return policy


def frame(value):
    return np.full((368, 640, 3), value, np.uint8)


def test_history_sends_exo_then_wrist_now_and_eight_steps_ago():
    policy = fake_policy()
    for step in range(12):
        policy.add_observation(frame(step), frame(100 + step))
    chunk = policy.predict_chunk(np.zeros(8), "Pick up the mug")
    assert policy._wrapper.calls[-1] == [3, 11, 103, 111]
    assert (chunk[:, 7] == checkpoint.GRIPPER_CLOSED).all()
    assert len(policy._history) == 9, "only the reachable frames are kept"


def test_history_at_episode_start_has_no_padding():
    policy = fake_policy()
    policy.add_observation(frame(0), frame(100))
    policy.predict_chunk(np.zeros(8), "x")
    assert policy._wrapper.calls[-1] == [0, 100]


def test_joint_delta_limit_keeps_direction():
    limited = checkpoint.limit_joint_delta(np.array([0.4, 0.1, 0, 0, 0, 0, 0]), np.zeros(7))
    np.testing.assert_allclose(limited[:2], [0.2, 0.05])
    assert math.isclose(np.abs(limited).max(), checkpoint.RELATIVE_MAX_JOINT_DELTA)
