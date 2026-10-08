"""
Franka -> Stretch 4 retargeting, checked on the robots' own MuJoCo models and shown in Rerun.

One scene holds Stretch 4 (the stretch4_mujoco MJCF) and the Franka DROID overlaid on it as a
see-through ghost, standing where the retargeter puts the virtual Franka. The Franka is
commanded through 20 poses, Stretch follows each one through the retargeter, and the test
asserts at every pose that Stretch's `grasp_center_link`, read from MuJoCo (not from
stretch4_kinematics), sits on the Franka's `grasp_site` (plus the tool's default grasp offset, which lines the
fingers up) within TOLERANCE_MM / TOLERANCE_DEG, for the Stretch gripper and the parallel one,
and that mapping Stretch back gives the Franka's TCP again.

Where Stretch's wrist cannot turn the gripper as the Franka's is, the retargeter turns it half a
turn about the approach axis instead (`TCP_FLIP`): the same grasp for two symmetric fingers.
Those poses are compared against the Franka's TCP turned the same way.

Rerun opens a viewer when there is a display; set RERUN_SAVE=<file.rrd> to record instead.

    python -m pytest examples/vla/molmobot_droid/franka_retarget/test_retargetting.py -s

(`conftest.py` next to this file works around the repo root being a package named
stretch4_mujoco.)
"""

import math
import os

import mujoco
import numpy as np
import pytest

pytest.importorskip("molmo_spaces")
pytest.importorskip("stretch4_kinematics")
rr = pytest.importorskip("rerun")

from examples.vla.molmobot_droid import rerun_scene  # noqa: E402
from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS  # noqa: E402
from examples.vla.molmobot_droid.droid import (  # noqa: E402
    RobotPose,
    add_franka_ghost,
    body_transform,
    mat_to_quat,
    site_transform,
)
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (  # noqa: E402
    TCP_ALIGN,
    TCP_FLIP,
    FrankaStretchRetargeter,
    RetargetParams,
    StretchJoints,
    planar_transform,
)

NUM_POSES = 40
TOLERANCE_MM = 3.0
TOLERANCE_DEG = 1.0

ROBOT_POSE = RobotPose(0.5, -0.3, 0.4)
LINK0_HEIGHT = 0.45
"""Low enough that the Franka's home pose, pointing down, is within Stretch's lift."""


def build_scene(tool_name: str):
    """Stretch 4 at ROBOT_POSE with the ghost Franka at its retargeting pose, on a floor."""
    from examples.molmo_environment import add_stretch_to_scene

    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], rgba=[0.8, 0.8, 0.8, 1])
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    add_stretch_to_scene(
        spec, pos=[ROBOT_POSE.x, ROBOT_POSE.y, 0.0], quat=ROBOT_POSE.quat_wxyz, floor_geom_names=["floor"], tool_name=tool_name
    )
    franka = add_franka_ghost(spec, ROBOT_POSE, LINK0_HEIGHT)
    model = spec.compile()
    return model, mujoco.MjData(model), franka


def franka_trajectory(rng: np.random.Generator) -> list[np.ndarray]:
    """NUM_POSES Franka arm poses wandering away from home, as a policy would command them."""
    q = np.array(FRANKA_HOME_QPOS, dtype=float)
    poses = []
    for _ in range(NUM_POSES):
        q = q + rng.uniform(-0.12, 0.12, 7)
        q = np.clip(q, np.array(FRANKA_HOME_QPOS) - 0.5, np.array(FRANKA_HOME_QPOS) + 0.5)
        poses.append(q.copy())
    return poses


def pose_robots(model, data, franka, franka_q7, world_from_footprint, joints: StretchJoints):
    """Kinematically place both robots: the Franka at its joints, Stretch at its footprint and joints."""
    for i, value in enumerate(franka_q7):
        data.joint(f"{franka.prefix}fr3_joint{i + 1}").qpos = value
    data.joint("floating_base").qpos = np.concatenate(
        [world_from_footprint[:3, 3], mat_to_quat(world_from_footprint[:3, :3])]
    )
    data.joint("lift_joint").qpos = joints.lift
    for i in range(1, 5):
        data.joint(f"arm_l{i}_joint").qpos = joints.arm / 4
    data.joint("wrist_yaw_joint").qpos = joints.wrist_yaw
    data.joint("wrist_pitch_joint").qpos = joints.wrist_pitch
    data.joint("wrist_roll_joint").qpos = joints.wrist_roll
    mujoco.mj_kinematics(model, data)


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1.0, 1.0)))


@pytest.fixture
def rerun_recording(request):
    rr.init("stretch4_retargeting_test", spawn=False)
    save_to = os.environ.get("RERUN_SAVE")
    if save_to:
        rr.save(save_to)
    elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        rr.spawn(memory_limit="1GB")
    else:
        rr.save(str(request.config.rootpath / "outputs" / "test_retargetting.rrd"))
    yield rr


@pytest.mark.parametrize("use_parallel_gripper", [False, True], ids=["stretch_gripper", "parallel_gripper"])
def test_stretch_follows_franka_across_20_poses(rerun_recording, use_parallel_gripper):
    params = RetargetParams(use_parallel_gripper=use_parallel_gripper)
    model, data, franka = build_scene(params.tool_name)
    retargeter = FrankaStretchRetargeter(franka, params)

    # Stretch starts at its own home and follows from there, its base turning as IK asks.
    world_from_footprint = ROBOT_POSE.matrix()
    joints = StretchJoints(lift=0.6, arm=0.1, wrist_yaw=0.0, wrist_pitch=0.0, wrist_roll=0.0, gripper_open_fraction=1.0)
    pose_robots(model, data, franka, FRANKA_HOME_QPOS, world_from_footprint, joints)

    # The ghost is in a geom group MuJoCo hides by default; RerunScene draws every visual geom.
    scene = rerun_scene.RerunScene(model, data, ["stretch4", franka.base_name])

    errors = []
    flips = 0
    for index, franka_q7 in enumerate(franka_trajectory(np.random.default_rng(7))):
        targets = retargeter.franka_to_stretch(np.append(franka_q7, 0.0), world_from_footprint, joints)
        assert targets is not None, f"pose {index}: Stretch cannot reach it"

        # Stretch follows: turn the base, then the arm joints.
        world_from_footprint = world_from_footprint @ planar_transform(0, 0, targets.base_rotate_by)
        joints = StretchJoints(
            targets.lift, targets.arm, targets.wrist_yaw, targets.wrist_pitch, targets.wrist_roll, 1.0
        )
        pose_robots(model, data, franka, franka_q7, world_from_footprint, joints)

        # Where Stretch's tool belongs: the Franka's TCP, with the tool's default grasp offset that
        # lines its fingers up with the Robotiq's.
        franka_tcp = site_transform(data, f"{franka.prefix}gripper/grasp_site") @ TCP_ALIGN @ params.tcp_offset
        if targets.flipped:
            franka_tcp = franka_tcp @ TCP_FLIP
        stretch_tcp = body_transform(data, "grasp_center_link")
        position_mm = float(np.linalg.norm(franka_tcp[:3, 3] - stretch_tcp[:3, 3]) * 1000)
        rotation_deg = rotation_error_deg(franka_tcp, stretch_tcp)

        # And back: the Franka state the policy would see is this pose again.
        state8, converged = retargeter.stretch_to_franka(world_from_footprint, joints)
        reverse_mm = float(
            np.linalg.norm(retargeter.franka_kinematics.fk(state8[:7])[:3, 3] - retargeter.franka_kinematics.fk(franka_q7)[:3, 3])
            * 1000
        )

        rerun_scene.set_step(index)
        scene.log(data)
        rerun_scene.log_tool_poses(franka_tcp, stretch_tcp)
        rerun_scene.log_metrics(
            {"position_error_mm": position_mm, "rotation_error_deg": rotation_deg, "reverse_error_mm": reverse_mm}
        )
        errors.append((index, position_mm, rotation_deg, reverse_mm, targets.clamped, converged))
        flips += targets.flipped

    print(f"\n{params.tool_name}: {NUM_POSES} poses, {flips} with the gripper turned half a turn:")
    for index, position_mm, rotation_deg, reverse_mm, *_ in errors:
        print(f"  {index:2d}: {position_mm:.3f} mm, {rotation_deg:.3f} deg, back to the Franka {reverse_mm:.3f} mm")
    for index, position_mm, rotation_deg, reverse_mm, clamped, converged in errors:
        assert not clamped, f"pose {index}: Stretch only got close ({position_mm:.1f} mm, {rotation_deg:.1f} deg)"
        assert position_mm < TOLERANCE_MM, f"pose {index}: TCPs {position_mm:.2f} mm apart"
        assert rotation_deg < TOLERANCE_DEG, f"pose {index}: TCPs {rotation_deg:.2f} deg apart"
        assert converged and reverse_mm < TOLERANCE_MM, f"pose {index}: mapping back is {reverse_mm:.2f} mm off"
    assert len(errors) == NUM_POSES


if __name__ == "__main__":
    pytest.main([__file__])