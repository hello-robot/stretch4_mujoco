"""
Franka -> Stretch 4 retargeting, checked on the robots' own MuJoCo models and shown in Rerun.

One scene holds Stretch 4 (the stretch4_mujoco MJCF) and the Franka DROID overlaid on it as a
see-through ghost, standing where the retargeter puts the virtual Franka. The Franka is
commanded through 20 poses, Stretch follows each one through the retargeter, and the test
asserts at every pose that Stretch's `grasp_center_link`, read from MuJoCo (not from
stretch4_kinematics), sits on the Franka's `grasp_site` (plus the tool's default grasp offset, which lines the
fingers up) within TOLERANCE_MM / TOLERANCE_DEG, for the Stretch gripper and the parallel one,
and that mapping Stretch back gives the Franka's TCP again.


Rerun opens a viewer when there is a display; set RERUN_SAVE=<file.rrd> to record instead.

    python -m pytest examples/vla/molmobot_droid/franka_retarget/test_retargetting.py -s

(`conftest.py` next to this file works around the repo root being a package named
stretch4_mujoco.)
"""

import math
import os
import uuid

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


STRETCH_RANGES = {
    "lift": (0.25, 1.10),
    "arm": (0.0, 0.50),
    # 0.15 rad inside each wrist limit (yaw and pitch -65..245 degrees, roll -65..245 on the
    # robot); pitch stops short of folding the gripper back over the arm.
    "wrist_yaw": (-0.98, 4.12),
    "wrist_pitch": (-0.98, 2.40),
    "wrist_roll": (-0.98, 4.12),
}
"""Where the test poses come from: Stretch joint values spread across these ranges."""


def wide_franka_poses(rng: np.random.Generator, retargeter: FrankaStretchRetargeter) -> list[np.ndarray]:
    """
    NUM_POSES Franka arm poses whose TCPs are spread over Stretch's lift, arm and wrist ranges:
    random Stretch joint values within STRETCH_RANGES, mapped back to the Franka (its TCP where
    Stretch's tool is, less the grasp offset), keeping only those the Franka reaches too. So
    every pose is one both robots can take, and the wrist turns well beyond what a pick needs.
    """
    franka = retargeter.franka_kinematics
    footprint = ROBOT_POSE.matrix()
    seeds = [np.array(FRANKA_HOME_QPOS)] + [
        np.clip(np.array(FRANKA_HOME_QPOS) + rng.uniform(-1.0, 1.0, 7), franka.lower, franka.upper) for _ in range(4)
    ]
    poses = []
    while len(poses) < NUM_POSES:
        values = {name: rng.uniform(*bounds) for name, bounds in STRETCH_RANGES.items()}
        joints = StretchJoints(gripper_open_fraction=1.0, **values)
        tool_world = retargeter.stretch_tool_world(footprint, joints)
        franka_tcp = (
            np.linalg.inv(retargeter.franka.world_from_link0)
            @ tool_world
            @ np.linalg.inv(retargeter.tcp_offset)
            @ TCP_ALIGN.T
        )
        for seed in seeds:
            q, converged = franka.ik(franka_tcp, seed)
            if converged:
                poses.append(q)
                break
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
    # A recording of its own: `rr.init` reuses the process's recording id, so the parametrized
    # runs would share one, and the Stretch gripper's geoms the parallel gripper's scene lacks
    # would hang about where the first run left them.
    rr.init("stretch4_retargeting_test", recording_id=f"{request.node.name}-{uuid.uuid4().hex[:8]}", spawn=False)
    save_to = os.environ.get("RERUN_SAVE")
    if save_to:
        rr.save(save_to)
    elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        rr.spawn(memory_limit="1GB")
    else:
        rr.save(str(request.config.rootpath / "outputs" / "test_retargetting.rrd"))
    yield rr


@pytest.mark.parametrize("use_parallel_gripper", [False, True], ids=["stretch_gripper", "parallel_gripper"])
def test_stretch_follows_franka_across_wide_poses(rerun_recording, use_parallel_gripper):
    params = RetargetParams(use_parallel_gripper=use_parallel_gripper)
    model, data, franka = build_scene(params.tool_name)
    retargeter = FrankaStretchRetargeter(franka, params)

    # Stretch starts at its own home and follows from there, its base turning as IK asks.
    world_from_footprint = ROBOT_POSE.matrix()
    joints = StretchJoints(lift=0.6, arm=0.1, wrist_yaw=0.0, wrist_pitch=0.0, wrist_roll=0.0, gripper_open_fraction=1.0)
    pose_robots(model, data, franka, FRANKA_HOME_QPOS, world_from_footprint, joints)

    # The ghost is in a geom group MuJoCo hides by default; RerunScene draws every visual geom.
    scene = rerun_scene.RerunScene(model, data, ["stretch4", franka.base_name])

    errors, followed = [], []
    for index, franka_q7 in enumerate(wide_franka_poses(np.random.default_rng(7), retargeter)):
        # Each pose is a new target anywhere in the range, not the next step of a motion, so
        # the per-step jump limit (which keeps a policy's motion from flipping the wrist) is off.
        retargeter.at_start = True
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
        followed.append([joints.lift, joints.arm, joints.wrist_yaw, joints.wrist_pitch, joints.wrist_roll])

    spread = np.array(followed)
    print(f"\n{params.tool_name}: {NUM_POSES} poses; Stretch followed through")
    for name, column, scale, unit in [
        ("lift", 0, 1, "m"), ("arm", 1, 1, "m"),
        ("wrist yaw", 2, 180 / math.pi, "deg"), ("wrist pitch", 3, 180 / math.pi, "deg"), ("wrist roll", 4, 180 / math.pi, "deg"),
    ]:
        print(f"  {name:12s} {spread[:, column].min() * scale:7.2f} .. {spread[:, column].max() * scale:7.2f} {unit}")
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