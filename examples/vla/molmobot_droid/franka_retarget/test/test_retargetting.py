"""
Franka -> Stretch 4 retargeting, checked on the robots' own MuJoCo models and shown in Rerun.

One scene holds Stretch 4 (the stretch4_mujoco MJCF) and the Franka DROID overlaid on it as a
see-through ghost, standing where the retargeter puts the virtual Franka. The Franka is
commanded through 20 poses, Stretch follows each one through the retargeter, and the test
asserts at every pose that Stretch's `grasp_center_link`, read from MuJoCo (not from
stretch4_kinematics), sits on the Franka's `grasp_site` (plus the tool's default grasp offset, which lines the
fingers up) within TOLERANCE_MM / TOLERANCE_DEG, for the Stretch gripper and the parallel one,
and that mapping Stretch back gives the Franka's TCP again.

Every run records `outputs/retargeting/<test>.rrd` (RERUN_SAVE=<dir> to put them elsewhere), and
opens a viewer as well when there is a display (RERUN_SPAWN=0 to not). Its tabs show each robot's
joint space (joints over the poses, and each joint's position within its limits) and task space
(the TCP in 3D, as it moves and every pose so far, and its x, y, z, roll, pitch and yaw) for the
Franka alone, then Stretch 4 alone, then both overlaid (`RetargetSpacesLogger`). Open a recording
later with `rerun outputs/retargeting/<test>.rrd`.

    python -m pytest examples/vla/molmobot_droid/franka_retarget/test/test_retargetting.py -s

(`conftest.py` next to this file works around the repo root being a package named
stretch4_mujoco.)
"""

import math
import os
import uuid
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("molmo_spaces")
pytest.importorskip("stretch4_kinematics")
rr = pytest.importorskip("rerun")
rrb = pytest.importorskip("rerun.blueprint")

from examples.digital_twin import RerunMujocoRobot  # noqa: E402
from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS  # noqa: E402
from examples.vla.molmobot_droid.droid import (  # noqa: E402
    FrankaSpawn,
    RobotPose,
    add_franka_ghost,
    body_transform,
    mat_to_quat,
    site_transform,
)
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (  # noqa: E402
    ARM_RANGE,
    DROID_EXO_FOCAL_PX,
    LIFT_RANGE,
    TCP_ALIGN,
    WRIST_ROLL_RANGE,
    FrankaStretchRetargeter,
    RetargetParams,
    StretchJoints,
    head_rectifier,
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


FRANKA_COLOR = [255, 160, 0]
"""Orange: the Franka, and the TCP pose it asks for (as `rerun_scene.TARGET_COLOR`)."""
STRETCH_COLOR = [0, 200, 255]
"""Blue: Stretch 4, and where its tool is (as `rerun_scene.ACTUAL_COLOR`)."""
STRETCH_JOINTS = ("base_yaw", "lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll")
TASK_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
"""The TCP's position (m, world) and orientation (rad, ZYX Euler angles of its world rotation)."""

WORKSPACE_SAMPLES = 6000
"""Joint configurations sampled per robot, uniformly within its limits, to draw its reachable TCP positions."""
REACH_SAMPLES = 400
"""Of the Franka's, how many are checked for whether Stretch reaches the whole TCP pose (6-D IK)."""
FRANKA_SHOULDER_Z = 0.333
"""m. fr3_joint2's height above fr3_link0: the centre of the Franka's (near-spherical) reach."""
JOINT_BAR_WIDTH = 6.0
"""Width of each joint's range bar in the joint-space view (its lower limit at 0, its upper here)."""
JOINT_ROWS = {"franka": 0, "stretch": 8}
"""Joint-space view row of each robot's first joint: the Franka's 7 above Stretch's 6."""
REACHED_COLOR, NOT_REACHED_COLOR = [40, 200, 80], [230, 50, 50]


def annulus_mesh(center: np.ndarray, r_in: float, r_out: float, z_low: float, z_high: float, segments: int = 72):
    """
    A closed annular cylinder (a tube with walls and end rings) about the vertical through `center`.

    Args:
        center: World point on the axis (its z is ignored).
        r_in: Inner radius (m).
        r_out: Outer radius (m).
        z_low: Bottom height (m, world).
        z_high: Top height (m, world).
        segments: Facets around the axis.

    Returns:
        (vertices (N, 3), triangles (M, 3)) for `rr.Mesh3D`.
    """
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    rings = [(r_out, z_low), (r_out, z_high), (r_in, z_high), (r_in, z_low)]
    vertices = np.array([
        [center[0] + r * math.cos(a), center[1] + r * math.sin(a), z] for r, z in rings for a in angles
    ])
    triangles = []
    for ring in range(4):  # outer wall, top, inner wall, bottom: each joins ring to the next
        a0, b0 = ring * segments, ((ring + 1) % 4) * segments
        for i in range(segments):
            j = (i + 1) % segments
            triangles += [[a0 + i, a0 + j, b0 + i], [a0 + j, b0 + j, b0 + i]]
    return vertices, np.array(triangles)


def annulus_outline(center: np.ndarray, r_in: float, r_out: float, z_low: float, z_high: float, segments: int = 72):
    """
    The four circles bounding `annulus_mesh()`'s tube, and vertical lines joining them every 45 degrees.

    Args:
        center: World point on the axis (its z is ignored).
        r_in: Inner radius (m).
        r_out: Outer radius (m).
        z_low: Bottom height (m, world).
        z_high: Top height (m, world).
        segments: Points per circle.

    Returns:
        Line strips for `rr.LineStrips3D`.
    """
    angles = np.linspace(0, 2 * math.pi, segments + 1)
    strips = [
        [[center[0] + r * math.cos(a), center[1] + r * math.sin(a), z] for a in angles]
        for r in (r_in, r_out) for z in (z_low, z_high)
    ]
    for a in np.arange(0, 2 * math.pi, math.pi / 4):
        for r in (r_in, r_out):
            x, y = center[0] + r * math.cos(a), center[1] + r * math.sin(a)
            strips.append([[x, y, z_low], [x, y, z_high]])
    return strips


def euler_zyx(rotation: np.ndarray) -> tuple[float, float, float]:
    """
    Roll, pitch and yaw (ZYX Euler angles) of a rotation matrix.

    Args:
        rotation: A 3x3 (or the top-left of a 4x4) rotation matrix.

    Returns:
        (roll, pitch, yaw) in rad.
    """
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    return math.atan2(rotation[2, 1], rotation[2, 2]), pitch, math.atan2(rotation[1, 0], rotation[0, 0])


class RetargetSpacesLogger:
    """
    Logs the Franka and Stretch 4 to Rerun in joint space and in task space, each robot under
    its own entities so the blueprint (`blueprint()`) can show the Franka alone, Stretch alone,
    and both overlaid.

    Entities, with <robot> franka or stretch:
      world/<robot>/<geom>        the robot's geometry, posed each step
      world/<robot>/tcp           its TCP's pose (axes); world/<robot>/tcp_cloud every TCP so far
      world/<robot>/workspace     every TCP position it can reach (sampled), and its envelope: a
                                  spherical shell about the Franka's shoulder, an annular cylinder
                                  about Stretch's base axis (`log_workspaces()`)
      world/franka/workspace/stretch_reaches
                                  Franka TCP poses (sampled), green where Stretch reaches the whole
                                  pose, red where it does not
      joint_space/<robot>         each joint's range as a bar, every value so far and the current one
      joints/<robot>/<joint>      joint positions (rad or m)
      joint_range/<robot>/<joint> each joint's position within its limits, 0 at the lower, 1 at the upper
      limit_margin/<robot>        the least of those distances to a limit, over the robot's joints
      task/<axis>/<robot>         the TCP's TASK_AXES
    The Franka's TCP is its grasp site in Stretch's tool axes with the grasp offset applied
    (`TCP_ALIGN @ tcp_offset`), so where retargeting is exact the two robots' TCPs coincide.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        franka: FrankaSpawn,
        retargeter: FrankaStretchRetargeter,
        world_from_footprint: np.ndarray,
        current: StretchJoints,
    ):
        """
        Log both robots' geometry, their workspaces, and the limits their joints are normalized against.

        Args:
            model: The scene with Stretch 4 (root body "stretch4") and the ghost Franka.
            franka: Where the ghost Franka is, for its root body and its workspace's centre.
            retargeter: The retargeter, for both robots' kinematics and joint limits.
            world_from_footprint: Stretch's footprint, about whose vertical its workspace turns.
            current: Stretch's joints, the seed for checking which Franka poses it reaches.
        """
        self.model = model
        self.franka = franka
        self.retargeter = retargeter
        self.robots = {
            "franka": RerunMujocoRobot(rr, model, franka.base_name, "world/franka", tint=[c / 255 for c in FRANKA_COLOR], alpha=0.5),
            "stretch": RerunMujocoRobot(rr, model, "stretch4", "world/stretch"),
        }
        franka_kinematics = retargeter.franka_kinematics
        # q of the IK model is [base, lift, arm, yaw, pitch, roll].
        ik = retargeter.stretch_kinematics.model_ik
        self.limits = {
            "franka": list(zip(franka_kinematics.lower, franka_kinematics.upper)),
            "stretch": [
                (-math.pi, math.pi),
                LIFT_RANGE,
                ARM_RANGE,
                (ik.lowerPositionLimit[3], ik.upperPositionLimit[3]),
                (ik.lowerPositionLimit[4], ik.upperPositionLimit[4]),
                WRIST_ROLL_RANGE,
            ],
        }
        self.joint_names = {"franka": [f"q{i + 1}" for i in range(7)], "stretch": list(STRETCH_JOINTS)}
        self.units = {"franka": ["rad"] * 7, "stretch": ["rad", "m", "m", "rad", "rad", "rad"]}
        self.tcp_clouds: dict[str, list[np.ndarray]] = {"franka": [], "stretch": []}
        self.visited: dict[str, list[list[float]]] = {"franka": [], "stretch": []}
        """Per robot, every step's joint-space point (x along the bar, y the joint's row) so far."""
        self._log_series_styles()
        self._log_joint_bars()
        self.log_workspaces(world_from_footprint, current)

    def log_workspaces(self, world_from_footprint: np.ndarray, current: StretchJoints) -> None:
        """
        Log (static) every TCP position each robot can reach, from WORKSPACE_SAMPLES joint
        configurations uniform within its limits (above the floor only), and the shape enclosing
        them: for the Franka a spherical shell about its shoulder (fr3_joint2), between the nearest
        and furthest sample; for Stretch an annular cylinder about its footprint's vertical, which
        its base turns it about, between the samples' least and greatest radius and height. Then
        REACH_SAMPLES of the Franka's TCP poses coloured by whether Stretch's IK reaches the whole
        pose (position and orientation), which is what following needs.

        Args:
            world_from_footprint: Stretch's footprint in the world.
            current: Stretch's joints, the IK seed.

        Returns:
            None.
        """
        from stretch4_kinematics import StretchJointPositions

        from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import solve_stretch_ik

        rng = np.random.default_rng(0)
        retargeter, franka_kinematics = self.retargeter, self.retargeter.franka_kinematics
        franka_tcp_from_link0 = TCP_ALIGN @ retargeter.tcp_offset

        # Franka: FK of random joints, its TCP in Stretch's tool axes with the grasp offset.
        franka_q = rng.uniform(franka_kinematics.lower, franka_kinematics.upper, (WORKSPACE_SAMPLES, 7))
        franka_poses = [self.franka.world_from_link0 @ franka_kinematics.fk(q) @ franka_tcp_from_link0 for q in franka_q]
        above = [pose for pose in franka_poses if pose[2, 3] > 0.0]
        franka_points = np.array([pose[:3, 3] for pose in above])
        shoulder = (self.franka.world_from_link0 @ [0.0, 0.0, FRANKA_SHOULDER_Z, 1.0])[:3]
        distances = np.linalg.norm(franka_points - shoulder, axis=1)
        r_in, r_out = float(distances.min()), float(distances.max())
        rr.log("world/franka/workspace/samples", rr.Points3D(franka_points, colors=[FRANKA_COLOR + [70]], radii=0.004), static=True)
        rr.log(
            "world/franka/workspace/envelope",
            rr.Ellipsoids3D(
                centers=[shoulder, shoulder], half_sizes=[[r_out] * 3, [r_in] * 3], colors=[FRANKA_COLOR, FRANKA_COLOR],
                fill_mode="MajorWireframe", line_radii=0.002,
                labels=[f"Franka reach: {r_out:.2f} m from the shoulder", f"{r_in:.2f} m"],
            ),
            static=True,
        )

        # Stretch: FK of random joints, the base turned anywhere about its footprint.
        low, high = zip(*self.limits["stretch"])
        stretch_q = rng.uniform(low, high, (WORKSPACE_SAMPLES, 6))
        stretch_points = []
        for base_yaw, *arm in stretch_q:
            joints = StretchJoints(*arm, gripper_open_fraction=1.0)
            pose = retargeter.stretch_tool_world(world_from_footprint @ planar_transform(0, 0, base_yaw), joints)
            if pose[2, 3] > 0.0:
                stretch_points.append(pose[:3, 3])
        stretch_points = np.array(stretch_points)
        axis = world_from_footprint[:3, 3]
        radii = np.linalg.norm(stretch_points[:, :2] - axis[:2], axis=1)
        tube = (float(radii.min()), float(radii.max()), float(stretch_points[:, 2].min()), float(stretch_points[:, 2].max()))
        rr.log("world/stretch/workspace/samples", rr.Points3D(stretch_points, colors=[STRETCH_COLOR + [70]], radii=0.004), static=True)
        vertices, triangles = annulus_mesh(axis, *tube)
        rr.log("world/stretch/workspace/envelope", rr.Mesh3D(vertex_positions=vertices, triangle_indices=triangles,
                                                             albedo_factor=STRETCH_COLOR + [40]), static=True)
        rr.log(
            "world/stretch/workspace/outline",
            rr.LineStrips3D(annulus_outline(axis, *tube), colors=[STRETCH_COLOR], radii=0.002),
            static=True,
        )
        rr.log(
            "world/stretch/workspace/label",
            rr.Points3D([[axis[0] + tube[1], axis[1], tube[3]]], colors=[STRETCH_COLOR], radii=0.004, show_labels=True,
                        labels=[f"Stretch reach: r {tube[0]:.2f}..{tube[1]:.2f} m, z {tube[2]:.2f}..{tube[3]:.2f} m"]),
            static=True,
        )

        # Which Franka TCP poses Stretch reaches exactly, its base free to turn.
        footprint_from_world = np.linalg.inv(world_from_footprint)
        seed = StretchJointPositions(0, 0, 0, current.lift, current.arm - retargeter.arm_offset,
                                     current.wrist_yaw, current.wrist_pitch, current.wrist_roll)
        checked, reached = [], []
        for pose in above[:REACH_SAMPLES]:
            goal = footprint_from_world @ pose @ retargeter.tool_correction_inv
            _, clamped = solve_stretch_ik(retargeter.stretch_kinematics, goal, seed, prefer_margin=True)
            checked.append(pose[:3, 3])
            reached.append(not clamped)
        share = float(np.mean(reached))
        rr.log(
            "world/franka/workspace/stretch_reaches",
            rr.Points3D(checked, colors=[REACHED_COLOR if r else NOT_REACHED_COLOR for r in reached], radii=0.01),
            static=True,
        )
        rr.log("metrics/franka_workspace_stretch_reaches_share", rr.Scalars(share), static=True)
        print(f"\nStretch reaches {share:.0%} of {len(checked)} sampled Franka TCP poses (6-D); "
              f"Franka shell {r_in:.2f}..{r_out:.2f} m, Stretch tube r {tube[0]:.2f}..{tube[1]:.2f} m, z {tube[2]:.2f}..{tube[3]:.2f} m")

    def _log_joint_bars(self) -> None:
        """
        Log (static) each joint's range as a bar in the joint-space view: one row per joint, the
        Franka's above Stretch's, the lower limit at x 0 and the upper at JOINT_BAR_WIDTH, labelled
        with the limits in the joint's units.

        Returns:
            None.
        """
        for robot, color in (("franka", FRANKA_COLOR), ("stretch", STRETCH_COLOR)):
            rows = [JOINT_ROWS[robot] + i for i in range(len(self.joint_names[robot]))]
            labels = [
                f"{robot} {joint} [{low:.2f}, {high:.2f}] {unit}"
                for joint, (low, high), unit in zip(self.joint_names[robot], self.limits[robot], self.units[robot])
            ]
            rr.log(
                f"joint_space/{robot}/limits",
                rr.Boxes2D(mins=[[0.0, row - 0.3] for row in rows], sizes=[[JOINT_BAR_WIDTH, 0.6]] * len(rows),
                           colors=[color + [90]], labels=labels, show_labels=True),
                static=True,
            )

    def _log_joint_space(self, robot: str, values: list[float], color: list[int]) -> None:
        """
        Log one step in the joint-space view: the robot's current joint values on their bars, and
        every value so far.

        Args:
            robot: "franka" or "stretch".
            values: Its joint values, in `joint_names[robot]` order.
            color: The robot's colour.

        Returns:
            None.
        """
        points = [
            [JOINT_BAR_WIDTH * (value - low) / (high - low), JOINT_ROWS[robot] + i]
            for i, (value, (low, high)) in enumerate(zip(values, self.limits[robot]))
        ]
        self.visited[robot].extend(points)
        rr.log(f"joint_space/{robot}/visited", rr.Points2D(self.visited[robot], colors=[color + [110]], radii=0.05))
        rr.log(f"joint_space/{robot}/current", rr.Points2D(points, colors=[[255, 255, 255]], radii=0.12))

    def _log_series_styles(self) -> None:
        """
        Name and colour every time series once (static): the Franka's in oranges, Stretch's in blues.

        Returns:
            None.
        """
        for robot, base in (("franka", FRANKA_COLOR), ("stretch", STRETCH_COLOR)):
            names = self.joint_names[robot]
            for i, joint in enumerate(names):
                # Shades of the robot's colour, darker to lighter along the chain.
                shade = [int(c * (0.45 + 0.55 * i / max(len(names) - 1, 1))) for c in base]
                style = rr.SeriesLines(colors=[shade], names=[f"{robot} {joint}"])
                rr.log(f"joints/{robot}/{joint}", style, static=True)
                rr.log(f"joint_range/{robot}/{joint}", style, static=True)
            rr.log(f"limit_margin/{robot}", rr.SeriesLines(colors=[base], names=[robot]), static=True)
            for axis in TASK_AXES:
                rr.log(f"task/{axis}/{robot}", rr.SeriesLines(colors=[base], names=[f"{robot} {axis}"]), static=True)

    def log(
        self,
        data: mujoco.MjData,
        franka_q7: np.ndarray,
        franka_tcp: np.ndarray,
        stretch_base_yaw: float,
        joints: StretchJoints,
        stretch_tcp: np.ndarray,
    ) -> None:
        """
        Log one step for both robots, at the current Rerun time.

        Args:
            data: Both robots posed (`pose_robots()`).
            franka_q7: The Franka's 7 arm joints (rad).
            franka_tcp: The Franka's TCP in the world, in Stretch's tool axes with the grasp offset.
            stretch_base_yaw: Stretch's heading in the world (rad).
            joints: Stretch's lift, arm and wrist joints.
            stretch_tcp: Stretch's `grasp_center_link` in the world.

        Returns:
            None.
        """
        for robot in self.robots.values():
            for geom, path in robot._paths.items():
                rr.log(path, rr.Transform3D(translation=data.geom_xpos[geom], mat3x3=data.geom_xmat[geom].reshape(3, 3)))
        values = {
            "franka": list(franka_q7),
            "stretch": [
                math.remainder(stretch_base_yaw, 2 * math.pi), joints.lift, joints.arm,
                joints.wrist_yaw, joints.wrist_pitch, joints.wrist_roll,
            ],
        }
        tcps = {"franka": franka_tcp, "stretch": stretch_tcp}
        for robot, color in (("franka", FRANKA_COLOR), ("stretch", STRETCH_COLOR)):
            margins = []
            for joint, value, (low, high) in zip(self.joint_names[robot], values[robot], self.limits[robot]):
                fraction = (value - low) / (high - low)
                rr.log(f"joints/{robot}/{joint}", rr.Scalars(float(value)))
                rr.log(f"joint_range/{robot}/{joint}", rr.Scalars(float(fraction)))
                if joint != "base_yaw":  # it turns without limit
                    margins.append(min(fraction, 1 - fraction))
            rr.log(f"limit_margin/{robot}", rr.Scalars(float(min(margins))))
            self._log_joint_space(robot, values[robot], color)

            tcp = tcps[robot]
            self.tcp_clouds[robot].append(tcp[:3, 3].copy())
            rr.log(f"world/{robot}/tcp", rr.Transform3D(translation=tcp[:3, 3], mat3x3=tcp[:3, :3]), rr.TransformAxes3D(0.08))
            rr.log(f"world/{robot}/tcp_cloud", rr.Points3D(self.tcp_clouds[robot], colors=[color], radii=0.008))
            for axis, value in zip(TASK_AXES, [*tcp[:3, 3], *euler_zyx(tcp)]):
                rr.log(f"task/{axis}/{robot}", rr.Scalars(float(value)))
        rr.log(
            "world/tcp_error",
            rr.LineStrips3D([[franka_tcp[:3, 3], stretch_tcp[:3, 3]]], colors=[[230, 230, 230]], radii=0.002),
        )


def blueprint() -> rrb.Blueprint:
    """
    `RetargetSpacesLogger`'s layout: a tab for the Franka alone, one for Stretch 4 alone, and one
    with both overlaid, each with the robot(s) and TCPs in 3D beside joint space and task space.

    Returns:
        The blueprint, to send to the recording.
    """

    def robot_tab(robot: str, name: str) -> rrb.Horizontal:
        """One robot alone: its 3D view, joint positions, their place within the limits, and its TCP."""
        return rrb.Horizontal(
            rrb.Spatial3DView(origin="world", contents=[f"+ world/{robot}/**"], name=f"{name}: task space (3D)"),
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"joint_space/{robot}", name=f"{name}: joint space (ranges, visited, current)"),
                rrb.TimeSeriesView(origin=f"joints/{robot}", name=f"{name}: joints"),
                rrb.TimeSeriesView(origin="task", contents=[f"+ task/x/{robot}", f"+ task/y/{robot}", f"+ task/z/{robot}"],
                                   name=f"{name}: TCP position (m)"),
                rrb.TimeSeriesView(origin="task", contents=[f"+ task/roll/{robot}", f"+ task/pitch/{robot}", f"+ task/yaw/{robot}"],
                                   name=f"{name}: TCP orientation (rad)"),
            ),
            column_shares=[3, 2],
            name=name,
        )

    overlap = rrb.Horizontal(
        rrb.Spatial3DView(origin="world", name="Both: task space (3D)"),
        rrb.Vertical(
            rrb.Grid(*[rrb.TimeSeriesView(origin=f"task/{axis}", name=f"TCP {axis}") for axis in TASK_AXES]),
            rrb.Horizontal(
                rrb.Spatial2DView(origin="joint_space", name="Both: joint space (ranges, visited, current)"),
                rrb.TimeSeriesView(origin="joint_range", name="Both: joints within limits (0..1)"),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="limit_margin", name="Closest to a joint limit (0..0.5)"),
                rrb.TimeSeriesView(origin="metrics", name="Retargeting error"),
            ),
            row_shares=[3, 2, 2],
        ),
        column_shares=[3, 2],
        name="Both overlaid",
    )
    return rrb.Blueprint(
        rrb.Tabs(robot_tab("franka", "Franka"), robot_tab("stretch", "Stretch 4"), overlap, active_tab=2),
        collapse_panels=True,
    )


@pytest.fixture
def rerun_recording(request):
    """
    A recording of its own for each test, saved to `<RERUN_SAVE or outputs/retargeting>/<test>.rrd`
    and, with a display and RERUN_SPAWN not 0, streamed to a viewer too, with `blueprint()`'s layout.

    Args:
        request: pytest's request, for the test's name and the repo root.

    Returns:
        The rerun module, logging to that recording.
    """
    # A recording of its own: `rr.init` reuses the process's recording id, so the parametrized
    # runs would share one, and the Stretch gripper's geoms the parallel gripper's scene lacks
    # would hang about where the first run left them.
    rr.init("stretch4_retargeting_test", recording_id=f"{request.node.name}-{uuid.uuid4().hex[:8]}", spawn=False)
    out_dir = Path(os.environ.get("RERUN_SAVE") or request.config.rootpath / "outputs" / "retargeting")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{request.node.name.replace('[', '_').rstrip(']')}.rrd"
    sinks = [rr.FileSink(str(path))]
    if os.environ.get("RERUN_SPAWN", "1") != "0" and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        rr.spawn(memory_limit="1GB", connect=False)
        sinks.append(rr.GrpcSink())
    rr.set_sinks(*sinks)
    rr.send_blueprint(blueprint())
    yield rr
    rr.disconnect()
    print(f"\nRerun recording: {path}  (rerun {path})")


@pytest.mark.parametrize("use_parallel_gripper", [False, True], ids=["stretch_gripper", "parallel_gripper"])
def test_stretch_follows_franka_across_wide_poses(rerun_recording, use_parallel_gripper):
    params = RetargetParams(use_parallel_gripper=use_parallel_gripper)
    model, data, franka = build_scene(params.tool_name)
    retargeter = FrankaStretchRetargeter(franka, params)

    # Stretch starts at its own home and follows from there, its base turning as IK asks.
    world_from_footprint = ROBOT_POSE.matrix()
    joints = StretchJoints(lift=0.6, arm=0.1, wrist_yaw=0.0, wrist_pitch=0.0, wrist_roll=0.0, gripper_open_fraction=1.0)
    pose_robots(model, data, franka, FRANKA_HOME_QPOS, world_from_footprint, joints)

    # The ghost is in a geom group MuJoCo hides by default; RerunMujocoRobot draws every visual geom.
    spaces = RetargetSpacesLogger(model, franka, retargeter, world_from_footprint, joints)

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

        rr.set_time("pose", sequence=index)
        base_yaw = math.atan2(world_from_footprint[1, 0], world_from_footprint[0, 0])
        spaces.log(data, franka_q7, franka_tcp, base_yaw, joints, stretch_tcp)
        for name, value in {"position_error_mm": position_mm, "rotation_error_deg": rotation_deg, "reverse_error_mm": reverse_mm}.items():
            rr.log(f"metrics/{name}", rr.Scalars(value))
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



@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("sensor_size", [(1920, 1200), (1280, 800)])
def test_rectified_fisheye_is_the_droid_exo_camera(side, sensor_size):
    """
    --rectify: a dot where OpenCV's fisheye model puts a ray in the landscape sensor frame, turned
    upright as the cameras' frames are, lands where a pinhole with the DROID exo camera's
    intrinsics puts that ray.
    """
    import cv2

    from examples.vla.molmobot_droid.droid import HEAD_CAMERAS

    camera = HEAD_CAMERAS[side]
    settings = camera.initial_camera_settings
    scale = sensor_size[0] / settings.width
    k = np.array([[settings.focal[0], 0, settings.optical_center[0]], [0, settings.focal[1], settings.optical_center[1]], [0, 0, 1]])
    k[:2] *= scale
    rectifier = head_rectifier(side)
    for u, v in [(319.5, 183.5), (100, 60), (550, 300), (20, 340)]:
        x, y = (u - 319.5) / DROID_EXO_FOCAL_PX, (v - 183.5) / DROID_EXO_FOCAL_PX
        # The upright image is the sensor's turned counterclockwise rotate_number_of_times.
        ray = {1: (-y, x, 1.0), 3: (y, -x, 1.0)}[settings.rotate_number_of_times % 4]
        pixel, _ = cv2.fisheye.projectPoints(np.array([[ray]]), np.zeros(3), np.zeros(3), k, np.array(settings.distortion_params[:4]))
        sensor = np.zeros((sensor_size[1], sensor_size[0], 3), np.uint8)
        cv2.circle(sensor, tuple(int(round(c)) for c in pixel[0, 0]), 4, (255, 255, 255), -1)
        out = rectifier(np.ascontiguousarray(np.rot90(sensor, settings.rotate_number_of_times))).sum(axis=2).astype(float)
        rows, cols = np.nonzero(out > out.max() / 2)
        weights = out[rows, cols]
        found = (cols @ weights / weights.sum(), rows @ weights / weights.sum())
        assert np.hypot(found[0] - u, found[1] - v) < 1.5, f"({u}, {v}) came out at {found}"


if __name__ == "__main__":
    pytest.main([__file__])