"""
Stretch 4 following a Franka DROID blind, to turn MolmoBot-DROID's Franka episodes into Stretch 4
demonstrations: the policy drives the Franka and sees only the Franka's cameras, and Stretch 4
does each of its actions in the world through retargeting (`FrankaStretchRetargeter`).

- The Franka is simulated alone, in a model of its own (`franka_alone()`): its joints move as
  on the Franka DROID, but it touches nothing, so only Stretch moves things.
- The policy's views are the Franka's DROID exo and wrist cameras, rendered in the scene that
  Stretch's simulator runs (`SceneMirror`), with the Franka drawn where its joints are and
  Stretch hidden. Stretch's own cameras never see the Franka (it is in the ghost's geom group).
- The policy's state is the Franka's own joints, not Stretch's mapped back. Only the gripper's
  opening is Stretch's: the Franka's Robotiq would close on nothing, where Stretch's stops on the
  object it holds, and the Robotiq is drawn closed as far.
- Stretch's cameras are recorded raw and as a policy would be shown them, the head fisheyes
  also rectified to the DROID exo camera, and the fisheye on the tool's side as
  `run_stretch4_sim.py` picks its exo view (`fisheye_tool`) all three ways (`stretch_views()`),
  and its joints and joint targets as `finetune.stretch4_joints` values (`trajectory`), for
  `save_trajectory()`.

Used by `run_follow_franka.py` and `run_benchmark_follow_franka.py`.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from examples.vla.molmobot_droid.checkpoint import FRANKA_HOME_QPOS, GRIPPER_CLOSED, robotiq_driver_angle
from examples.vla.molmobot_droid.droid import (
    FRANKA_PEDESTAL_HEIGHT,
    FrankaDroidEnv,
    Observation,
    RobotPose,
    spawn_franka_droid,
)
from examples.vla.molmobot_droid.finetune import stretch4_joints
from examples.vla.molmobot_droid.finetune.dataset import StretchTrajectory, write_episode
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import (
    MIN_BASE_ROTATION,
    RetargetParams,
    Stretch4SimEnv,
    StretchTargets,
    _wrap,
    head_rectifier,
    planar_from_pose,
    to_droid_frame,
    wrist_view,
)
from examples.vla.molmobot_droid.molmospaces.custom_scene import DROID_EXO_IN_STRETCH_SCENE, STRETCH_ROOT_BODY, CustomScene
from stretch4_mujoco.enums.stretch_cameras import StretchCameras

STRETCH_CAMERAS: dict[str, StretchCameras] = {
    "fisheye_left": StretchCameras.cam_nav_rgb_se4_left,
    "fisheye_right": StretchCameras.cam_nav_rgb_se4_right,
    # The simulator renders the low-resolution one (1280x965) under this name.
    "center": StretchCameras.cam_nav_rgb_se4_center,
    "gripper_left": StretchCameras.cam_gripper_se4_left_rgb,
    "gripper_right": StretchCameras.cam_gripper_se4_right_rgb,
}
"""Every Stretch camera recorded, by the name its videos and the h5's `sensor_data` use."""

UNCROPPED = ("center",)
"""Recorded only as rendered: the others are also recorded as a policy would be shown them."""

RECTIFIED = ("fisheye_left", "fisheye_right")
"""Also recorded redrawn as the DROID exo camera would see them (`<name>_rectified`, as --rectify does)."""

TOOL_FISHEYE = "fisheye_tool"
"""The head fisheye on the side the tool is toward (`head_camera_for_tool()`), as
`run_stretch4_sim.py` and the real robot pick the exo view: `fisheye_left` or `fisheye_right`
each step, raw, cropped and rectified as they are."""

STRETCH_VIEWS: tuple[str, ...] = tuple(
    view
    for name in (*STRETCH_CAMERAS, TOOL_FISHEYE)
    for view in ((name,) if name in UNCROPPED else (f"{name}_raw", name))
    + ((f"{name}_rectified",) if name in (*RECTIFIED, TOOL_FISHEYE) else ())
)
"""What `stretch_views()` returns given a `tool_side`, in order."""

GRID_EXCLUDED: tuple[str, ...] = (
    tuple(
        view for view in STRETCH_VIEWS
        if view.endswith("_raw") or (view.endswith("_rectified") and view != f"{TOOL_FISHEYE}_rectified")
    )
    + UNCROPPED
)
"""Left out of the grid videos (their own videos have them): it shows the Franka's views, Stretch's
cropped ones, and TOOL_FISHEYE's cropped and rectified."""


def stretch_views(
    images: dict[str, np.ndarray], rectify: bool = False, tool_side: str | None = None
) -> dict[str, np.ndarray]:
    """
    Each Stretch camera's image as rendered (`<name>_raw`) and as a policy would be shown it
    (`<name>`): the head fisheyes DROID-framed as --head-crop droid frames them (with `rectify`,
    redrawn as the DROID exo camera, as --rectify does), the gripper cameras as the retargeted
    policy's wrist view. The center camera only as rendered. The head fisheyes also redrawn as
    the DROID exo camera whatever `rectify` is (`<name>_rectified`). With `tool_side` (left or
    right), that side's fisheye views again as TOOL_FISHEYE's.
    """
    views = {}
    for name, image in images.items():
        if name in UNCROPPED:
            views[name] = image
            continue
        views[f"{name}_raw"] = image
        if name.startswith("gripper"):
            views[name] = wrist_view(image)
            continue
        rectified = head_rectifier(name.removeprefix("fisheye_"))(image)
        views[name] = rectified if rectify else to_droid_frame(image)
        views[f"{name}_rectified"] = rectified
    if tool_side is not None:
        source = f"fisheye_{tool_side}"
        for suffix in ("_raw", "", "_rectified"):
            views[f"{TOOL_FISHEYE}{suffix}"] = views[f"{source}{suffix}"]
    return views


def follow_flags(params: RetargetParams, custom_franka_start_pose: bool) -> dict[str, object]:
    """The run's flags, for run names and reports: `params.flags()` without the ones for which of
    Stretch's cameras the policy sees, since it sees the Franka's."""
    flags = {k: v for k, v in params.flags().items() if k not in ("exo_camera", "gripper_camera", "head-crop", "head_follows_tool")}
    if custom_franka_start_pose:
        flags["custom_franka_start_pose"] = True
    return flags


def franka_alone(scene: CustomScene) -> FrankaDroidEnv:
    """
    The Franka DROID by itself, on molmospaces' pedestal with nothing else in its model,
    simulated at the policy's rate as `FrankaDroidEnv` simulates it in `scene`'s house: what the
    policy drives while Stretch follows. Where it stands does not change how its joints move, so
    it stands at the origin; its views are rendered where the virtual Franka is.

    It takes the house's physics options (`<option>`: integrator, cone, impratio, noslip,
    gravity), without which its joints follow the same actions differently from the Franka's in
    the house (by up to 0.03 rad over a reach), and the policy is shown another arm's state.
    """
    spec = mujoco.MjSpec()
    house = mujoco.MjSpec.from_file(scene.scene_xml).option
    for name in dir(house):
        if not name.startswith("_"):
            setattr(spec.option, name, getattr(house, name))
    spawn = spawn_franka_droid(spec, RobotPose(0.0, 0.0, 0.0), FRANKA_PEDESTAL_HEIGHT, exo_camera=None)
    return FrankaDroidEnv(spec.compile(), spawn, render_cameras=False)


class FollowFrankaEnv(Stretch4SimEnv):
    """
    The policy drives `franka`; Stretch 4 follows each action blind (see the module docstring).
    Stepped by the rollout loop and the benchmark like any env; success is judged on Stretch
    (`robot_root`), since it is all that touches anything.

    `reset()` puts both robots at `start_q7` and starts a new `trajectory`.
    """

    def __init__(self, sim, stretch_scene, params: RetargetParams, mirror):
        if not stretch_scene.include_franka:
            raise ValueError("Following the Franka needs it in the scene (include_franka) to render its cameras")
        super().__init__(sim, stretch_scene, params, mirror)
        self.franka = franka_alone(stretch_scene.scene)
        self.trajectory = StretchTrajectory()
        self.start_yaw = 0.0
        """Stretch's heading at `reset()`: its state's base_yaw is from there."""
        self.tool_errors: list[float] = []
        """m. After each step, how far Stretch's tool is from where the Franka's action put it."""
        # `exo_camera` follows the tool, for TOOL_FISHEYE: the policy's views are the Franka's either way.
        self.head_follows_tool = True

    def cameras(self) -> list[StretchCameras]:
        return list(STRETCH_CAMERAS.values())

    def base_pose(self, world_from_footprint: np.ndarray) -> np.ndarray:
        """x, y (m) and yaw (rad) of Stretch's footprint in the world."""
        return np.array([
            world_from_footprint[0, 3], world_from_footprint[1, 3],
            math.atan2(world_from_footprint[1, 0], world_from_footprint[0, 0]),
        ])

    def move_to_franka_pose(self, franka_q7=FRANKA_HOME_QPOS, timeout: float = 30.0) -> StretchTargets:
        """Stretch to where the Franka at `franka_q7` has its tool (its base stays where it is), then
        the Franka put there. If Stretch cannot get there, neither moves."""
        targets = super().move_to_franka_pose(franka_q7, timeout)
        self.franka.reset(franka_q7)
        return targets

    def reset(self) -> None:
        """Both robots to `start_q7` (`move_to_franka_pose()`), starting a new `trajectory`."""
        self.move_to_franka_pose(self.start_q7)
        self.start_yaw = float(self.base_pose(self.world_from_footprint())[2])
        self.trajectory = StretchTrajectory()
        self.tool_errors = []

    def observe(self) -> Observation:
        camera_data = self.fresh_camera_data()
        status = self.sim.pull_status()
        poses = self.sim.pull_body_poses()
        footprint = planar_from_pose(*poses[STRETCH_ROOT_BODY], self.stretch_scene.scene.floor_z)
        joints = self.joints(status)
        self.observed_state = (status.time, footprint, joints)
        self.mirror.update(status, poses)
        self.update_exo_side(joints)

        q7 = self.franka.state8[:7]
        driver = robotiq_driver_angle(joints.gripper_open_fraction)
        state8 = np.append(q7, driver)
        self.mirror.set_ghost(q7)
        # The Franka's own cameras, with the Franka in its own look and none of Stretch.
        exo = self.mirror.render_with_franka(DROID_EXO_IN_STRETCH_SCENE, driver, "franka_alone")
        wrist = self.mirror.render_with_franka(self.stretch_scene.franka.wrist_camera_name, driver, "franka_alone")

        images = {name: camera_data.get_camera_data(camera, auto_correct_rgb=False) for name, camera in STRETCH_CAMERAS.items()}
        base = self.base_pose(footprint)
        self.trajectory.observe(
            stretch4_joints.state8(_wrap(base[2] - self.start_yaw), joints), state8, base, status.time
        )
        return Observation(exo_rgb=exo, wrist_rgb=wrist, state8=state8, extra_cameras=stretch_views(images, self.params.rectify, self.exo_camera))

    def step(self, action8: np.ndarray) -> StretchTargets | None:
        joints = self.joints()
        footprint = self.world_from_footprint()
        targets = self.retargeter.franka_to_stretch(action8, footprint, joints)
        closed = bool(action8[7] >= GRIPPER_CLOSED / 2)
        start = self.sim.pull_status().time
        if targets is not None:
            self.send(targets)
        else:
            self.send_gripper(closed)
        self.franka.step(action8)
        self.wait(start)

        yaw = _wrap(self.base_pose(footprint)[2] - self.start_yaw)
        if targets is None:  # Stretch holds still
            stretch_action = stretch4_joints.action8(
                yaw, joints.lift, joints.arm, joints.wrist_yaw, joints.wrist_pitch, joints.wrist_roll, closed
            )
        else:
            # The base turns only by more than MIN_BASE_ROTATION (`send()`).
            turn = targets.base_rotate_by if abs(targets.base_rotate_by) > MIN_BASE_ROTATION else 0.0
            stretch_action = stretch4_joints.action8(
                yaw + turn, targets.lift, targets.arm, targets.wrist_yaw, targets.wrist_pitch, targets.wrist_roll,
                targets.gripper_closed,
            )
        self.trajectory.act(stretch_action, action8, self._reach(targets))
        error = self.tool_error_m()
        if error is not None:
            self.tool_errors.append(error)
        return targets

    def _reach(self, targets: StretchTargets | None) -> str:
        if targets is None:
            return "unreachable"
        miss = self.retargeter.last_miss
        if miss is not None and miss.kind == "twisted":
            return "twisted"
        return "clamped" if targets.clamped else "reached"

    def stats(self) -> dict[str, float | int]:
        stats = super().stats()
        stats.pop("reverse_ik_failures")  # the policy's state is the Franka's own, never Stretch's mapped back
        stats["ik_twisted"] = self.retargeter.ik_twisted
        if self.tool_errors:
            stats["tool_error_mean_m"] = float(np.mean(self.tool_errors))
            stats["tool_error_max_m"] = float(np.max(self.tool_errors))
        return stats

    def close(self) -> None:
        super().close()
        self.franka.close()


def save_trajectory(env: FollowFrankaEnv, out_dir: Path, prefix: str, instruction: str, task_type: str,
                    attrs: dict) -> Path:
    """
    `env.trajectory` as `<prefix>.h5` in `out_dir`, pointing at the videos of Stretch's cameras
    `benchmark.EpisodeRecorder` wrote there for the same episode (`<prefix>_<view>.mp4`).
    """
    from examples.vla.molmobot_droid.molmospaces.benchmark import video_path

    videos = {view: video_path(out_dir, prefix, view).name for view in STRETCH_VIEWS}
    missing = [name for name in videos.values() if not (out_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"No videos {missing} in {out_dir}")
    attrs = {
        "tool_name": env.params.tool_name,
        "grasp_offset_mm": list(env.params.effective_grasp_offset_mm),
        "grasp_offset_deg": list(env.params.grasp_offset_deg),
        "start_q7": [float(q) for q in env.start_q7],
        "franka_link0_height": float(env.stretch_scene.franka.world_from_link0[2, 3]),
        **attrs,
    }
    return write_episode(out_dir / f"{prefix}.h5", env.trajectory, videos, instruction, task_type, attrs)
