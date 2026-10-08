"""
Rerun view of a MolmoBot-DROID run: the whole MolmoSpaces scene in 3D, built from the same
MuJoCo model the simulation runs (so every mesh comes straight from molmospaces), the robot
and target object moving in it, the ghost Franka when there is one, the cameras the policy
sees, and the TCP target vs. where the tool actually is.

The geometry is drawn by `examples/digital_twin.py`'s `RerunMujocoRobot`.
"""

from __future__ import annotations

import mujoco
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from examples.digital_twin import RerunMujocoRobot

SCENE_PATH = "world/scene"
CAMERAS_PATH = "cameras"


def init_rerun(app_name: str, camera_names: list[str], spawn: bool = True) -> None:
    rr.init(app_name, spawn=False)
    if spawn:
        rr.spawn(memory_limit="5GB")
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="Scene"),
                rrb.Vertical(
                    *[rrb.Spatial2DView(origin=f"{CAMERAS_PATH}/{name}", name=name) for name in camera_names],
                    rrb.TimeSeriesView(origin="metrics", name="Metrics"),
                ),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )
    )


class RerunScene:
    """
    Logs every visual geom of `model` once, then only the ones under `moving_bodies` (the
    robot, the target object, the ghost) on each `log()`.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, moving_bodies: list[str]):
        self.model = model
        self.scene = RerunMujocoRobot(rr, model, "world", SCENE_PATH, tint=None)
        self.scene.log(data)
        moving = {model.body(name).id for name in moving_bodies if _has_body(model, name)}
        self.moving_geoms = [
            geom for geom in self.scene.geoms if _ancestor_in(model, int(model.geom_bodyid[geom]), moving)
        ]

    def log(self, data: mujoco.MjData) -> None:
        for geom in self.moving_geoms:
            rr.log(
                self.scene._paths[geom],
                rr.Transform3D(translation=data.geom_xpos[geom], mat3x3=data.geom_xmat[geom].reshape(3, 3)),
            )


def set_step(step: int) -> None:
    rr.set_time("step", sequence=step)


def log_cameras(images: dict[str, np.ndarray]) -> None:
    """RGB images, downscaled so the viewer stays responsive."""
    import cv2

    for name, image in images.items():
        if image is None:
            continue
        scale = 960 / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rr.log(f"{CAMERAS_PATH}/{name}", rr.Image(image).compress(jpeg_quality=85))


def log_tool_poses(target_world: np.ndarray | None, actual_world: np.ndarray | None) -> None:
    """Where the policy wants the tool (from the Franka) vs. where it is."""
    for name, pose, color in (("target", target_world, [255, 160, 0]), ("actual", actual_world, [0, 200, 255])):
        if pose is None:
            continue
        rr.log(f"world/tool/{name}", rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]))
        rr.log(f"world/tool/{name}", rr.TransformAxes3D(0.08), static=True)
        rr.log(f"world/tool/{name}/point", rr.Points3D([[0, 0, 0]], colors=[color], radii=0.01))


def log_metrics(values: dict[str, float]) -> None:
    for name, value in values.items():
        rr.log(f"metrics/{name}", rr.Scalars(float(value)))


def _has_body(model: mujoco.MjModel, name: str) -> bool:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) != -1


def _ancestor_in(model: mujoco.MjModel, body: int, bodies: set[int]) -> bool:
    while body != 0:
        if body in bodies:
            return True
        body = int(model.body_parentid[body])
    return False
