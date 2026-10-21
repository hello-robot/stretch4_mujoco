"""
--apply_franka_gripper_image_mismatch_fix: Stretch 4's gripper camera image, redrawn as DROID's
wrist camera on the Franka would see the same scene, for the policy's wrist view.

With the grasps lined up (`TCP_ALIGN` and the grasp offset), DROID's wrist camera is pitched 19.5
degrees further toward the fingers than Stretch's, its focal length is 341 px at DROID_IMAGE_SIZE
against Stretch's 397 px, and the two are about 40 mm apart. So in Stretch's image things sit
lower and look bigger than the policy is used to.

The turn and the zoom are the same at every distance; the parallax of the cameras being apart is
not. A homography maps one image onto the other exactly for the points of one plane: here the
plane through the Franka's TCP, facing Stretch's camera, so what is between the fingers, about to
be grasped, is where DROID's camera would see it. Further out things are off by the parallax,
growing to about 50 px at 35 cm past the TCP (making it right everywhere would need depth).

A grasp offset cannot stand in for the turn: moving the tool sideways shifts the image as the
19.5 degrees do at one distance only, so a tuned offset works in some parts of the workspace and
not in others. With this fix, the grasp offset only has to line up the fingers.

The warp would leave Stretch's fingers where DROID's camera would not see them, and DROID's
camera sees further toward its fingers than Stretch's can. So:

1. Stretch's fingers are masked out of its image,
2. the rest is warped into DROID's camera,
3. anything there Stretch's camera did not see is filled with the colours around it, blurred,
4. the Robotiq is drawn over it as DROID's camera sees it.

Both cameras are rigid on their grippers, so where the fingers are in either image depends on
the opening alone: they are rendered once (from stretch4_mujoco's tool and molmospaces' Franka
DROID), at OPENING_STEPS openings, and the fix needs no simulator, so it runs on the real robot
as in sim.

The models' cameras are pinholes; the real gripper camera's wide lens is not. Given its
calibration (the robot's own, which the publisher sends with every image), the warp starts from
the lens's own distorted image: each of DROID's pixels is traced back through the homography and the
lens to the robot's image, and sampled there once. Without it, Stretch's fingertips, near the
image's corners, show past their mask, the lens having drawn them in toward the centre.
"""

from __future__ import annotations

import functools
import math

import cv2
import mujoco
import numpy as np

from examples.vla.molmobot_droid.checkpoint import DROID_IMAGE_SIZE, ROBOTIQ_DRIVER_CLOSED, ROBOTIQ_DRIVER_OPEN
from examples.vla.molmobot_droid.droid import (
    FRANKA_PREFIX,
    GHOST_GEOM_GROUP,
    FrankaKinematics,
    body_transform,
    make_transform,
    robotiq_joint_positions,
    site_transform,
    subtree_geoms,
)
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import STRETCH_TCP, TCP_ALIGN
from stretch4_mujoco.enums.stretch_cameras import StretchCameras

FIX_FLAG = "--apply_franka_gripper_image_mismatch_fix"

OPENING_STEPS = 21
"""Openings, closed to open, the fingers are rendered at; the nearest is used."""

FINGER_MARGIN_PX = 5
"""Stretch's fingers are masked out this much wider, for their edges, shading and, on the real
robot, calibration."""

FILL_LEVELS = 6
"""Pyramid levels `_fill()` averages over: at 1/64 of the image, every hole has colours around it."""

CV_FROM_MJ = np.diag([1.0, -1.0, -1.0, 1.0])
"""MuJoCo's camera axes (looking down -z, y up) to OpenCV's (looking down +z, y down), and back."""

TOOL_ROOT = "tool_attachment_site_link"
"""Everything of the tool's, fingers included, but not the gripper camera."""

FINGER_ACTUATORS = ("gripper_left_finger", "gripper_right_finger")


def _camera_pose(model: mujoco.MjModel, data: mujoco.MjData, camera: str) -> np.ndarray:
    """The camera's world pose, in OpenCV's camera axes."""
    i = model.camera(camera).id
    return make_transform(data.cam_xmat[i].reshape(3, 3), data.cam_xpos[i]) @ CV_FROM_MJ


def _pinhole(width: int, height: int, fovy_deg: float) -> np.ndarray:
    """K of a camera as MuJoCo renders it: square pixels, centred."""
    f = height / 2 / math.tan(math.radians(fovy_deg) / 2)
    return np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])


def _hide_all_but(model: mujoco.MjModel, shown: list[int]) -> None:
    keep = set(shown)
    model.geom_group[[g for g in range(model.ngeom) if g not in keep]] = GHOST_GEOM_GROUP


def _render(model, data, camera: str, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(RGB, where any shown geom is) of `camera` at `size` (width, height)."""
    width, height = size
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render()
    return rgb, segmentation[..., 0] >= 0


@functools.cache
def _droid_wrist_camera() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DROID's wrist camera: its pose in the Robotiq's `grasp_site` (OpenCV axes), its K at
    DROID_IMAGE_SIZE, and the Robotiq as it sees it at each of OPENING_STEPS openings, closed to
    open: (RGB, mask).
    """
    kinematics = FrankaKinematics()
    model, data = kinematics.model, kinematics.data
    camera = f"{FRANKA_PREFIX}gripper/wrist_camera"
    mujoco.mj_camlight(model, data)
    pose = np.linalg.inv(site_transform(data, f"{FRANKA_PREFIX}gripper/grasp_site")) @ _camera_pose(model, data, camera)
    k = _pinhole(*DROID_IMAGE_SIZE, model.cam_fovy[model.camera(camera).id])

    _hide_all_but(model, subtree_geoms(model, f"{FRANKA_PREFIX}gripper/base"))
    rgbs, masks = [], []
    for closed in np.linspace(1.0, 0.0, OPENING_STEPS):
        driver = ROBOTIQ_DRIVER_OPEN + closed * (ROBOTIQ_DRIVER_CLOSED - ROBOTIQ_DRIVER_OPEN)
        for name, value in robotiq_joint_positions(driver).items():
            data.joint(FRANKA_PREFIX + name).qpos = value
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        rgb, mask = _render(model, data, camera, DROID_IMAGE_SIZE)
        rgbs.append(rgb)
        masks.append(mask)
    return pose, k, (np.stack(rgbs), np.stack(masks))


@functools.cache
def _stretch_gripper_camera(tool_name: str, camera: StretchCameras) -> tuple[np.ndarray, np.ndarray]:
    """
    Stretch's gripper camera with `tool_name`: its pose in the tool's `grasp_center_link`
    (OpenCV axes), and where the tool is in its image (as MuJoCo renders it: centred) at each of
    OPENING_STEPS openings, closed to open.
    """
    from examples.vla.molmobot_droid.droid import apply_stretch_camera_settings
    from examples.vla.molmobot_droid.franka_retarget.stretch4_hardware_on_franka import _finger_range
    from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

    spec = mujoco.MjSpec.from_file(Stretch4MujocoSimulator.get_robot_xml_path(tool_name))
    for pair in list(spec.pairs):  # with the scene's floor, which is not loaded
        spec.delete(pair)
    model = spec.compile()
    data = mujoco.MjData(model)
    name = camera.camera_name_in_mjcf
    apply_stretch_camera_settings(model, name, camera)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    pose = np.linalg.inv(body_transform(data, STRETCH_TCP)) @ _camera_pose(model, data, name)

    _hide_all_but(model, subtree_geoms(model, TOOL_ROOT))
    settings = camera.initial_camera_settings
    finger_closed, finger_open = _finger_range(tool_name)
    joints = [model.joint(model.actuator_trnid[model.actuator(a).id, 0]).name for a in FINGER_ACTUATORS]
    masks = []
    for opening in np.linspace(0.0, 1.0, OPENING_STEPS):
        for joint in joints:
            data.joint(joint).qpos = finger_closed + opening * (finger_open - finger_closed)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        masks.append(_render(model, data, name, (settings.width, settings.height))[1])
    return pose, np.stack(masks)


class GripperImageFix:
    """
    Stretch's gripper camera image (`camera`, on `tool_name`) as DROID's wrist camera would see
    it, DROID_IMAGE_SIZE: the policy's wrist view in place of `stretch4_retarget.wrist_view()`.

    `calibration`: the robot's camera, (camera matrix, OpenCV distortion coefficients) at its
    image size. None: the camera is a pinhole, as the simulator renders it.
    """

    def __init__(
        self,
        tool_name: str,
        camera: StretchCameras,
        calibration: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        self.franka_camera, self.k_franka, (self.robotiq_rgbs, self.robotiq_masks) = _droid_wrist_camera()
        self.stretch_camera, finger_masks = _stretch_gripper_camera(tool_name, camera)
        settings = camera.initial_camera_settings
        self.size = (settings.width, settings.height)
        k_rendered = _pinhole(*self.size, settings.field_of_view_vertical_in_degrees)
        if calibration is not None:
            self.k_stretch = np.asarray(calibration[0], dtype=float).reshape(3, 3)
            self.distortion = np.asarray(calibration[1], dtype=float).ravel()
        else:
            self.k_stretch = k_rendered
            self.distortion = np.zeros(5)
        # The fingers as this camera would see them, from as the simulator rendered them.
        to_camera = self.k_stretch @ np.linalg.inv(k_rendered)
        margin = np.ones((2 * FINGER_MARGIN_PX + 1,) * 2, np.uint8)
        self.stretch_seen = np.stack([
            cv2.warpPerspective(cv2.dilate(mask.astype(np.uint8), margin), to_camera, self.size, flags=cv2.INTER_NEAREST) == 0
            for mask in finger_masks
        ])
        """Per opening, where Stretch's camera (as a pinhole) sees past its fingers."""
        self.lens_radius = self._lens_radius()
        self._maps: tuple[bytes, np.ndarray, np.ndarray, np.ndarray] | None = None

    def homography(self, tcp_offset: np.ndarray | None = None) -> np.ndarray:
        """
        Stretch's image to DROID's, exact on the plane through the Franka's TCP facing Stretch's
        camera. `tcp_offset`: Stretch's tool in the Franka's TCP (`RetargetParams.tcp_offset`,
        in Stretch's axes); default none.
        """
        tool = TCP_ALIGN if tcp_offset is None else TCP_ALIGN @ tcp_offset
        franka_from_stretch = np.linalg.inv(self.franka_camera) @ tool @ self.stretch_camera
        rotation, translation = franka_from_stretch[:3, :3], franka_from_stretch[:3, 3]
        # The TCP's depth in front of Stretch's camera: the plane is z = depth there.
        depth = (np.linalg.inv(tool @ self.stretch_camera) @ np.array([0.0, 0.0, 0.0, 1.0]))[2]
        plane = rotation + np.outer(translation, [0.0, 0.0, 1.0 / depth])
        return self.k_franka @ plane @ np.linalg.inv(self.k_stretch)

    def __call__(self, image: np.ndarray, driver_angle: float, tcp_offset: np.ndarray | None = None) -> np.ndarray:
        """
        `image` (RGB) as DROID's wrist camera would see it, with the Robotiq at `driver_angle`
        (the Franka state's last value) where Stretch's fingers were, at the same opening.
        """
        if (image.shape[1], image.shape[0]) != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        closed = (driver_angle - ROBOTIQ_DRIVER_OPEN) / (ROBOTIQ_DRIVER_CLOSED - ROBOTIQ_DRIVER_OPEN)
        opening = int(round((1.0 - float(np.clip(closed, 0.0, 1.0))) * (OPENING_STEPS - 1)))
        h = self.homography(tcp_offset)
        map_x, map_y, in_image = self._lens_maps(h)
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        seen = cv2.warpPerspective(
            self.stretch_seen[opening].astype(np.uint8), h, DROID_IMAGE_SIZE, flags=cv2.INTER_NEAREST
        ).astype(bool)
        out = _fill(warped, ~(seen & in_image))
        robotiq = self.robotiq_masks[opening]
        out[robotiq] = self.robotiq_rgbs[opening][robotiq]
        return out


    def _lens_maps(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For each of DROID's pixels, where in the robot's (distorted) image `h` takes it from:
        (x, y, whether that is in the image). Made again only when `h` changes.
        """
        key = h.tobytes()
        if self._maps is not None and self._maps[0] == key:
            return self._maps[1:]
        width, height = DROID_IMAGE_SIZE
        u, v = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
        pinhole = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3) @ np.linalg.inv(h).T
        rays = pinhole @ np.linalg.inv(self.k_stretch).T
        ahead = rays[:, 2] > 1e-6
        rays[~ahead] = [0.0, 0.0, 1.0]
        rays /= rays[:, 2:]
        # A lens model holds only as far out as the image goes: beyond, a rational one can fold back
        # into the image. So rays further from the axis than any of the image's edge are not in it.
        within_lens = np.hypot(rays[:, 0], rays[:, 1]) <= self.lens_radius
        rays[~within_lens] = [0.0, 0.0, 1.0]
        points = _project(rays, self.k_stretch, self.distortion, self.lens_radius)
        map_x, map_y = (points.reshape(height, width, 2).astype(np.float32)[..., i] for i in (0, 1))
        in_image = (ahead & within_lens).reshape(height, width)
        in_image &= (map_x >= 0) & (map_x <= self.size[0] - 1) & (map_y >= 0) & (map_y <= self.size[1] - 1)
        self._maps = (key, map_x, map_y, in_image)
        return map_x, map_y, in_image

    def _lens_radius(self) -> float:
        """How far from the axis (tan of the angle) the image's edge sees, at most, with 5% to spare."""
        width, height = self.size
        edge = [(x, y) for x in np.linspace(0, width - 1, 33) for y in (0, height - 1)]
        edge += [(x, y) for y in np.linspace(0, height - 1, 33) for x in (0, width - 1)]
        rays = cv2.undistortPoints(np.array(edge, np.float32).reshape(-1, 1, 2), self.k_stretch, self.distortion)
        return 1.05 * float(np.hypot(rays[..., 0], rays[..., 1]).max())


POLE_DENOMINATOR = 0.1
"""Where the rational model's denominator is smaller than this, `_project()` bridges its radial
factor from either side instead."""


def _project(rays: np.ndarray, k: np.ndarray, distortion: np.ndarray, max_radius: float) -> np.ndarray:
    """
    Pixels of `rays` (N x 3, z = 1) through OpenCV's rational lens model (k1 k2 p1 p2 k3 k4 k5 k6),
    as `cv2.projectPoints()` would, but for its poles. The radial factor (1 + k1 r^2 + k2 r^4 +
    k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6) of a fit like stretch-se4-4024's gripper camera's has
    both near zero at some radius (r = 0.87 there), where it shoots off; so it is tabulated out to
    `max_radius` and, where the denominator is near zero, bridged from the radii either side.
    """
    coefficients = np.zeros(14)
    coefficients[: len(distortion)] = distortion
    if np.any(coefficients[8:]):  # thin prism or tilt: not modelled here
        points, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), k, coefficients)
        return points.reshape(-1, 2)
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients[:8]
    r_table = np.linspace(0.0, max_radius, 4096)
    r2 = r_table**2
    numerator = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    denominator = 1 + k4 * r2 + k5 * r2**2 + k6 * r2**3
    good = np.abs(denominator) >= POLE_DENOMINATOR
    factor = np.interp(r_table, r_table[good], numerator[good] / denominator[good])

    x, y = rays[:, 0], rays[:, 1]
    r2 = x * x + y * y
    radial = np.interp(np.sqrt(r2), r_table, factor)
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([k[0, 0] * xd + k[0, 1] * yd + k[0, 2], k[1, 1] * yd + k[1, 2]], axis=-1)


def _fill(image: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """
    `image` with `holes` filled with the colours around them, blurred: the average of what is not
    a hole, over neighbourhoods that grow until they reach some (a push-pull pyramid). Unlike
    inpainting, it does not draw an object at a hole's edge out across the hole.
    """
    if not holes.any():
        return image.copy()
    weight = (~holes).astype(np.float32)
    levels = [(image.astype(np.float32) * weight[..., None], weight)]
    for _ in range(FILL_LEVELS):
        colour, weight = levels[-1]
        levels.append((cv2.pyrDown(colour), cv2.pyrDown(weight)))
    colour, weight = levels[-1]
    fill = colour / np.maximum(weight, 1e-6)[..., None]
    for colour, weight in reversed(levels[:-1]):
        coarser = cv2.resize(fill, (weight.shape[1], weight.shape[0]), interpolation=cv2.INTER_LINEAR)
        known = np.clip(weight, 0.0, 1.0)[..., None]
        fill = known * colour / np.maximum(weight, 1e-6)[..., None] + (1.0 - known) * coarser
    out = image.copy()
    out[holes] = np.clip(fill[holes], 0, 255).astype(np.uint8)
    return out


@functools.cache
def gripper_image_fix(tool_name: str, camera: StretchCameras) -> GripperImageFix:
    """A `GripperImageFix` without a calibration, made once per process (its renders take a few
    seconds the first time; a `GripperImageFix` made with one reuses them)."""
    return GripperImageFix(tool_name, camera)


def reject_with_fix(fix: bool, **flags: bool) -> None:
    """The policy's wrist view comes from the fix or from one of `flags` (by parameter name), not both."""
    import click

    for name, given in flags.items():
        if fix and given:
            raise click.BadParameter(f"cannot be used with --{name}", param_hint=FIX_FLAG)
