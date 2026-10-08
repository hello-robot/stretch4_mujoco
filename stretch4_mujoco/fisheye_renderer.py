from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

FISHEYE_VIEW_RESOLUTION_SCALE = 0.75
"""View resolution as a fraction of the one that matches the lens.

At 1.0 each view is sized so that nowhere it is sampled is it coarser than the
fisheye frame, so no detail is lost. 0.75 renders a little over half the pixels
for a barely visible softening, which on an integrated GPU takes a head camera
from ~27ms to ~19ms a frame.
"""

FISHEYE_LENS_PROTRUSION_M = 0.006
"""How far in front of the camera frame the views are rendered from.

The head lenses stand proud of the shell, so the real cameras only see the
shell where it bulges out at the bottom of the frame. The optical frame sits
just behind the shell's opening, and rendering from there rings the whole image
circle with the edge of that opening.
"""

FISHEYE_VIGNETTE_WIDTH_PX = 30.0
"""Width of the ring over which the image circle fades to black.

Measured on the SE4 head cameras, where the brightness at the edge of the image
circle drops from full to black over roughly 30 pixels.
"""

View = tuple[tuple[float, float, float], tuple[float, float, float]]
"""A pinhole view as (forward, image-down) in the optical frame (x right, y down, z forward)."""


def views_yawed_about_y(*yaw_degrees: float) -> tuple[View, ...]:
    """Views fanned out across the frame's long (x) axis."""
    return tuple(
        ((float(np.sin(np.radians(yaw))), 0.0, float(np.cos(np.radians(yaw)))), (0.0, 1.0, 0.0))
        for yaw in yaw_degrees
    )


FISHEYE_VIEWS = views_yawed_about_y(0, -60, 60)
"""One view straight ahead and one either side along the frame's long axis.

Every view costs a fixed ~3ms on top of its pixels, but a pinhole view also
oversamples its edges by sec^2 of the angle off its axis, so fewer, wider views
pay for themselves in pixels instead. Three is the cheapest: two views ~50
degrees either side are slower, and the five faces of a cube are no faster.
"""


@dataclass
class _View:
    forward: np.ndarray
    """Forward axis in the optical frame."""
    down: np.ndarray
    """Image-down axis in the optical frame."""
    u_min: float
    v_min: float
    pixels_per_unit: float
    rect: mujoco.MjrRect
    """Where in the framebuffer the view renders the part of it the lens sees."""


def _smallest_step(u: np.ndarray, v: np.ndarray) -> float:
    """The least a one-pixel step in the frame moves `(u, v)`, in any direction.

    `u` and `v` are per-pixel maps, NaN where the view is not sampled. The
    answer is the smallest singular value of the map's Jacobian over the pixels
    the view serves, which is the finest detail the view has to resolve.
    """
    du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / 2
    dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / 2
    du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / 2
    dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / 2
    # Smallest eigenvalue of J^T J, in closed form.
    a = du_dx**2 + dv_dx**2
    b = du_dx * du_dy + dv_dx * dv_dy
    d = du_dy**2 + dv_dy**2
    smallest = 0.5 * (a + d) - np.sqrt(0.25 * (a - d) ** 2 + b**2)
    return float(np.sqrt(np.nanmin(np.maximum(smallest, 0.0))))


def fisheye_angles(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion_params: tuple,
    width: int,
    height: int,
    max_normalized_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(theta, phi, r_d)` for every pixel of a Kannala-Brandt fisheye frame.

    `theta` is the ray's angle off the optical axis, `phi` its direction around
    it and `r_d` the pixel's normalized distance from the optical centre. The
    model maps theta to `r_d = theta * (1 + k1 theta^2 + ... + k4 theta^8)`, so
    going from pixels to rays means inverting that polynomial, done here with
    Newton's method. `r_d` is clamped to `max_normalized_radius` first: past the
    image circle the polynomial is extrapolating, and those pixels are black.
    """
    k1, k2, k3, k4 = distortion_params[:4]
    grid_y, grid_x = np.indices((height, width), dtype=np.float64)
    x_d = (grid_x - cx) / fx
    y_d = (grid_y - cy) / fy
    r_d = np.hypot(x_d, y_d)

    target = np.minimum(r_d, max_normalized_radius)
    theta = target.copy()
    for _ in range(10):
        theta2 = theta * theta
        residual = theta * (1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))) - target
        slope = 1 + theta2 * (3 * k1 + theta2 * (5 * k2 + theta2 * (7 * k3 + theta2 * 9 * k4)))
        theta -= residual / slope

    return theta, np.arctan2(y_d, x_d), r_d


class FisheyeRenderer:
    """Renders a wide-angle fisheye camera from a few pinhole views.

    A single pinhole render cannot be warped into the SE4 head cameras: their
    lenses see ~98 degrees off axis (a ~196 degree image circle), and a pinhole
    camera puts anything at 90 degrees at infinity. So the scene is rendered
    from several pinhole views fanned out around the camera, and every fisheye
    pixel looks up the view nearest its ray. Each view renders only the part of
    the scene the lens sees through it.

    The frame comes back at the calibration's resolution and intrinsics, with
    no crop, so the published K and D describe it exactly, and with the image
    circle and its vignette where the real lens puts them.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        distortion_params: tuple,
        width: int,
        height: int,
        image_circle_radius_px: float,
        views: tuple[View, ...] = FISHEYE_VIEWS,
        view_resolution_scale: float = FISHEYE_VIEW_RESOLUTION_SCALE,
        lens_protrusion_m: float = FISHEYE_LENS_PROTRUSION_M,
    ):
        self.lens_protrusion_m = lens_protrusion_m
        focal = max(fx, fy)

        circle_radius = image_circle_radius_px / focal
        theta, phi, r_d = fisheye_angles(
            fx, fy, cx, cy, distortion_params, width, height, circle_radius
        )
        rays = np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
        )

        fade = np.clip((circle_radius - r_d) * focal / FISHEYE_VIGNETTE_WIDTH_PX, 0.0, 1.0)
        fade = fade * fade * (3.0 - 2.0 * fade)
        seen = fade > 0
        vignette = np.round(fade * 255).astype(np.uint8)
        self._vignette = cv2.merge([vignette] * 3)

        forwards = np.array([forward for forward, _ in views], dtype=np.float64)
        view_index = (rays @ forwards.T).argmax(axis=-1)

        self.map_x = np.full((height, width), -1.0, dtype=np.float32)
        self.map_y = np.full((height, width), -1.0, dtype=np.float32)
        self._views: list[_View] = []
        atlas_x = 0
        for index, (forward, down) in enumerate(views):
            forward = np.array(forward, dtype=np.float64)
            down = np.array(down, dtype=np.float64)
            right = np.cross(down, forward)

            in_view = (view_index == index) & seen
            if not in_view.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                depth = rays @ forward
                u_all = np.where(in_view, (rays @ right) / depth, np.nan)
                v_all = np.where(in_view, (rays @ down) / depth, np.nan)
            u = u_all[in_view]
            v = v_all[in_view]

            # Face pixels per unit of u and v. A pinhole view samples its edges
            # far more densely than its centre, so rather than a fixed size,
            # each view gets just enough that no fisheye pixel it serves is
            # coarser than the lens itself.
            pixels_per_unit = view_resolution_scale / _smallest_step(u_all, v_all)

            # Pad by a couple of pixels so bilinear sampling never reads past
            # the part of the view that was rendered. Stopping exactly at the
            # boundary with the next view leaves a dark line along every seam.
            pad = 2.0 / pixels_per_unit
            u_min = u.min() - pad
            v_min = v.min() - pad
            rect_width = int(np.ceil((u.max() + pad - u_min) * pixels_per_unit))
            rect_height = int(np.ceil((v.max() + pad - v_min) * pixels_per_unit))

            # The views render side by side into one framebuffer that is read
            # back in one go. OpenGL's rows run bottom up, so v is flipped here
            # rather than flipping the pixels after every read.
            self.map_x[in_view] = atlas_x + (u - u_min) * pixels_per_unit - 0.5
            self.map_y[in_view] = rect_height - 0.5 - (v - v_min) * pixels_per_unit

            self._views.append(
                _View(
                    forward=forward,
                    down=down,
                    u_min=u_min,
                    v_min=v_min,
                    pixels_per_unit=pixels_per_unit,
                    rect=mujoco.MjrRect(atlas_x, 0, rect_width, rect_height),
                )
            )
            atlas_x += rect_width

        atlas_height = max(view.rect.height for view in self._views)
        self._atlas_rect = mujoco.MjrRect(0, 0, atlas_x, atlas_height)
        self._atlas = np.zeros((atlas_height, atlas_x, 3), dtype=np.uint8)

    @property
    def render_size(self) -> tuple[int, int]:
        """`(width, height)` the `mujoco.Renderer` handed to `render_views` needs."""
        return (self._atlas_rect.width, self._atlas_rect.height)

    def render_views(self, renderer: mujoco.Renderer, data: mujoco.MjData, camera_name: str):
        """Render the views. Reads `data`, so call it under the sim's lock."""
        renderer.update_scene(data=data, camera=camera_name)
        scene = renderer.scene

        # A headlight is lit along whichever way the GL camera faces, so each
        # view would be lit differently and the seams would show. Pin it in the
        # world, pointing where the real camera looks.
        for light_index in range(scene.nlight):
            scene.lights[light_index].headlight = 0

        gl_camera = scene.camera[0]
        # MuJoCo's camera looks down -z with +y up; the optical frame is that
        # rotated 180 degrees about x.
        z_axis = np.array(gl_camera.forward, dtype=np.float64)
        y_axis = -np.array(gl_camera.up, dtype=np.float64)
        optical_to_world = np.stack([np.cross(y_axis, z_axis), y_axis, z_axis], axis=1)
        position = np.array(gl_camera.pos, dtype=np.float64) + z_axis * self.lens_protrusion_m
        near = gl_camera.frustum_near

        if renderer._gl_context:
            renderer._gl_context.make_current()

        for view in self._views:
            forward = optical_to_world @ view.forward
            up = -(optical_to_world @ view.down)
            u_max = view.u_min + view.rect.width / view.pixels_per_unit
            v_max = view.v_min + view.rect.height / view.pixels_per_unit
            for camera in (scene.camera[0], scene.camera[1]):
                camera.pos[:] = position
                camera.forward[:] = forward
                camera.up[:] = up
                camera.frustum_center = 0.5 * (view.u_min + u_max) * near
                camera.frustum_width = 0.5 * (u_max - view.u_min) * near
                # GL's y points up, the image's v points down.
                camera.frustum_bottom = -v_max * near
                camera.frustum_top = -view.v_min * near

            mujoco.mjr_render(view.rect, scene, renderer._mjr_context)

        mujoco.mjr_readPixels(self._atlas, None, self._atlas_rect, renderer._mjr_context)

    def project(self) -> np.ndarray:
        """Warp the last `render_views` into the fisheye frame."""
        frame = cv2.remap(
            self._atlas,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return cv2.multiply(frame, self._vignette, scale=1 / 255)
