from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

FISHEYE_FACE_RESOLUTION_SCALE = 1.0
"""Cube face resolution as a fraction of the one that matches the lens.

A face `2 * f` pixels across samples the scene as densely at its centre as the
fisheye does at its optical centre (f pixels per radian), so 1.0 loses no
detail. Lower it to trade sharpness for render time: every face costs a render,
and the face count is fixed by the lens' field of view.
"""

FISHEYE_VIGNETTE_WIDTH_PX = 30.0
"""Width of the ring over which the image circle fades to black.

Measured on the SE4 head cameras, where the brightness at the edge of the image
circle drops from full to black over roughly 30 pixels.
"""

# Cube faces in the optical frame (x right, y down, z forward), as
# (forward, image-down). There is no back face: the head lenses see ~98 degrees
# off axis, and the four side faces already reach 135 degrees.
_FACES = (
    ((0, 0, 1), (0, 1, 0)),
    ((1, 0, 0), (0, 1, 0)),
    ((-1, 0, 0), (0, 1, 0)),
    ((0, 1, 0), (0, 0, -1)),
    ((0, -1, 0), (0, 0, 1)),
)


@dataclass
class _Face:
    forward: np.ndarray
    """Forward axis in the optical frame."""
    down: np.ndarray
    """Image-down axis in the optical frame."""
    u_min: float
    v_min: float
    rect: mujoco.MjrRect
    """The part of the face the lens actually sees, in pixels."""
    atlas_x: int
    pixels: np.ndarray


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
    """Renders a wide-angle fisheye camera from a cubemap.

    A single pinhole render cannot be warped into the SE4 head cameras: their
    lenses see ~98 degrees off axis (a ~196 degree image circle), and a pinhole
    camera puts anything at 90 degrees at infinity. So the scene is rendered
    onto five faces of a cube around the camera, and every fisheye pixel looks
    up the face its ray passes through. Only the part of each side face the
    lens actually sees is rendered.

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
        face_resolution_scale: float = FISHEYE_FACE_RESOLUTION_SCALE,
    ):
        focal = max(fx, fy)
        self.face_resolution = int(round(2 * focal * face_resolution_scale))
        n = self.face_resolution

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

        forwards = np.array([forward for forward, _ in _FACES], dtype=np.float64)
        face_index = (rays @ forwards.T).argmax(axis=-1)

        self.map_x = np.full((height, width), -1.0, dtype=np.float32)
        self.map_y = np.full((height, width), -1.0, dtype=np.float32)
        self._faces: list[_Face] = []
        atlas_x = 0
        for index, (forward, down) in enumerate(_FACES):
            forward = np.array(forward, dtype=np.float64)
            down = np.array(down, dtype=np.float64)
            right = np.cross(down, forward)

            on_face = (face_index == index) & seen
            if not on_face.any():
                continue
            face_rays = rays[on_face]
            depth = face_rays @ forward
            u = (face_rays @ right) / depth
            v = (face_rays @ down) / depth

            # Pad by a couple of pixels so bilinear sampling never reads past
            # the part of the face that was rendered. The pad runs past the
            # cube's edge where it has to, overlapping the next face a little;
            # stopping at the edge leaves a dark line along every seam.
            pad = 2.0 / n
            u_min = u.min() - pad
            v_min = v.min() - pad
            rect_width = int(np.ceil((u.max() + pad - u_min) * n / 2))
            rect_height = int(np.ceil((v.max() + pad - v_min) * n / 2))

            self.map_x[on_face] = atlas_x + (u - u_min) * n / 2 - 0.5
            self.map_y[on_face] = (v - v_min) * n / 2 - 0.5

            self._faces.append(
                _Face(
                    forward=forward,
                    down=down,
                    u_min=u_min,
                    v_min=v_min,
                    rect=mujoco.MjrRect(0, 0, rect_width, rect_height),
                    atlas_x=atlas_x,
                    pixels=np.empty((rect_height, rect_width, 3), dtype=np.uint8),
                )
            )
            atlas_x += rect_width

        self._render_width = max(face.rect.width for face in self._faces)
        self._render_height = max(face.rect.height for face in self._faces)
        self._atlas = np.zeros((self._render_height, atlas_x, 3), dtype=np.uint8)

    @property
    def render_size(self) -> tuple[int, int]:
        """`(width, height)` the `mujoco.Renderer` handed to `render_faces` needs."""
        return (self._render_width, self._render_height)

    def render_faces(self, renderer: mujoco.Renderer, data: mujoco.MjData, camera_name: str):
        """Render the cube faces. Reads `data`, so call it under the sim's lock."""
        renderer.update_scene(data=data, camera=camera_name)
        scene = renderer.scene

        # A headlight is lit along whichever way the GL camera faces, so each
        # face would be lit differently and the seams would show. Pin it in the
        # world, pointing where the real camera looks.
        for light_index in range(scene.nlight):
            scene.lights[light_index].headlight = 0

        gl_camera = scene.camera[0]
        # MuJoCo's camera looks down -z with +y up; the optical frame is that
        # rotated 180 degrees about x.
        z_axis = np.array(gl_camera.forward, dtype=np.float64)
        y_axis = -np.array(gl_camera.up, dtype=np.float64)
        optical_to_world = np.stack([np.cross(y_axis, z_axis), y_axis, z_axis], axis=1)
        near = gl_camera.frustum_near

        if renderer._gl_context:
            renderer._gl_context.make_current()

        half_face = self.face_resolution / 2
        for face in self._faces:
            forward = optical_to_world @ face.forward
            up = -(optical_to_world @ face.down)
            u_max = face.u_min + face.rect.width / half_face
            v_max = face.v_min + face.rect.height / half_face
            for camera in (scene.camera[0], scene.camera[1]):
                camera.forward[:] = forward
                camera.up[:] = up
                camera.frustum_center = 0.5 * (face.u_min + u_max) * near
                camera.frustum_width = 0.5 * (u_max - face.u_min) * near
                # GL's y points up, the image's v points down.
                camera.frustum_bottom = -v_max * near
                camera.frustum_top = -face.v_min * near

            mujoco.mjr_render(face.rect, scene, renderer._mjr_context)
            mujoco.mjr_readPixels(face.pixels, None, face.rect, renderer._mjr_context)
            self._atlas[: face.rect.height, face.atlas_x : face.atlas_x + face.rect.width] = face.pixels[::-1]

    def project(self) -> np.ndarray:
        """Warp the last `render_faces` into the fisheye frame."""
        frame = cv2.remap(
            self._atlas,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return cv2.multiply(frame, self._vignette, scale=1 / 255)
