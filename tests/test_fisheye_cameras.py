import time

import numpy as np

from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.fisheye_renderer import fisheye_angles
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

HEAD_CAMERAS = [StretchCameras.cam_nav_rgb_se4_left, StretchCameras.cam_nav_rgb_se4_right]


def test_fisheye_angles_invert_kannala_brandt():
    settings = StretchCameras.cam_nav_rgb_se4_left.initial_camera_settings
    fx, fy = settings.focal
    cx, cy = settings.optical_center
    k1, k2, k3, k4 = settings.distortion_params
    theta, _, r_d = fisheye_angles(
        fx, fy, cx, cy, settings.distortion_params, settings.width, settings.height, 10.0
    )
    theta2 = theta * theta
    np.testing.assert_allclose(
        theta * (1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4)))), r_d, atol=1e-9
    )


def test_head_cameras_render_the_whole_image_circle():
    sim = Stretch4MujocoSimulator(cameras_to_use=HEAD_CAMERAS)
    sim.start(headless=True)
    try:
        time.sleep(2.0)
        cam_data = sim.pull_camera_data()

        for camera in HEAD_CAMERAS:
            settings = camera.initial_camera_settings
            image = cam_data.get_camera_data(camera, auto_rotate=False)
            assert image is not None
            # Calibration resolution, so the published K and D describe it.
            assert image.shape == (settings.height, settings.width, 3)

            cx, cy = settings.optical_center
            radius = settings.image_circle_radius_px

            # Past the image circle is black, like the real lens.
            assert not image[0, 0].any() and not image[-1, -1].any()
            assert not image[int(cy), int(cx + radius + 5)].any()

            # Just inside the circle along the long axis is more than 90
            # degrees off axis, which a single pinhole render cannot reach.
            band = image[int(cy) - 20 : int(cy) + 20, int(cx + radius - 80) : int(cx + radius - 50)]
            assert band.mean() > 5
    finally:
        sim.stop()
