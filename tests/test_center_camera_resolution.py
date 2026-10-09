import time
from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator


def test_center_camera_outside_world_mapping():
    # To the outside world, get_all_cameras() returns cam_nav_rgb_se4_center
    all_cams = Stretch4MujocoSimulator.get_all_cameras()
    assert StretchCameras.cam_nav_rgb_se4_center in all_cams
    assert StretchCameras.cam_nav_rgb_se4_center_low_rez not in all_cams

    # 1. Test low-rez mode (default, use_full_center_camera_resolution=False)
    sim_low = Stretch4MujocoSimulator(
        cameras_to_use=[StretchCameras.cam_nav_rgb_se4_center],
        use_full_center_camera_resolution=False,
    )
    sim_low.start(headless=True)
    try:
        time.sleep(1.0)
        cam_data = sim_low.pull_camera_data()
        all_rendered = cam_data.get_all()

        # Outside world MUST see cam_nav_rgb_se4_center, NOT cam_nav_rgb_se4_center_low_rez
        assert StretchCameras.cam_nav_rgb_se4_center in all_rendered
        assert StretchCameras.cam_nav_rgb_se4_center_low_rez not in all_rendered

        img_raw = cam_data.get_camera_data(StretchCameras.cam_nav_rgb_se4_center, auto_rotate=False)
        assert img_raw is not None
        assert img_raw.shape[0] == 965
        assert img_raw.shape[1] == 1280
        print(f"Low-rez mode outside world image shape for cam_nav_rgb_se4_center: {img_raw.shape}")
    finally:
        sim_low.stop()

    # 2. Test full-rez mode (use_full_center_camera_resolution=True)
    sim_full = Stretch4MujocoSimulator(
        cameras_to_use=[StretchCameras.cam_nav_rgb_se4_center],
        use_full_center_camera_resolution=True,
    )
    sim_full.start(headless=True)
    try:
        time.sleep(1.0)
        cam_data = sim_full.pull_camera_data()
        all_rendered = cam_data.get_all()

        # Outside world MUST see cam_nav_rgb_se4_center, NOT cam_nav_rgb_se4_center_low_rez
        assert StretchCameras.cam_nav_rgb_se4_center in all_rendered
        assert StretchCameras.cam_nav_rgb_se4_center_low_rez not in all_rendered

        img_raw = cam_data.get_camera_data(StretchCameras.cam_nav_rgb_se4_center, auto_rotate=False)
        assert img_raw is not None
        assert img_raw.shape[0] == 3040
        assert img_raw.shape[1] == 4034
        print(f"Full-rez mode outside world image shape for cam_nav_rgb_se4_center: {img_raw.shape}")
    finally:
        sim_full.stop()

    print("All outside world center camera mapping tests passed successfully!")


if __name__ == "__main__":
    test_center_camera_outside_world_mapping()
