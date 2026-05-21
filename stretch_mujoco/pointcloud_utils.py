import numpy as np

from stretch_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch_mujoco.enums.stretch_cameras import StretchCameras
from stretch_mujoco.utils import Rx, Ry, Rz
from scipy.spatial.transform import Rotation

# TODO: Move this cam_hemilidar_top45 settings pull somewhere less ambiguous
camera_settings = StretchCameras.cam_hemilidar_right.initial_camera_settings

fx, fy, cx, cy = (
    camera_settings.focal[0],
    camera_settings.focal[1],
    camera_settings.width / 2,
    camera_settings.height / 2,
)


def _depth_to_point_cloud(depth_image, fx, fy, cx, cy):
    height, width = depth_image.shape

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    valid = (depth_image > 0) & np.isfinite(depth_image)

    z = depth_image[valid]
    x = (xx[valid] - cy) * z / fx
    y = (yy[valid] - cx) * z / fy

    points = np.stack((x, y, z), axis=-1)
    return points.reshape(-1, 3)


def _get_pointcloud(depth_buffer):
    points = _depth_to_point_cloud(depth_buffer, fx, fy, cx, cy)
    return points


def get_pointcloud_from_camera_status(
    camera_status: StatusStretchCameras, in_world_frame: bool
) -> list[tuple[str, np.ndarray]]:
    """Calculates a pointcloud from the simulated lidar depth cameras, returned in the world frame if requested.

    Get `camera_status` from `sim.pull_camera_data()`"""

    if not in_world_frame:
        return [
                (lidar.name, _get_pointcloud(camera_status.get_camera_data(lidar, use_depth_color_map=False)))
                for lidar in StretchCameras.hemispherical_lidars()
            ]

    all_points = []

    left_translation = np.array(
        [0.0387442, 0.124414, 1.50375]
    )  # from mjcf
    left_quat = np.array(
        [
            -0.853553,
            -0.353553,
            -0.339444,
            0.176704,
        ]
    )  # w last, from mjcv
    left_rot_inv = Rotation.from_quat(left_quat).inv().as_matrix()

    right_translation = np.array(
        [0.0387442, -0.124414, 1.50375]
    )  # from mjcf
    right_quat = np.array(
        [
            -0.353553,
            -0.853553,
            0.176704,
            -0.339444,
        ]
    )  # w last, from mjcv
    right_rot_inv = Rotation.from_quat(right_quat).inv().as_matrix()


    for lidar in StretchCameras.hemispherical_lidars():
        try:
            points = _get_pointcloud(
                camera_status.get_camera_data(lidar, use_depth_color_map=False)
            )
            if "left" in lidar.name:
                # rot_matrix = Rotation.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
                # points = (points) @ rot_matrix @ left_rot_inv + left_translation
                points = (points) @ left_rot_inv + left_translation
            elif "right" in lidar.name:
                rot_matrix = Rotation.from_euler("z", 180, degrees=True).as_matrix()
                points = (points) @ rot_matrix @ right_rot_inv + right_translation
            else:
                raise Exception(f"Unknown lidar, can't do pointcloud transform: {lidar.name}")

            all_points.append((lidar.name, points))
            print(f"{lidar.name}: {points.shape}")
        except ValueError:
            ...
        except Exception as e:
            print(f"Failed to get pointcloud for {lidar.name}, {e=}")

    return all_points
