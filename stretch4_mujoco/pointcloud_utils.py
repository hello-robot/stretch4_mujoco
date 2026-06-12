import numpy as np

from stretch4_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.utils import Rx, Ry, Rz, URDFmodel
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


_cached_urdf_model = None


def _get_default_urdf_model():
    global _cached_urdf_model
    if _cached_urdf_model is None:
        import tempfile
        from stretch4_urdf import get_urdf
        from stretch4_mujoco.utils import URDFmodel

        model_name = "SE4"
        batch_name = "francis"
        tool_name = "eoa_wrist_dw4_tool_sg4"

        out_dir = tempfile.mkdtemp()
        urdf_path = get_urdf(
            model_name=model_name,
            batch_name=batch_name,
            tool_name=tool_name,
            output_dir=out_dir,
            description="mujoco_stretch_4",
        )
        _cached_urdf_model = URDFmodel(urdf_path)
    return _cached_urdf_model


def get_pointcloud_from_camera_status(
    camera_status: StatusStretchCameras,
    in_world_frame: bool,
    urdf_model: URDFmodel = None,
    joint_positions: dict[str, float] = None,
    base_pose: tuple[float, float, float] = None,
) -> list[tuple[str, np.ndarray]]:
    """Calculates a pointcloud from the simulated lidar depth cameras, returned in the world frame if requested.

    Get `camera_status` from `sim.pull_camera_data()`"""

    if not in_world_frame:
        return [
            (
                lidar.name,
                _get_pointcloud(
                    camera_status.get_camera_data(lidar, use_depth_color_map=False)
                ),
            )
            for lidar in StretchCameras.hemispherical_lidars()
        ]

    all_points = []

    if urdf_model is None:
        urdf_model = _get_default_urdf_model()

    if joint_positions is None:
        joint_positions = {
            "wrist_yaw": 0.0,
            "wrist_pitch": 0.0,
            "wrist_roll": 0.0,
            "lift": 0.0,
            "arm": 0.0,
            "head_pan": 0.0,
            "head_tilt": 0.0,
        }

    T_base_left = urdf_model.get_transform(joint_positions, "lidar_left_link")
    T_base_right = urdf_model.get_transform(joint_positions, "lidar_right_link")

    # We transform the point cloud to be in base_link frame
    T_world_left = T_base_left
    T_world_right = T_base_right

    left_translation = T_world_left[:3, 3]
    left_rot_inv = T_world_left[:3, :3].T

    right_translation = T_world_right[:3, 3]
    right_rot_inv = T_world_right[:3, :3].T

    for lidar in StretchCameras.hemispherical_lidars():
        try:
            points = _get_pointcloud(
                camera_status.get_camera_data(lidar, use_depth_color_map=False)
            )
            if "left" in lidar.name:
                points = (points) @ left_rot_inv + left_translation
            elif "right" in lidar.name:
                points = (points) @ right_rot_inv + right_translation
            else:
                raise Exception(f"Unknown lidar, can't do pointcloud transform: {lidar.name}")

            all_points.append((lidar.name, points))
            print(f"{lidar.name}: {points.shape}")
        except ValueError:
            ...
        except Exception as e:
            print(f"Failed to get pointcloud for {lidar.name}, {e=}")

    return all_points
