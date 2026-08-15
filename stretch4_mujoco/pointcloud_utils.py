import numpy as np

from stretch4_mujoco.datamodels.status_stretch_camera import StatusStretchCameras


def depth_to_points(depth_image, fx, fy, cx, cy):
    height, width = depth_image.shape

    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    valid = (depth_image > 0) & np.isfinite(depth_image)

    z = depth_image[valid]
    x = (xx[valid] - cy) * z / fx
    y = (yy[valid] - cx) * z / fy

    return (x, y, z)


def depth_to_point_cloud(depth_image, fx, fy, cx, cy):
    x, y, z = depth_to_points(depth_image, fx, fy, cx, cy)
    return np.stack((x, y, z), axis=-1)

