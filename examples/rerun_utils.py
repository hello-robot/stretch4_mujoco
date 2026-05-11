import numpy as np
import rerun as rr


def init_pointcloud_viz():
    rr.init("Stretch4 Mujoco", spawn=False)
    rr.spawn(memory_limit='5GB')
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
    rr.log(
        "world/xyz",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
    )



def update_pointcloud_viz(points: np.ndarray | list[tuple[str, np.ndarray]], label):
    if isinstance(points, list):
        for points_name, points_instance in points:
            rr.log(f"{label}/{points_name}",
            rr.Points3D(points_instance))
        return

    rr.log(label,
       rr.Points3D(points))
