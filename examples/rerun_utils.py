import queue
import threading
import numpy as np
import rerun as rr


import rerun.blueprint as rrb


import cv2


def get_default_blueprint(use_stretch_3:bool) -> rrb.Blueprint:
    # 1. Top row: Horizontally stack left, center, right navigation cameras
    nav_cameras_row = rrb.Horizontal(
        rrb.Spatial2DView(origin="world/cameras/cam_nav_rgb_se4_left", name="Nav Left"),
        rrb.Spatial2DView(origin="world/cameras/cam_nav_rgb_se4_center", name="Nav Center"),
        rrb.Spatial2DView(origin="world/cameras/cam_nav_rgb_se4_right", name="Nav Right"),
    )
    if use_stretch_3:
        nav_cameras_row = rrb.Horizontal(
            rrb.Spatial2DView(origin="world/cameras/cam_nav_rgb", name="Nav"),
        )

    # 2. Below top row: Vertically stack left, right, and depth gripper cameras
    gripper_cameras_col = rrb.Horizontal(
        rrb.Spatial2DView(origin="world/cameras/cam_gripper_se4_left_rgb", name="Gripper Left"),
        rrb.Spatial2DView(origin="world/cameras/cam_gripper_se4_right_rgb", name="Gripper Right"),
        rrb.Spatial2DView(origin="world/cameras/cam_gripper_se4_stereo_depth", name="Gripper Depth"),
    )
    if use_stretch_3:
        gripper_cameras_col = rrb.Horizontal(
        rrb.Spatial2DView(origin="world/cameras/cam_gripper_rgb", name="Gripper RGB"),
        rrb.Spatial2DView(origin="world/cameras/cam_gripper_depth", name="Gripper Depth"),
        )

    # Stack navigation row on top, gripper column below
    cameras_panel = rrb.Vertical(
        nav_cameras_row,
        gripper_cameras_col,
    )

    # 3. Side element: 3D point cloud spatial view
    pointcloud_view = rrb.Spatial3DView(origin="world", name="Lidar 3D Point Cloud")

    # Combine cameras panel on left, 3D view on right
    return rrb.Blueprint(
        rrb.Horizontal(
            cameras_panel,
            pointcloud_view,
        )
    )


class RerunLogger:
    def __init__(self, maxsize: int = 5):
        self._log_queue = queue.Queue(maxsize=maxsize)
        self._latest_images = {}
        self._latest_images_lock = threading.Lock()
        self._log_thread = None
        self._stop_event = threading.Event()
        self._initialized = False

    def _logging_worker(self):
        while not self._stop_event.is_set():
            # 1. Log queued pointclouds
            try:
                item = self._log_queue.get_nowait()
                item_type = item[0]
                if item_type == "pointcloud":
                    _, points, label = item
                    if isinstance(points, dict):
                        for points_name, points_instance in points.items():
                            if len(points_instance) > 0:
                                # Left lidar cyan, right lidar amber
                                c = [0, 220, 255] if "left" in points_name else [255, 180, 0]
                                rr.log(
                                    f"{label}/{points_name}",
                                    rr.Points3D(positions=points_instance, radii=0.008, colors=c),
                                )
                    elif isinstance(points, list):
                        for points_name, points_instance in points:
                            if len(points_instance) > 0:
                                # Left lidar cyan, right lidar amber
                                c = [0, 220, 255] if "left" in points_name else [255, 180, 0]
                                rr.log(
                                    f"{label}/{points_name}",
                                    rr.Points3D(positions=points_instance, radii=0.008, colors=c),
                                )
                    elif len(points) > 0:
                        rr.log(
                            label,
                            rr.Points3D(positions=points, radii=0.008, colors=[0, 220, 255]),
                        )
                self._log_queue.task_done()
            except queue.Empty:
                pass

            # 2. Log latest camera images (drops older backlog frames for zero lag)
            with self._latest_images_lock:
                current_images = self._latest_images.copy()
                self._latest_images.clear()

            for camera_name, pixels in current_images.items():
                rr.log(f"world/cameras/{camera_name}", rr.Image(pixels))

            threading.Event().wait(0.01)

    def init_rerun(self, use_stretch_3: bool = False):
        if not self._initialized:
            rr.init("Stretch4 Mujoco", spawn=False)
            rr.spawn(memory_limit='5GB')
            try:
                rr.send_blueprint(get_default_blueprint(use_stretch_3))
            except Exception as e:
                print(f"Warning: Failed to send Rerun blueprint: {e}")
            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # Set an up-axis
            rr.log(
                "world/xyz",
                rr.Arrows3D(
                    vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                ),
            )
            self._stop_event.clear()
            self._log_thread = threading.Thread(target=self._logging_worker, daemon=True)
            self._log_thread.start()
            self._initialized = True

    def init_pointcloud_viz(self, use_stretch_3: bool):
        self.init_rerun(use_stretch_3)

    def _put_queue(self, item):
        if self._stop_event.is_set():
            return
        while not self._stop_event.is_set():
            try:
                self._log_queue.put_nowait(item)
                break
            except queue.Full:
                try:
                    self._log_queue.get_nowait()
                except queue.Empty:
                    pass

    def update_pointcloud_viz(
        self,
        points: np.ndarray | list[tuple[str, np.ndarray]],
        label: str,
        use_stretch_3: bool = False,
    ):
        if not self._initialized:
            self.init_rerun(use_stretch_3)
        self._put_queue(("pointcloud", points, label))

    def update_camera_images(self, camera_data):
        if not self._initialized:
            self.init_rerun()
        images = camera_data.get_all(auto_rotate=True, auto_correct_rgb=False, use_depth_color_map=True)
        with self._latest_images_lock:
            for camera, pixels in images.items():
                if pixels is not None:
                    # Downsample high-res frames (e.g. 4032x3040 center camera) for real-time zero-lag IPC
                    h, w = pixels.shape[:2]
                    if max(h, w) > 1280:
                        scale = 1280.0 / max(h, w)
                        pixels = cv2.resize(pixels, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    self._latest_images[camera.name] = pixels

    def stop(self):
        self._stop_event.set()
        if self._log_thread:
            self._log_thread.join(timeout=2.0)


