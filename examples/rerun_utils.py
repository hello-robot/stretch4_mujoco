import queue
import threading
import numpy as np
import rerun as rr


import rerun.blueprint as rrb


import cv2


# Fallbacks for when the caller doesn't say which cameras are streaming.
# These mirror `StretchCameras.all_stretch4()` / `all_stretch3()`.
_DEFAULT_SE4_CAMERAS = [
    "cam_nav_rgb_se4_left",
    "cam_nav_rgb_se4_right",
    "cam_nav_rgb_se4_center",
    "cam_gripper_se4_left_rgb",
    "cam_gripper_se4_right_rgb",
    "cam_gripper_se4_stereo_depth",
]
_DEFAULT_SE3_CAMERAS = [
    "cam_nav_rgb",
    "cam_d435i_rgb",
    "cam_d435i_depth",
    "cam_gripper_rgb",
    "cam_gripper_depth",
]

# Prettify the raw StretchCameras enum names for the view tabs.
_CAMERA_DISPLAY_NAMES = {
    "cam_nav_rgb": "Nav",
    "cam_nav_rgb_se4_left": "Nav Left",
    "cam_nav_rgb_se4_right": "Nav Right",
    "cam_nav_rgb_se4_center": "Nav Center",
    "cam_nav_rgb_se4_center_low_rez": "Nav Center (low res)",
    "cam_d435i_rgb": "D435i RGB",
    "cam_d435i_depth": "D435i Depth",
    "cam_gripper_rgb": "Gripper RGB",
    "cam_gripper_depth": "Gripper Depth",
    "cam_gripper_se4_left_rgb": "Gripper Left",
    "cam_gripper_se4_right_rgb": "Gripper Right",
    "cam_gripper_se4_stereo_depth": "Gripper Depth",
}


def _camera_view(camera_name: str) -> rrb.Spatial2DView:
    return rrb.Spatial2DView(
        origin=f"world/cameras/{camera_name}",
        name=_CAMERA_DISPLAY_NAMES.get(camera_name, camera_name),
    )


def get_default_blueprint(
    use_stretch_3: bool, camera_names: list[str] | None = None
) -> rrb.Blueprint:
    """
    Build the viewer layout: camera feeds on the left, 3D lidar and metrics on
    the right.

    Pass `camera_names` (the `StretchCameras` member names that are actually
    being logged) to get a view per live camera and nothing else -- otherwise
    the layout covers every camera the robot has, and the ones that are not
    streaming show up as empty panels.
    """
    if camera_names is None:
        camera_names = _DEFAULT_SE3_CAMERAS if use_stretch_3 else _DEFAULT_SE4_CAMERAS

    # Head/nav cameras on one row, wrist/gripper cameras on the next.
    nav_cameras = [name for name in camera_names if "nav" in name or "d435i" in name]
    gripper_cameras = [name for name in camera_names if "gripper" in name]

    camera_rows = []
    if nav_cameras:
        camera_rows.append(rrb.Horizontal(*[_camera_view(n) for n in nav_cameras], name="Head"))
    if gripper_cameras:
        camera_rows.append(
            rrb.Horizontal(*[_camera_view(n) for n in gripper_cameras], name="Wrist")
        )

    # The camera images live under world/cameras/**, so they have to be
    # excluded or the 3D view tries to render them alongside the point cloud.
    pointcloud_view = rrb.Spatial3DView(
        origin="world",
        contents=["world/**", "- world/cameras/**"],
        name="Lidar Point Cloud",
    )

    metrics_row = rrb.Horizontal(
        rrb.TimeSeriesView(origin="metrics", name="FPS"),
        rrb.TextLogView(origin="logs", name="Sim vs Real"),
        column_shares=[2, 1],
    )

    right_panel = rrb.Vertical(pointcloud_view, metrics_row, row_shares=[3, 1])

    if not camera_rows:
        # No cameras streaming, so give the 3D view and metrics the full window.
        return rrb.Blueprint(right_panel, collapse_panels=True)

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(*camera_rows),
            right_panel,
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


class RerunLogger:
    def __init__(self, maxsize: int = 1):
        # maxsize=1: `_put_queue()` drops a still-pending item in favor of the
        # newest one, so the logging thread (which does the expensive rr.log
        # serialization/IPC) never falls behind and builds a backlog of stale
        # pointclouds that would otherwise get logged one-by-one after the
        # fact, dragging out the visible lag between sim and viewer.
        self._log_queue = queue.Queue(maxsize=maxsize)
        self._latest_images = {}
        self._latest_images_lock = threading.Lock()
        self._latest_metrics = {}
        self._latest_metrics_message = None
        self._latest_metrics_lock = threading.Lock()
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

            # 3. Log latest metrics (drops older backlog values for zero lag)
            with self._latest_metrics_lock:
                current_metrics = self._latest_metrics.copy()
                self._latest_metrics.clear()
                current_message = self._latest_metrics_message
                self._latest_metrics_message = None

            for name, value in current_metrics.items():
                rr.log(f"metrics/{name}", rr.Scalars(value))
            if current_message is not None:
                # Kept out of metrics/** so the FPS time-series view stays
                # purely scalar, see `get_default_blueprint()`.
                rr.log("logs/sim_to_real_ratio", rr.TextLog(current_message))

            threading.Event().wait(0.01)

    def init_rerun(self, use_stretch_3: bool = False, cameras_to_use=None):
        """
        Spawn the Rerun viewer and send the layout.

        Pass `cameras_to_use` (the `StretchCameras` list handed to the
        simulator) so the blueprint only lays out the cameras that are
        actually streaming, see `get_default_blueprint()`.
        """
        if not self._initialized:
            rr.init("Stretch4 Mujoco", spawn=False)
            rr.spawn(memory_limit='5GB')
            camera_names = (
                [camera.name for camera in cameras_to_use]
                if cameras_to_use is not None
                else None
            )
            try:
                rr.send_blueprint(get_default_blueprint(use_stretch_3, camera_names))
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

    def update_metrics(self, metrics: dict[str, float], message: str | None = None):
        """
        Log scalar metrics (e.g. fps) as Rerun time-series, plus an optional
        one-line text message (e.g. a sim-to-real ratio summary that isn't a
        bare number).
        """
        if not self._initialized:
            self.init_rerun()
        with self._latest_metrics_lock:
            self._latest_metrics.update(metrics)
            if message is not None:
                self._latest_metrics_message = message

    def stop(self):
        self._stop_event.set()
        if self._log_thread:
            self._log_thread.join(timeout=2.0)


