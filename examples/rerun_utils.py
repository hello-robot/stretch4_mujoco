import queue
import threading
import numpy as np
import rerun as rr


class RerunLogger:
    def __init__(self, maxsize: int = 2):
        self._log_queue = queue.Queue(maxsize=maxsize)
        self._log_thread = None
        self._stop_event = threading.Event()

    def _logging_worker(self):
        while not self._stop_event.is_set():
            try:
                item = self._log_queue.get(timeout=0.1)
                points, label = item
                if isinstance(points, list):
                    for points_name, points_instance in points:
                        rr.log(f"{label}/{points_name}", rr.Points3D(points_instance))
                else:
                    rr.log(label, rr.Points3D(points))
                self._log_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def init_pointcloud_viz(self):
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
        self._stop_event.clear()
        self._log_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self._log_thread.start()

    def update_pointcloud_viz(self, points: np.ndarray | list[tuple[str, np.ndarray]], label: str):
        if self._log_thread is None or not self._log_thread.is_alive():
            raise RuntimeError(
                "RerunLogger must be initialized via init_pointcloud_viz() before calling update_pointcloud_viz()."
            )
        if self._stop_event.is_set():
            return
        while not self._stop_event.is_set():
            try:
                self._log_queue.put_nowait((points, label))
                break
            except queue.Full:
                try:
                    self._log_queue.get_nowait()
                except queue.Empty:
                    pass

    def stop(self):
        self._stop_event.set()
        if self._log_thread:
            self._log_thread.join(timeout=2.0)


