"""Check that every Stretch 4 camera renders a usable image, and how fast.

Slow rendering is reported as a `CameraPerformanceWarning` rather than a
failure, since it depends on the machine. Only a wrong or blank image fails.

The baselines were measured with EGL on an Intel i9-13900KF with an NVIDIA RTX
4090 (driver 595.84), on the default Stretch 4 scene. Rerun with `-s` to see
the tables while updating them.
"""

import time
import warnings
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.mujoco_server_camera_manager import MujocoServerCameraManagerSync
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

BASELINE_RENDER_MS = {
    StretchCameras.cam_nav_rgb_se4_left: 3.8,
    StretchCameras.cam_nav_rgb_se4_right: 3.9,
    StretchCameras.cam_nav_rgb_se4_center: 9.0,
    StretchCameras.cam_nav_rgb_se4_center_low_rez: 0.78,
    StretchCameras.cam_gripper_se4_left_rgb: 0.26,
    StretchCameras.cam_gripper_se4_right_rgb: 0.26,
    StretchCameras.cam_gripper_se4_stereo_depth: 0.68,
}
"""Median milliseconds to render one frame, scene update through readback."""

BASELINE_SERVER_CAMERA_FPS = 30.0
"""Camera batches per second the headless sim renders while nobody pulls them."""

BASELINE_DELIVERED_FPS = {
    "cam_nav_rgb_se4_left": 30.0,
    "cam_nav_rgb_se4_right": 30.0,
    "cam_nav_rgb_se4_center_low_rez": 10.0,
    "cam_gripper_se4_left_rgb": 30.0,
    "cam_gripper_se4_right_rgb": 30.0,
    "cam_gripper_se4_stereo_depth": 30.0,
}
"""New frames per second a client pulling at `CLIENT_PULL_HZ` receives."""

SLOWER_THAN_BASELINE_TOLERANCE = 1.5
"""How many times slower than the baseline a measurement may be before warning."""

RENDER_FRAMES = 30
MEASURE_WINDOW_S = 4.0
CLIENT_PULL_HZ = 100.0
"""Faster than any camera, so that a pull landing twice between two frames does
not count as a dropped frame, as it would for a client pulling at 30 Hz."""


class CameraPerformanceWarning(UserWarning):
    """Camera rendering is slower than on the baseline machine."""


def _check_image(camera: StretchCameras, image: np.ndarray | None) -> None:
    """
    Assert that a camera's image has the right shape and is not blank.

    Args:
        camera: The camera that rendered the image.
        image: The image as rendered, before any auto-rotation.

    Returns:
        None.
    """
    settings = camera.initial_camera_settings
    assert image is not None, f"{camera.name} returned no image"

    if camera.is_depth:
        assert image.shape == (settings.height, settings.width), camera.name
        assert np.isfinite(image).all(), f"{camera.name} has non-finite depth"
        assert (image >= 0).all(), f"{camera.name} has negative depth"
        # Depth past the limit is zeroed, but the gripper camera sees its own
        # fingers and the floor, so a real frame is never all zero or flat.
        assert (image > 0).mean() > 0.05, f"{camera.name} sees nothing within range"
        assert image[image > 0].std() > 1e-3, f"{camera.name} depth is flat"
        return

    assert image.shape == (settings.height, settings.width, 3), camera.name
    assert image.dtype == np.uint8, camera.name
    assert image.mean() > 5, f"{camera.name} is black"
    assert image.std() > 5, f"{camera.name} is a single colour"

    if camera.is_fisheye:
        # Past the image circle is black, like the real lens.
        cx, cy = settings.optical_center
        radius = settings.image_circle_radius_px
        assert not image[0, 0].any() and not image[-1, -1].any(), camera.name
        assert not image[int(cy), int(cx + radius + 5)].any(), camera.name


def _print_table(title: str, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """
    Print a fixed-width table.

    Args:
        title: Line printed above the table.
        header: Column names.
        rows: One tuple of already formatted cells per row.

    Returns:
        None.
    """
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    print(f"\n{title}")
    for row in [header, *rows]:
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)))


@pytest.fixture(scope="module")
def camera_manager():
    """
    Create renderers for every Stretch 4 camera on the default scene.

    Generates the robot MJCF first, overwriting any previously generated one.

    Both centre camera resolutions are included, since the simulator renders
    whichever one `use_full_center_camera_resolution` selects.

    Returns:
        The camera manager, closed after the module's tests.
    """
    cameras = list(BASELINE_RENDER_MS)
    # The scene includes the robot MJCF, which is generated rather than checked
    # in, so a fresh checkout has none until something generates it.
    Stretch4MujocoSimulator.get_robot_xml_path()
    model = mujoco.MjModel.from_xml_path(Stretch4MujocoSimulator.get_scene_xml_path())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    manager = MujocoServerCameraManagerSync(
        camera_hz=30, cameras_to_use=cameras, mujoco_server=SimpleNamespace(mjmodel=model, mjdata=data)
    )
    yield manager
    manager.close()


def test_each_camera_renders_correctly_and_in_time(camera_manager, capsys):
    """
    Render every camera, check its image, and compare its render time to the baseline.

    Args:
        camera_manager: The fixture's camera manager.
        capsys: Pytest's output capture, disabled to always show the table.

    Returns:
        None.
    """
    rows = []
    slow = []
    for camera, baseline_ms in BASELINE_RENDER_MS.items():
        renderer = camera_manager.camera_renderers[camera]
        for _ in range(3):  # Warm up the GL pipeline.
            camera_manager._render_camera(renderer, camera)

        timings_ms = []
        for _ in range(RENDER_FRAMES):
            start = time.perf_counter()
            _, image = camera_manager._render_camera(renderer, camera)
            timings_ms.append((time.perf_counter() - start) * 1e3)

        _check_image(camera, image)

        median_ms = float(np.median(timings_ms))
        is_slow = median_ms > baseline_ms * SLOWER_THAN_BASELINE_TOLERANCE
        if is_slow:
            slow.append(f"{camera.name} {median_ms:.1f} ms (baseline {baseline_ms:.1f} ms)")
        rows.append(
            (
                camera.name,
                f"{camera.target_hz:.0f}",
                f"{1e3 / baseline_ms:.0f}",
                f"{1e3 / median_ms:.0f}",
                f"{median_ms:.2f}",
                "SLOW" if is_slow else "ok",
            )
        )

    with capsys.disabled():
        _print_table(
            "Camera render rate, one camera at a time:",
            ("camera", "target Hz", "baseline FPS", "measured FPS", "ms/frame", ""),
            rows,
        )

    if slow:
        warnings.warn(
            CameraPerformanceWarning(
                "Cameras render more than "
                f"{SLOWER_THAN_BASELINE_TOLERANCE}x slower than the baseline machine: " + ", ".join(slow)
            )
        )


def test_cameras_deliver_frames_at_expected_rate(capsys):
    """
    Run a headless sim and measure its camera rate, then the rate a client sees.

    The sim's own rate is read from the status it publishes, sampled rarely so
    that reading it does not slow the sim down. The delivered rate counts new
    frames per camera while pulling at `CLIENT_PULL_HZ`, so it also catches
    pulls slowing the sim down.

    Args:
        capsys: Pytest's output capture, disabled to always show the table.

    Returns:
        None.
    """
    sim = Stretch4MujocoSimulator(cameras_to_use=Stretch4MujocoSimulator.get_all_cameras())
    sim.start(headless=True)
    try:
        assert sim.is_running(), "The simulator did not start"

        # FpsCounter reports the batches counted over the last whole second.
        time.sleep(MEASURE_WINDOW_S / 2)
        server_fps = sim.pull_camera_data().fps

        frame_times: dict[str, set[float]] = {name: set() for name in BASELINE_DELIVERED_FPS}
        start = time.perf_counter()
        next_pull = start
        while time.perf_counter() - start < MEASURE_WINDOW_S:
            cam_data = sim.pull_camera_data()
            for name, sim_time in cam_data.camera_times.items():
                frame_times.setdefault(name, set()).add(sim_time)
            next_pull += 1 / CLIENT_PULL_HZ
            time.sleep(max(0.0, next_pull - time.perf_counter()))
        elapsed = time.perf_counter() - start

        for camera in Stretch4MujocoSimulator.get_all_cameras():
            _check_image(
                StretchCameras.cam_nav_rgb_se4_center_low_rez
                if camera == StretchCameras.cam_nav_rgb_se4_center
                else camera,
                cam_data.get_camera_data(camera, auto_rotate=False),
            )
    finally:
        sim.stop()

    rows = []
    slow = []
    if server_fps * SLOWER_THAN_BASELINE_TOLERANCE < BASELINE_SERVER_CAMERA_FPS:
        slow.append(f"sim camera loop {server_fps:.1f} FPS (baseline {BASELINE_SERVER_CAMERA_FPS:.0f})")
    for name, baseline_fps in BASELINE_DELIVERED_FPS.items():
        target_hz = StretchCameras[name].target_hz
        fps = len(frame_times[name]) / elapsed
        is_slow = fps * SLOWER_THAN_BASELINE_TOLERANCE < baseline_fps
        if is_slow:
            slow.append(f"{name} {fps:.1f} FPS (baseline {baseline_fps:.0f})")
        rows.append(
            (name, f"{target_hz:.0f}", f"{baseline_fps:.0f}", f"{fps:.1f}", "SLOW" if is_slow else "ok")
        )

    with capsys.disabled():
        print(
            f"\nHeadless sim camera loop, unread: {server_fps:.1f} FPS "
            f"(baseline {BASELINE_SERVER_CAMERA_FPS:.0f}, camera_hz {sim.camera_hz:.0f})"
        )
        _print_table(
            f"Frames delivered to a client pulling at {CLIENT_PULL_HZ:.0f} Hz over {elapsed:.1f}s:",
            ("camera", "target Hz", "baseline FPS", "measured FPS", ""),
            rows,
        )

    for name, times in frame_times.items():
        assert times, f"{name} delivered no frames"

    if slow:
        warnings.warn(
            CameraPerformanceWarning(
                "Cameras deliver frames more than "
                f"{SLOWER_THAN_BASELINE_TOLERANCE}x slower than the baseline machine: " + ", ".join(slow)
            )
        )
