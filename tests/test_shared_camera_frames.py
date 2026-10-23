"""Tests for passing camera frames between processes through shared memory."""

import multiprocessing
import time
from multiprocessing import shared_memory
from multiprocessing.managers import SyncManager

import numpy as np
import pytest

import stretch4_mujoco.shared_camera_frames as shared_camera_frames
from stretch4_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.mujoco_server import MujocoServerProxies

HEAD_SHAPE = (1200, 1920, 3)
DEPTH_SHAPE = (400, 640)


@pytest.fixture(scope="module")
def manager():
    """
    Start a Manager for the shared dicts.

    Returns:
        The running Manager, shut down after the module's tests.
    """
    manager = multiprocessing.get_context("spawn").Manager()
    yield manager
    manager.shutdown()


def _batch(value: int, sim_time: float) -> StatusStretchCameras:
    """
    Build a batch whose every pixel holds `value`.

    Args:
        value: The value to fill the frames with.
        sim_time: The simulation time to stamp the batch with.

    Returns:
        A batch with a head RGB frame and a gripper depth frame.
    """
    status = StatusStretchCameras(time=sim_time, fps=30.0)
    status.camera_times = {
        StretchCameras.cam_nav_rgb_se4_left.name: sim_time,
        StretchCameras.cam_gripper_se4_stereo_depth.name: sim_time,
    }
    status.set_camera_data(StretchCameras.cam_nav_rgb_se4_left, np.full(HEAD_SHAPE, value % 256, np.uint8))
    status.set_camera_data(
        StretchCameras.cam_gripper_se4_stereo_depth, np.full(DEPTH_SHAPE, float(value), np.float32)
    )
    return status


def _segment_exists(segment_name: str) -> bool:
    """
    Check whether a shared memory segment still exists.

    Args:
        segment_name: The segment's name.

    Returns:
        True if the segment can be attached to.
    """
    try:
        segment = shared_camera_frames._attach_untracked(segment_name)
    except FileNotFoundError:
        return False
    segment.close()
    return True


def test_round_trip_keeps_frames_and_metadata(manager: SyncManager):
    """
    A published batch is read back with the same frames and timing.

    Args:
        manager: The fixture's Manager.

    Returns:
        None.
    """
    proxies = MujocoServerProxies.default(manager)
    sent = _batch(7, sim_time=1.5)
    proxies.set_cameras(sent)

    received = proxies.get_cameras()

    assert received.time == 1.5 and received.fps == 30.0
    assert received.camera_times == sent.camera_times
    np.testing.assert_array_equal(received.cam_nav_rgb_se4_left, sent.cam_nav_rgb_se4_left)
    np.testing.assert_array_equal(received.cam_gripper_se4_stereo_depth, sent.cam_gripper_se4_stereo_depth)
    assert received.cam_gripper_se4_stereo_depth.dtype == np.float32
    assert received.cam_nav_rgb_se4_right is None
    # The pixels went through shared memory, not the Manager.
    assert manager_status_has_no_frames(proxies)
    proxies.close_camera_frames()


def manager_status_has_no_frames(proxies: MujocoServerProxies) -> bool:
    """
    Check that the status stored in the Manager carries no frames.

    Args:
        proxies: The proxies whose Manager dict to inspect.

    Returns:
        True if every camera field of the stored status is None.
    """
    status = proxies._cameras[shared_camera_frames.STATUS_KEY]
    return not shared_camera_frames._frames_in(status)


def test_frames_read_are_copies(manager: SyncManager):
    """
    A frame already read is not changed by the next batch, and the batch sent is not modified.

    Args:
        manager: The fixture's Manager.

    Returns:
        None.
    """
    proxies = MujocoServerProxies.default(manager)
    sent = _batch(1, sim_time=0.1)
    proxies.set_cameras(sent)
    first = proxies.get_cameras()

    proxies.set_cameras(_batch(2, sim_time=0.2))
    second = proxies.get_cameras()

    assert (first.cam_nav_rgb_se4_left == 1).all()
    assert (second.cam_nav_rgb_se4_left == 2).all()
    assert sent.cam_nav_rgb_se4_left is not None
    proxies.close_camera_frames()


def test_shape_change_replaces_segment(manager: SyncManager):
    """
    A camera whose frame changes shape gets a new segment, and the old one is unlinked.

    Args:
        manager: The fixture's Manager.

    Returns:
        None.
    """
    proxies = MujocoServerProxies.default(manager)
    proxies.set_cameras(_batch(3, sim_time=0.1))
    proxies.get_cameras()
    old_name = proxies._cameras[shared_camera_frames.LAYOUTS_KEY][StretchCameras.cam_nav_rgb_se4_left.name].segment_name

    resized = _batch(4, sim_time=0.2)
    resized.set_camera_data(StretchCameras.cam_nav_rgb_se4_left, np.full((965, 1280, 3), 4, np.uint8))
    proxies.set_cameras(resized)
    received = proxies.get_cameras()

    assert received.cam_nav_rgb_se4_left.shape == (965, 1280, 3)
    assert (received.cam_nav_rgb_se4_left == 4).all()
    assert not _segment_exists(old_name)
    proxies.close_camera_frames()


def test_close_unlinks_segments_and_leftovers(manager: SyncManager):
    """
    The writer unlinks its segments on close, and a reader can clean up after a writer that never closed.

    Args:
        manager: The fixture's Manager.

    Returns:
        None.
    """
    proxies = MujocoServerProxies.default(manager)
    proxies.set_cameras(_batch(5, sim_time=0.1))
    proxies.get_cameras()
    names = [layout.segment_name for layout in proxies._cameras[shared_camera_frames.LAYOUTS_KEY].values()]
    proxies.close_camera_frames()
    assert not any(_segment_exists(name) for name in names)

    # A writer that died without closing leaves its segments behind.
    leaky = MujocoServerProxies.default(manager)
    leaky.set_cameras(_batch(6, sim_time=0.1))
    leaky.get_cameras()
    names = [layout.segment_name for layout in leaky._cameras[shared_camera_frames.LAYOUTS_KEY].values()]
    for segment in leaky._camera_frame_writer._segments.values():
        segment.close()
    leaky._camera_frame_writer._segments.clear()

    leaky._camera_frame_reader.close(unlink_leftovers=True)
    assert not any(_segment_exists(name) for name in names)


def test_falls_back_to_manager_without_shared_memory(manager: SyncManager, monkeypatch: pytest.MonkeyPatch):
    """
    If segments cannot be created, frames still arrive, through the Manager.

    Args:
        manager: The fixture's Manager.
        monkeypatch: Pytest's monkeypatch fixture.

    Returns:
        None.
    """

    def unavailable(*args, **kwargs):
        """Fail as a container without /dev/shm would."""
        raise OSError("no shared memory")

    monkeypatch.setattr(shared_camera_frames.shared_memory, "SharedMemory", unavailable)
    proxies = MujocoServerProxies.default(manager)
    proxies.set_cameras(_batch(8, sim_time=0.3))
    received = proxies.get_cameras()

    assert proxies._camera_frame_writer.is_shared_memory_unavailable
    assert (received.cam_nav_rgb_se4_left == 8).all()
    assert received.time == 0.3


def _publish_batches(proxies: MujocoServerProxies, count: int) -> None:
    """
    Publish `count` batches back to back, as the sim process would.

    Pauses briefly between batches, still far faster than any camera rate. The
    lock is not fair, so on a slow machine a publisher that re-takes it at once
    can starve the reader until the last batch.

    Args:
        proxies: The proxies passed to this process when it was spawned.
        count: How many batches to publish.

    Returns:
        None.
    """
    for value in range(1, count + 1):
        proxies.set_cameras(_batch(value, sim_time=value * 0.01))
        time.sleep(0.001)
    proxies.close_camera_frames()


def test_no_torn_frames_across_processes(manager: SyncManager):
    """
    A reader pulling while another process publishes never sees a half-written batch.

    Every pixel of a batch holds the same value, and its camera times encode
    that value too, so any batch read mid-write shows up as mixed values.

    Args:
        manager: The fixture's Manager.

    Returns:
        None.
    """
    proxies = MujocoServerProxies.default(manager)
    proxies.set_cameras(_batch(0, sim_time=0.0))

    publisher = multiprocessing.get_context("spawn").Process(target=_publish_batches, args=(proxies, 300))
    publisher.start()

    values_seen = set()
    while publisher.is_alive() or not values_seen:
        received = proxies.get_cameras()
        head = received.cam_nav_rgb_se4_left
        depth = received.cam_gripper_se4_stereo_depth
        if head is None:
            continue  # The publisher closed its segments after its last batch.
        value = int(round(received.time / 0.01))
        assert head.min() == head.max() == value % 256, "head frame was torn"
        assert depth.min() == depth.max() == float(value), "depth frame does not match its batch"
        values_seen.add(value)
    publisher.join(timeout=10)

    assert publisher.exitcode == 0
    assert len(values_seen) > 10, f"only saw {len(values_seen)} distinct batches"
    proxies.close_camera_frames(unlink_leftovers=True)
