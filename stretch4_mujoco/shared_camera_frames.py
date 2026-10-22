"""Pass camera frames from the sim process to clients through shared memory.

Pickling a batch of frames into the `multiprocessing.Manager` process and back
out costs tens of milliseconds for the ~20MB the Stretch 4 cameras produce, and
the Manager serves one request at a time, so a client pulling frames delays the
sim publishing the next ones. Here the pixels are copied into one shared memory
segment per camera instead, and only a few hundred bytes describing them go
through the Manager.

One lock covers a whole batch: the sim holds it while it copies the batch in and
publishes the description, and a client holds it while it reads the description
and copies the batch out. Every pull is therefore one consistent batch, as when
the whole batch went through the Manager in one piece.
"""

import dataclasses
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Lock
import sys

import click
import numpy as np

from stretch4_mujoco.datamodels.status_stretch_camera import StatusStretchCameras
from stretch4_mujoco.enums.stretch_cameras import StretchCameras

LOCK_TIMEOUT_S = 2.0
"""How long to wait for the other process to finish copying a batch.

Copying holds the lock for a few milliseconds, so waiting this long means the
process holding it died mid-copy."""

STATUS_KEY = "val"
LAYOUTS_KEY = "frame_layouts"


@dataclass(frozen=True)
class SharedFrameLayout:
    """Where one camera's frame is in shared memory, and how to read it."""

    segment_name: str
    shape: tuple[int, ...]
    dtype: str


def _frames_in(status: StatusStretchCameras) -> dict[str, np.ndarray]:
    """
    Collect the frames a camera status holds.

    Args:
        status: The camera status to read.

    Returns:
        Each frame keyed by its camera's name, which is also its field name.
    """
    frames = {}
    for camera in StretchCameras:
        frame = getattr(status, camera.name, None)
        if frame is not None:
            frames[camera.name] = frame
    return frames


def _attach_untracked(segment_name: str) -> shared_memory.SharedMemory:
    """
    Attach to an existing segment without Python's resource tracker owning it.

    Before Python 3.13 attaching registers the segment with the resource
    tracker, which unlinks it, and warns that it leaked, when the attaching
    process exits, even though the sim process created it and still uses it.

    Args:
        segment_name: The name of the segment to attach to.

    Returns:
        The attached segment.
    """
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=segment_name, track=False)  # type: ignore[call-arg]

    segment = shared_memory.SharedMemory(name=segment_name)
    resource_tracker.unregister(segment._name, "shared_memory")  # type: ignore[attr-defined]
    return segment


class SharedCameraFrameWriter:
    """Publishes camera batches from the sim process, which owns the segments."""

    def __init__(self, lock: Lock, cameras_proxy: DictProxy) -> None:
        """
        Prepare to publish, creating segments as cameras first appear.

        Args:
            lock: The lock shared with every reader.
            cameras_proxy: The Manager dict the batch descriptions go in.
        """
        self._lock = lock
        self._cameras_proxy = cameras_proxy

        self._segments: dict[str, shared_memory.SharedMemory] = {}
        self._layouts: dict[str, SharedFrameLayout] = {}

        self.is_shared_memory_unavailable = False
        """Set if a segment could not be created, after which every batch goes
        through the Manager whole, as before shared memory was used."""
        self._is_lock_timeout_reported = False

    def write(self, status: StatusStretchCameras) -> None:
        """
        Publish a batch of frames.

        Args:
            status: The batch to publish. It is not modified.

        Returns:
            None.
        """
        frames = _frames_in(status)
        retired = self._prepare_segments(frames) if not self.is_shared_memory_unavailable else []

        if self.is_shared_memory_unavailable:
            description, layouts = status, {}
        else:
            description = dataclasses.replace(status, **{name: None for name in frames})
            layouts = {name: self._layouts[name] for name in frames}

        if not self._lock.acquire(timeout=LOCK_TIMEOUT_S):
            if not self._is_lock_timeout_reported:
                self._is_lock_timeout_reported = True
                click.secho(
                    "Camera frames were not published: a client died while copying them and still holds the lock.",
                    fg="red",
                )
            return
        try:
            for name, layout in layouts.items():
                segment = self._segments[name]
                np.ndarray(layout.shape, dtype=layout.dtype, buffer=segment.buf)[...] = frames[name]
            self._cameras_proxy.update({STATUS_KEY: description, LAYOUTS_KEY: layouts})
        finally:
            self._lock.release()

        # No reader can still be copying out of a replaced segment: they copy
        # under the lock, and the description they read it from is gone.
        for segment in retired:
            self._release(segment)

    def _prepare_segments(self, frames: dict[str, np.ndarray]) -> list[shared_memory.SharedMemory]:
        """
        Make sure every frame has a segment of its shape and dtype.

        A camera whose frame changed shape gets a new segment. Falls back to
        the Manager for good if one cannot be created.

        Args:
            frames: The batch's frames, keyed by camera name.

        Returns:
            The segments that were replaced, to release once the batch that
            replaces them is published.
        """
        retired = []
        for name, frame in frames.items():
            layout = self._layouts.get(name)
            if layout is not None and layout.shape == frame.shape and layout.dtype == frame.dtype.str:
                continue
            try:
                segment = shared_memory.SharedMemory(create=True, size=max(frame.nbytes, 1))
            except OSError as error:
                self.is_shared_memory_unavailable = True
                click.secho(
                    f"Shared memory is unavailable ({error!r}). Camera frames go through the multiprocessing Manager instead, which is slower.",
                    fg="yellow",
                )
                return retired
            if name in self._segments:
                retired.append(self._segments[name])
            self._segments[name] = segment
            self._layouts[name] = SharedFrameLayout(segment.name, frame.shape, frame.dtype.str)
        return retired

    def close(self) -> None:
        """
        Unlink every segment. Readers still attached keep their mappings.

        Returns:
            None.
        """
        for segment in self._segments.values():
            self._release(segment)
        self._segments.clear()
        self._layouts.clear()

    @staticmethod
    def _release(segment: shared_memory.SharedMemory) -> None:
        """
        Close and unlink one of this writer's segments.

        Args:
            segment: The segment to release.

        Returns:
            None.
        """
        segment.close()
        try:
            segment.unlink()
        except FileNotFoundError:
            pass


class SharedCameraFrameReader:
    """Pulls camera batches in a client process, attaching to segments as needed."""

    def __init__(self, lock: Lock, cameras_proxy: DictProxy) -> None:
        """
        Prepare to pull batches.

        Args:
            lock: The lock shared with the writer.
            cameras_proxy: The Manager dict the batch descriptions are in.
        """
        self._lock = lock
        self._cameras_proxy = cameras_proxy
        self._segments: dict[str, shared_memory.SharedMemory] = {}
        self._seen_segment_names: set[str] = set()

    def read(self) -> StatusStretchCameras:
        """
        Pull the latest batch.

        Returns:
            The batch, with frames the caller owns: they are copies, so a later
            batch does not change them.

        Raises:
            TimeoutError: If the sim process died while publishing a batch.
        """
        if not self._lock.acquire(timeout=LOCK_TIMEOUT_S):
            raise TimeoutError("Timed out waiting for camera frames: the sim process died while publishing them.")
        try:
            published = self._cameras_proxy.copy()
            status: StatusStretchCameras = published[STATUS_KEY]
            layouts: dict[str, SharedFrameLayout] = published.get(LAYOUTS_KEY, {})
            for name, layout in layouts.items():
                segment = self._attach(layout.segment_name)
                frame = np.ndarray(layout.shape, dtype=layout.dtype, buffer=segment.buf)
                setattr(status, name, frame.copy())
        finally:
            self._lock.release()

        self._detach_unused({layout.segment_name for layout in layouts.values()})
        return status

    def _attach(self, segment_name: str) -> shared_memory.SharedMemory:
        """
        Attach to a segment, reusing an earlier attachment.

        Args:
            segment_name: The name of the segment.

        Returns:
            The attached segment.
        """
        segment = self._segments.get(segment_name)
        if segment is None:
            segment = _attach_untracked(segment_name)
            self._segments[segment_name] = segment
            self._seen_segment_names.add(segment_name)
        return segment

    def _detach_unused(self, in_use: set[str]) -> None:
        """
        Close attachments to segments the writer has replaced.

        Args:
            in_use: Names of the segments the latest batch uses.

        Returns:
            None.
        """
        for segment_name in list(self._segments):
            if segment_name not in in_use:
                self._segments.pop(segment_name).close()

    def close(self, unlink_leftovers: bool = False) -> None:
        """
        Close every attachment.

        Args:
            unlink_leftovers: Also unlink any segment this reader has seen that
                still exists. Only do this once the sim process has exited: it
                unlinks its own segments when it closes, so anything left was
                leaked by a crash.

        Returns:
            None.
        """
        for segment in self._segments.values():
            segment.close()
        self._segments.clear()

        if not unlink_leftovers:
            return
        for segment_name in self._seen_segment_names:
            try:
                # Attach tracked, so that unlinking also clears the sim
                # process's registration with the shared resource tracker.
                leftover = shared_memory.SharedMemory(name=segment_name)
            except FileNotFoundError:
                continue
            leftover.close()
            leftover.unlink()
        self._seen_segment_names.clear()
