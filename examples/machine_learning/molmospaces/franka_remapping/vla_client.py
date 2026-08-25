"""
Transport for the VLA inference server.

MolmoSpaces already ships a working client for a chunked websocket policy
server: `DreamZeroWebsocketClient`, which packs msgpack-numpy over a websocket,
waits for a server that is still starting, falls back from `ws://` to `wss://`,
and reconnects when a long inference call drops the socket. All of that is worth
having and none of it is worth writing twice, so this subclasses it.

The one thing that is not universal is the `endpoint` field DreamZero's server
routes on. openpi's `websocket_policy_server` passes the whole unpacked dict
into the policy's input transform instead, where an unexpected key can be a hard
error, so `include_endpoint` makes it optional.
"""

from __future__ import annotations

import logging

import numpy as np
from molmo_spaces.policy.learned_policy.dreamzero_policy import DreamZeroWebsocketClient
from molmo_spaces.policy.learned_policy.utils import resize_with_pad

log = logging.getLogger(__name__)

__all__ = ["VLAWebsocketClient", "letterbox"]


def letterbox(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize to `(height, width)` keeping the aspect ratio, zero-padding the rest.

    `resize_with_pad` from MolmoSpaces, under a name that says what it does at
    the call site. Every Franka-space baseline preprocesses its images this way,
    and matching it matters more than it looks: a model trained on letterboxed
    320x180 frames and fed stretched ones sees objects at the wrong aspect
    ratio, which is exactly the cue it uses for depth.
    """
    return resize_with_pad(np.asarray(image), height, width)


class VLAWebsocketClient(DreamZeroWebsocketClient):
    """A chunked VLA inference server, with the routing field made optional."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        include_endpoint: bool = False,
    ) -> None:
        self._include_endpoint = include_endpoint
        super().__init__(host=host, port=port)
        log.info(
            f"[franka-vla] connected to inference server at {host}:{port} "
            f"({'dreamzero' if include_endpoint else 'openpi'} protocol)"
        )

    def infer(self, observation: dict) -> dict:
        if self._include_endpoint:
            return super().infer(dict(observation))
        return self._call(dict(observation))

    def reset(self, reset_info: dict | None = None) -> None:
        """End the episode, whichever way this server understands that.

        With endpoint routing there is a `reset` message to send. Without it --
        MolmoBot's `olmo/eval/websocket_server.py`, and openpi's server -- the
        only reset there is happens when a client *connects*: the server calls
        `policy.reset()` once and then loops on requests. So reconnecting is the
        reset, and skipping it would carry one episode's action buffer into the
        next for the whole benchmark run.
        """
        if self._include_endpoint:
            return super().reset(dict(reset_info or {}))
        try:
            self._ws.close()
        except Exception as error:  # noqa: BLE001 - a closed socket is the goal
            log.debug(f"[franka-vla] closing before reset raised {error!r}; reconnecting anyway")
        self._reconnect()
        return None

    def _call(self, payload: dict) -> dict:
        """One request/response round trip without the `endpoint` field."""
        import msgpack_numpy
        import websockets.exceptions

        data = self._packer.pack(payload)
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosedError:
            self._reconnect()
            self._ws.send(data)
            response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)
