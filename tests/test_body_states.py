"""Tests for the watched-body pose and contact reports (`StretchMujocoSimulator.watch_bodies()`).

These drive a server in-process, like `test_guarded_contact.py`, with a free box dropped
next to the robot so it lands on the floor.
"""

import multiprocessing
import threading

import mujoco
import numpy as np
import pytest

from stretch4_mujoco.mujoco_server import MujocoServer, MujocoServerProxies
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator

BOX = "test_box"
BOX_START = (1.0, 1.0, 0.5)


@pytest.fixture
def server():
    spec = mujoco.MjSpec.from_file(Stretch4MujocoSimulator.get_scene_xml_path())
    box = spec.worldbody.add_body(name=BOX, pos=BOX_START)
    box.add_freejoint()
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.03], mass=0.1)

    manager = multiprocessing.Manager()
    server = MujocoServer(
        scene_xml_path=None,
        model=spec.compile(),
        stop_mujoco_process_event=threading.Event(),
        data_proxies=MujocoServerProxies.default(manager),
        start_translation=None,
        start_rotation_quat=None,
    )
    yield server
    server.request_to_stop()
    server.sensor_manager.sensors_thread.join(timeout=5)
    manager.shutdown()


def step(server: MujocoServer, cycles: int):
    for _ in range(cycles):
        server._physics_step(server.sensor_manager.sensor_lock)


def test_nothing_is_published_until_a_body_is_watched(server):
    step(server, 20)
    assert server.data_proxies.get_body_states() == {}


def test_dropped_box_pose_and_floor_contact(server):
    server.data_proxies.set_watched_bodies([BOX, "stretch4", "no_such_body"])
    step(server, 200)  # 2 s of sim time: the box falls 0.5 m and comes to rest

    states = server.data_proxies.get_body_states()
    assert set(states) == {BOX, "stretch4"}, "unknown bodies are skipped"

    box = states[BOX]
    np.testing.assert_allclose(box["pos"][:2], BOX_START[:2], atol=0.01)
    assert box["pos"][2] == pytest.approx(0.03, abs=0.01), "resting on the floor"
    np.testing.assert_allclose(np.linalg.norm(box["quat"]), 1.0)
    model = server.mjmodel
    floor_root = model.body(model.body_rootid[model.geom("floor").bodyid[0]]).name
    assert box["contacts"] == [floor_root]

    # The robot is resting on its wheels.
    assert floor_root in states["stretch4"]["contacts"]


def test_box_on_robot_reports_robot_contact(server):
    """A box resting on the robot reports contact with its root body, not the link."""
    server.data_proxies.set_watched_bodies([BOX])
    model, data = server.mjmodel, server.mjdata
    adr = model.jnt_qposadr[model.body(BOX).jntadr[0]]
    with server.sensor_manager.sensor_lock:
        top = data.body("stretch4").xpos.copy()
        data.qpos[adr : adr + 3] = [top[0], top[1], 2.0]
        mujoco.mj_forward(model, data)
    step(server, 300)

    contacts = server.data_proxies.get_body_states()[BOX]["contacts"]
    assert "stretch4" in contacts
