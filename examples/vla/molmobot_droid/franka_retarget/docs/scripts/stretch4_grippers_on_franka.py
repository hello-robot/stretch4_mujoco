"""
docs/stretch4_grippers_on_franka.png: Stretch 4's grippers on the Franka (`run_franka.py
--use_stretch4_gripper`, `--use_stretch4_gripper_camera`), at home over the counter, with the
tennis ball moved to straight under its TCP. A row per gripper (the stock Robotiq, SG4, PG4);
columns: outside, DROID's wrist camera, Stretch 4's gripper camera (DROID-framed, as the policy
is given it).

    MUJOCO_GL=egl python -m examples.vla.molmobot_droid.franka_retarget.docs.scripts.stretch4_grippers_on_franka
"""

import math

import mujoco
import numpy as np

from examples.vla.molmobot_droid.checkpoint import DROID_IMAGE_SIZE
from examples.vla.molmobot_droid.droid import FrankaDroidEnv
from examples.vla.molmobot_droid.franka_retarget.docs.scripts.figures import label, load_scene, save
from examples.vla.molmobot_droid.franka_retarget.stretch4_retarget import wrist_view
from examples.vla.molmobot_droid.molmospaces.custom_scene import load_custom_scene_franka_droid

GRIPPERS = ((None, "Robotiq (stock)"), ("sg4", "SG4"), ("pg4", "PG4"))


def object_under_tcp(env: FrankaDroidEnv, body: str) -> None:
    """`body` (a free one) moved to straight under the TCP, at the height it rests at."""
    address = env.model.jnt_qposadr[env.model.body(body).jntadr[0]]
    env.data.qpos[address : address + 2] = env.tcp_world()[:2, 3]
    mujoco.mj_forward(env.model, env.data)


def outside(env: FrankaDroidEnv, yaw: float) -> np.ndarray:
    """The gripper from in front and to the side of the robot, which faces `yaw` (rad)."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = env.tcp_world()[:3, 3] + [0.0, 0.0, 0.05]
    camera.distance, camera.elevation, camera.azimuth = 0.75, -15, math.degrees(yaw) + 120
    width, height = DROID_IMAGE_SIZE
    with mujoco.Renderer(env.model, height, width) as renderer:
        renderer.update_scene(env.data, camera=camera)
        return renderer.render()


def main():
    scene, pose = load_scene()
    rows = []
    for gripper, name in GRIPPERS:
        # With Stretch 4's gripper camera (and its housing, for the outside view), and without.
        with_camera = load_custom_scene_franka_droid(scene, pose, stretch4_gripper=gripper, stretch4_gripper_camera=True)
        without = load_custom_scene_franka_droid(scene, pose, stretch4_gripper=gripper)
        try:
            for env in (with_camera, without):
                object_under_tcp(env, scene.object_name)
            stretch = wrist_view(with_camera.observe().wrist_rgb)
            droid = without.observe().wrist_rgb
            # The stock Franka is shown as it comes, with no Stretch camera on it.
            view = outside(without if gripper is None else with_camera, pose.yaw)
        finally:
            with_camera.close()
            without.close()
        rows.append([label(view, f"{name}: outside"), label(droid, f"{name}: DROID wrist camera"),
                     label(stretch, f"{name}: Stretch 4 gripper camera")])
    save(rows, "stretch4_grippers_on_franka")


if __name__ == "__main__":
    main()
