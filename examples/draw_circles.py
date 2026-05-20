import threading
import time
import click
import numpy as np

from examples.camera_feeds import show_camera_feeds_sync
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
from stretch4_mujoco.stretch4_mujoco_simulator import StretchMujocoSimulator


def draw_circle(n, diameter_m, arm_init, lift_init, sim: StretchMujocoSimulator):
    """
    From https://forum.hello-robot.com/t/creating-smooth-motion-using-trajectories/671
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=True)
    x = (diameter_m / 2) * np.cos(t) + arm_init
    y = (diameter_m / 2) * np.sin(t) + lift_init
    circle_mat = np.c_[x, y]
    for pt in circle_mat:
        print(f"Moving to {pt}")
        sim.move_to(Actuators.arm, pt[0])
        sim.move_to(Actuators.lift, pt[1])

        sim.wait_until_at_setpoint(Actuators.arm)
        sim.wait_until_at_setpoint(Actuators.lift)


def _run_draw_circle(sim: StretchMujocoSimulator, use_head_pan_and_tilt: bool):
    time.sleep(2)
    try:
        while sim.is_running():
            if use_head_pan_and_tilt:
                sim.move_to(Actuators.head_tilt, -1.5707)
                sim.move_to(Actuators.head_pan, -0.7853)

            sim.move_to(Actuators.wrist_yaw, 1.5707)
            sim.move_to(Actuators.gripper, 0.5)
            sim.wait_until_at_setpoint(Actuators.wrist_yaw)
            sim.wait_until_at_setpoint(Actuators.gripper)

            sim.move_to(Actuators.gripper, pos=-0.15)
            sim.wait_until_at_setpoint(Actuators.gripper)

            status = sim.pull_status()
            draw_circle(25, 0.2, status.arm.pos, status.lift.pos, sim)
            time.sleep(1)
            sim.home()
            time.sleep(2)
    except ConnectionError:
        ...


@click.command()
@click.option("--use_stretch_3", type=bool, is_flag=True, help="Use Stretch 3")
def main(use_stretch_3: bool):
    cameras_to_use = (
        StretchCameras.rgb_stretch3() if use_stretch_3 else StretchCameras.rgb_stretch4()
    )

    sim = (
        StretchMujocoSimulator(cameras_to_use=cameras_to_use)
        if use_stretch_3
        else Stretch4MujocoSimulator(cameras_to_use=cameras_to_use)
    )

    use_head_pan_and_tilt = use_stretch_3

    sim.start(headless=False)

    threading.Thread(target=_run_draw_circle, daemon=False, args=[sim, use_head_pan_and_tilt]).start()

    try:
        while sim.is_running():
            show_camera_feeds_sync(sim, True)

    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
