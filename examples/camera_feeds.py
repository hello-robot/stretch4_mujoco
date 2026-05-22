import random
import threading
import time
import click
import cv2
from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
from stretch4_mujoco.stretch4_mujoco_simulator import StretchMujocoSimulator


def show_camera_feeds_sync(sim: StretchMujocoSimulator, print_fps: bool):
    """
    Pull camera data from the simulator and display it using OpenCV.

    Call this after calling StretchMujocoSimulator::start().
    """

    camera_data = sim.pull_camera_data()

    if print_fps:
        print(
            f"Physics fps: {sim.pull_status().fps}. Camera FPS: {camera_data.fps}. {sim.pull_status().sim_to_real_time_ratio_msg}"
        )

    for camera_name, pixels in camera_data.get_all(use_depth_color_map=True).items():
        cv2.namedWindow(camera_name.name, cv2.WINDOW_NORMAL)
        cv2.imshow(camera_name.name, pixels)

    cv2.waitKey(1)


def my_control_loop(sim: StretchMujocoSimulator,):
    while sim.is_running():
        sim.move_to(Actuators.lift, random.random())
        time.sleep(3)


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

    sim.start(headless=False)

    threading.Thread(target=my_control_loop, daemon=True, args=[sim]).start()

    try:
        while sim.is_running():
            show_camera_feeds_sync(sim, True)

    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
