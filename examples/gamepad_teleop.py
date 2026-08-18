import time

import click
import cv2
from examples.camera_feeds import show_camera_feeds_sync

from examples.rerun_utils import RerunLogger
from examples.laser_scan import show_laser_scan
from stretch4_mujoco import StretchMujocoSimulator
from stretch4_mujoco.enums.stretch_cameras import StretchCameras
from stretch4_mujoco.enums.stretch_sensors import StretchSensors
from stretch4_mujoco.sim_teleop import GamepadTeleop
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator


@click.command()
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option("--select_env", is_flag=True, help="Interactively select an environment")
@click.option("--headless", is_flag=True, help="Run in headless mode")
@click.option("--imagery", is_flag=True, help="Show all the cameras' imagery")
@click.option("--lidar2d", is_flag=True, help="Show the lidar scan in Matplotlib")
@click.option("--lidar3d", is_flag=True, help="Show the point cloud in Rerun")
@click.option("--print-ratio", is_flag=True, help="Print the sim-to-real time ratio to the cli.")
@click.option("--use_stretch_3", type=bool, is_flag=True, help="Use Stretch 3")
def main(
    scene_xml_path: str | None,
    select_env: bool,
    headless: bool,
    imagery: bool,
    lidar2d: bool,
    lidar3d: bool,
    print_ratio: bool,
    use_stretch_3: bool,
):
    rerun_logger = RerunLogger()

    simulator_class = StretchMujocoSimulator if use_stretch_3 else Stretch4MujocoSimulator

    cameras_to_use = simulator_class.get_rgb_cameras() if imagery else  [StretchCameras.cam_gripper_rgb]

    if lidar3d and not simulator_class is Stretch4MujocoSimulator:
        raise NotImplementedError("3D Lidar is only supported in Stretch4MujocoSimulator.")

    if lidar3d:
        rerun_logger.init_pointcloud_viz(use_stretch_3)

    use_imagery = len(cameras_to_use) > 0

    model = None

    if select_env:
        from stretch4_mujoco.robocasa_gen import model_generation_wizard

        model, xml, objects_info = model_generation_wizard(
            stretch_xml_absolute=simulator_class.get_robot_xml_path(),
            objects_list=["apple", "cup", "can", "milk"],
        )

    sim = simulator_class(
        model=model,
        scene_xml_path=scene_xml_path,
        cameras_to_use=cameras_to_use,
        camera_hz=10.00 if lidar3d else 30.0,
    )

    teleop = None
    try:
        sim.start(headless=headless)
        teleop = GamepadTeleop(sim)
        teleop.start()

        while sim.is_running():
            if not lidar2d and not use_imagery:
                time.sleep(0.05)

            if print_ratio:
                print(f"{sim.pull_status().sim_to_real_time_ratio_msg}")

            if use_imagery:
                show_camera_feeds_sync(sim, False)

            if lidar3d:
                rerun_logger.update_pointcloud_viz(
                    sim.pull_lidar_points(), "world/lidar_points"
                )

            if lidar2d:
                sensor_data = sim.pull_sensor_data()

                try:
                    show_laser_scan(
                        scan_data=sensor_data.get_data(StretchSensors.base_lidar),
                        is_se4=simulator_class is Stretch4MujocoSimulator,
                    )
                except:
                    ...
    except KeyboardInterrupt:
        pass
    finally:
        rerun_logger.stop()
        if teleop is not None:
            teleop.stop()
        sim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
