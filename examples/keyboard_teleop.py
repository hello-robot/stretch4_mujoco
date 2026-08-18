from time import sleep, perf_counter

import click
import cv2

from examples.camera_feeds import show_camera_feeds_sync
from examples.rerun_utils import RerunLogger
from examples.laser_scan import show_laser_scan
from stretch4_mujoco import StretchMujocoSimulator
from stretch4_mujoco.enums.stretch_sensors import StretchSensors
from stretch4_mujoco.sim_teleop import KeyboardTeleop, print_keyboard_help
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator


@click.command()
@click.option("--scene-xml-path", type=str, default=None, help="Path to the scene xml file")
@click.option("--select_env", is_flag=True, help="Use robocasa environment")
@click.option("--imagery", is_flag=True, help="Show all the cameras' imagery")
@click.option(
    "--lidar",
    is_flag=True,
    help="Show the lidar scan: a 3D point cloud in Rerun for Stretch4MujocoSimulator, "
    "or a 2D scan in Matplotlib for Stretch3.",
)
@click.option(
    "--opencv",
    is_flag=True,
    help="Show camera imagery in OpenCV windows instead of Rerun.",
)
@click.option("--show_metrics", is_flag=True, help="Print the sim-to-real time ratio to the cli.")
@click.option("--use_stretch_3", type=bool, is_flag=True, help="Use Stretch 3")
def main(
    scene_xml_path: str | None,
    select_env: bool,
    imagery: bool,
    lidar: bool,
    opencv: bool,
    show_metrics: bool,
    use_stretch_3: bool,
):

    rerun_logger = RerunLogger()

    simulator_class = StretchMujocoSimulator if use_stretch_3 else Stretch4MujocoSimulator

    cameras_to_use = simulator_class.get_all_cameras() if imagery else []

    show_lidar_3d = lidar and simulator_class is Stretch4MujocoSimulator
    show_lidar_2d = lidar and not show_lidar_3d

    use_imagery = len(cameras_to_use) > 0

    if show_lidar_3d or ((use_imagery or show_metrics) and not opencv):
        # With --opencv the frames go to OpenCV windows, so Rerun shouldn't lay
        # out camera panels that will never receive an image.
        rerun_logger.init_rerun(
            use_stretch_3, cameras_to_use=[] if opencv else cameras_to_use
        )

    model = None

    if select_env:
        from stretch4_mujoco.robocasa_gen import model_generation_wizard

        model, xml, objects_info = model_generation_wizard(
            stretch_xml_absolute=simulator_class.get_robot_xml_path()
        )

    rate = 30.0

    sim = simulator_class(
        model=model,
        scene_xml_path=scene_xml_path,
        cameras_to_use=cameras_to_use,
        camera_hz=rate,
    )

    teleop = None
    try:
        sim.start()

        print_keyboard_help()

        teleop = KeyboardTeleop(sim, rate_hz=rate)
        teleop.start()

        last_loop = perf_counter()
        loop_time = 1.0/rate #hz->sec

        while sim.is_running():

            # rate limit loop
            elapsed = perf_counter()-last_loop
            if elapsed < loop_time:
                sleep(loop_time-elapsed)
            last_loop = perf_counter()

            camera_data = None
            if use_imagery:
                if opencv:
                    show_camera_feeds_sync(sim, False)
                else:
                    camera_data = sim.pull_camera_data()
                    rerun_logger.update_camera_images(camera_data)

            if show_metrics:
                if opencv:
                    print(f"{sim.pull_status().sim_to_real_time_ratio_msg}")
                else:
                    status = sim.pull_status()
                    metrics = {"physics_fps": status.fps}
                    if camera_data is not None:
                        metrics["camera_fps"] = camera_data.fps
                    rerun_logger.update_metrics(metrics, message=status.sim_to_real_time_ratio_msg)

            if show_lidar_3d:
                rerun_logger.update_pointcloud_viz(
                    sim.pull_lidar_points(), "world/lidar_points"
                )

            if show_lidar_2d:
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
