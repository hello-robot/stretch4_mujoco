from examples.rerun_utils import RerunLogger
import stretch4_mujoco
import click
import cv2

from stretch4_mujoco.enums.stretch_cameras import StretchCameras


@click.command()
@click.option("--scene-xml-path", help="Path to a scene xml file")
@click.option("--headless", is_flag=True, help="Run the simulation headless")
@click.option("--imagery", is_flag=True, help="Show the cameras' imagery")
@click.option("--lidar3d", is_flag=True, help="Show the point cloud in Rerun")
@click.option("--use_stretch_3", type=bool, is_flag=True, help="Use Stretch 3, otherwise use Stretch 4")
def main(
    scene_xml_path: str,
    headless: bool,
    imagery: bool,
    lidar3d: bool,
    use_stretch_3:bool
):

    rerun_logger = RerunLogger()

    cameras_to_use =( StretchCameras.rgb_stretch4() if not use_stretch_3 else StretchCameras.all_stretch3()) if imagery else []

    if lidar3d and use_stretch_3:
        raise NotImplementedError("3D Lidar is only supported in Stretch4MujocoSimulator. Do not use the --use_stretch_3 flag.")

    if lidar3d:
        cameras_to_use += StretchCameras.hemispherical_lidars()
        rerun_logger.init_pointcloud_viz()

    sim = stretch4_mujoco.StretchMujocoSimulator(scene_xml_path,cameras_to_use=cameras_to_use) if use_stretch_3 else stretch4_mujoco.Stretch4MujocoSimulator(scene_xml_path, cameras_to_use=cameras_to_use)

    sim.start(headless=headless)
    try:
        while sim.is_running():
            camera_data = sim.pull_camera_data()

            for camera in cameras_to_use:
                cv2.namedWindow(camera.name, cv2.WINDOW_NORMAL)
                cv2.imshow(camera.name, camera_data.get_camera_data(camera))

            if lidar3d:
                rerun_logger.update_pointcloud_viz(sim.pull_hemi_lidar_points(in_world_frame=True), "world/lidar_points")

            cv2.waitKey(10)

    except KeyboardInterrupt:
        pass
    finally:
        rerun_logger.stop()
        sim.stop()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
