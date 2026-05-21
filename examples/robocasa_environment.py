import click
import cv2

from examples.camera_feeds import show_camera_feeds_sync
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.robocasa_gen import model_generation_wizard
from stretch_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator


@click.command()
@click.option("--task", type=str, default="PickPlaceCounterToCabinet", help="task")
@click.option("--layout", type=int, default=None, help="kitchen layout (choose number 0-9)")
@click.option("--style", type=int, default=None, help="kitchen style (choose number 0-11)")
@click.option("--write-to-file", type=str, default=None, help="write to file")
@click.option("--use_stretch_3", type=bool, is_flag=True, help="Use Stretch 3")
def main(task: str, layout: int, style: int, write_to_file:str, use_stretch_3:bool):

    simulator_class = StretchMujocoSimulator if use_stretch_3 else Stretch4MujocoSimulator

    cameras_to_use = simulator_class.get_rgb_cameras()

    model, xml, objects_info = model_generation_wizard(
        stretch_xml_absolute=simulator_class.get_robot_xml_path(),
        task=task,
        layout=layout,
        style=style,
        write_to_file=write_to_file,
    )
    sim = simulator_class(model=model, cameras_to_use=cameras_to_use)
    sim.start()
    # display camera feeds
    try:
        while sim.is_running():
            show_camera_feeds_sync(sim, True)
    except KeyboardInterrupt:
        sim.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
