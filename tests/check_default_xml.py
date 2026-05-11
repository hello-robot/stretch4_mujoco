import click
import pathlib
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
import stretch_mujoco.utils as utils
from mujoco._structs import MjModel


if __name__ == "__main__":
    for simulator_class in [Stretch4MujocoSimulator, StretchMujocoSimulator]:
        for xml_path_method in [simulator_class.get_scene_xml_path,
                                simulator_class.get_robot_xml_path]:
            try:
                xml_path = pathlib.Path(xml_path_method())
                assert xml_path.exists()
            except Exception as e:
                click.secho(f'Call to {xml_path_method.__name__} failed: {e}', fg='red')
                continue

            short_path = xml_path.relative_to(utils.models_path)
            try:
                model = MjModel.from_xml_path(str(xml_path))
                click.secho(f'Loaded {short_path}', fg='green')
            except Exception as e:
                click.secho(f'Failed to load {short_path}: {e}', fg='red')

        urdf_path = simulator_class.get_urdf_path()
        urdf_path = pathlib.Path(urdf_path)
        if urdf_path.exists():
            click.secho(f'URDF {urdf_path} Exists!', fg='green')
        else:
            click.secho(f'URDF {urdf_path} does not exist!', fg='red')
            continue

        model = utils.URDFmodel(urdf_path)
        click.secho(f'URDF has {len(model.joint_names)} joints defined.', fg='cyan')
