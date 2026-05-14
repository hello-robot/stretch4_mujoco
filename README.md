# Overview

This repo provides a simulation stack for Stretch 4, built on [MuJoCo](https://github.com/google-deepmind/mujoco). The simulation for Stretch 3 can be found in the [stretch_mujoco](https://github.com/hello-robot/stretch_mujoco) repo. The simulation includes position control for the arm and gripper joints, velocity control for mobile base, calibrated camera RGB + depth imagery, 3D lidar clouds, and more. There is a visualizer that supports user interaction, or a more efficient headless mode. There is a [ROS2 package](https://github.com/hello-robot/stretch4_ros2/tree/jazzy/stretch_simulation), built on this library, that works with Nav2 and more. There is 100s of permutations of Robocasa-provided kitchen environments that Stretch can spawn into. The MuJoCo API can be used for features like deformables, procedural model generation, SDF collisions, cloth simulation, and more.

Check out a video of Stretch 4 in Robocasa environments in Mujoco:

https://github.com/user-attachments/assets/ea683561-998b-44d3-9d45-41ab1b1664ab

## Getting Started

First, install [`uv`](https://docs.astral.sh/uv/#getting-started). Uv is a package manager that we'll use to run this project.

Then, clone this repo:

```
git clone https://github.com/hello-robot/stretch4_mujoco --recurse-submodules
cd stretch4_mujoco
```

> If you've already cloned the repo without `--recurse-submodules`, run `git submodule update --init` to pull the submodule.

Then, install this repo:

```
uv venv
uv pip install -e .
```

Lastly, run the simulation:

```
uv run launch_sim.py
```

> Note: If you see a build error mentioning `evdev` on linux, please run `sudo apt install python3-dev`.

To exit, press `Ctrl+C` in the terminal.

> On MacOS, if `mjpython` fails to locate `libpython3.10.dylib` and `libz.1.dylib`, run these commands:
```shell
# Before proceeding, please reload your terminal and/or IDE window, to make sure the correct UV environment variables are loaded.

source .venv/bin/activate

# When `libpython3.10.dylib` is missing, run:
PYTHON_LIB_DIR=$(python3 -c 'from distutils.sysconfig import get_config_var; print(get_config_var("LIBDIR"))')
ln -s "$PYTHON_LIB_DIR/libpython3.10.dylib" ./.venv/lib/libpython3.10.dylib

# When `libz.1.dylib` is missing, run:
export DYLD_LIBRARY_PATH=/usr/lib:$DYLD_LIBRARY_PATH
```

## GPU Acceleration

On Linux, by default, the Python code will attempt to use GPU acceleration by setting the necessary environment variables at runtime. However, if you are running examples outside the typical Python execution flow or need to ensure these are set system-wide, you can manually export the following environment variables:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_FLAGS=--xla_gpu_triton_gemm_any=true
```

These variables ensure that MuJoCo and any associated rendering libraries utilize hardware acceleration, which is highly recommended for performance when working with cameras or complex scenes.

## Example Scripts

[Keyboard teleop](./examples/keyboard_teleop.py)

```
uv run examples/keyboard_teleop.py
```

[Gamepad teleop](./examples/gamepad_teleop.py)

Control Stretch in simulation using any xbox type gamepad (uses xinput)

```
uv run examples/gamepad_teleop.py
```

[Robocasa environments](./examples/robocasa_environment.py)

```
# Setup
uv pip install -e "robocasa@third_party/robocasa"
uv pip install -e "robosuite@third_party/robosuite"
uv run third_party/robosuite/robosuite/scripts/setup_macros.py
uv run third_party/robocasa/robocasa/scripts/setup_macros.py
uv run third_party/robocasa/robocasa/scripts/download_kitchen_assets.py

# Run sim
uv run examples/keyboard_teleop.py --select_env
uv run examples/gamepad_teleop.py --select_env
uv run examples/robocasa_environment.py
```

Ignore any warnings.

## Writing Code

Use the Stretch4MujocoSimulator class to:

 * start the simulation
 * position control the robot's ranged joints
 * velocity control the robot's mobile base
 * read joint states
 * read camera imagery

Try the code below using `uv run ipython`. For advanced Mujoco users, the class also exposes the `mjModel` and `mjData`. See the [official Mujoco documentation](https://mujoco.readthedocs.io/en/stable/python.html).

```python
from stretch_mujoco import Stretch4MujocoSimulator

if __name__ == "__main__":
    sim = Stretch4MujocoSimulator()
    sim.start(headless=False) # This will open a Mujoco-Viewer window

    # Poses
    sim.stow()
    sim.home()

    # Position Control
    sim.move_to('lift', 1.0)
    sim.move_by('head_pan', -1.1)
    sim.move_by('base_translate', 0.1)

    sim.wait_until_at_setpoint('lift')
    sim.wait_while_is_moving('base_translate')

    # Base Velocity control
    sim.set_base_velocity(0.3, -0.1)

    # Get Joint Status
    from pprint import pprint
    pprint(sim.pull_status())

    # Get Camera Frames
    camera_data = sim.pull_camera_data()
    pprint(camera_data)

    # Kills simulation process
    sim.stop()
```

### Loading Robocasa Kitchen Scenes

The `stretch_mujoco.robocasa_gen.model_generation_wizard()` method gives you:

- Wizard/API to generate a kitchen model for a given task, layout, and style.
- If layout and style are not provided, it will take you through a wizard to choose them in the terminal.
- If robot_spawn_pose is not provided, it will spawn the robot to the default pose from robocasa fixtures.
- You can also write the generated xml model with absolutepaths to a file.

```python
from stretch_mujoco import Stretch4MujocoSimulator
from stretch_mujoco.robocasa_gen import model_generation_wizard

# Use the wizard:
model, xml, objects_info = model_generation_wizard(stretch_xml_absolute=StretchMujocoSimulator.get_robot_xml_path(),)

# Or, launch a specific task/layout/style
model, xml = model_generation_wizard(
    stretch_xml_absolute=StretchMujocoSimulator.get_robot_xml_path(),
    task=<task_name>,
    layout=<layout_id>,
    style=<style_id>,
    wrtie_to_file=<filename>,
)

sim = Stretch4MujocoSimulator(model=model)
sim.start()
```

### ROS2

You can use this simulation in ROS2 using the [`stretch_simulation` package](https://github.com/hello-robot/stretch4_ros2/tree/jazzy/stretch_simulation) in `stretch4_ros2`.

### Docs

Check out the following documentation resources:

- [Architecture](./docs/architecture.md)
- [Stretch 4 MJCF and URDF](./stretch_mujoco/models/stretch_4/README.md)

### Feature Requests and Bug reporting

All the enhancements/bugfixes are tracked by [Github Issues](https://github.com/hello-robot/stretch4_mujoco/issues) filed. Please feel free to file an issue if you would like to report a bug or request a feature addition.

## Acknowledgment

The assets in this repository contain significant contributions and efforts from [Kevin Zakka](https://github.com/kevinzakka) and [Google Deepmind](https://github.com/google-deepmind), along with others in Hello Robot Inc. who helped us in modeling Stretch in Mujoco. Thank you for your contributions.


## License

The license covering the code and assets in this repo can be found in the [LICENSE](./LICENSE) file.
