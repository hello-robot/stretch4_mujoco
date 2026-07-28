import math
import time

from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator


def main():

    sim = Stretch4MujocoSimulator()

    sim.start(headless=False)

    try:
        sim.stow()

        while sim.is_running():

            status = sim.pull_status()

            sim.base.translate_by(0.5, 0.0)
            
            sim.wait_command()

            sim.base.translate_by(0.0, 0.5)
            
            sim.wait_command()

            sim.base.translate_by(-0.5, -0.5)
            
            sim.wait_command()

            sim.base.rotate_by(math.radians(-45))

            sim.wait_command()

            sim.base.rotate_by(math.radians(90))

            sim.wait_command()

            sim.base.set_velocity(0.5, 0.5, 0.0)

            time.sleep(2)

            sim.base.set_velocity(-0.5, -0.5, 0.5)

            time.sleep(2)
            

    except KeyboardInterrupt:
        sim.stop()

if __name__ == "__main__":
    main()