import random
import threading
import click
import numpy as np

from stretch4_mujoco.enums.actuators import Actuators
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
from stretch4_mujoco.stretch4_mujoco_simulator import StretchMujocoSimulator


def lift_sequence(
    sim: StretchMujocoSimulator,
):
    while sim.is_running():
        sim.end_of_arm.wrist_pitch.move_to(random.random() * 2 - 1)
        sim.end_of_arm.wrist_roll.move_to(random.random() * 2 - 1)
        sim.end_of_arm.wrist_yaw.move_to(random.random() * 2 - 1)
        gripper_target = (random.random() * 50.0) if isinstance(sim, Stretch4MujocoSimulator) else (random.random() / 2)
        sim.end_of_arm.stretch_gripper.move_to(gripper_target)
        if hasattr(sim, "head"):
            sim.head.head_pan.move_to(random.random() - 0.5)
            sim.head.head_tilt.move_to(random.random() - 0.5)

        LIFT_START_POS = 0.1
        MOVE_LIFT_BY = 0.5

        sim.lift.move_to(LIFT_START_POS)
        sim.wait_until_at_setpoint(Actuators.lift)

        start_lift_position = sim.pull_status().lift.pos

        if not np.isclose(start_lift_position, LIFT_START_POS, atol=0.05):
            print(
                f"The lift did not move to the starting position. Should be at {LIFT_START_POS}, but is at {start_lift_position:.2f} instead."
            )

        sim.lift.move_by(MOVE_LIFT_BY)
        sim.wait_while_is_moving(Actuators.lift)

        current_lift_position = sim.pull_status().lift.pos

        if not np.isclose(start_lift_position + MOVE_LIFT_BY, current_lift_position, atol=0.05):
            print(
                f"The lift did not move by the specified amount. Asked to move from {start_lift_position:.4f} by {MOVE_LIFT_BY}, but ended up at {current_lift_position:.4f}. Should be {start_lift_position + MOVE_LIFT_BY :.4f}"
            )

        if hasattr(sim, "head"):
            sim.wait_until_at_setpoint(Actuators.head_pan, timeout=0.1)
            sim.wait_until_at_setpoint(Actuators.head_tilt, timeout=0.1)

        sim.wait_until_at_setpoint(Actuators.wrist_pitch, timeout=0.1)
        sim.wait_until_at_setpoint(Actuators.wrist_roll, timeout=0.1)
        sim.wait_until_at_setpoint(Actuators.wrist_yaw, timeout=0.1)
        sim.wait_until_at_setpoint(Actuators.gripper, timeout=0.1)


def set_base_velocity(sim: StretchMujocoSimulator, v_linear: float, omega: float):
    """
    Set the base velocity of the robot.
    """
    if isinstance(sim, Stretch4MujocoSimulator):
        sim.base.set_velocity(v_linear, 0.0, omega / 3)
    else:
        sim.base.set_velocity(v_linear, 0.0, omega)


@click.command()
@click.option("--use_stretch_3", type=bool, is_flag=True, help="Use Stretch 3")
def main(use_stretch_3: bool):

    sim = StretchMujocoSimulator() if use_stretch_3 else Stretch4MujocoSimulator()

    sim.start(headless=False)

    try:
        sim.stow()

        set_base_velocity(sim, v_linear=0.5, omega=5)

        thread = threading.Thread(target=lift_sequence, daemon=False, args=[sim])
        thread.start()

        target = 1.1  # m
        while sim.is_running():

            status = sim.pull_status()

            current_position = status.base.x

            # print(f"{status=}")

            if target > 0 and current_position > target:
                target *= -1
                set_base_velocity(sim, v_linear=-0.5, omega=-5)
            elif target < 0 and current_position < target:
                target *= -1
                set_base_velocity(sim, v_linear=0.5, omega=5)

        thread.join()

    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
