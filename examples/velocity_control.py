#!/usr/bin/env python3
"""
Example demonstrating how to use the set_velocity() API for velocity control of 
both the mobile base and individual robot joints (e.g., lift and arm).
"""

import time
import click

from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
from stretch4_mujoco.stretch_mujoco_simulator import StretchMujocoSimulator


@click.command()
@click.option("--use_stretch_3", type=bool, is_flag=True, help="Use Stretch 3 instead of Stretch 4")
def main(use_stretch_3: bool):
    # Initialize the corresponding simulator class
    if use_stretch_3:
        click.secho("Initializing Stretch 3 simulator...", fg="yellow")
        sim = StretchMujocoSimulator()
    else:
        click.secho("Initializing Stretch 4 simulator...", fg="yellow")
        sim = Stretch4MujocoSimulator()

    # Start the simulator with GUI
    sim.start(headless=False)

    try:
        click.secho("Stowing robot to a safe start position...", fg="cyan")
        sim.stow()
        sim.wait_command()

        # =====================================================================
        # 1. Base Velocity Control
        # =====================================================================
        click.secho("\n--- 1. Base Velocity Control ---", fg="green", bold=True)
        
        if use_stretch_3:
            # Stretch 3: Differential drive base (forwards/backwards and rotation)
            # set_velocity(vx_m, vy_m, w_r) where vy_m must be 0.0
            click.secho("Moving forward (vx = 0.2 m/s)...", fg="cyan")
            sim.base.set_velocity(vx_m=0.2, vy_m=0.0, w_r=0.0)
            time.sleep(2.0)
            
            click.secho("Rotating counter-clockwise (w_r = 0.5 rad/s)...", fg="cyan")
            sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.5)
            time.sleep(2.0)
        else:
            # Stretch 4: Omnidirectional base (supports sideways/crab walk translation)
            click.secho("Moving diagonally (vx = 0.4 m/s, vy = 0.4 m/s)...", fg="cyan")
            sim.base.set_velocity(vx_m=0.4, vy_m=0.4, w_r=0.0)
            time.sleep(5.0)
            
            click.secho("Crab-walking sideways to the left (vy = -0.4 m/s)...", fg="cyan")
            sim.base.set_velocity(vx_m=0.0, vy_m=-0.4, w_r=0.0)
            time.sleep(5.0)

            click.secho("Crab-walking backward to the left (vx = -0.4 m/s)...", fg="cyan")
            sim.base.set_velocity(vx_m=-0.4, vy_m=0.0, w_r=0.0)
            time.sleep(5.0)
            
            click.secho("Rotating while moving forward (vx = 0.5 m/s, w_r = 1 rad/s)...", fg="cyan")
            sim.base.set_velocity(vx_m=0.5, vy_m=0.0, w_r=1)
            time.sleep(5.0)

        click.secho("Stopping the base...", fg="cyan")
        sim.base.set_velocity(0.0, 0.0, 0.0)
        time.sleep(0.5)

        # =====================================================================
        # 2. Joint Velocity Control
        # =====================================================================
        click.secho("\n--- 2. Joint Velocity Control ---", fg="green", bold=True)
        
        # Lift joint velocity control (upwards)
        click.secho("Moving lift upwards (v = 0.1 m/s)...", fg="cyan")
        sim.lift.set_velocity(0.1)
        
        # Monitor the lift position and velocity for 2 seconds
        for _ in range(20):
            if not sim.is_running():
                break
            status = sim.pull_status()
            print(f"Lift -> Pos: {status.lift.pos:.4f} m | Vel: {status.lift.vel:.4f} m/s")
            time.sleep(0.1)

        # Stop lift
        click.secho("Stopping lift...", fg="cyan")
        sim.lift.set_velocity(0.0)
        time.sleep(0.5)

        # Arm joint velocity control (extending)
        click.secho("Extending arm outwards (v = 0.08 m/s)...", fg="cyan")
        sim.arm.set_velocity(0.08)

        # Monitor the arm position and velocity for 2 seconds
        for _ in range(20):
            if not sim.is_running():
                break
            status = sim.pull_status()
            print(f"Arm  -> Pos: {status.arm.pos:.4f} m | Vel: {status.arm.vel:.4f} m/s")
            time.sleep(0.1)

        # Stop arm
        click.secho("Stopping arm...", fg="cyan")
        sim.arm.set_velocity(0.0)
        time.sleep(0.5)

        # Retracting arm and lowering lift back
        click.secho("Retracting arm and lowering lift...", fg="cyan")
        sim.arm.set_velocity(-0.08)
        sim.lift.set_velocity(-0.1)
        time.sleep(1.5)

        # Stop all joint movements
        sim.arm.set_velocity(0.0)
        sim.lift.set_velocity(0.0)
        time.sleep(0.5)

        # =====================================================================
        # 3. End-of-Arm Subsystem Velocity Control
        # =====================================================================
        click.secho("\n--- 3. End-of-Arm Subsystem Velocity Control ---", fg="green", bold=True)

        # Wrist Yaw velocity control
        click.secho("Rotating wrist yaw (v = 0.5 rad/s)...", fg="cyan")
        sim.end_of_arm.wrist_yaw.set_velocity(-0.5)
        time.sleep(1.5)
        sim.end_of_arm.wrist_yaw.set_velocity(0.0)
        time.sleep(0.5)

        # Wrist Pitch velocity control
        click.secho("Rotating wrist pitch (v = -0.5 rad/s)...", fg="cyan")
        sim.end_of_arm.wrist_pitch.set_velocity(-0.5)
        time.sleep(1.5)
        sim.end_of_arm.wrist_pitch.set_velocity(0.0)
        time.sleep(0.5)

        # Wrist Roll velocity control
        click.secho("Rotating wrist roll (v = 0.5 rad/s)...", fg="cyan")
        sim.end_of_arm.wrist_roll.set_velocity(0.5)
        time.sleep(1.5)
        sim.end_of_arm.wrist_roll.set_velocity(0.0)
        time.sleep(0.5)

        # Gripper velocity control (opening and closing)
        click.secho("Opening gripper (v = 0.2 rad/s)...", fg="cyan")
        sim.end_of_arm.stretch_gripper.set_velocity(0.2)
        
        # Monitor gripper opening
        for _ in range(50):
            if not sim.is_running():
                break
            status = sim.pull_status()
            print(f"Gripper -> Pos: {status.gripper.pos:.4f} rad | Vel: {status.gripper.vel:.4f} rad/s")
            time.sleep(0.1)

        click.secho("Closing gripper (v = -0.2 rad/s)...", fg="cyan")
        sim.end_of_arm.stretch_gripper.set_velocity(-0.2)
        time.sleep(1.0)
        sim.end_of_arm.stretch_gripper.set_velocity(0.0)
        time.sleep(0.5)
        
        click.secho("\nStowing robot safely before shutdown...", fg="cyan")
        sim.stow()
        sim.wait_command()
        click.secho("Demonstration complete!", fg="green", bold=True)

    except KeyboardInterrupt:
        click.secho("\nInterrupted by user.", fg="yellow")
    finally:
        # Gracefully stop the simulator and server processes
        click.secho("Shutting down simulator...", fg="red")
        sim.stop()


if __name__ == "__main__":
    main()
