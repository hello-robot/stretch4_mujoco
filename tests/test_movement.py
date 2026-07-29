import time
import click
import numpy as np
from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator


def wait_for_sim_time(sim, duration_s):
    t_start = sim.pull_status().time
    while sim.pull_status().time - t_start < duration_s:
        time.sleep(0.01)


def get_perf_str(sim):
    """Returns formatted sim FPS and real-time-ratio string."""
    status = sim.pull_status()
    fps = status.fps
    rtr = status.sim_to_real_time_ratio_msg or "1.0x"
    return f"FPS: {fps:.1f}, RTR: {rtr}"


def run_movement_tests(sim, test_cameras=False):
    """Runs the core movement test suite on an active simulator instance."""
    # ==========================================
    # 1. Base Tests
    # ==========================================
    click.secho("\n--- Testing Base Movement ---", fg="yellow")

    # 1a. Base translate_by
    start_x = sim.base.status.x
    sim.base.translate_by(0.20)
    time.sleep(0.3)
    assert sim.wait_command(timeout=5.0, position_tolerance=0.001)
    end_x = sim.base.status.x
    disp_x = end_x - start_x
    click.secho(
        f"Base translate_by(0.20): start={start_x:.4f}, end={end_x:.4f}, disp={disp_x:.4f}m [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_x, 0.20, atol=0.04), f"Base translate_by failed: disp={disp_x:.4f}m"

    # 1b. Base rotate_by
    start_theta = sim.base.status.theta
    rotate_by = np.radians(90)
    sim.base.rotate_by(rotate_by)
    assert sim.wait_command(timeout=5.0, position_tolerance=0.001)
    end_theta = sim.base.status.theta
    disp_theta = end_theta - start_theta
    click.secho(
        f"Base rotate_by({rotate_by:.4f}): start={start_theta:.4f}, end={end_theta:.4f}, disp={disp_theta:.4f}rad [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_theta, rotate_by, atol=0.08), f"Base rotate_by failed: disp={disp_theta:.4f}rad"

    # ==========================================
    # Omnibase Velocity & Error/Bias Metrics
    # ==========================================
    click.secho("\n--- Testing Omnibase Velocity & Error Metrics ---", fg="yellow")

    # Helper for converting world delta to local frame relative to initial heading
    def world_to_local(dx_world, dy_world, th0):
        cos_th = np.cos(th0)
        sin_th = np.sin(th0)
        disp_x_local = dx_world * cos_th + dy_world * sin_th
        disp_y_local = -dx_world * sin_th + dy_world * cos_th
        return disp_x_local, disp_y_local

    # 1c. Base set_velocity X (Forward 0.20 m/s for 4.0s -> Expected 0.80m)
    start_x = sim.base.status.x
    start_y = sim.base.status.y
    start_th = sim.base.status.theta
    sim.base.set_velocity(vx_m=0.20, vy_m=0.0, w_r=0.0)
    wait_for_sim_time(sim, 4.0)
    sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.0)
    time.sleep(0.5)
    end_x = sim.base.status.x
    end_y = sim.base.status.y
    end_th = sim.base.status.theta

    disp_x, disp_y = world_to_local(end_x - start_x, end_y - start_y, start_th)
    expected_x = 0.80
    err_x = abs(disp_x - expected_x)
    pct_err_x = (err_x / expected_x) * 100.0
    drift_y = abs(disp_y)
    drift_th = abs(end_th - start_th)
    click.secho(
        f"Omnibase Vel X (0.20 m/s x 4s): disp_x={disp_x:.4f}m (exp {expected_x:.2f}m), "
        f"err={err_x:.4f}m ({pct_err_x:.2f}%), drift_y={drift_y:.4f}m, drift_th={drift_th:.4f}rad [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_x, expected_x, atol=0.05), f"Base X velocity failed: disp_x={disp_x:.4f}m"
    assert drift_y < 0.03, f"Base X velocity transverse drift Y too high: drift_y={drift_y:.4f}m"
    assert drift_th < 0.05, f"Base X velocity heading drift too high: drift_th={drift_th:.4f}rad"

    # 1d. Base set_velocity Y (Lateral 0.20 m/s for 3.0s -> Expected 0.60m)
    start_x = sim.base.status.x
    start_y = sim.base.status.y
    start_th = sim.base.status.theta
    sim.base.set_velocity(vx_m=0.0, vy_m=0.20, w_r=0.0)
    wait_for_sim_time(sim, 3.0)
    sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.0)
    time.sleep(0.5)
    end_x = sim.base.status.x
    end_y = sim.base.status.y
    end_th = sim.base.status.theta

    disp_x, disp_y = world_to_local(end_x - start_x, end_y - start_y, start_th)
    expected_y = 0.60
    err_y = abs(disp_y - expected_y)
    pct_err_y = (err_y / expected_y) * 100.0
    drift_x = abs(disp_x)
    drift_th = abs(end_th - start_th)
    click.secho(
        f"Omnibase Vel Y (0.20 m/s x 3s): disp_y={disp_y:.4f}m (exp {expected_y:.2f}m), "
        f"err={err_y:.4f}m ({pct_err_y:.2f}%), drift_x={drift_x:.4f}m, drift_th={drift_th:.4f}rad [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_y, expected_y, atol=0.05), f"Base Y velocity failed: disp_y={disp_y:.4f}m"
    assert drift_x < 0.03, f"Base Y velocity transverse drift X too high: drift_x={drift_x:.4f}m"
    assert drift_th < 0.05, f"Base Y velocity heading drift too high: drift_th={drift_th:.4f}rad"

    # 1e. Base set_velocity Omnidirectional (vx=0.15, vy=0.15 m/s for 3.0s -> Expected dx=0.45m, dy=0.45m, dist=0.6364m)
    start_x = sim.base.status.x
    start_y = sim.base.status.y
    start_th = sim.base.status.theta
    sim.base.set_velocity(vx_m=0.15, vy_m=0.15, w_r=0.0)
    wait_for_sim_time(sim, 3.0)
    sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.0)
    time.sleep(0.5)
    end_x = sim.base.status.x
    end_y = sim.base.status.y
    end_th = sim.base.status.theta

    dx, dy = world_to_local(end_x - start_x, end_y - start_y, start_th)
    disp_dist = np.hypot(dx, dy)
    expected_dist = float(np.hypot(0.45, 0.45))
    err_dist = abs(disp_dist - expected_dist)
    pct_err_dist = (err_dist / expected_dist) * 100.0
    drift_th = abs(end_th - start_th)
    click.secho(
        f"Omnibase Vel Omni (0.15, 0.15 m/s x 3s): disp_dist={disp_dist:.4f}m (exp {expected_dist:.4f}m, dx={dx:.4f}m, dy={dy:.4f}m), "
        f"err={err_dist:.4f}m ({pct_err_dist:.2f}%), drift_th={drift_th:.4f}rad [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(dx, 0.45, atol=0.05), f"Base Omni velocity dx failed: dx={dx:.4f}m"
    assert np.isclose(dy, 0.45, atol=0.05), f"Base Omni velocity dy failed: dy={dy:.4f}m"
    assert np.isclose(disp_dist, expected_dist, atol=0.05), f"Base Omni velocity total dist failed: disp_dist={disp_dist:.4f}m"
    assert drift_th < 0.05, f"Base Omni velocity heading drift too high: drift_th={drift_th:.4f}rad"

    # 1f. Base set_velocity Angular (w_r=0.50 rad/s for 3.0s -> Expected ~1.32-1.50 rad with accel ramp)
    start_x = sim.base.status.x
    start_y = sim.base.status.y
    start_th = sim.base.status.theta
    sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.50)
    wait_for_sim_time(sim, 3.0)
    sim.base.set_velocity(vx_m=0.0, vy_m=0.0, w_r=0.0)
    time.sleep(0.5)
    end_x = sim.base.status.x
    end_y = sim.base.status.y
    end_th = sim.base.status.theta
    disp_th = end_th - start_th
    expected_th = 1.50
    err_th = abs(disp_th - expected_th)
    pct_err_th = (err_th / expected_th) * 100.0
    drift_pos = np.hypot(end_x - start_x, end_y - start_y)
    click.secho(
        f"Omnibase Vel Angular (0.50 rad/s x 3s): disp_th={disp_th:.4f}rad (exp {expected_th:.2f}rad), "
        f"err={err_th:.4f}rad ({pct_err_th:.2f}%), drift_pos={drift_pos:.4f}m [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_th, expected_th, atol=0.22), f"Base Angular velocity failed: disp_th={disp_th:.4f}rad"
    assert drift_pos < 0.04, f"Base Angular velocity position drift too high: drift_pos={drift_pos:.4f}m"

    # ==========================================
    # 2. Arm Tests
    # ==========================================
    click.secho("\n--- Testing Arm Movement ---", fg="yellow")

    # 2a. Arm move_by
    start_arm = sim.arm.status.pos
    sim.arm.move_by(0.10)
    time.sleep(0.3)
    sim.wait_command(timeout=5.0, position_tolerance=0.0001)
    end_arm = sim.arm.status.pos
    disp_arm = end_arm - start_arm
    click.secho(
        f"Arm move_by(0.10): start={start_arm:.4f}, end={end_arm:.4f}, disp={disp_arm:.4f}m [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_arm, 0.10, atol=0.03), f"Arm move_by failed: disp={disp_arm:.4f}m"

    # 2b. Arm set_velocity
    start_arm = sim.arm.status.pos
    v_arm = 0.08
    sim.arm.set_velocity(v_arm)
    wait_for_sim_time(sim, 2.0)
    sim.arm.set_velocity(0.0)
    time.sleep(0.5)
    end_arm = sim.arm.status.pos
    disp_arm = end_arm - start_arm
    click.secho(
        f"Arm set_velocity(0.08 m/s for 2s): start={start_arm:.4f}, end={end_arm:.4f}, disp={disp_arm:.4f}m (Expected ~0.16m) [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_arm, 0.16, atol=0.03), f"Arm set_velocity failed: disp={disp_arm:.4f}m"

    # ==========================================
    # 3. Lift Tests
    # ==========================================
    click.secho("\n--- Testing Lift Movement ---", fg="yellow")

    # 3a. Lift move_by
    start_lift = sim.lift.status.pos
    sim.lift.move_by(0.10)
    time.sleep(0.3)
    sim.wait_command(timeout=5.0, position_tolerance=0.0001)
    end_lift = sim.lift.status.pos
    disp_lift = end_lift - start_lift
    click.secho(
        f"Lift move_by(0.10): start={start_lift:.4f}, end={end_lift:.4f}, disp={disp_lift:.4f}m [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_lift, 0.10, atol=0.03), f"Lift move_by failed: disp={disp_lift:.4f}m"

    # 3b. Lift set_velocity
    start_lift = sim.lift.status.pos
    v_lift = 0.08
    sim.lift.set_velocity(v_lift)
    wait_for_sim_time(sim, 1.5)
    sim.lift.set_velocity(0.0)
    time.sleep(0.5)
    end_lift = sim.lift.status.pos
    disp_lift = end_lift - start_lift
    click.secho(
        f"Lift set_velocity(0.08 m/s for 1.5s): start={start_lift:.4f}, end={end_lift:.4f}, disp={disp_lift:.4f}m (Expected ~0.12m) [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_lift, 0.12, atol=0.03), f"Lift set_velocity failed: disp={disp_lift:.4f}m"

    # ==========================================
    # 4. Wrist Yaw Tests
    # ==========================================
    click.secho("\n--- Testing Wrist Yaw Movement ---", fg="yellow")

    # 4a. Wrist Yaw move_by
    start_yaw = sim.end_of_arm.wrist_yaw.status.pos
    sim.end_of_arm.wrist_yaw.move_by(0.20)
    time.sleep(0.3)
    sim.wait_command(timeout=5.0, position_tolerance=0.0001)
    end_yaw = sim.end_of_arm.wrist_yaw.status.pos
    disp_yaw = end_yaw - start_yaw
    click.secho(
        f"Wrist Yaw move_by(0.20): start={start_yaw:.4f}, end={end_yaw:.4f}, disp={disp_yaw:.4f}rad [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_yaw, 0.20, atol=0.04), f"Wrist Yaw move_by failed: disp={disp_yaw:.4f}rad"

    # 4b. Wrist Yaw set_velocity
    start_yaw = sim.end_of_arm.wrist_yaw.status.pos
    v_yaw = -0.20
    sim.end_of_arm.wrist_yaw.set_velocity(v_yaw)
    wait_for_sim_time(sim, 1.0)
    sim.end_of_arm.wrist_yaw.set_velocity(0.0)
    time.sleep(0.5)
    end_yaw = sim.end_of_arm.wrist_yaw.status.pos
    disp_yaw = end_yaw - start_yaw
    click.secho(
        f"Wrist Yaw set_velocity(-0.20 rad/s for 1.0s): start={start_yaw:.4f}, end={end_yaw:.4f}, disp={disp_yaw:.4f}rad (Expected ~-0.20rad) [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_yaw, -0.20, atol=0.04), f"Wrist Yaw set_velocity failed: disp={disp_yaw:.4f}rad"

    # ==========================================
    # 5. Wrist Pitch Tests
    # ==========================================
    click.secho("\n--- Testing Wrist Pitch Movement ---", fg="yellow")

    # 5a. Wrist Pitch move_by
    start_pitch = sim.end_of_arm.wrist_pitch.status.pos
    sim.end_of_arm.wrist_pitch.move_by(0.20)
    time.sleep(0.3)
    sim.wait_command(timeout=5.0, position_tolerance=0.0001)
    end_pitch = sim.end_of_arm.wrist_pitch.status.pos
    disp_pitch = end_pitch - start_pitch
    click.secho(
        f"Wrist Pitch move_by(0.20): start={start_pitch:.4f}, end={end_pitch:.4f}, disp={disp_pitch:.4f}rad [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_pitch, 0.20, atol=0.04), f"Wrist Pitch move_by failed: disp={disp_pitch:.4f}rad"

    # 5b. Wrist Pitch set_velocity
    start_pitch = sim.end_of_arm.wrist_pitch.status.pos
    v_pitch = -0.20
    sim.end_of_arm.wrist_pitch.set_velocity(v_pitch)
    wait_for_sim_time(sim, 1.0)
    sim.end_of_arm.wrist_pitch.set_velocity(0.0)
    time.sleep(0.5)
    end_pitch = sim.end_of_arm.wrist_pitch.status.pos
    disp_pitch = end_pitch - start_pitch
    click.secho(
        f"Wrist Pitch set_velocity(-0.20 rad/s for 1.0s): start={start_pitch:.4f}, end={end_pitch:.4f}, disp={disp_pitch:.4f}rad (Expected ~-0.20rad) [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_pitch, -0.20, atol=0.04), f"Wrist Pitch set_velocity failed: disp={disp_pitch:.4f}rad"

    # ==========================================
    # 6. Wrist Roll Tests
    # ==========================================
    click.secho("\n--- Testing Wrist Roll Movement ---", fg="yellow")

    # 6a. Wrist Roll move_by
    start_roll = sim.end_of_arm.wrist_roll.status.pos
    sim.end_of_arm.wrist_roll.move_by(0.20)
    time.sleep(0.3)
    sim.wait_command(timeout=5.0, position_tolerance=0.0001)
    end_roll = sim.end_of_arm.wrist_roll.status.pos
    disp_roll = end_roll - start_roll
    click.secho(
        f"Wrist Roll move_by(0.20): start={start_roll:.4f}, end={end_roll:.4f}, disp={disp_roll:.4f}rad [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_roll, 0.20, atol=0.04), f"Wrist Roll move_by failed: disp={disp_roll:.4f}rad"

    # 6b. Wrist Roll set_velocity
    start_roll = sim.end_of_arm.wrist_roll.status.pos
    v_roll = 0.20
    sim.end_of_arm.wrist_roll.set_velocity(v_roll)
    wait_for_sim_time(sim, 1.0)
    sim.end_of_arm.wrist_roll.set_velocity(0.0)
    time.sleep(0.5)
    end_roll = sim.end_of_arm.wrist_roll.status.pos
    disp_roll = end_roll - start_roll
    click.secho(
        f"Wrist Roll set_velocity(0.20 rad/s for 1.0s): start={start_roll:.4f}, end={end_roll:.4f}, disp={disp_roll:.4f}rad (Expected ~0.20rad) [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_roll, 0.20, atol=0.04), f"Wrist Roll set_velocity failed: disp={disp_roll:.4f}rad"

    # ==========================================
    # 7. Gripper Tests
    # ==========================================
    click.secho("\n--- Testing Gripper Movement ---", fg="yellow")

    # 7a. Gripper move_by (open)
    start_grip = sim.end_of_arm.stretch_gripper.status.pos
    sim.end_of_arm.stretch_gripper.move_by(0.10)
    time.sleep(0.3)
    sim.wait_command(timeout=5.0, position_tolerance=0.0001)
    end_grip = sim.end_of_arm.stretch_gripper.status.pos
    disp_grip = end_grip - start_grip
    click.secho(
        f"Gripper move_by(0.10): start={start_grip:.4f}, end={end_grip:.4f}, disp={disp_grip:.4f} [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_grip, 0.10, atol=0.03), f"Gripper move_by failed: disp={disp_grip:.4f}"

    # 7b. Gripper set_velocity (close)
    start_grip = sim.end_of_arm.stretch_gripper.status.pos
    sim.end_of_arm.stretch_gripper.set_velocity(-0.10)
    wait_for_sim_time(sim, 1.0)
    sim.end_of_arm.stretch_gripper.set_velocity(0.0)
    time.sleep(0.5)
    end_grip = sim.end_of_arm.stretch_gripper.status.pos
    disp_grip = end_grip - start_grip
    click.secho(
        f"Gripper set_velocity(-0.10 for 1.0s): start={start_grip:.4f}, end={end_grip:.4f}, disp={disp_grip:.4f} [{get_perf_str(sim)}]",
        fg="green",
    )
    assert np.isclose(disp_grip, -0.10, atol=0.03), f"Gripper set_velocity failed: disp={disp_grip:.4f}"

    # Optional camera check
    if test_cameras:
        click.secho("\n--- Verifying Camera Feeds ---", fg="yellow")
        cam_data = sim.pull_camera_data()
        assert cam_data is not None, "Failed to pull camera data from active simulator."
        click.secho(f"Camera feed active at sim_time={cam_data.time:.2f}s [{get_perf_str(sim)}]", fg="green")

    click.secho("\nTESTS PASSED!", fg="green", bold=True)


def test_joint_and_base_movement():
    """Headless movement test suite."""
    click.secho("\n=== Starting Headless Movement Test Suite ===", fg="cyan", bold=True)
    sim = Stretch4MujocoSimulator()
    sim.start(headless=True)
    time.sleep(1.0)
    try:
        run_movement_tests(sim)
    finally:
        sim.stop()


def test_joint_and_base_movement_with_viewer():
    """Movement test suite with passive viewer enabled."""
    click.secho("\n=== Starting Movement Test Suite (With Passive Viewer) ===", fg="cyan", bold=True)
    sim = Stretch4MujocoSimulator()
    sim.start(headless=False)
    time.sleep(1.0)
    try:
        run_movement_tests(sim)
    finally:
        sim.stop()


def test_joint_and_base_movement_with_cameras():
    """Movement test suite with all Stretch 4 cameras running."""
    click.secho("\n=== Starting Movement Test Suite (With All Cameras Enabled) ===", fg="cyan", bold=True)
    all_cameras = Stretch4MujocoSimulator.get_all_cameras()
    sim = Stretch4MujocoSimulator(cameras_to_use=all_cameras)
    sim.start(headless=True)
    time.sleep(1.0)
    try:
        run_movement_tests(sim, test_cameras=True)
    finally:
        sim.stop()


if __name__ == "__main__":
    click.secho("===================================================", fg="blue", bold=True)
    click.secho("RUNNING STRETCH 4 MOVEMENT TEST SUITE VARIATIONS", fg="blue", bold=True)
    click.secho("===================================================", fg="blue", bold=True)

    click.secho("\n[1/3] Headless Test Routine", fg="cyan", bold=True)
    test_joint_and_base_movement()

    click.secho("\n[2/3] Viewer Test Routine", fg="cyan", bold=True)
    test_joint_and_base_movement_with_viewer()

    click.secho("\n[3/3] All Cameras Test Routine", fg="cyan", bold=True)
    test_joint_and_base_movement_with_cameras()

    click.secho("\n===================================================", fg="green", bold=True)
    click.secho("ALL 3 MOVEMENT TEST SUITES PASSED SUCCESSFULLY!", fg="green", bold=True)
    click.secho("===================================================", fg="green", bold=True)

