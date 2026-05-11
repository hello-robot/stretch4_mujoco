robot_settings = {
    "wheel_diameter": 0.1016,
    "wheel_separation": 0.3153,
    "gripper_min_max": (-0.376, 0.56),
    "sim_gripper_min_max": (-0.02, 0.04),
}
robot_settings_se4 = {
    "gripper_min_max": (-0.376, 0.56),
    "sim_gripper_min_max": (-0.02, 0.04),
    'gripper_conversion': {
        'finger_length_m': 0.18205,# Straight length from `link_gripper_finger_right.STL`, ignoring the bend in the metal
        'aperture_open_m': 0.177,  # Measured by hand, from fingertip to fingertip
        'aperture_closed_m': 0.0,
        'urdf_open_rad': 0.5,# From URDF, note: this is offset by 0.5, because the URDF maps 0 for open and -0.5 for closed, but that messes up the math
        'urdf_closed_rad': 0,# From URDF, note: this is offset by 0.5, because the URDF maps 0 for open and -0.5 for closed, but that messes up the math
        'urdf_offset': 0.5  # This is offset is subtracted from the final mapping
    },
    'omnibase': {
        'forward_dir': 'calder',
        'gr': 6,
        'motion': {
            'default': {
                'accel_w_r': 2.0,
                'vel_w_r': 2.0,
                'accel_xy_m': 0.25,
                'vel_xy_m': 0.3},
            'fast': {
                'accel_w_r': 3.0,
                'vel_w_r': 3.0,
                'accel_xy_m': 0.4,
                'vel_xy_m': 0.4},
            'max': {
                'accel_w_r': 4.0,
                'vel_w_r': 4.0,
                'accel_xy_m': 0.5,
                'vel_xy_m': 0.6},
            'slow': {
                'accel_w_r': 1.0,
                'vel_w_r': 1.0,
                'accel_xy_m': 0.1,
                'vel_xy_m': 0.1},

            'trajectory_max': {
                'vel_r': 50.0,
                'accel_r': 30.0}
        },
        "wheel_diameter_m": 0.20,
        "base_radius_m": 0.174,
        "wheel0_polarity": 1,
        "wheel1_polarity": 1,
        "wheel2_polarity": 1,
    }
}


depth_limits = {"gripper": 1, "d435i": 10} # Keep d435i for depth cameras.
