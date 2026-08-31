import math

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
    # Per-joint trapezoidal motion profiles, mirrored from stretch4_body's
    # `robot/robot_params_SE4.py`. Keys and units are kept identical to the
    # source so the two can be diffed by eye:
    #   prismatic joints use `vel_m`/`accel_m` (m/s, m/s^2)
    #   revolute joints use `vel`/`accel`     (rad/s, rad/s^2)
    # Everything here is in the *joint* frame, not the motor frame -- stretch_body
    # multiplies by `gr` on its way to servo ticks (see
    # `FeetechSMHello.world_rad_to_ticks_per_sec`), so no gear ratio is applied.
    #
    # MuJoCo has no notion of a joint velocity or acceleration limit, so these are
    # enforced by shaping the actuator setpoint before it is written to `ctrl`:
    # `MujocoServer._update_joint_profiles()` for the standalone simulator, and
    # `examples/.../molmospaces/stretch/motion_limits.py` for MolmoSpaces.
    'lift': {
        'motion': {
            'default': {'accel_m': 0.3, 'vel_m': 0.3},
            'fast': {'accel_m': 0.5, 'vel_m': 0.4},
            'max': {'accel_m': 1.0, 'vel_m': 0.5},
            'slow': {'accel_m': 0.2, 'vel_m': 0.15}}},
    # The arm's numbers are for the *total* telescoping extension, which is what
    # both the real joint and the MJCF's "extend" tendon are measured in.
    'arm': {
        'motion': {
            'default': {'accel_m': 0.4, 'vel_m': 0.4},
            'fast': {'accel_m': 0.6, 'vel_m': 0.6},
            'max': {'accel_m': 0.7, 'vel_m': 0.7},
            'slow': {'accel_m': 0.1, 'vel_m': 0.1}}},
    'wrist_yaw': {
        'motion': {
            'default': {'accel': 7.0, 'vel': 7.0},
            'fast': {'accel': 9.0, 'vel': 9.0},
            'max': {'accel': 12.0, 'vel': 12.0},
            'slow': {'accel': 4.0, 'vel': 4.0}}},
    'wrist_pitch': {
        'motion': {
            'default': {'accel': 7.0, 'vel': 7.0},
            'fast': {'accel': 9.0, 'vel': 9.0},
            'max': {'accel': 12.0, 'vel': 12.0},
            'slow': {'accel': 4.0, 'vel': 4.0}}},
    'wrist_roll': {
        'motion': {
            'default': {'accel': 7.0, 'vel': 7.0},
            'fast': {'accel': 9.0, 'vel': 9.0},
            'max': {'accel': 12.0, 'vel': 12.0},
            'slow': {'accel': 4.0, 'vel': 4.0}}},
    # Gripper limits are in the *servo* frame, over the servo's `range_deg`
    # sweep. `gripper_servo_range_deg` records that sweep so the limits can be
    # rescaled into the aperture and URDF finger frames MuJoCo is commanded in;
    # see `get_actuator_motion_limits()`.
    'stretch_gripper': {
        'gripper_servo_range_deg': (-100.0, 300.0),
        'motion': {
            'default': {'accel': 6.0, 'vel': 6.0},
            'fast': {'accel': 6.0, 'vel': 6.0},
            'max': {'accel': 6.0, 'vel': 6.0},
            'slow': {'accel': 4.0, 'vel': 1.0}}},
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
        # Negative because the wheel joint axes follow the URDF, whose axles point
        # opposite the direction H0's kinematics assumes. See mjcf_generator.py step 5.
        "wheel0_polarity": -1,
        "wheel1_polarity": -1,
        "wheel2_polarity": -1,
    }
}


depth_limits = {"gripper": 1, "d435i": 10} # Keep d435i for depth cameras.


# ---------------------------------------------------------------------------
# Joint motion limits
# ---------------------------------------------------------------------------

DEFAULT_MOTION_PROFILE = "default"
"""Which of the `'default' | 'fast' | 'max' | 'slow'` profiles above to enforce.

`'default'` matches what stretch_body uses when a caller passes no explicit
`v_m`/`a_m`, so a sim move looks like the same move on hardware.
"""

# MJCF actuator name -> the `robot_settings_se4` key holding its motion profiles.
# Wheels are absent on purpose: the base runs on its own trapezoidal profiles out
# of `robot_settings_se4['omnibase']['motion']`, see `BaseController`.
_ACTUATOR_MOTION_KEYS = {
    "lift": "lift",
    "arm": "arm",
    "wrist_yaw": "wrist_yaw",
    "wrist_pitch": "wrist_pitch",
    "wrist_roll": "wrist_roll",
    "gripper": "stretch_gripper",
    "gripper_left_finger": "stretch_gripper",
    "gripper_right_finger": "stretch_gripper",
}


def _aperture_open_rad(settings: dict) -> float:
    """The aperture angle, in radians, at which the gripper is fully open.

    The same chord-over-radius the gripper conversions in `mujoco_server.py` and
    stretch_body's `GripperConversion` use, restated here so the scale factors
    below do not have to import either.
    """
    conversion = settings["gripper_conversion"]
    return 2 * math.asin(
        conversion["aperture_open_m"] / (2 * conversion["finger_length_m"])
    )


def _gripper_rate_scale(actuator_name: str, settings: dict) -> float:
    """Servo-frame rate -> the frame `actuator_name` is commanded in.

    Both hops are linear maps, so one factor covers velocity and acceleration:

    * servo -> aperture angle: the servo's full `range_deg` sweep spans the
      aperture angle from closed to fully open (`GripperConversion`).
    * aperture angle -> URDF finger angle: `map_between_ranges` over
      `gripper_conversion`'s `urdf_closed_rad`..`urdf_open_rad`, which is what
      `MujocoServer.aperture_angle_radians_to_urdf_angle_radians` applies.
    """
    gripper = settings["stretch_gripper"]
    servo_min_deg, servo_max_deg = gripper["gripper_servo_range_deg"]
    servo_span_rad = math.radians(servo_max_deg - servo_min_deg)
    scale = _aperture_open_rad(settings) / servo_span_rad

    if actuator_name in ("gripper_left_finger", "gripper_right_finger"):
        conversion = settings["gripper_conversion"]
        scale *= (conversion["urdf_open_rad"] - conversion["urdf_closed_rad"]) / (
            _aperture_open_rad(settings)
        )
    return scale


def get_actuator_motion_limits(
    actuator_name: str,
    profile: str = DEFAULT_MOTION_PROFILE,
    settings: dict | None = None,
) -> tuple[float, float] | None:
    """Max velocity and acceleration for one MuJoCo actuator, in its ctrl units.

    Args:
        actuator_name: an MJCF actuator name, e.g. `"lift"` or
            `"gripper_left_finger"`.
        profile: one of `'default' | 'fast' | 'max' | 'slow'`.
        settings: the robot settings dict to read from. Defaults to
            `robot_settings_se4`; pass `robot_settings` (Stretch 3) and you get
            `None` back, since that model's limits are not recorded here.

    Returns:
        `(max_vel, max_accel)`, or `None` if this actuator has no recorded limits
        and should be left unshaped.
    """
    settings = robot_settings_se4 if settings is None else settings

    key = _ACTUATOR_MOTION_KEYS.get(actuator_name)
    if key is None or key not in settings:
        return None

    motion = settings[key]["motion"].get(profile)
    if motion is None:
        raise KeyError(
            f"No '{profile}' motion profile for '{key}'; "
            f"have {sorted(settings[key]['motion'])}."
        )

    # Prismatic joints are recorded in metres, revolute ones in radians.
    vel = motion["vel_m"] if "vel_m" in motion else motion["vel"]
    accel = motion["accel_m"] if "accel_m" in motion else motion["accel"]

    if key == "stretch_gripper":
        scale = _gripper_rate_scale(actuator_name, settings)
        vel *= scale
        accel *= scale

    return vel, accel


def get_base_motion_limits(
    profile: str = DEFAULT_MOTION_PROFILE, settings: dict | None = None
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Translational and rotational limits for the base.

    Returns:
        `((vel_xy_m, accel_xy_m), (vel_w_r, accel_w_r))`, i.e. m/s, m/s^2 and
        then rad/s, rad/s^2.
    """
    settings = robot_settings_se4 if settings is None else settings
    motion = settings["omnibase"]["motion"][profile]
    return (
        (motion["vel_xy_m"], motion["accel_xy_m"]),
        (motion["vel_w_r"], motion["accel_w_r"]),
    )
