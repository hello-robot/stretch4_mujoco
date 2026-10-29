from enum import IntEnum

# Six PixArt J3 line sensors ring the base, each projecting a laser line
# down and out onto the floor. The values below mirror `line_sensor_geometry`
# in stretch4_body's robot_params_SE4.py, and the projection in its
# `LineSensorGeometry.get_sensor_points_in_robot_frame()`.

#: Sensors are `sensor_0` ... `sensor_5`, clockwise from robot forward.
NUM_LINE_SENSORS = 6

#: Range bins per sensor (`pixart_report_num`).
LINE_SENSOR_NUM_BINS = 320

#: Width of the fan the bins span (`sensor_horizontal_fov_degrees`).
LINE_SENSOR_FOV_DEG = 103.0

#: Furthest depth the chip reports as a distance. Past this a return comes back
#: as `LineSensorCode.beyond_limit`. Not in the robot params; the floor sits at
#: ~0.23m, so this only matters for cliffs and walls.
LINE_SENSOR_MAX_RANGE_M = 1.0

#: Where the MJCF site each sensor is ray cast from is placed, in its URDF
#: `line_sensor_N_link` frame. The link is the receiver, but the measured
#: plane fans out of the emitter, 24mm above and 3.6mm ahead of it. That puts
#: the site at `emitter_height_above_floor_mm` (100.67) and half the
#: `emitter_pitch_diameter_mm` (404.04) out, so a flat floor reads the same
#: ~0.2296m depth the real sensors are tared to.
LINE_SENSOR_EMITTER_OFFSET = (0.0036, 0.0, 0.024)

#: What the real sensors report, in metres, in place of a distance. The real
#: robot NaNs these out of `ranges` and keeps them in `codes`; so does the sim.
LINE_SENSOR_RANGE_BEYOND_LIMIT_M = 5.09
LINE_SENSOR_RANGE_NO_RETURN_M = 5.11


def line_sensor_site_name(index: int) -> str:
    """The MJCF site `mjcf_generator` emits at sensor `index`'s emitter."""
    return f"line_sensor_{index}"


class LineSensorCode(IntEnum):
    """
    What a line sensor bin carries, matching stretch4_body's `protocol.CODE_*`.
    """

    #: A real distance measurement.
    valid = 0
    #: 5.09 -- a return from past the range limit: the floor dropped away.
    beyond_limit = 1
    #: 5.11 -- nothing reflected back at all.
    no_return = 2
    #: Zero, negative, non-finite, or an unknown code. The sim never emits it.
    other_invalid = 3
