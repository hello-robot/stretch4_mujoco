import math
import time

import numpy as np


class TrapezoidalProfile:
    """
    Generates a trapezoidal velocity profile for a single joint.
    Mimics the behavior of a stepper motor driver.

    Two modes, both bounded by the same `max_vel`/`max_accel`:

    * velocity: `set_target_velocity()`, the position is whatever integrating
      the ramped velocity produces. Used for the wheels and for jogging.
    * position: `set_target_position()`, the profile accelerates towards the
      goal and decelerates into it. Used to shape the position actuators'
      setpoints, which MuJoCo would otherwise step to instantly.
    """

    VELOCITY = "velocity"
    POSITION = "position"

    def __init__(self, max_vel: float = 10.0, max_accel: float = 10.0, dt: float = 0.002):
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.dt = dt

        self.current_pos = 0.0
        self.current_vel = 0.0
        self.target_vel = 0.0
        self.target_pos = 0.0
        self.mode = self.VELOCITY
        self.last_update_time = time.perf_counter()

    def set_target_velocity(self, target_vel: float):
        """
        Sets the target velocity for the profile.
        """
        self.mode = self.VELOCITY
        self.target_vel = max(-self.max_vel, min(self.max_vel, target_vel))

    def set_target_position(self, target_pos: float):
        """
        Sets the position the profile should drive to, and switches to position mode.
        """
        self.mode = self.POSITION
        self.target_pos = target_pos

    def update(self, dt: float | None = None) -> float:
        """
        Updates the profile state and returns the new position.
        Call this every simulation step.
        """
        if dt is None:
            now = time.perf_counter()
            dt = now - self.last_update_time
            self.last_update_time = now

        if self.mode == self.POSITION:
            self._update_position_target()

        # Calculate velocity error
        vel_error = self.target_vel - self.current_vel

        # Determine acceleration to apply
        if abs(vel_error) < 1e-6:
            accel = 0.0
        else:
            # Ramp velocity towards target
            accel_direction = 1.0 if vel_error > 0 else -1.0
            accel = accel_direction * self.max_accel

            # Don't overshoot target velocity in this step
            if abs(accel * dt) > abs(vel_error):
                accel = vel_error / dt

        # Update velocity
        self.current_vel += accel * dt

        # Clamp velocity (safety)
        self.current_vel = max(-self.max_vel, min(self.max_vel, self.current_vel))

        # Snap to zero if target is zero and we are close (prevents drift)
        if self.target_vel == 0.0 and abs(self.current_vel) < 1e-4:
            self.current_vel = 0.0

        # Update position
        previous_pos = self.current_pos
        self.current_pos += self.current_vel * dt

        if self.mode == self.POSITION:
            # Land exactly on the goal rather than dithering around it: with a
            # discrete step the deceleration ramp below can only get within
            # O(max_accel * dt^2) of it on its own.
            crossed = (self.target_pos - previous_pos) * (self.target_pos - self.current_pos) < 0
            if crossed or abs(self.target_pos - self.current_pos) < 1e-9:
                self.current_pos = self.target_pos
                self.current_vel = 0.0

        return self.current_pos

    def _update_position_target(self) -> None:
        """Pick the velocity that heads for `target_pos` and still stops on it.

        `sqrt(2 * a * distance)` is the fastest we can be going and still bleed
        off all of it within the distance left, so taking the smaller of that and
        `max_vel` gives the cruise-then-brake shape of a trapezoid without having
        to plan the whole move up front -- which matters because the goal can be
        moved on any step.
        """
        error = self.target_pos - self.current_pos
        if self.max_accel <= 0.0:
            braking_vel = self.max_vel
        else:
            braking_vel = math.sqrt(2.0 * self.max_accel * abs(error))
        self.target_vel = math.copysign(min(self.max_vel, braking_vel), error)

    @property
    def is_settled(self) -> bool:
        """Whether the profile has nothing left to do.

        A joint whose profile is still ramping has not finished moving even when
        it has not visibly moved yet -- a trapezoid starts from rest, so for the
        first tick of a move the position barely changes. Callers that decide
        "motion is over" from position stability need this, or they conclude a
        move is done before it has begun.
        """
        if self.mode == self.POSITION:
            return self.current_vel == 0.0 and self.current_pos == self.target_pos
        return self.current_vel == 0.0 and self.target_vel == 0.0

    def set_position(self, pos: float):
        """
        Hard reset of the position (e.g. for initialization).
        """
        self.current_pos = pos
        self.target_pos = pos


class TrapezoidalSetpointLimiter:
    """Shapes a *stream* of position setpoints so they respect vel/accel limits.

    A vector wrapper over `TrapezoidalProfile` in position mode, for callers that
    reissue an absolute target every control step -- MolmoSpaces' controllers --
    rather than handing over one discrete goal at a time.

    It decelerates into the goal, and that matters more than it sounds. The
    obvious cheaper shaper -- cap the speed, cap how fast it may speed up, let it
    stop dead on arrival -- is wrong here, because Stretch's wrist actuators are
    `gainprm=[20, 0, 0] biasprm=[0, -20, 0]`: kp only, *no velocity feedback*. The
    only thing damping the joint is `damping="2"` on the joint itself, so a
    setpoint that stops dead leaves the wrist to carry its own momentum through
    the target. Measured on the compiled model, driving wrist yaw 1.5 rad:

        step ctrl (unshaped)   peak 11.9 rad/s, no overshoot,      settles 0.50s
        stop-dead rate limit   peak  3.9 rad/s, 0.99 rad (57deg!), settles 2.99s
        this, decelerating     peak  2.8 rad/s, no overshoot,      settles 1.13s

    The middle row is a joint visibly swinging past where it was sent and ringing
    back -- slower than before by the numbers, and much worse to watch.

    Args:
        max_vel: per-joint velocity limit.
        max_accel: per-joint acceleration limit.
        angular: per-joint flag marking a continuous revolute DOF. Targets for
            those are unwrapped onto the revolution the setpoint is currently on
            before the error is taken, so a target the far side of +-pi is chased
            the short way round. Only ever the base yaw here -- the wrist joints
            are range-limited hinges whose travel exceeds pi, so wrapping them
            would fold a legitimate target back on itself.
    """

    def __init__(self, max_vel, max_accel, angular=None) -> None:
        max_vel = np.atleast_1d(np.asarray(max_vel, dtype=float))
        max_accel = np.atleast_1d(np.asarray(max_accel, dtype=float))
        if max_vel.shape != max_accel.shape:
            raise ValueError(
                f"max_vel {max_vel.shape} and max_accel {max_accel.shape} must match."
            )

        self._profiles = [
            TrapezoidalProfile(max_vel=v, max_accel=a)
            for v, a in zip(max_vel, max_accel, strict=True)
        ]
        if angular is None:
            self._angular = np.zeros(max_vel.shape, dtype=bool)
        else:
            self._angular = np.atleast_1d(np.asarray(angular, dtype=bool))
            if self._angular.shape != max_vel.shape:
                raise ValueError(
                    f"angular {self._angular.shape} must match max_vel {max_vel.shape}."
                )

    def __len__(self) -> int:
        return len(self._profiles)

    @property
    def position(self) -> np.ndarray:
        """The setpoint as it currently stands."""
        return np.array([p.current_pos for p in self._profiles])

    def reset(self, position) -> None:
        """Jump the setpoint to `position` and stop, with no ramp."""
        for profile, value in zip(
            self._profiles, np.atleast_1d(np.asarray(position, dtype=float)), strict=True
        ):
            profile.set_position(float(value))
            profile.current_vel = 0.0

    def step(self, target, dt: float) -> np.ndarray:
        """Advance the setpoint one control interval towards `target`."""
        target = np.atleast_1d(np.asarray(target, dtype=float))
        shaped = np.empty(len(self._profiles))
        for i, profile in enumerate(self._profiles):
            goal = float(target[i])
            if self._angular[i]:
                goal = profile.current_pos + _wrap_to_pi(goal - profile.current_pos)
            profile.set_target_position(goal)
            shaped[i] = profile.update(dt)
        return shaped


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
