import time

class TrapezoidalProfile:
    """
    Generates a trapezoidal velocity profile for a single joint.
    Mimics the behavior of a stepper motor driver.
    """
    def __init__(self, max_vel: float = 10.0, max_accel: float = 10.0, dt: float = 0.002):
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.dt = dt

        self.current_pos = 0.0
        self.current_vel = 0.0
        self.target_vel = 0.0
        self.last_update_time = time.perf_counter()

    def set_target_velocity(self, target_vel: float):
        """
        Sets the target velocity for the profile.
        """
        self.target_vel = max(-self.max_vel, min(self.max_vel, target_vel))

    def update(self, dt: float | None = None) -> float:
        """
        Updates the profile state and returns the new position.
        Call this every simulation step.
        """
        if dt is None:
            now = time.perf_counter()
            dt = now - self.last_update_time
            self.last_update_time = now

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

        # Update position
        self.current_pos += self.current_vel * dt

        return self.current_pos

    def set_position(self, pos: float):
        """
        Hard reset of the position (e.g. for initialization).
        """
        self.current_pos = pos
