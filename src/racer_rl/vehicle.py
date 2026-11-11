"""The race car, in two versions: a kinematic and a dynamic bicycle model.

Both have the same controls, the same grip limit and the same steering lag. They
differ in how the car responds to the steering:

- ``RaceCar`` (kinematic): the car turns exactly as its front wheels point.
- ``DynamicRaceCar``: the wheels build up tyre forces, and the forces have to push and
  rotate a car that has mass and inertia. The response takes time, and the car can
  slide.
"""

from __future__ import annotations

import math

import numpy as np
from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


class RaceCar(Vehicle):
    """
    A kinematic bicycle model whose total acceleration is limited by tyre grip.

    HighwayEnv's stock ``Vehicle`` can take any corner at any speed. Here, the
    longitudinal and lateral accelerations must stay inside a friction circle of
    radius ``GRIP``: the car cannot turn tighter than ``speed**2 / lateral grip``, and
    whatever grip is spent on braking or accelerating is not available for turning.
    A car that enters a corner too fast therefore understeers and runs wide, which is
    what makes braking points and racing lines matter.

    The front wheels do not jump to the commanded steering angle: they turn towards
    it at ``STEERING_RATE``, so the steering cannot be thrown from side to side.

    The "acceleration" command is a pedal position in [-1, 1]: positive values are a
    fraction of ``MAX_ACCELERATION``, negative values a fraction of ``MAX_BRAKING``.
    """

    LENGTH = 4.88
    """ Overall length [m].  """
    WIDTH = 1.93
    """ Overall width [m] """
    WHEELBASE = 2.97
    """ Distance between the front and rear axles [m]. The centre of gravity is
    midway. With the steering at full lock (0.25 rad), the tightest turn has a radius
    of 11.7 m. """
    MAX_SPEED = 80.0
    """ Top speed [m/s] """
    MIN_SPEED = 0.0
    """ The car cannot reverse [m/s] """
    MAX_ACCELERATION = 10.0
    """ Acceleration at full throttle [m/s2] """
    MAX_BRAKING = 20.0
    """ Deceleration at full brake [m/s2] """
    GRIP = 20.0
    """ Radius of the friction circle, i.e. friction coefficient times g [m/s2] """
    STEERING_RATE = 0.5
    """ How fast the front wheels turn towards the commanded angle [rad/s] """

    def __init__(
        self, road: Road, position: Vector, heading: float = 0, speed: float = 0
    ) -> None:
        super().__init__(road, position, heading, speed)
        self.lateral_acceleration = 0.0
        self.steering_angle = 0.0  # where the front wheels are [rad]
        self.sideslip = 0.0  # angle between where the car points and where it goes
        self.yaw_rate = 0.0  # [rad/s]

    def act(self, action: dict | str | None = None) -> None:
        if action:
            pedal = float(np.clip(action["acceleration"], -1.0, 1.0))
            scale = self.MAX_ACCELERATION if pedal >= 0 else self.MAX_BRAKING
            self.action = {
                "steering": float(action["steering"]),
                "acceleration": pedal * scale,
            }

    def step(self, dt: float) -> None:
        half_length = self.WHEELBASE / 2  # from the centre of gravity to each axle
        acceleration = float(
            np.clip(self.action["acceleration"], -self.GRIP, self.GRIP)
        )
        self._turn_wheels(dt)
        # Slip angle at the centre of gravity requested by the steering angle
        sin_beta = np.sin(np.arctan(0.5 * np.tan(self.steering_angle)))
        # Friction circle: the lateral acceleration speed**2 * sin(beta) / half_length
        # cannot exceed the grip left over by the longitudinal acceleration.
        lateral_grip = np.sqrt(self.GRIP**2 - acceleration**2)
        if self.speed > 1e-3:
            max_sin_beta = min(1.0, lateral_grip * half_length / self.speed**2)
            sin_beta = float(np.clip(sin_beta, -max_sin_beta, max_sin_beta))
        beta = np.arcsin(sin_beta)

        direction = np.array([np.cos(self.heading + beta), np.sin(self.heading + beta)])
        self.position = self.position + self.speed * direction * dt
        self.heading += self.speed * sin_beta / half_length * dt
        self.lateral_acceleration = self.speed**2 * sin_beta / half_length
        self.sideslip = float(beta)
        self.yaw_rate = self.speed * sin_beta / half_length
        self.speed = float(
            np.clip(self.speed + acceleration * dt, self.MIN_SPEED, self.MAX_SPEED)
        )
        self.on_state_update()

    def _turn_wheels(self, dt: float) -> None:
        """Turn the front wheels towards the commanded angle, at a limited rate."""
        travel = self.STEERING_RATE * dt
        self.steering_angle += float(
            np.clip(self.action["steering"] - self.steering_angle, -travel, travel)
        )


class DynamicRaceCar(RaceCar):
    """
    A dynamic bicycle model: the tyres push a car that has mass and yaw inertia.

    Its state has two more variables than the kinematic model: the sideways velocity
    and the yaw rate. Each axle produces a sideways force that grows with its slip
    angle (the angle between where the tyre points and where it is going), then
    levels off and drops a little past the limit of grip. As a result:

    - cornering builds up over a few tenths of a second instead of at once;
    - steering more than the tyres can use loses grip instead of being free;
    - the car can slide, and spin if it is thrown around.

    The limits are those of ``RaceCar``: in a steady corner the car reaches the same
    ``GRIP``, and braking or accelerating takes grip away from cornering. The rear
    tyres have slightly more grip than the front ones, so the car runs wide before
    it spins.

    Below ``BLEND_SPEEDS`` the tyre model is faded out in favour of the kinematic one,
    because slip angles are not defined for a car that is standing still.
    """

    YAW_INERTIA = 1.4
    """ Yaw moment of inertia divided by the mass [m2]: about 1,050 kg m2 for a car of
    750 kg """
    TYRE_STIFFNESS = 16.0
    """ How steeply the tyre force rises with the slip angle [1/rad] """
    TYRE_SHAPE = 1.4
    """ Shape of the tyre curve: above 1, the force drops past its peak """
    REAR_GRIP = 1.1
    """ Grip of the rear tyres relative to the front ones """
    BLEND_SPEEDS = (5.0, 12.0)
    """ Speeds between which the model goes from kinematic to dynamic [m/s] """
    SUBSTEPS = 2
    """ Integration steps per simulation step """

    def __init__(
        self, road: Road, position: Vector, heading: float = 0, speed: float = 0
    ) -> None:
        super().__init__(road, position, heading, speed)
        self.lateral_speed = 0.0  # sideways velocity, in the frame of the car [m/s]

    def step(self, dt: float) -> None:
        self._turn_wheels(dt)
        pedal = float(np.clip(self.action["acceleration"], -self.GRIP, self.GRIP))
        state = (
            float(self.position[0]),
            float(self.position[1]),
            self.heading,
            self.speed,
            self.lateral_speed,
            self.yaw_rate,
        )
        h = dt / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):  # fourth-order Runge-Kutta
            k1 = self._derivative(state, pedal)
            k2 = self._derivative(_add(state, k1, h / 2), pedal)
            k3 = self._derivative(_add(state, k2, h / 2), pedal)
            k4 = self._derivative(_add(state, k3, h), pedal)
            state = tuple(
                x + h / 6 * (a + 2 * b + 2 * c + d)
                for x, a, b, c, d in zip(state, k1, k2, k3, k4)
            )
        x, y, heading, speed, lateral_speed, yaw_rate = state
        self.position = np.array([x, y])
        self.heading = heading
        self.speed = min(max(speed, self.MIN_SPEED), self.MAX_SPEED)
        self.lateral_speed, self.yaw_rate = lateral_speed, yaw_rate
        self.sideslip = math.atan2(lateral_speed, max(self.speed, 1.0))
        self.lateral_acceleration = self._derivative(state, pedal)[4] + (
            yaw_rate * self.speed
        )
        self.on_state_update()

    def _derivative(self, state: tuple, pedal: float) -> tuple:
        """Time derivative of (x, y, heading, speed, sideways speed, yaw rate)."""
        _, _, heading, speed, lateral_speed, yaw_rate = state
        steering = self.steering_angle
        arm = self.WHEELBASE / 2  # from the centre of gravity to each axle [m]
        low, high = self.BLEND_SPEEDS
        dynamic = min(max((speed - low) / (high - low), 0.0), 1.0)  # 0: kinematic

        # Tyre forces, per unit of mass. Each axle carries half of the weight, and
        # what it uses to brake or accelerate is not available sideways.
        front_force = rear_force = 0.0
        if dynamic > 0:
            front_grip = math.sqrt(max((self.GRIP / 2) ** 2 - (pedal / 2) ** 2, 0.0))
            rear_grip = math.sqrt(
                max((self.REAR_GRIP * self.GRIP / 2) ** 2 - (pedal / 2) ** 2, 0.0)
            )
            front_slip = steering - math.atan2(lateral_speed + arm * yaw_rate, speed)
            rear_slip = -math.atan2(lateral_speed - arm * yaw_rate, speed)
            front_force = front_grip * self._tyre(front_slip)
            rear_force = rear_grip * self._tyre(rear_slip)

        # What the kinematic model would do, towards which the car is pulled at low
        # speed
        tan_beta = 0.5 * math.tan(steering)
        relaxation = 0.1  # [s]
        kinematic_lateral = (speed * tan_beta - lateral_speed) / relaxation
        kinematic_yaw = (speed * tan_beta / arm - yaw_rate) / relaxation

        cos, sin = math.cos(steering), math.sin(steering)
        if speed >= self.MAX_SPEED:
            pedal = min(pedal, 0.0)
        acceleration = pedal + dynamic * (yaw_rate * lateral_speed - front_force * sin)
        lateral = dynamic * (front_force * cos + rear_force - yaw_rate * speed)
        yaw = dynamic * arm * (front_force * cos - rear_force) / self.YAW_INERTIA
        return (
            speed * math.cos(heading) - lateral_speed * math.sin(heading),
            speed * math.sin(heading) + lateral_speed * math.cos(heading),
            yaw_rate,
            acceleration,
            lateral + (1 - dynamic) * kinematic_lateral,
            yaw + (1 - dynamic) * kinematic_yaw,
        )

    def _tyre(self, slip: float) -> float:
        """Sideways tyre force as a fraction of the grip, for a slip angle [rad]."""
        return math.sin(self.TYRE_SHAPE * math.atan(self.TYRE_STIFFNESS * slip))


def _add(state: tuple, derivative: tuple, step: float) -> tuple:
    return tuple(x + step * dx for x, dx in zip(state, derivative))
