"""The race car: a dynamic bicycle model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Where each quantity is in the state of a car
X, Y, HEADING, SPEED, LATERAL_SPEED, YAW_RATE, WHEELS = range(7)


class RaceCar:
    """
    A car pushed around by its tyres, with mass and yaw inertia.

    Its state is the array (x, y, heading, speed, sideways speed, yaw rate, wheel
    angle), in metres, radians and seconds, the two speeds being those of the centre
    of gravity in the frame of the car. An action is (pedal, steering), each in
    [-1, 1]: a positive pedal is a share of ``MAX_ACCELERATION``, a negative one a
    share of ``MAX_BRAKING``, and the steering is a share of ``STEERING_LOCK``.

    Each axle makes a sideways force that grows with its slip angle (the angle between
    where the tyre points and where it is going), then levels off and drops a little
    past the limit of grip. As a result:

    - cornering builds up over a few tenths of a second;
    - steering more than the tyres can use loses grip;
    - braking or accelerating takes grip away from cornering, the total staying
      inside a friction circle of radius ``GRIP``;
    - the car can slide, and spin if it is thrown around. The rear tyres have a
      little more grip than the front ones, so it runs wide before it spins.

    The front wheels turn towards the commanded angle at ``STEERING_RATE``, so the
    steering cannot be thrown from side to side.
    """

    LENGTH = 4.88  # [m]
    WIDTH = 1.93  # [m]
    WHEELBASE = 2.97  # [m], the centre of gravity being midway between the axles
    MAX_SPEED = 80.0  # [m/s]
    MIN_SPEED = 0.0  # [m/s]: the car cannot reverse
    MAX_ACCELERATION = 10.0  # at full throttle [m/s2]
    MAX_BRAKING = 20.0  # at full brake [m/s2]
    GRIP = 20.0  # radius of the friction circle: friction coefficient times g [m/s2]
    STEERING_LOCK = 0.25  # largest wheel angle [rad]: the tightest turn is 11.7 m
    STEERING_RATE = 0.5  # of the front wheels [rad/s]
    YAW_INERTIA = 1.4  # over the mass [m2]: about 1,050 kg m2 for a car of 750 kg
    TYRE_STIFFNESS = (
        16.0  # how steeply the tyre force rises with the slip angle [1/rad]
    )
    TYRE_SHAPE = 1.4  # above 1, the tyre force drops past its peak
    REAR_GRIP = 1.1  # of the rear tyres, relative to the front ones
    BLEND_SPEEDS = (5.0, 12.0)  # between which the tyres take over [m/s], see below
    SUBSTEPS = 2  # integration steps per simulation step

    @classmethod
    def state(cls, position, heading, speed=0.0) -> jax.Array:
        position = jnp.asarray(position, dtype=float)
        rest = jnp.asarray([heading, speed, 0.0, 0.0, 0.0], dtype=float)
        return jnp.concatenate([position, rest])

    @classmethod
    def step(cls, car: jax.Array, action: jax.Array, dt: float) -> jax.Array:
        action = jnp.clip(jnp.asarray(action, dtype=float), -1.0, 1.0)
        pedal = cls.acceleration(action[0])
        travel = cls.STEERING_RATE * dt
        target = action[1] * cls.STEERING_LOCK
        steering = car[WHEELS] + jnp.clip(target - car[WHEELS], -travel, travel)

        state, h = car[:WHEELS], dt / cls.SUBSTEPS
        for _ in range(cls.SUBSTEPS):  # fourth-order Runge-Kutta
            k1 = cls.derivative(state, pedal, steering)
            k2 = cls.derivative(state + h * k1 / 2, pedal, steering)
            k3 = cls.derivative(state + h * k2 / 2, pedal, steering)
            k4 = cls.derivative(state + h * k3, pedal, steering)
            state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        speed = jnp.clip(state[SPEED], cls.MIN_SPEED, cls.MAX_SPEED)
        return jnp.concatenate([state.at[SPEED].set(speed), steering[None]])

    @classmethod
    def derivative(cls, state, pedal, steering) -> jax.Array:
        _, _, heading, speed, lateral_speed, yaw_rate = state
        arm = cls.WHEELBASE / 2
        # Slip angles are not defined for a car that is standing still: below
        # BLEND_SPEEDS the car simply turns as its wheels point, and the tyres take
        # over between the two speeds.
        low, high = cls.BLEND_SPEEDS
        tyres = jnp.clip((speed - low) / (high - low), 0.0, 1.0)

        # Forces per unit of mass. Each axle carries half of the weight, and what it
        # uses to brake or accelerate is not available sideways.
        front_grip = jnp.sqrt(jnp.maximum((cls.GRIP / 2) ** 2 - (pedal / 2) ** 2, 0))
        rear_grip = jnp.sqrt(
            jnp.maximum((cls.REAR_GRIP * cls.GRIP / 2) ** 2 - (pedal / 2) ** 2, 0)
        )
        front_slip = steering - jnp.arctan2(lateral_speed + arm * yaw_rate, speed)
        rear_slip = -jnp.arctan2(lateral_speed - arm * yaw_rate, speed)
        front = front_grip * cls.tyre(front_slip)
        rear = rear_grip * cls.tyre(rear_slip)

        rolling = speed * jnp.tan(steering) / 2  # sideways speed without any slip
        relaxation = 0.1  # [s]
        cos, sin = jnp.cos(steering), jnp.sin(steering)
        pedal = jnp.where(speed >= cls.MAX_SPEED, jnp.minimum(pedal, 0.0), pedal)
        return jnp.stack(
            [
                speed * jnp.cos(heading) - lateral_speed * jnp.sin(heading),
                speed * jnp.sin(heading) + lateral_speed * jnp.cos(heading),
                yaw_rate,
                pedal + tyres * (yaw_rate * lateral_speed - front * sin),
                tyres * (front * cos + rear - yaw_rate * speed)
                + (1 - tyres) * (rolling - lateral_speed) / relaxation,
                tyres * arm * (front * cos - rear) / cls.YAW_INERTIA
                + (1 - tyres) * (rolling / arm - yaw_rate) / relaxation,
            ]
        )

    @classmethod
    def tyre(cls, slip):
        """Sideways force of a tyre as a share of its grip, for a slip angle [rad]."""
        return jnp.sin(cls.TYRE_SHAPE * jnp.arctan(cls.TYRE_STIFFNESS * slip))

    @classmethod
    def acceleration(cls, pedal):
        """What a pedal position in [-1, 1] asks for [m/s2]."""
        scale = jnp.where(pedal >= 0, cls.MAX_ACCELERATION, cls.MAX_BRAKING)
        return jnp.clip(pedal * scale, -cls.GRIP, cls.GRIP)

    @classmethod
    def sideslip(cls, car):
        """Angle between where the car points and where it is going [rad]."""
        return jnp.arctan2(car[LATERAL_SPEED], jnp.maximum(car[SPEED], 1.0))

    @classmethod
    def lateral_acceleration(cls, car, action):
        """Cornering acceleration [m/s2] of a car driven with an action."""
        pedal = cls.acceleration(jnp.clip(action[0], -1.0, 1.0))
        sideways = cls.derivative(car[:WHEELS], pedal, car[WHEELS])[LATERAL_SPEED]
        return sideways + car[YAW_RATE] * car[SPEED]
