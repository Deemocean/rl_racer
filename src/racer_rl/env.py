"""A time-trial racing environment built on HighwayEnv."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import action_factory
from highway_env.road.road import Road

from racer_rl.observation import TrackObservation
from racer_rl.rendering import RaceViewer
from racer_rl.track import Track
from racer_rl.vehicle import DynamicRaceCar, RaceCar

CARS = {"kinematic": RaceCar, "dynamic": DynamicRaceCar}


@dataclass(frozen=True)
class Step:
    """
    What happened during one step of the environment: the input of a reward function.

    "Right" and "left" are from the driver's seat.
    """

    progress: float
    """ Distance covered along the centerline during the step [m]. Negative when the
    car goes backwards around the track. """
    speed: float
    """ Speed at the end of the step [m/s] """
    lateral: float
    """ Offset from the centerline [m], positive to the right """
    heading_error: float
    """ Heading relative to the direction of the track [rad], positive to the right """
    on_track: bool
    """ False on the step in which the car leaves the track, which ends the episode """
    pedal: float
    """ Pedal command applied during the step, in [-1, 1]: throttle if positive """
    steering: float
    """ Steering command applied during the step, in [-1, 1]: right if positive """
    steering_actual: float
    """ Where the front wheels are at the end of the step, in [-1, 1]. They follow the
    command at a limited rate, so this lags behind ``steering``. """
    previous_pedal: float
    """ Pedal command of the previous step (0 on the first step) """
    previous_steering: float
    """ Steering command of the previous step (0 on the first step) """
    longitudinal_acceleration: float
    """ Change of speed during the step, per second [m/s2] """
    lateral_acceleration: float
    """ Cornering acceleration at the end of the step [m/s2], positive to the right """
    sideslip: float
    """ Angle between where the car points and where it is going [rad]. Small when
    the car grips, large when it slides. """
    yaw_rate: float
    """ How fast the car is turning [rad/s], positive to the right """
    curvature: float
    """ Curvature of the centerline at the car [1/m], positive in right-hand corners """
    lap_time: float | None
    """ Time of the lap completed during this step [s], or None """
    time: float
    """ Time since the start of the episode [s] """
    dt: float
    """ Duration of a step [s] """
    max_speed: float
    """ Top speed of the car [m/s] """
    grip: float
    """ Largest total acceleration the tyres allow [m/s2] """
    half_width: float
    """ Half of the track width where the car is [m]: the car is on track while
    |lateral| <= half_width. It varies along ``laguna_seca``. """


class RacingEnv(AbstractEnv):
    """
    Drive a single car around a closed track as fast as possible.

    The agent controls the pedals and the steering wheel. An episode ends when the
    car leaves the track (terminated), or when the time limit or the requested number
    of laps is reached (truncated).

    The reward is not part of the environment: it is computed by the function given
    as the ``reward`` config entry, which maps a ``Step`` to a number. Without one,
    the reward is always zero.

    The ``info`` dict returned by ``step`` contains, among others:

    - ``progress``: arc length along the lap [m];
    - ``distance``: centerline distance covered since the start of the episode [m];
    - ``laps``: number of completed laps;
    - ``lap_times``: the time of each completed lap [s];
    - ``on_track``: whether the centre of the car is within the track limits.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "track": "laguna_seca",  # a name from racer_rl.track.TRACKS
                # A constant width [m]. None: the width of the track itself, which
                # is 12 m for "oval" and "hairpins" and varies along "laguna_seca".
                "track_width": None,
                # How the car responds to the steering, see racer_rl.vehicle.
                # "kinematic": it turns as the wheels point, at once.
                # "dynamic": tyre forces move a car with inertia; it can slide.
                "dynamics": "kinematic",
                "observation": {"type": "Track"},
                "action": {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True,
                    # Pedal position in [-1, 1], see racer_rl.vehicle.RaceCar
                    "acceleration_range": [-1.0, 1.0],
                    "steering_range": [-0.25, 0.25],  # [rad]
                },
                "simulation_frequency": 20,  # [Hz]
                "policy_frequency": 10,  # [Hz]
                "duration": 60.0,  # time limit [s]
                "laps": None,  # stop after this many laps (None: no limit)
                # True: start anywhere on the track with a random speed (training).
                # False: standing start on the start/finish line (time trial).
                "random_start": True,
                "reward": None,  # a function: Step -> float
                # "track": fixed view of the whole track. "follow": follow the car.
                # "both": the two side by side.
                "camera": "both",
                "theme": "dark",  # or "light"
                "satellite": True,  # draw the aerial image of the track, if it has one
                "screen_width": 1200,  # in points
                "screen_height": 800,
                # Pixels per point in rendered frames and videos (a window uses 1)
                "pixel_ratio": 2,
                "centering_position": [0.5, 0.5],
                "scaling": 2.5,  # [points/m] of the "follow" camera
                # True: the "follow" camera turns with the car, which always points up.
                # False: north stays up.
                "heading_up": True,
            }
        )
        return config

    def define_spaces(self) -> None:
        if self.config["observation"]["type"] != "Track":
            # Any HighwayEnv observation works too, e.g. "OccupancyGrid" or "Lidar"
            return super().define_spaces()
        self.observation_type = TrackObservation(self, **self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reset(self) -> None:
        self.track = Track.from_name(self.config["track"], self.config["track_width"])
        self.road = Road(
            network=self.track.to_network(),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        if self.config["random_start"]:
            start = self.np_random.uniform(0, self.track.length)
            lateral = self.np_random.uniform(-0.5, 0.5) * self.track.half_width(start)
            speed = self.np_random.uniform(0, 20)
        else:
            start, lateral, speed = 0.0, 0.0, 0.0
        car = CARS[self.config["dynamics"]](
            self.road,
            self.track.position(start, lateral),
            heading=self.track.heading(start),
            speed=speed,
        )
        self.road.vehicles.append(car)
        self.controlled_vehicles = [car]

        self.segment = int(self.track.segment_at(start))
        self._locate()
        self.distance = 0.0
        self.step_distance = 0.0
        self.lap_times: list[float] = []
        self._lap_start_time = 0.0
        self._speed_before = speed
        self._lap_completed = False
        self._previous_command = (0.0, 0.0)  # (pedal, steering)
        self.last_step: Step | None = None

    def _locate(self) -> None:
        """Project the car onto the centerline."""
        self.segment, self.progress, self.lateral = self.track.locate(
            self.vehicle.position, self.segment
        )
        self.half_width = float(self.track.half_width(self.progress))
        self.on_track = abs(self.lateral) <= self.half_width

    def _simulate(self, action=None) -> None:
        self._speed_before = self.vehicle.speed
        super()._simulate(action)
        previous = self.progress
        self._locate()
        # Signed centerline distance covered during this step
        half_lap = self.track.length / 2
        self.step_distance = (
            self.progress - previous + half_lap
        ) % self.track.length - half_lap

        # Lap timing, interpolated within the step. `self.time` is the time at the end
        # of the step, and has already been incremented by `step()`.
        before = self.distance
        self.distance += self.step_distance
        finish = (len(self.lap_times) + 1) * self.track.length
        self._lap_completed = self.distance >= finish
        if self._lap_completed:
            dt = 1 / self.config["policy_frequency"]
            fraction = (finish - before) / (self.distance - before)
            crossing_time = self.time - dt * (1 - fraction)
            self.lap_times.append(float(crossing_time - self._lap_start_time))
            self._lap_start_time = crossing_time

    def _command(self) -> tuple[float, float]:
        """The (pedal, steering) command the car is executing, each in [-1, 1]."""
        car = self.vehicle
        acceleration = car.action["acceleration"]
        scale = car.MAX_ACCELERATION if acceleration >= 0 else car.MAX_BRAKING
        steering = car.action["steering"] / self.action_type.steering_range[1]
        return acceleration / scale, steering

    def _reward(self, action) -> float:
        car = self.vehicle
        dt = 1 / self.config["policy_frequency"]
        command = self._command()
        self.last_step = Step(
            progress=self.step_distance,
            speed=car.speed,
            lateral=self.lateral,
            heading_error=float(
                utils.wrap_to_pi(car.heading - self.track.heading(self.progress))
            ),
            on_track=self.on_track,
            pedal=command[0],
            steering=command[1],
            steering_actual=car.steering_angle / self.action_type.steering_range[1],
            previous_pedal=self._previous_command[0],
            previous_steering=self._previous_command[1],
            longitudinal_acceleration=(car.speed - self._speed_before) / dt,
            lateral_acceleration=car.lateral_acceleration,
            sideslip=car.sideslip,
            yaw_rate=car.yaw_rate,
            curvature=float(self.track.curvature(self.progress)),
            lap_time=self.lap_times[-1] if self._lap_completed else None,
            time=self.time,
            dt=dt,
            max_speed=car.MAX_SPEED,
            grip=car.GRIP,
            half_width=self.half_width,
        )
        self._previous_command = command
        reward: Callable[[Step], float] | None = self.config["reward"]
        return float(reward(self.last_step)) if reward is not None else 0.0

    def _is_terminated(self) -> bool:
        return not self.on_track

    def _is_truncated(self) -> bool:
        # Reaching the lap count is a truncation, not a termination: the car could
        # keep racing, so the value of the final state should not be treated as zero.
        laps = self.config["laps"]
        laps_done = laps is not None and len(self.lap_times) >= laps
        return laps_done or self.time >= self.config["duration"]

    def _info(self, obs, action=None) -> dict:
        info = super()._info(obs, action)
        info.update(
            {
                "progress": self.progress,
                "distance": self.distance,
                "lateral": self.lateral,
                "on_track": self.on_track,
                "laps": len(self.lap_times),
                "lap_times": list(self.lap_times),
                "time": self.time,
            }
        )
        return info

    def render(self):
        if self.render_mode is not None and self.viewer is None:
            self.viewer = RaceViewer(self)
        return super().render()


def make_env(
    track: str = "laguna_seca",
    reward: Callable[[Step], float] | None = None,
    render_mode: str | None = None,
    **config,
) -> gym.Env:
    """
    Create a racing environment.

    :param track: a name from ``racer_rl.track.TRACKS``
    :param reward: the reward function, mapping a ``Step`` to a number
    :param render_mode: None, "human" (a window) or "rgb_array" (frames)
    :param config: any other entry of ``RacingEnv.default_config()``
    """
    config = {"track": track, "reward": reward, **config}
    return gym.make("racer-v0", render_mode=render_mode, config=config)
