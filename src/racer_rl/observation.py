"""A compact, track-relative observation for racing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from gymnasium import spaces
from highway_env import utils
from highway_env.envs.common.observation import ObservationType

if TYPE_CHECKING:
    from racer_rl.env import RacingEnv


LOOKAHEAD = (0, 10, 20, 30, 45, 60, 80, 100, 130, 160, 200, 250)
""" Distances ahead of the car at which the curvature is observed [m] """
CURVATURE_SCALE = 50.0
""" Curvatures are multiplied by this length [m] to be of order one """


class TrackObservation(ObservationType):
    """
    The car state relative to the track centerline, and the road curvature ahead.

    The observation is the vector::

        [ speed / top speed,
          lateral offset / half track width,      (+-1 at the track edges)
          sin(heading error), cos(heading error), (heading relative to the centerline)
          steering angle / full lock,             (where the front wheels are)
          sideslip angle [rad], yaw rate [rad/s], (how the car is moving and turning)
          curvature_scale * curvature at each lookahead distance ]

    It does not depend on where the car is in the world, so the same policy can be
    run on any track.
    """

    def __init__(
        self,
        env: RacingEnv,
        lookahead: Sequence[float] = LOOKAHEAD,
        curvature_scale: float = CURVATURE_SCALE,
        **kwargs,
    ) -> None:
        super().__init__(env)
        self.lookahead = np.asarray(lookahead, dtype=float)
        self.curvature_scale = curvature_scale

    def space(self) -> spaces.Box:
        size = 7 + len(self.lookahead)
        return spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)

    def observe(self) -> np.ndarray:
        env, car = self.env, self.observer_vehicle
        heading_error = utils.wrap_to_pi(car.heading - env.track.heading(env.progress))
        curvatures = env.track.curvature(env.progress + self.lookahead)
        return np.concatenate(
            [
                [
                    car.speed / car.MAX_SPEED,
                    env.lateral / env.half_width,
                    np.sin(heading_error),
                    np.cos(heading_error),
                    car.steering_angle / env.action_type.steering_range[1],
                    car.sideslip,
                    car.yaw_rate,
                ],
                self.curvature_scale * curvatures,
            ]
        ).astype(np.float32)
