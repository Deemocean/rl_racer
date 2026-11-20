"""A time trial on a closed track.

The race is a few pure JAX functions of a ``State``: ``start`` places a car on a
track, ``transition`` moves it by one decision and ``observe`` is what a policy sees.
They can be compiled, batched and searched. ``RacingEnv`` plays them one decision at
a time, as a Gymnasium environment, and adds what an episode needs: lap times, a
reward, a time limit and a picture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from racer_rl.rendering import VIEW, RaceViewer
from racer_rl.track import Track, wrap_to_pi
from racer_rl.vehicle import HEADING, SPEED, WHEELS, YAW_RATE, RaceCar

DT = 0.1  # time between two decisions [s]
FRAMES = 2  # simulation steps per decision
LOOKAHEAD = (0, 10, 20, 30, 45, 60, 80, 100, 130, 160, 200, 250)
""" Distances ahead of the car at which the curvature is observed [m] """
CURVATURE_SCALE = 50.0
""" Curvatures are multiplied by this length [m] to be of order one """


class State(NamedTuple):
    car: jax.Array  # see RaceCar
    progress: jax.Array  # arc length along the lap [m]
    lateral: jax.Array  # offset from the centerline [m], positive to the right
    distance: jax.Array  # centerline distance covered since the start [m]
    segment: jax.Array  # index of the track segment the car is on
    on_track: jax.Array  # False once the car has left the track


def start(track: Track, progress=0.0, lateral=0.0, speed=0.0) -> State:
    """A car on the track, pointing along it."""
    position, heading = track.position(progress, lateral), track.heading(progress)
    car = RaceCar.state(position, heading, speed)
    return _locate(track, car, track.segment_at(progress))


def transition(track: Track, state: State, action) -> State:
    """
    The state one decision (``DT``) later, for an action (pedal, steering) in [-1, 1].

    A car that has left the track stays where it is.
    """
    car = simulate(state.car, action)[-1]
    moved = _locate(track, car, state.segment)
    half_lap = track.length / 2
    covered = (moved.progress - state.progress + half_lap) % track.length - half_lap
    moved = moved._replace(distance=state.distance + covered)
    return jax.tree.map(
        lambda new, old: jnp.where(state.on_track, new, old), moved, state
    )


def observe(track: Track, state: State) -> jax.Array:
    """
    The car relative to the centerline, and the curvature of the track ahead::

        [ speed / top speed,
          lateral offset / half track width,      (+-1 at the track edges)
          sin(heading error), cos(heading error), (heading relative to the centerline)
          wheel angle / full lock,
          sideslip angle [rad], yaw rate [rad/s], (how the car is moving and turning)
          CURVATURE_SCALE * curvature at each of the LOOKAHEAD distances ]

    It does not depend on where the car is in the world, so the same policy can be
    run on any track.
    """
    car, s = state.car, state.progress
    error = wrap_to_pi(car[HEADING] - track.heading(s))
    here = jnp.stack(
        [
            car[SPEED] / RaceCar.MAX_SPEED,
            state.lateral / track.half_width(s),
            jnp.sin(error),
            jnp.cos(error),
            car[WHEELS] / RaceCar.STEERING_LOCK,
            RaceCar.sideslip(car),
            car[YAW_RATE],
        ]
    )
    ahead = track.curvature(s + jnp.asarray(LOOKAHEAD, dtype=float))
    return jnp.concatenate([here, CURVATURE_SCALE * ahead])


def simulate(car: jax.Array, action) -> jax.Array:
    """The car after each of the simulation steps of one decision."""
    frames = []
    for _ in range(FRAMES):
        car = RaceCar.step(car, action, DT / FRAMES)
        frames.append(car)
    return jnp.stack(frames)


def _locate(track: Track, car: jax.Array, hint) -> State:
    """A car that has not covered any distance yet, found on the track."""
    segment, progress, lateral = track.locate(car[:HEADING], hint)
    on_track = jnp.abs(lateral) <= track.half_width(progress)
    return State(car, progress, lateral, jnp.zeros(()), segment, on_track)


def _telemetry(track: Track, state: State, action) -> jax.Array:
    """What the environment reports about a state besides the car, in one array."""
    car, s = state.car, state.progress
    return jnp.stack(
        [
            s,
            state.lateral,
            state.distance,
            state.on_track.astype(float),
            wrap_to_pi(car[HEADING] - track.heading(s)),
            RaceCar.lateral_acceleration(car, action),
            RaceCar.sideslip(car),
            track.curvature(s),
            track.half_width(s),
        ]
    )


@partial(jax.jit, static_argnums=0)
def _start(track: Track, progress, lateral, speed):
    state = start(track, progress, lateral, speed)
    return state, observe(track, state), _telemetry(track, state, jnp.zeros(2))


@partial(jax.jit, static_argnums=0)
def _advance(track: Track, state: State, action):
    moved = transition(track, state, action)
    frames = simulate(state.car, action)
    return moved, frames, observe(track, moved), _telemetry(track, moved, action)


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


class RacingEnv(gym.Env):
    """
    Drive a single car around a closed track as fast as possible.

    The agent controls the pedals and the steering wheel, ten times per second. An
    episode ends when the car leaves the track (terminated), or when the time limit
    or the requested number of laps is reached (truncated).

    The reward is not part of the environment: it is computed by the ``reward``
    function, which maps a ``Step`` to a number. Without one, the reward is zero.

    ``state`` is the ``State`` of the race, from which ``transition`` can look ahead.
    The ``info`` dict returned by ``step`` contains:

    - ``progress``: arc length along the lap [m];
    - ``distance``: centerline distance covered since the start of the episode [m];
    - ``lateral``: offset from the centerline [m];
    - ``speed``: [m/s];
    - ``laps``: number of completed laps;
    - ``lap_times``: the time of each completed lap [s];
    - ``on_track``: whether the centre of the car is within the track limits;
    - ``time``: time since the start of the episode [s].
    """

    metadata: ClassVar[dict] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 40,
    }

    def __init__(
        self,
        *,
        track: str = "laguna_seca",
        track_width: float | None = None,
        reward: Callable[[Step], float] | None = None,
        duration: float = 60.0,
        laps: int | None = None,
        random_start: bool = True,
        render_mode: str | None = None,
        **view,
    ) -> None:
        """
        :param track: a name from ``racer_rl.track.TRACKS``
        :param track_width: a constant width [m], instead of the width of the track
            itself: 12 m for "oval" and "hairpins", varying along "laguna_seca"
        :param reward: the reward function, mapping a ``Step`` to a number
        :param duration: time limit [s]
        :param laps: stop after this many laps (None: no limit)
        :param random_start: start anywhere on the track with a random speed, for
            training, instead of standing on the start/finish line, for a time trial
        :param render_mode: None, "human" (a window) or "rgb_array" (frames)
        :param view: how the race is drawn, see ``racer_rl.rendering.VIEW``
        """
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"unknown render mode {render_mode!r}")
        if unknown := sorted(view.keys() - VIEW.keys()):
            raise TypeError(f"unknown options {unknown}: the view has {sorted(VIEW)}")
        self.track = Track.from_name(track, track_width)
        self.reward = reward
        self.duration = duration
        self.laps = laps
        self.random_start = random_start
        self.render_mode = render_mode
        self.view = {**VIEW, **view}
        self.viewer: RaceViewer | None = None
        self.closed = False  # closing the window, or close(), ends the picture
        self.recorder = None  # shown the simulation steps between two decisions
        size = 7 + len(LOOKAHEAD)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        progress = lateral = speed = 0.0
        if self.random_start:
            progress = self.np_random.uniform(0, self.track.length)
            half_width = float(self.track.half_width(progress))
            lateral = self.np_random.uniform(-0.5, 0.5) * half_width
            speed = self.np_random.uniform(0, 20)
        self.state, obs, telemetry = _start(self.track, progress, lateral, speed)
        self.time = 0.0
        self.steps = 0  # simulation steps
        self.lap_times: list[float] = []
        self.command = (0.0, 0.0)  # the (pedal, steering) the car is executing
        self.last_step: Step | None = None
        self._lap_start = 0.0
        self._read(telemetry)
        if self.render_mode == "human":
            self.render()
        return np.asarray(obs, dtype=np.float32), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        previous, self.command = self.command, (float(action[0]), float(action[1]))
        before, speed_before = self.distance, float(self.car[SPEED])
        self.time += DT
        self.state, frames, obs, telemetry = _advance(self.track, self.state, action)
        for car in np.asarray(frames)[:-1]:
            self._show(car)
        error, sideslip, curvature, half_width = self._read(telemetry)
        self.steps += 1

        # Lap timing, interpolated within the step, at the end of which is `time`
        finish = (len(self.lap_times) + 1) * self.track.length
        lap_time = None
        if self.distance >= finish:
            fraction = (finish - before) / (self.distance - before)
            crossing = self.time - DT * (1 - fraction)
            lap_time = crossing - self._lap_start
            self.lap_times.append(lap_time)
            self._lap_start = crossing

        speed, yaw_rate, wheels = self.car[[SPEED, YAW_RATE, WHEELS]].tolist()
        self.last_step = Step(
            progress=self.distance - before,
            speed=speed,
            lateral=self.lateral,
            heading_error=error,
            on_track=self.on_track,
            pedal=self.command[0],
            steering=self.command[1],
            steering_actual=wheels / RaceCar.STEERING_LOCK,
            previous_pedal=previous[0],
            previous_steering=previous[1],
            longitudinal_acceleration=(speed - speed_before) / DT,
            lateral_acceleration=self.lateral_acceleration,
            sideslip=sideslip,
            yaw_rate=yaw_rate,
            curvature=curvature,
            lap_time=lap_time,
            time=self.time,
            dt=DT,
            max_speed=RaceCar.MAX_SPEED,
            grip=RaceCar.GRIP,
            half_width=half_width,
        )
        reward = float(self.reward(self.last_step)) if self.reward else 0.0
        terminated = not self.on_track
        # Reaching the lap count is a truncation, not a termination: the car could
        # keep racing, so the value of the final state should not be treated as zero.
        laps_done = self.laps is not None and len(self.lap_times) >= self.laps
        truncated = laps_done or self.time >= self.duration
        if self.render_mode == "human":
            self.render()
        obs = np.asarray(obs, dtype=np.float32)
        return obs, reward, terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None or self.closed:
            return None
        if self.viewer is None:
            self.viewer = RaceViewer(self)
        self.viewer.display()
        if self.render_mode == "rgb_array":
            return self.viewer.image()
        return None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
        self.viewer, self.closed = None, True

    def _read(self, telemetry) -> list[float]:
        """Keep what is reported about the state, and return what only a step uses."""
        self.car = np.asarray(self.state.car)
        progress, lateral, distance, on_track, error, sideways, *rest = (
            telemetry.tolist()
        )
        self.progress, self.lateral, self.distance = progress, lateral, distance
        self.on_track = bool(on_track)
        self.lateral_acceleration = sideways
        return [error, *rest]

    def _show(self, car: np.ndarray) -> None:
        """A simulation step between two decisions, for whoever is watching."""
        self.car = car
        self.steps += 1
        if self.render_mode == "human":
            self.render()
        elif self.recorder is not None:
            self.recorder.capture()

    def _info(self) -> dict:
        return {
            "progress": self.progress,
            "distance": self.distance,
            "lateral": self.lateral,
            "speed": float(self.car[SPEED]),
            "on_track": self.on_track,
            "laps": len(self.lap_times),
            "lap_times": list(self.lap_times),
            "time": self.time,
        }


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
    :param config: any other argument of ``RacingEnv``
    """
    return gym.make(
        "racer-v0", track=track, reward=reward, render_mode=render_mode, **config
    )
