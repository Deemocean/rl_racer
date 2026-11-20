"""A time-trial race in JAX: a track, a car, and a Gymnasium environment to drive it."""

import jax
from gymnasium.envs.registration import register

# Positions are metres on a track a kilometre across: 32 bits would leave them coarse
jax.config.update("jax_enable_x64", True)

from racer_rl.env import (
    RacingEnv,
    State,
    Step,
    make_env,
    observe,
    start,
    transition,
)
from racer_rl.evaluation import TimeTrialResult, rollout, show_video, time_trial
from racer_rl.rendering import format_time
from racer_rl.track import TRACKS, Track
from racer_rl.vehicle import RaceCar

__all__ = [
    "TRACKS",
    "RaceCar",
    "RacingEnv",
    "State",
    "Step",
    "TimeTrialResult",
    "Track",
    "format_time",
    "make_env",
    "observe",
    "rollout",
    "show_video",
    "start",
    "time_trial",
    "transition",
]

register(id="racer-v0", entry_point="racer_rl.env:RacingEnv")
