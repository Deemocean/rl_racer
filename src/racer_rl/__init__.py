"""Time-trial racing environments for reinforcement learning, built on HighwayEnv."""

from gymnasium.envs.registration import register

from racer_rl.env import RacingEnv, Step, make_env
from racer_rl.evaluation import TimeTrialResult, rollout, show_video, time_trial
from racer_rl.rendering import format_time
from racer_rl.track import TRACKS, Track
from racer_rl.vehicle import RaceCar

__all__ = [
    "TRACKS",
    "RaceCar",
    "RacingEnv",
    "Step",
    "TimeTrialResult",
    "Track",
    "format_time",
    "make_env",
    "rollout",
    "show_video",
    "time_trial",
]

register(id="racer-v0", entry_point="racer_rl.env:RacingEnv")
