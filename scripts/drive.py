"""Drive the car yourself with the arrow keys, to get a feel for the task.

    uv run python scripts/drive.py --track laguna_seca

Up/down: throttle/brake. Left/right: steer. Close the window to quit.
"""

from __future__ import annotations

import argparse

import gymnasium as gym

import racer_rl  # noqa: F401  (registers racer-v0)
from racer_rl import TRACKS, format_time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track", default="laguna_seca", choices=sorted(TRACKS))
    parser.add_argument("--camera", default="both", choices=["both", "follow", "track"])
    args = parser.parse_args()

    config = {
        "track": args.track,
        "camera": args.camera,
        "random_start": False,
        "duration": float("inf"),
        "manual_control": True,
        "real_time_rendering": True,
    }
    env = gym.make("racer-v0", render_mode="human", config=config)
    env.reset()
    laps = 0
    while True:
        # The action passed to step() is ignored: the keyboard drives the car
        _, _, terminated, _, info = env.step(env.action_space.sample())
        if env.unwrapped.viewer is None:  # the window was closed
            break
        if info["laps"] > laps:
            laps = info["laps"]
            print(f"lap {laps}: {format_time(info['lap_times'][-1])}")
        if terminated:
            print("off track!")
            env.reset()
            laps = 0


if __name__ == "__main__":
    main()
