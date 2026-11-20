"""Drive the car yourself with the arrow keys, to get a feel for the task.

    uv run python scripts/drive.py --track laguna_seca

Up/down: throttle/brake. Left/right: steer. Close the window to quit.
"""

from __future__ import annotations

import argparse

import numpy as np
import pygame

from racer_rl import TRACKS, format_time, make_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track", default="laguna_seca", choices=sorted(TRACKS))
    parser.add_argument("--camera", default="both", choices=["both", "follow", "track"])
    args = parser.parse_args()

    env = make_env(
        args.track,
        render_mode="human",
        camera=args.camera,
        random_start=False,
        duration=float("inf"),
        real_time_rendering=True,
    )
    env.reset()
    laps = 0
    while not env.unwrapped.closed:  # until the window is
        keys = pygame.key.get_pressed()
        pedal = int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])
        steering = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
        _, _, terminated, _, info = env.step(0.7 * np.array([pedal, steering]))
        if info["laps"] > laps:
            laps = info["laps"]
            print(f"lap {laps}: {format_time(info['lap_times'][-1])}")
        if terminated and not env.unwrapped.closed:
            print("off track!")
            env.reset()
            laps = 0


if __name__ == "__main__":
    main()
