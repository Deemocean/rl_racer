"""The time trial: how policies are scored."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from racer_rl.env import Step, make_env
from racer_rl.rendering import format_time


@dataclass(frozen=True)
class TimeTrialResult:
    """The outcome of a time trial. ``print`` it for a summary."""

    completed: bool
    """ Whether all the laps were completed """
    lap_times: list[float]
    """ Time of each completed lap [s] """
    total_time: float | None
    """ The score: total time of the laps [s], or None if they were not completed """
    laps_covered: float
    """ How far the car got, in laps """
    off_track: bool
    """ Whether the trial ended because the car left the track """
    video: str | None
    """ Path of the recorded video, if any """
    saved: str | None = None
    """ Path of the file in which the trial was saved, if any """

    def __str__(self) -> str:
        lines = [
            f"lap {i}: {format_time(lap_time)}"
            for i, lap_time in enumerate(self.lap_times, start=1)
        ]
        if self.completed:
            lines.append(f"total: {format_time(self.total_time)}")
        elif self.off_track:
            lines.append(f"went off track after {self.laps_covered:.2f} laps")
        else:
            lines.append(f"ran out of time after {self.laps_covered:.2f} laps")
        return "\n".join(lines)


class VideoRecorder:
    """
    Record an environment to an mp4 file, one frame per simulation step.

    Frames are encoded as they are rendered, so memory use does not grow with the
    length of the video. The video plays at twice real time.
    """

    def __init__(self, env, path: Path) -> None:
        self.env, self.path = env, path
        self.frames_per_sec = 30  # overwritten by the environment, see below
        self._encoder = None
        # HighwayEnv then calls _capture_frame() for the frames between two decisions
        env.unwrapped.set_record_video_wrapper(self)

    def capture(self) -> None:
        """Render and store the current frame."""
        frame = self.env.render()
        if self._encoder is None:
            height, width = frame.shape[:2]
            self._encoder = imageio_ffmpeg.write_frames(
                str(self.path),
                (width, height),
                fps=self.frames_per_sec,
                quality=6,
                macro_block_size=2,
                output_params=["-preset", "veryfast"],
            )
            self._encoder.send(None)
        self._encoder.send(frame)

    _capture_frame = capture

    def close(self) -> None:
        if self._encoder is not None:
            self.capture()  # the final frame
            self._encoder.close()


def time_trial(
    policy: Callable[[np.ndarray], np.ndarray],
    track: str = "laguna_seca",
    laps: int = 1,
    video: str | None = None,
    render: bool = False,
    save_trial: str | None = None,
    **config,
) -> TimeTrialResult:
    """
    Time a policy over a few laps, from a standing start on the start/finish line.

    :param policy: a function mapping an observation to an action
    :param track: a name from ``racer_rl.track.TRACKS``
    :param laps: number of laps to complete
    :param video: a folder in which to save a video of the trial
    :param render: show the trial in a window, in real time
    :param save_trial: a file (.npz) in which to save the whole trial, to replay it
        or draw it later, as ``scripts/race_video.py`` does. Read it with ``np.load``.
        It holds the ``actions``, one row per decision, and the state of the car
        before each of them and after the last one: ``time`` [s], ``position`` [m],
        ``heading`` [rad], ``speed`` [m/s], ``steering`` (the angle of the front
        wheels [rad]), ``lateral_speed`` [m/s] and ``yaw_rate`` [rad/s]. Also
        ``lap_times`` [s] of the laps that were completed, ``laps`` (how many were
        asked for), ``completed`` and ``off_track`` as in the result, ``track``,
        ``dynamics`` and ``dt`` (the time between two decisions [s]).
    :param config: other entries of the environment config, e.g. ``camera="follow"``
        or ``dynamics="dynamic"``
    """
    env = make_env(
        track,
        render_mode="human" if render else "rgb_array" if video else None,
        random_start=False,
        laps=laps,
        duration=200.0 * laps,
        real_time_rendering=render,
        **config,
    )
    recorder = None
    if video:
        path = Path(video) / f"{track}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        recorder = VideoRecorder(env, path)

    obs, info = env.reset()
    done = False
    states, actions = [_state(env)], []
    while not done:
        if recorder:
            recorder.capture()
        action = np.clip(np.asarray(policy(obs), dtype=float), -1.0, 1.0)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        actions.append(action)
        states.append(_state(env))
    track_length = env.unwrapped.track.length
    if recorder:
        recorder.close()
    details = {
        "track": track,
        "dynamics": env.unwrapped.config["dynamics"],
        "dt": 1 / env.unwrapped.config["policy_frequency"],
    }
    env.close()

    completed = info["laps"] == laps
    result = TimeTrialResult(
        completed=completed,
        lap_times=info["lap_times"],
        total_time=sum(info["lap_times"]) if completed else None,
        laps_covered=float(min(info["distance"] / track_length, laps)),
        off_track=not info["on_track"],
        video=str(recorder.path) if recorder else None,
    )
    if save_trial:
        result = replace(
            result, saved=_save(save_trial, result, states, actions, laps, details)
        )
    return result


def _state(env) -> tuple:
    """The time and the state of the car: enough to carry on from there."""
    car = env.unwrapped.vehicle
    return (
        env.unwrapped.time,
        *car.position,
        car.heading,
        car.speed,
        car.steering_angle,
        getattr(car, "lateral_speed", 0.0),  # the kinematic car has none
        car.yaw_rate,
    )


def _save(
    file: str,
    result: TimeTrialResult,
    states: list[tuple],
    actions: list[np.ndarray],
    laps: int,
    details: dict,
) -> str:
    """
    Save a time trial.

    :param states: the state before each decision and after the last one
    :param actions: the decisions
    """
    path = Path(file).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    states = np.array(states)
    np.savez(
        path,
        actions=np.array(actions),
        time=states[:, 0],
        position=states[:, 1:3],
        heading=states[:, 3],
        speed=states[:, 4],
        steering=states[:, 5],
        lateral_speed=states[:, 6],
        yaw_rate=states[:, 7],
        lap_times=np.array(result.lap_times),
        laps=laps,
        completed=result.completed,
        off_track=result.off_track,
        **details,
    )
    return str(path)


def show_video(path: str, width: int = 900):
    """
    Show a video in a notebook: make it the last line of a cell.

    Every time trial writes its video to the same file. A notebook that is given the
    same file name twice keeps showing the video it has already loaded, so the name is
    followed by the time the file was written, which changes when the video does.
    """
    from IPython.display import Video

    return Video(url=f"{path}?v={Path(path).stat().st_mtime_ns}", width=width)


def rollout(
    policy: Callable[[np.ndarray], np.ndarray],
    reward: Callable[[Step], float],
    track: str = "laguna_seca",
    duration: float = 60.0,
    **config,
) -> dict:
    """
    Drive from a standing start for at most ``duration`` seconds, and add up a reward.

    Useful to check what a reward function makes of a given way of driving.

    :param config: other entries of the environment config, e.g. ``dynamics="dynamic"``
    :return: the sum of the rewards ("return"), the time driven [s] ("time"), the
        distance covered along the track [m] ("distance"), and whether the car left
        the track ("off_track")
    """
    env = make_env(
        track, reward=reward, random_start=False, duration=duration, **config
    )
    obs, info = env.reset()
    total, done = 0.0, False
    while not done:
        obs, step_reward, terminated, truncated, info = env.step(policy(obs))
        total += step_reward
        done = terminated or truncated
    env.close()
    return {
        "return": total,
        "time": info["time"],
        "distance": float(info["distance"]),
        "off_track": not info["on_track"],
    }
