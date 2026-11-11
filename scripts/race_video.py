"""Put several saved time trials on the track together, as one video.

    uv run python scripts/race_video.py laps/ --output videos/race.mp4

The trials are the files written by ``time_trial(..., save_trial="laps/name")``: give
a folder of them, or the files themselves. Each driver is named after their file.
The race is over three laps (``--laps``). Of a trial with more laps than that, the
fastest three in a row are used. A car that did not complete them races until it
leaves the track, and is greyed out from then on. All the cars start their first lap
at the same moment. The left of the picture shows the
whole track; the right follows the car that is furthest ahead, and the table lists
the drivers in order, with the time by which each trails the leader.
"""

from __future__ import annotations

import argparse
import os
from itertools import pairwise
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no window is needed

import imageio_ffmpeg
import numpy as np
import pygame

from racer_rl import RaceCar, Track, format_time
from racer_rl.rendering import (
    SPRITE_SCALE,
    THEMES,
    Type,
    car_icon,
    draw_polyline,
    draw_track,
)

SIZE = (1920, 1080)
MARGIN = 36
TOP_BAND = 132
GUTTER = 24
TABLE_WIDTH = 440
TRACK_VIEW_WIDTH = 620
FOLLOW_SCALE = 4.0  # [px/m]
CAR_SCALE = 6.0  # [px/m]: the cars are drawn larger than life, to stay readable
TRAIL_SECONDS = 3.0
TRAIL_FAINTEST = 0.15  # opacity of the trail of a car that is standing still
CAMERA_LAG = 0.4  # [s] the camera takes to settle on a new leader
HOLD_SECONDS = 3.0  # on the final table
# One colour per driver: the only colours in the picture
COLORS = [
    (0, 112, 243),
    (245, 130, 20),
    (240, 50, 120),
    (60, 190, 110),
    (150, 70, 230),
    (235, 195, 40),
    (40, 195, 215),
    (230, 80, 60),
    (160, 200, 60),
    (220, 120, 220),
]


class Run:
    """Laps in a row from one saved trial, on a clock that starts with the first."""

    def __init__(self, path: Path, track: Track, laps: int) -> None:
        data = np.load(path)
        self.name = path.stem
        self.dynamics = str(data["dynamics"])
        lap_times = data["lap_times"]
        self.finished = len(lap_times) >= laps
        self.reason = "off track" if data["off_track"] else "out of time"
        self.first_lap, self.total = 1, float("inf")
        if self.finished:  # the fastest laps in a row, if the trial has more of them
            totals = [
                lap_times[i : i + laps].sum() for i in range(len(lap_times) - laps + 1)
            ]
            self.first_lap = int(np.argmin(totals)) + 1
            self.total = float(min(totals))
        self.lap_times = lap_times[self.first_lap - 1 :][:laps]
        position, time = data["position"], data["time"]

        # How far the car has gone since the start of the trial, at each instant
        segment = int(track.segment_at(0.0))
        along = np.zeros(len(position))
        for i, point in enumerate(position):
            segment, along[i], _ = track.locate(point, segment)
        half = track.length / 2
        first = along[0] - track.length if along[0] > half else along[0]
        steps = (np.diff(along) + half) % track.length - half
        progress = np.maximum.accumulate(
            first + np.concatenate([[0.0], np.cumsum(steps)])
        )

        # ...and since the start of the first of the laps used
        before = (self.first_lap - 1) * track.length
        self.time = time - np.interp(before, progress, time)
        self.progress = progress - before
        self.position = position
        self.heading = np.unwrap(data["heading"])
        self.speed = data["speed"]
        self.distance = laps * track.length
        # When the car completes the race, if it does, and when it stops in any case
        self.finish = float("inf")
        if self.finished:
            self.finish = float(np.interp(self.distance, self.progress, self.time))
        self.end = min(self.finish, float(self.time[-1]))
        self.covered = float(min(self.progress[-1], self.distance))  # in the end [m]

    def at(self, time: float) -> tuple[np.ndarray, float, float, float]:
        """Position, heading, speed and distance covered at a time on the race clock."""
        return (
            np.array([np.interp(time, self.time, self.position[:, k]) for k in (0, 1)]),
            float(np.interp(time, self.time, self.heading)),
            float(np.interp(time, self.time, self.speed)),
            float(np.interp(time, self.time, self.progress)),
        )

    def time_at(self, progress: float) -> float:
        """When the car had covered that distance."""
        return float(np.interp(progress, self.progress, self.time))

    def out(self, time: float) -> bool:
        """Whether the car has stopped by then without completing the race."""
        return not self.finished and time >= self.end


def load(paths: list[str], laps: int) -> tuple[Track, list[Run]]:
    files: list[Path] = []
    for path in map(Path, paths):
        files += sorted(path.glob("*.npz")) if path.is_dir() else [path]
    if not files:
        raise SystemExit("No trial files found.")
    usable = []
    for file in files:
        if "lap_times" in np.load(file):
            usable.append(file)
        else:
            print(f"left out: {file.stem} (not saved by time_trial(save_trial=...))")
    if not usable:
        raise SystemExit("None of the files is a saved time trial.")
    tracks = {str(np.load(file)["track"]) for file in usable}
    if len(tracks) > 1:
        raise SystemExit(f"The trials are on different tracks: {sorted(tracks)}")
    track = Track.from_name(tracks.pop())
    runs = [Run(file, track, laps) for file in usable]
    for run in runs:
        if run.finished and abs(run.finish - run.total) > 0.05:
            print(f"warning: {run.name}: the positions give {run.finish:.3f} s")
    if len({run.dynamics for run in runs}) > 1:
        print("note: the trials were not all driven with the same car model")
    return track, runs


class Picture:
    """Draws the frames."""

    def __init__(self, track: Track, runs: list[Run], theme: str, title: str) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))
        self.track, self.laps, self.title = track, runs, title
        self.distance = runs[0].distance  # of the race [m]
        self.theme = THEMES[theme]
        self.surface = pygame.Surface(SIZE)
        self.colors = {lap.name: COLORS[i % len(COLORS)] for i, lap in enumerate(runs)}
        self.label = Type("Regular", 22)
        self.name = Type("Medium", 24)
        self.value = Type("SemiBold", 24)
        self.clock = Type("SemiBold", 72)
        self.small = Type("Medium", 18)
        # The trails are the more opaque the faster the car was going
        self.top_speed = max(float(lap.speed.max()) for lap in runs)

        top = TOP_BAND
        height = SIZE[1] - top - MARGIN
        self.track_view = pygame.Rect(MARGIN, top, TRACK_VIEW_WIDTH, height)
        follow_left = self.track_view.right + GUTTER
        follow_width = SIZE[0] - MARGIN - TABLE_WIDTH - GUTTER - follow_left
        self.follow_view = pygame.Rect(follow_left, top, follow_width, height)
        self.table_left = self.follow_view.right + GUTTER

        # The whole track, scaled to fit its view with room for the turn labels
        low, high = track.bounds()
        padding = 30
        self.track_scale = float(
            np.min((np.array(self.track_view.size) - 2 * padding) / (high - low))
        )
        self.track_image, self.track_origin = draw_track(
            track,
            self.track_scale,
            self.theme,
            satellite=True,
            visible=(low - 60, high + 60),
            pixel_ratio=2,  # for edges two pixels thick
        )
        self.track_center = (low + high) / 2
        # The track again, larger, for the view that follows the leader. Its edges
        # are drawn on each frame instead, as smooth lines: one point every metre.
        self.follow_image, self.follow_origin = draw_track(
            track,
            FOLLOW_SCALE,
            self.theme,
            satellite=True,
            visible=(low - 200, high + 200),
            lines=False,
        )
        along = np.arange(0.0, track.length + 1.0)
        self.edges = [
            np.array([track.position(s, side * track.half_width(s)) for s in along])
            for side in (1, -1)
        ]
        self.icons = {
            lap.name: car_icon({**self.theme, "text": self.colors[lap.name]})
            for lap in runs
        }
        self.grey = self.theme["edge"]  # for a car that has stopped
        self.grey_icon = car_icon({**self.theme, "text": self.grey})
        self.camera: np.ndarray | None = None
        self._offset = np.zeros(2)  # of the camera from the car it follows [m]
        self._followed: Run | None = None

    def frame(self, time: float, dt: float) -> np.ndarray:
        theme, surface, length = self.theme, self.surface, self.distance
        surface.fill(theme["canvas"])
        states = {lap.name: lap.at(time) for lap in self.laps}
        running = [lap for lap in self.laps if time < lap.end]
        out = {lap.name for lap in self.laps if lap.out(time)}
        order = sorted(
            self.laps, key=lambda lap: (-min(states[lap.name][3], length), lap.finish)
        )
        leader = max(running, key=lambda lap: states[lap.name][3]) if running else None
        # Cars stay in the picture for a moment after the finish, then leave it. One
        # that stopped before the finish stays where it stopped, in grey.
        shown = [lap for lap in self.laps if time < lap.finish + 0.5]

        # The camera sits on the leader, and glides over when the lead changes
        if leader is not None:
            target = states[leader.name][0]
            if self._followed not in (None, leader):
                self._offset = self.camera - target
            self._followed = leader
            self._offset = self._offset * np.exp(-dt / CAMERA_LAG)
            self.camera = target + self._offset

        self._draw_track_view(states, shown, leader, out)
        self._draw_follow_view(time, states, shown, out)
        self._draw_header(time, leader, order)
        self._draw_legend()
        self._draw_table(time, states, order, out)
        return np.frombuffer(pygame.image.tobytes(surface, "RGB"), np.uint8).reshape(
            SIZE[1], SIZE[0], 3
        )

    def _draw_track_view(
        self, states: dict, shown: list[Run], leader: Run | None, out: set[str]
    ) -> None:
        view, scale = self.track_view, self.track_scale
        corner = np.array(view.center) - (self.track_center - self.track_origin) * scale
        self.surface.set_clip(view)
        self.surface.blit(self.track_image, np.round(corner))
        # The stopped cars underneath, the leader on top
        for lap in sorted(shown, key=lambda lap: (lap.name not in out, lap is leader)):
            x, y = np.round(
                np.array(view.center)
                + (states[lap.name][0] - self.track_center) * scale
            ).astype(int)
            stopped = lap.name in out
            radius = 9 if lap is leader else 5 if stopped else 7
            color = self.grey if stopped else self.colors[lap.name]
            pygame.draw.circle(self.surface, self.theme["canvas"], (x, y), radius + 2)
            pygame.draw.circle(self.surface, color, (x, y), radius)
        self.surface.set_clip(None)

    def _draw_follow_view(
        self, time: float, states: dict, shown: list[Run], out: set[str]
    ) -> None:
        if self.camera is None:
            return
        view, surface = self.follow_view, self.surface

        def to_pixels(points) -> np.ndarray:
            return (
                np.array(view.center)
                + (np.asarray(points) - self.camera) * FOLLOW_SCALE
            )

        surface.set_clip(view)
        surface.blit(self.follow_image, np.round(to_pixels(self.follow_origin)))
        reach = np.hypot(*view.size) / 2 / FOLLOW_SCALE + 5.0  # of the view [m]
        for edge in self.edges:
            near = np.flatnonzero(np.abs(edge - self.camera).max(axis=1) < reach)
            # One line for each stretch of the edge that is in view
            for run in np.split(near, np.flatnonzero(np.diff(near) > 1) + 1):
                if len(run) > 1:
                    points = to_pixels(edge[run[0] : run[-1] + 1])
                    draw_polyline(surface, self.theme["text"], points, 3)
        line = to_pixels([self.edges[0][0], self.edges[1][0]])  # the start/finish line
        pygame.draw.aaline(surface, self.theme["text"], line[0], line[1], 4)
        # Where each car has just been: a trail in its colour, drawn on a clear sheet
        # that is then laid over the picture, so that it can be partly transparent
        sheet = pygame.Surface(view.size, pygame.SRCALPHA)
        for lap in shown:
            if lap.name in out:
                continue
            times = np.linspace(max(time - TRAIL_SECONDS, lap.time[0]), time, 36)
            past = [lap.at(t) for t in times]
            trail = to_pixels([position for position, *_ in past]) - view.topleft
            for (start, end), (_, _, speed, _) in zip(pairwise(trail), past[1:]):
                color = (*self.colors[lap.name], round(255 * self._opacity(speed)))
                pygame.draw.aaline(sheet, color, start, end, 4)
        surface.blit(sheet, view.topleft)
        for lap in shown:
            position, heading, _, _ = states[lap.name]
            sprite = self.grey_icon if lap.name in out else self.icons[lap.name]
            icon = pygame.transform.rotozoom(
                sprite, -np.degrees(heading), CAR_SCALE / SPRITE_SCALE
            )
            center = to_pixels(position)
            surface.blit(icon, icon.get_rect(center=center.round()))
        reach = CAR_SCALE * RaceCar.LENGTH / 2 + 8
        placed: list[pygame.Rect] = []
        for lap in shown:  # names last, over everything else
            center = to_pixels(states[lap.name][0])
            box = pygame.Rect(0, 0, self.small.width(lap.name) + 12, 26)
            box.midleft = (round(center[0] + reach), round(center[1] - reach))
            while box.collidelist(placed) >= 0:  # below a name that is in the way
                box.top = placed[box.collidelist(placed)].bottom + 2
            placed.append(box)
            backing = pygame.Surface(box.size, pygame.SRCALPHA)
            backing.fill((*self.theme["canvas"], 190))
            surface.blit(backing, box)
            color = self.grey if lap.name in out else self.colors[lap.name]
            self.small.draw(surface, lap.name, color, box.left + 6, box.top + 19)
        surface.set_clip(None)

    def _opacity(self, speed: float) -> float:
        """How opaque the trail is where the car was going at that speed."""
        return TRAIL_FAINTEST + (1 - TRAIL_FAINTEST) * min(speed / self.top_speed, 1.0)

    def _draw_legend(self) -> None:
        """What the opacity of a trail means, under the view that has the trails."""
        theme, surface = self.theme, self.surface
        baseline = self.follow_view.bottom + 26
        x = self.label.draw(
            surface, "Trail", theme["text_secondary"], self.follow_view.left, baseline
        )
        x = self.label.draw(surface, "0", theme["text_secondary"], x + 24, baseline)
        bar = pygame.Surface((240, 6), pygame.SRCALPHA)
        for column in range(240):
            opacity = self._opacity(self.top_speed * column / 239)
            bar.fill((*theme["text"], round(255 * opacity)), (column, 0, 1, 6))
        surface.blit(bar, (round(x + 12), round(baseline - 11)))
        top = f"{self.top_speed:.0f} m/s"
        self.label.draw(
            surface, top, theme["text_secondary"], x + 12 + 240 + 12, baseline
        )

    def _draw_header(self, time: float, leader: Run | None, order: list[Run]) -> None:
        theme, surface = self.theme, self.surface
        self.label.draw(surface, self.title, theme["text_secondary"], MARGIN, 46)
        shown = (
            max(time, 0.0) if leader is not None else max(lap.end for lap in self.laps)
        )
        self.clock.draw(surface, format_time(shown, 1), theme["text"], MARGIN - 4, 112)
        x = self.follow_view.left
        if leader is not None:
            self.label.draw(surface, "Following", theme["text_secondary"], x, 46)
            end = self.name.draw(surface, leader.name, self.colors[leader.name], x, 104)
            covered = max(leader.at(time)[3], 0.0) / self.track.length
            laps = round(self.distance / self.track.length)
            text = f"lap {min(int(covered) + 1, laps)} of {laps}"
            self.label.draw(surface, text, theme["text_secondary"], end + 16, 104)
        else:
            text = "Fastest" if order[0].finished else "Furthest"
            self.label.draw(surface, text, theme["text_secondary"], x, 46)
            self.name.draw(surface, order[0].name, self.colors[order[0].name], x, 104)

    def _draw_table(
        self, time: float, states: dict, order: list[Run], out: set[str]
    ) -> None:
        theme, surface = self.theme, self.surface
        left, right = self.table_left, SIZE[0] - MARGIN
        self.label.draw(surface, "Order", theme["text_secondary"], left, 46)
        self.label.draw(
            surface, "Gap or total time", theme["text_secondary"], right, 46, "right"
        )
        row = min(52, (SIZE[1] - TOP_BAND - MARGIN) / max(len(order), 1))
        first = order[0]
        for i, lap in enumerate(order):
            baseline = TOP_BAND + 30 + i * row
            progress = states[lap.name][3]
            self.value.draw(
                surface,
                f"{i + 1}",
                theme["text_secondary"],
                left + 30,
                baseline,
                "right",
            )
            stopped = lap.name in out
            chip = self.grey if stopped else self.colors[lap.name]
            pygame.draw.circle(surface, chip, (left + 54, round(baseline - 8)), 8)
            ink = self.grey if stopped else theme["text"]
            font = self.value
            if time >= lap.finish:
                text, color = format_time(lap.total), theme["text"]
            elif stopped:
                on_lap = int(lap.covered / self.track.length) + 1
                text, color, font = f"{lap.reason}, lap {on_lap}", self.grey, self.small
            elif lap is first:
                text, color = "leader", theme["text_secondary"]
            else:  # how long ago the car in front was where this one is now
                gap = time - first.time_at(max(progress, 0.0))
                text, color = f"+{max(gap, 0.0):.2f}", theme["text_secondary"]
            font.draw(surface, text, color, right, baseline, "right")
            # The name, cut short if it would run into what is on its right
            room = right - font.width(text) - 20 - (left + 76)
            name = lap.name
            while self.name.width(name) > room and len(name) > 1:
                name = name[:-2] + "…"
            self.name.draw(surface, name, ink, left + 76, baseline)
            if i < len(order) - 1 and row >= 40:
                y = round(baseline + row / 2 - 10)
                pygame.draw.line(surface, theme["rule"], (left, y), (right, y), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trials", nargs="+", help="trial files, or folders of them")
    parser.add_argument("--laps", type=int, default=3, help="length of the race")
    parser.add_argument(
        "--output", default="videos/race.mp4", help="the video to write"
    )
    parser.add_argument("--title", help="shown above the clock")
    parser.add_argument("--fps", type=int, default=30, help="frames per second")
    parser.add_argument(
        "--speed", type=float, default=1.0, help="2 plays twice as fast"
    )
    parser.add_argument("--theme", default="dark", choices=sorted(THEMES))
    args = parser.parse_args()

    track, laps = load(args.trials, args.laps)
    print(f"{len(laps)} drivers over {args.laps} laps of {track.length:.0f} m:")
    for run in sorted(laps, key=lambda run: (run.total, -run.covered)):
        if run.finished:
            last = run.first_lap + args.laps - 1
            times = ", ".join(format_time(t) for t in run.lap_times)
            print(
                f"  {format_time(run.total)}  {run.name}"
                f"  (laps {run.first_lap} to {last}: {times})"
            )
        else:
            done = run.covered / track.length
            print(f"  {run.reason} after {done:.2f} laps  {run.name}")

    title = args.title or f"Time trial, {args.laps} laps"
    picture = Picture(track, laps, args.theme, title)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = imageio_ffmpeg.write_frames(
        str(output),
        SIZE,
        fps=args.fps,
        quality=6,
        macro_block_size=2,
        output_params=["-preset", "veryfast"],
    )
    encoder.send(None)
    dt = args.speed / args.fps  # of the race, per frame
    start = -1.0  # the cars wait on the line, or come up to it, for a second
    end = max(lap.end for lap in laps) + HOLD_SECONDS * args.speed
    frames = int(np.ceil((end - start) / dt))
    for k in range(frames):
        encoder.send(picture.frame(start + k * dt, dt))
        if k % (10 * args.fps) == 0:
            print(f"\r{100 * k / frames:3.0f}%", end="", flush=True)
    encoder.close()
    print(f"\r100%\n{output} ({frames / args.fps:.0f} s)")


if __name__ == "__main__":
    main()
