"""The picture of a race, drawn with pygame.

The design is monochrome and typographic: readouts sit directly on the canvas, on a
shared grid, above and below the track. Colour is reserved for data: the trail behind
the car is coloured by speed, on the scale shown in the bottom right corner.
"""

from __future__ import annotations

from collections import deque
from importlib import resources
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pygame
import pygame.gfxdraw

from racer_rl.track import Track
from racer_rl.vehicle import HEADING, SPEED, WHEELS, RaceCar

if TYPE_CHECKING:
    from racer_rl.env import RacingEnv

# How a race is drawn: the options of ``RacingEnv`` besides those of the race itself
VIEW = {
    "camera": "both",  # "track": the whole track. "follow": the car. "both": the two.
    "theme": "dark",  # or "light"
    "satellite": True,  # draw the aerial image of the track, if it has one
    "screen_width": 1200,  # in points
    "screen_height": 800,
    "pixel_ratio": 2,  # pixels per point in rendered frames and videos (a window: 1)
    "scaling": 2.5,  # [points/m] of the "follow" camera
    "heading_up": True,  # the "follow" camera turns with the car, which points up
    "real_time_rendering": False,  # a window plays no faster than the race
}
FRAME_RATE = 20  # simulation steps per second of race

# Colours, per theme
THEMES = {
    "dark": {
        "canvas": (10, 10, 10),
        "track": (40, 40, 40),
        "edge": (104, 104, 104),
        "centerline": (64, 64, 64),
        "text": (237, 237, 237),
        "text_secondary": (161, 161, 161),
        "rule": (56, 56, 56),
        "error": (255, 110, 110),
    },
    "light": {
        "canvas": (255, 255, 255),
        "track": (234, 234, 234),
        "edge": (160, 160, 160),
        "centerline": (206, 206, 206),
        "text": (23, 23, 23),
        "text_secondary": (102, 102, 102),
        "rule": (220, 220, 220),
        "error": (200, 40, 46),
    },
}
# Trail colour, from standstill to top speed. Readable on both track surfaces.
SPEED_COLORS = np.array(
    [(0, 112, 243), (150, 70, 230), (240, 50, 120), (245, 130, 20)], dtype=float
)

# Layout, in points: one point is ``pixel_ratio`` pixels (see ``RaceViewer.px``).
# The readouts share the left margin and two column edges.
MARGIN = 24
TOP_BAND = 104
BOTTOM_BAND = 116
COLUMNS = 12
TRACK_PADDING = 40  # around the track in the "track" camera, room for turn labels

SUPERSAMPLING = 2
TRAIL_SECONDS = 30.0
MIN_CAR_LENGTH = 22  # so that the car stays visible when zoomed out
GUTTER = 16  # between the two views of the "both" camera
SPRITE_SCALE = 40  # [px/m] at which the car icon is drawn before being scaled down


def speed_color(fraction: float) -> tuple[int, int, int]:
    """Colour for a speed given as a fraction of the top speed."""
    x = float(np.clip(fraction, 0, 1)) * (len(SPEED_COLORS) - 1)
    i = min(int(x), len(SPEED_COLORS) - 2)
    color = SPEED_COLORS[i] + (x - i) * (SPEED_COLORS[i + 1] - SPEED_COLORS[i])
    return tuple(int(c) for c in color)


def draw_polyline(surface: pygame.Surface, color, points, width: int) -> None:
    """A line through points [px], ``width`` pixels thick, with smooth edges."""
    for start, end in pairwise(points):
        pygame.draw.aaline(surface, color, start, end, width)


def format_time(seconds: float, digits: int = 3) -> str:
    """Format a lap time, e.g. 83.456 -> '1:23.456'."""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}:{seconds:0{digits + 3}.{digits}f}"


class Type:
    """Text in Geist, with digits of equal width so that changing numbers do not jitter."""

    def __init__(self, weight: str, size: int) -> None:
        path = resources.files("racer_rl") / "assets" / "fonts" / f"Geist-{weight}.ttf"
        with resources.as_file(path) as file:
            self.font = pygame.font.Font(str(file), size)
        self.digit_width = max(self.font.metrics(d)[0][4] for d in "0123456789")
        self._glyphs: dict[tuple, pygame.Surface] = {}

    def _advance(self, character: str) -> int:
        if character.isdigit():
            return self.digit_width
        return self.font.metrics(character)[0][4]

    def width(self, text: str) -> int:
        return sum(self._advance(c) for c in text)

    def draw(self, surface, text: str, color, x: float, baseline: float, align="left"):
        """Draw text with its baseline at ``baseline``; return its right edge."""
        if align == "right":
            x -= self.width(text)
        elif align == "center":
            x -= self.width(text) / 2
        top = round(baseline - self.font.get_ascent())
        for character in text:
            key = (character, color)
            if key not in self._glyphs:
                self._glyphs[key] = self.font.render(character, True, color)
            glyph = self._glyphs[key]
            advance = self._advance(character)
            # Digits are centred in their cell; other characters keep their spacing
            offset = (advance - self.font.metrics(character)[0][4]) / 2
            surface.blit(glyph, (round(x + offset), top))
            x += advance
        return x


class RaceViewer:
    """Draws a ``RacingEnv``. Created by ``RacingEnv.render``."""

    def __init__(self, env: RacingEnv) -> None:
        self.env = env
        view = self.view = env.view
        self.theme = THEMES[view["theme"]]
        self.offscreen = env.render_mode != "human"
        # Frames for images and videos are drawn with several pixels per point, to be
        # sharp. A window is drawn at one pixel per point, to fit on the screen.
        self.pixel_ratio = int(view["pixel_ratio"]) if self.offscreen else 1
        px = self.px
        self.size = np.array([px(view["screen_width"]), px(view["screen_height"])])
        pygame.display.init()
        pygame.font.init()
        if not self.offscreen:
            pygame.display.set_caption("racer-rl")
            self.screen = pygame.display.set_mode(self.size.tolist())
        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface(self.size)
        area = pygame.Rect(
            0, px(TOP_BAND), self.size[0], self.size[1] - px(TOP_BAND + BOTTOM_BAND)
        )
        # The views to draw, as (where in the window, which camera)
        camera = view["camera"]
        if camera == "both":
            half = (area.w - px(GUTTER)) // 2
            left = pygame.Rect(area.x, area.y, half, area.h)
            right = pygame.Rect(area.right - half, area.y, half, area.h)
            self.views = [(left, "track"), (right, "follow")]
        else:
            self.views = [(area, camera)]
        self.viewport = area  # the view being drawn
        self.label = Type("Regular", px(14))
        self.value = Type("SemiBold", px(20))
        self.value_large = Type("SemiBold", px(40))
        self._backgrounds: dict[tuple, tuple[pygame.Surface, np.ndarray]] = {}
        self._car = car_icon(self.theme)
        self._trail: deque[tuple[np.ndarray, tuple]] = deque(
            maxlen=int(TRAIL_SECONDS * FRAME_RATE)
        )
        self._last_step = -1
        track = env.track
        self._on_image = view["satellite"] and track.satellite is not None
        # Both edges of the track, one point per metre
        s = np.arange(0.0, track.length + 1.0, 1.0)
        self._limits = [
            np.asarray(track.position(s, side * track.half_width(s)))
            for side in (1, -1)
        ]
        # What is named, just outside the track: (name, point of the edge, heading of
        # the track there, side of the track)
        names, at = zip(("Start", 0.0), *track.turns)
        at = np.array(at)
        sides = np.append(1.0, -np.sign(track.curvature(at[1:])))
        edges = np.asarray(track.position(at, sides * track.half_width(at)))
        self._labels = list(zip(names, edges, np.asarray(track.heading(at)), sides))

    def px(self, points: float) -> int:
        """A length in points, in pixels."""
        return round(points * self.pixel_ratio)

    def column(self, index: float) -> float:
        """Left edge [px] of a column of the layout grid."""
        margin = self.px(MARGIN)
        return margin + index * (self.size[0] - 2 * margin) / COLUMNS

    def image(self) -> np.ndarray:
        """The rendered frame as an array of shape (height, width, 3)."""
        width, height = self.canvas.get_size()
        pixels = pygame.image.tobytes(self.canvas, "RGB")
        return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)

    # -- camera ---------------------------------------------------------------------

    def _aim(self, viewport: pygame.Rect, camera: str) -> None:
        """Point the camera for the view about to be drawn."""
        self.viewport = viewport
        low, high = self.env.track.bounds()
        reach = np.array(viewport.size) / 2  # [px] from the centre of the view
        self.rotation = 0.0  # [rad] by which the world is turned, clockwise on screen
        if camera == "track":
            room = np.array(viewport.size) - 2 * self.px(TRACK_PADDING)
            self.scale = float(np.min(room / (high - low)))
            self.center = (low + high) / 2
            self.visible = (
                self.center - reach / self.scale,
                self.center + reach / self.scale,
            )
        else:
            self.scale = float(self.view["scaling"]) * self.pixel_ratio
            self.center = self.env.car[:HEADING]
            if self.view["heading_up"]:
                # Turn the world so that the car points up, and keep enough of it to
                # fill the view whichever way it is turned
                self.rotation = -np.pi / 2 - self.env.car[HEADING]
                reach = np.full(2, np.hypot(*viewport.size) / 2)
            self.visible = low - reach / self.scale, high + reach / self.scale

    def _to_pixels(self, points) -> np.ndarray:
        """World positions [m] to window positions [px]."""
        cos, sin = np.cos(self.rotation), np.sin(self.rotation)
        turn = np.array([[cos, -sin], [sin, cos]])
        offsets = (np.asarray(points) - self.center) @ turn.T
        return offsets * self.scale + self.viewport.center

    def _draw_background(self, surface: pygame.Surface) -> None:
        if self.rotation == 0.0:
            background, origin = self._track_surface()
            surface.blit(background, self._to_pixels(origin).round())
            return
        # Cut out the square around the car that covers the view at any angle, turn
        # it, and centre it on the view. Turning blurs thin lines, so the track
        # limits are left out of the picture and drawn on top afterwards.
        background, origin = self._track_surface(lines=False)
        side = int(np.ceil(np.hypot(*self.viewport.size)))
        patch = pygame.Surface((side, side))
        patch.fill(self.theme["canvas"])
        patch.blit(background, np.round((origin - self.center) * self.scale + side / 2))
        turned = pygame.transform.rotate(patch, -np.degrees(self.rotation))
        surface.blit(turned, turned.get_rect(center=self.viewport.center))
        self._draw_limits(surface)

    # -- drawing --------------------------------------------------------------------

    def display(self) -> None:
        self._update_trail()

        surface = self.canvas
        surface.fill(self.theme["canvas"])
        for viewport, camera in self.views:
            self._aim(viewport, camera)
            surface.set_clip(viewport)
            self._draw_background(surface)
            self._draw_labels(surface)
            self._draw_trail(surface)
            self._draw_car(surface)
        surface.set_clip(None)
        self._draw_timing(surface)
        self._draw_controls(surface)

        if not self.offscreen:
            self.screen.blit(surface, (0, 0))
            if self.view["real_time_rendering"]:
                self.clock.tick(FRAME_RATE)
            pygame.display.flip()
            if any(event.type == pygame.QUIT for event in pygame.event.get()):
                self.env.close()

    def close(self) -> None:
        pygame.quit()

    def _track_surface(self, lines: bool = True) -> tuple[pygame.Surface, np.ndarray]:
        """
        The track drawn at the current zoom level, and the world position of its
        top-left corner. Cached: it never changes.

        :param lines: include the track limits and the start line
        """
        key = (self.scale, lines)
        if key not in self._backgrounds:
            self._backgrounds[key] = draw_track(
                self.env.track,
                self.scale,
                self.theme,
                self.view["satellite"],
                self.visible,
                self.pixel_ratio,
                lines,
            )
        return self._backgrounds[key]

    def _draw_limits(self, surface: pygame.Surface) -> None:
        """The track limits and the start line, drawn as lines in the current view."""
        color = self.theme["text" if self._on_image else "edge"]
        width = max(self.px(1), round(0.25 * self.scale))
        reach = np.hypot(*self.viewport.size) / 2 / self.scale + 5.0  # [m]
        for edge in self._limits:
            near = np.flatnonzero(np.abs(edge - self.center).max(axis=1) < reach)
            if len(near) == 0:
                continue
            # One polyline per stretch of the edge that is in view
            for run in np.split(near, np.flatnonzero(np.diff(near) > 1) + 1):
                points = self._to_pixels(edge[run[0] : run[-1] + 1])
                draw_polyline(surface, color, points, width)
        start = self._to_pixels([self._limits[0][0], self._limits[1][0]])
        line_width = max(self.px(2), round(0.6 * self.scale))
        pygame.draw.aaline(surface, self.theme["text"], start[0], start[1], line_width)

    def _draw_labels(self, surface: pygame.Surface) -> None:
        """Name the start line and the corners, just outside the track."""
        color = self.theme["text" if self._on_image else "text_secondary"]
        for name, edge, heading, side in self._labels:
            edge = self._to_pixels(edge)
            heading = heading + self.rotation  # on screen
            outward = side * np.array([-np.sin(heading), np.cos(heading)])
            # Far enough for the text box to clear the edge of the track
            half_size = np.array([self.label.width(name), self.px(10)]) / 2
            x, y = edge + outward * (self.px(8) + np.abs(outward) @ half_size)
            self.label.draw(surface, name, color, x, y + self.px(5), align="center")

    def _update_trail(self) -> None:
        env = self.env
        if env.steps < self._last_step:  # the environment was reset
            self._trail.clear()
        if env.steps != self._last_step:
            color = speed_color(env.car[SPEED] / RaceCar.MAX_SPEED)
            self._trail.append((env.car[:HEADING].copy(), color))
        self._last_step = env.steps

    def _draw_trail(self, surface: pygame.Surface) -> None:
        if len(self._trail) < 2:
            return
        points = self._to_pixels(np.array([position for position, _ in self._trail]))
        width = max(self.px(3), round(0.5 * self.scale))
        for i in range(len(points) - 1):
            color = self._trail[i + 1][1]
            pygame.draw.line(surface, color, points[i], points[i + 1], width)

    def _draw_car(self, surface: pygame.Surface) -> None:
        position, heading = self.env.car[:HEADING], self.env.car[HEADING]
        # Drawn to scale, or larger when that would make it too small to see, but
        # never wider than most of the track
        scale = max(self.scale, self.px(MIN_CAR_LENGTH) / RaceCar.LENGTH)
        scale = min(scale, 0.7 * self.env.track.min_width / RaceCar.WIDTH * self.scale)
        # The icon points right; headings turn clockwise on screen, rotozoom the other way
        icon = pygame.transform.rotozoom(
            self._car, -np.degrees(heading + self.rotation), scale / SPRITE_SCALE
        )
        surface.blit(icon, icon.get_rect(center=self._to_pixels(position).round()))

    def _draw_timing(self, surface: pygame.Surface) -> None:
        """Lap timer, lap times and speed, above the track."""
        env, theme = self.env, self.theme
        lap_times = env.lap_times
        px = self.px
        label_baseline, value_baseline = px(36), px(82)

        x = self.column(0)
        end = self.label.draw(
            surface,
            f"Lap {len(lap_times) + 1}",
            theme["text_secondary"],
            x,
            label_baseline,
        )
        if not env.on_track:
            self.label.draw(
                surface, "Off track", theme["error"], end + px(12), label_baseline
            )
        current = format_time(env.time - sum(lap_times), digits=1)
        self.value_large.draw(surface, current, theme["text"], x, value_baseline)

        placeholder = "–:––.–––"
        peers = [
            ("Last lap", format_time(lap_times[-1]) if lap_times else placeholder),
            ("Best lap", format_time(min(lap_times)) if lap_times else placeholder),
        ]
        for column, (name, text) in zip((4, 6.5), peers):
            x = self.column(column)
            self.label.draw(surface, name, theme["text_secondary"], x, label_baseline)
            color = theme["text"] if lap_times else theme["text_secondary"]
            self.value.draw(surface, text, color, x, value_baseline)

        x = self.column(9)
        self.label.draw(surface, "Speed", theme["text_secondary"], x, label_baseline)
        end = self.value_large.draw(
            surface,
            f"{env.car[SPEED]:4.1f}".replace(" ", "0"),
            theme["text"],
            x,
            value_baseline,
        )
        self.label.draw(
            surface, "m/s", theme["text_secondary"], end + px(6), value_baseline
        )

    def _draw_controls(self, surface: pygame.Surface) -> None:
        """Grip use, pedals and steering, and the speed scale, below the track."""
        env, car, theme = self.env, RaceCar, self.theme
        pedal, _ = env.command
        px = self.px
        top = self.size[1] - px(BOTTOM_BAND)
        label_baseline = top + px(26)

        # Friction circle: the dot is the car's acceleration, the ring its grip limit
        radius = px(32)
        cx, cy = round(self.column(0)) + radius, top + px(16) + radius
        for i in range(self.pixel_ratio):  # a ring one point thick
            pygame.gfxdraw.aacircle(
                surface, cx, cy, radius - i, theme["text_secondary"]
            )
        hair = radius - px(1)
        pygame.draw.line(
            surface, theme["rule"], (cx - hair, cy), (cx + hair, cy), px(1)
        )
        pygame.draw.line(
            surface, theme["rule"], (cx, cy - hair), (cx, cy + hair), px(1)
        )
        forwards = pedal * (car.MAX_ACCELERATION if pedal >= 0 else car.MAX_BRAKING)
        acceleration = np.array([env.lateral_acceleration, -forwards])
        dot = np.round(np.array([cx, cy]) + acceleration / car.GRIP * radius).astype(
            int
        )
        pygame.gfxdraw.filled_circle(surface, dot[0], dot[1], px(4), theme["text"])
        pygame.gfxdraw.aacircle(surface, dot[0], dot[1], px(4), theme["text"])
        x = cx + radius + px(16)
        self.label.draw(
            surface, "Grip used", theme["text_secondary"], x, label_baseline
        )
        used = min(1.0, float(np.linalg.norm(acceleration)) / car.GRIP)
        self.value.draw(
            surface, f"{100 * used:.0f}%", theme["text"], x, label_baseline + px(30)
        )
        self.label.draw(
            surface,
            f"of {car.GRIP:.0f} m/s²",
            theme["text_secondary"],
            x,
            label_baseline + px(52),
        )

        # Pedals and steering: one label lane, one bar lane and one value lane
        steering = env.car[WHEELS] / car.STEERING_LOCK
        side = "" if abs(steering) < 0.005 else " right" if steering > 0 else " left"
        rows = [
            ("Throttle", max(pedal, 0.0), f"{100 * max(pedal, 0):.0f}%"),
            ("Brake", max(-pedal, 0.0), f"{100 * max(-pedal, 0):.0f}%"),
            ("Steering", steering, f"{100 * abs(steering):.0f}%{side}"),
        ]
        label_x, bar_x, bar_end = self.column(4), self.column(5.5), self.column(7.5)
        for i, (name, amount, text) in enumerate(rows):
            baseline = label_baseline + px(26) * i
            self.label.draw(surface, name, theme["text_secondary"], label_x, baseline)
            bar = pygame.Rect(bar_x, baseline - px(9), bar_end - bar_x, px(6))
            pygame.draw.rect(surface, theme["rule"], bar)
            if name == "Steering":  # from the middle of the bar, towards either side
                fill = pygame.Rect(bar.centerx, bar.y, round(amount * bar.w / 2), bar.h)
                fill.normalize()
                pygame.draw.rect(surface, theme["text"], fill)
                pygame.draw.line(
                    surface,
                    theme["text_secondary"],
                    (bar.centerx, bar.y - px(3)),
                    (bar.centerx, bar.bottom + px(2)),
                    px(1),
                )
            else:
                pygame.draw.rect(
                    surface, theme["text"], (bar.x, bar.y, round(amount * bar.w), bar.h)
                )
            self.label.draw(surface, text, theme["text"], bar_end + px(10), baseline)

        # Speed scale of the trail
        x, end = round(self.column(9)), self.size[0] - px(MARGIN)
        self.label.draw(
            surface, "Trail colour", theme["text_secondary"], x, label_baseline
        )
        for pixel in range(x, end):
            color = speed_color((pixel - x) / (end - x - 1))
            pygame.draw.line(
                surface, color, (pixel, top + px(43)), (pixel, top + px(48))
            )
        baseline = label_baseline + px(46)
        self.label.draw(surface, "0", theme["text_secondary"], x, baseline)
        self.label.draw(
            surface,
            f"{car.MAX_SPEED:.0f} m/s",
            theme["text_secondary"],
            end,
            baseline,
            "right",
        )


def draw_track(
    track: Track,
    scale: float,
    theme: dict,
    satellite: bool = True,
    visible: tuple[np.ndarray, np.ndarray] | None = None,
    pixel_ratio: int = 1,
    lines: bool = True,
) -> tuple[pygame.Surface, np.ndarray]:
    """
    Draw a track on a new surface.

    :param track: the track
    :param scale: zoom level [px/m]
    :param theme: colours, an entry of ``THEMES``
    :param satellite: draw the aerial image of the track behind it, if it has one
    :param visible: the part of the world that can come into view, as its (top-left,
        bottom-right) corners [m]: the aerial image is cropped to it
    :param pixel_ratio: pixels per point, for line widths
    :param lines: draw the track limits, the centerline and the start line
    :return: the surface, and the world position [m] of its top-left corner. Without
        an aerial image, the surface is transparent off the track.
    """
    satellite = satellite and track.satellite is not None
    low, high = track.bounds()
    border = 2.0  # [m]
    origin = low - border
    # Drawn larger, then scaled down to smooth the edges, unless already zoomed in
    supersampling = SUPERSAMPLING if scale < 3 else 1
    zoom = scale * supersampling
    size = np.ceil((high + border - origin) * zoom).astype(int)
    # Transparent around the track, so that the surface blends into the canvas
    surface = pygame.Surface(size, pygame.SRCALPHA)
    surface.fill((*theme["edge"], 0))

    def to_pixels(points) -> np.ndarray:
        return (np.asarray(points) - origin) * zoom

    def ribbon(margin: float, color) -> None:
        """Fill the track, and a margin [m] beyond each of its edges."""
        s = np.arange(0.0, track.length + 0.5, 0.5)
        reach = track.half_width(s) + margin
        left = to_pixels(track.position(s, reach))
        right = to_pixels(track.position(s, -reach))
        for i in range(len(s) - 1):
            pygame.draw.polygon(
                surface, color, [left[i], left[i + 1], right[i + 1], right[i]]
            )

    edge_line = max(0.25, 1.2 * pixel_ratio / scale)  # [m], at least about a point
    if lines:
        ribbon(edge_line, theme["text" if satellite else "edge"])
    # Over the aerial image, only tint the track to let the image show through
    ribbon(0.0, (*theme["track"], 96) if satellite else theme["track"])

    if lines and not satellite:  # dashed centerline
        starts = np.arange(0.0, track.length - 3.0, 9.0)
        for dash in to_pixels(track.position(starts[:, None] + np.linspace(0, 3, 4))):
            pygame.draw.lines(
                surface, theme["centerline"], False, dash, max(1, round(0.25 * zoom))
            )

    if lines:  # start/finish line
        half_width = float(track.half_width(0.0))
        line = to_pixels(track.position(0.0, np.array([half_width, -half_width])))
        pygame.draw.line(
            surface,
            theme["text"],
            line[0],
            line[1],
            max(2 * pixel_ratio * supersampling, round(0.6 * zoom)),
        )

    smooth = surface
    if supersampling > 1:
        smooth = pygame.transform.smoothscale(surface, tuple(size // supersampling))
    if not satellite:
        return smooth, origin
    image, image_origin = satellite_image(track, scale, theme, visible)
    image.blit(smooth, np.round((origin - image_origin) * scale))
    return image, image_origin


def satellite_image(
    track: Track,
    scale: float,
    theme: dict,
    visible: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[pygame.Surface, np.ndarray]:
    """
    The aerial image of a track, in grey and toned towards the canvas colour.

    The picture stays in the background that way: the trail keeps the only colours.

    :param visible: crop the image to this part of the world, given by its (top-left,
        bottom-right) corners [m]
    :return: the surface, and the world position [m] of its top-left corner
    """
    name, low, high = track.satellite
    path = resources.files("racer_rl") / "assets" / "tracks" / name
    with resources.as_file(path) as file:
        image = pygame.image.load(str(file))
    if visible is not None:
        pixels_per_metre = np.array(image.get_size()) / (high - low)
        corner = np.floor((visible[0] - low) * pixels_per_metre).astype(int)
        opposite = np.ceil((visible[1] - low) * pixels_per_metre).astype(int)
        crop = pygame.Rect(*corner, *(opposite - corner)).clip(image.get_rect())
        image = image.subsurface(crop).copy()
        high = low + np.array(crop.bottomright) / pixels_per_metre
        low = low + np.array(crop.topleft) / pixels_per_metre
    pixels = pygame.surfarray.pixels3d(image)
    grey = pixels @ np.array([0.299, 0.587, 0.114])
    canvas = np.mean(theme["canvas"])
    toned = canvas + (grey - canvas) * 0.5  # halve the contrast with the canvas
    pixels[:] = toned[..., None].astype(np.uint8)
    del pixels  # unlock the surface
    size = np.round((high - low) * scale).astype(int)
    return pygame.transform.smoothscale(image, tuple(size)), low


COCKPIT = (-0.1, 0.9, 0.44)  # of the car icon: its middle, length and width [m]


def car_parts() -> list[list[tuple[float, float]]]:
    """The race car seen from above, as polygons [m]. It points right, about (0, 0)."""

    def box(x0: float, x1: float, y0: float, y1: float) -> list:
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    half_body = [(-2.2, 0.32), (-1.2, 0.42), (-1.05, 0.62), (0.2, 0.62), (0.55, 0.3)]
    half_body += [(1.8, 0.13), (2.3, 0.1)]
    body = half_body + [(x, -y) for x, y in reversed(half_body)]
    tyre = 0.42  # [m] tyre width
    return [
        body,
        box(-2.5, -2.2, -0.85, 0.85),  # rear wing
        box(2.2, 2.5, -0.95, 0.95),  # front wing
        box(-1.6, -1.45, -0.7, 0.7),  # rear axle
        box(1.28, 1.42, -0.7, 0.7),  # front axle
        box(-2.0, -1.05, -1.0, -1.0 + tyre),  # rear wheels
        box(-2.0, -1.05, 1.0 - tyre, 1.0),
        box(0.95, 1.75, -1.0, -1.0 + tyre),  # front wheels
        box(0.95, 1.75, 1.0 - tyre, 1.0),
    ]


def car_icon(theme: dict) -> pygame.Surface:
    """
    The race car as a picture: one flat colour, with a thin outline in the canvas
    colour to keep it readable over the trail and the aerial image.
    """
    length, width = RaceCar.LENGTH, RaceCar.WIDTH
    margin = 0.2  # [m] around the car, for the outline
    size = np.round(
        np.array([length, width]) * SPRITE_SCALE + 2 * margin * SPRITE_SCALE
    )
    surface = pygame.Surface(size.astype(int), pygame.SRCALPHA)

    def to_pixels(points) -> np.ndarray:
        return (np.asarray(points, dtype=float) + [length / 2, width / 2] + margin) * (
            SPRITE_SCALE
        )

    parts = car_parts()
    outline = round(0.12 * SPRITE_SCALE)
    for part in parts:  # the outline: every part again, thicker, underneath
        pygame.draw.polygon(surface, theme["canvas"], to_pixels(part))
        pygame.draw.polygon(surface, theme["canvas"], to_pixels(part), 2 * outline)
    for part in parts:
        pygame.draw.polygon(surface, theme["text"], to_pixels(part))
    middle, long, wide = COCKPIT
    cockpit = pygame.Rect(0, 0, long * SPRITE_SCALE, wide * SPRITE_SCALE)
    cockpit.center = to_pixels((middle, 0.0))
    pygame.draw.ellipse(surface, theme["canvas"], cockpit)
    return surface
