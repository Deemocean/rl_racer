"""Closed race tracks built from HighwayEnv lane primitives.

A track is a closed, tangent-continuous centerline made of straights and arcs, and a
width, which may vary along the lap. There are two ways to describe one:

- a closed polygon whose corners are rounded with circular arcs (fillets), and a
  constant width: a new track takes a few vertices;
- its two edges, as files of points. The centerline then runs midway between them.

Conventions (inherited from HighwayEnv):
- world coordinates are in metres, and the y axis points *down* on screen;
- ``s`` is the arc length along the centerline, starting at the start/finish line;
- the lateral offset is positive on the side towards which heading increases, which
  is also the inside of a positive-curvature corner. Positive steering turns the car
  towards positive lateral offsets.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import cache
from importlib import resources

import numpy as np
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.road import RoadNetwork

DEFAULT_WIDTH = 12.0
""" Width of a track described by a polygon [m] """

# name -> description of the track. Either:
# - "vertices" [m] and "radii" [m] (one value, or one per vertex) of a rounded polygon.
#   The start/finish line is in the middle of the edge from the last vertex to the
#   first. "scale" multiplies all the dimensions.
# - "boundaries": two files in assets/tracks, each with a "lat,lon" column pair per
#   point of one edge of the track, going around the lap, point i of one file facing
#   point i of the other. "origin" is the latitude and longitude of the point (0, 0),
#   and the start/finish line is where the centerline passes closest to "start" [m].
# Optional entries: "turns" names corners for display (polygon: vertex index -> label;
# boundaries: label -> a point near the corner [m]), and "satellite" is an aerial image
# to draw behind the track: a file in assets/tracks, and the (x0, y0, x1, y1) it
# covers [m].
TRACKS: dict[str, dict] = {
    # Two straights and two wide 180-degree ends: the easy one. 1 km long.
    "oval": {
        "vertices": [(160, 0), (160, 70), (0, 70), (0, 0)],
        "radii": 35.0,
        "scale": 2.5,
    },
    # WeatherTech Raceway Laguna Seca at full size, flattened (the real track climbs
    # 55 m and drops through the Corkscrew). Both edges of the track are given, a point
    # about every half metre, so the width is the real one: from 7.4 m to 17.6 m.
    # Turn 7, a slight kink before the Corkscrew, is not labelled.
    # The aerial image is from the USGS National Agriculture Imagery Program (public
    # domain), at 0.6 m per pixel. North is up.
    "laguna_seca": {
        "boundaries": ("ls_wall_inner.csv", "ls_wall_outer.csv"),
        "origin": (36.584, -121.753),
        "start": (-257.1, -365.2),
        "satellite": {
            "image": "laguna_seca.jpg",
            "bounds": (-855.7, -710.0, 785.7, 650.0),
        },
        "turns": {
            "1": (-351, -234),
            "2": (-393, 168),
            "3": (-337, -100),
            "4": (-84, -42),
            "5": (-109, 437),
            "6": (275, 400),
            "8": (343, -99),
            "8A": (308, -122),
            "9": (271, -312),
            "10": (9, -277),
            "11": (-136, -508),
        },
    },
    # A different layout, useful to test how a policy generalises to an unseen track.
    "hairpins": {
        "vertices": [
            (180, 0),
            (180, 50),
            (60, 50),
            (60, 100),
            (200, 100),
            (230, 160),
            (-20, 160),
            (0, 0),
        ],
        "radii": [16, 16, 14, 14, 30, 30, 45, 30],
        "scale": 2.5,
    },
}


class Track:
    """A closed track, with fast centerline queries. See ``TRACKS`` for the arguments."""

    def __init__(
        self,
        vertices: Sequence[Sequence[float]] | None = None,
        radii: float | Sequence[float] | None = None,
        width: float | None = None,
        scale: float = 1.0,
        turns: dict | None = None,
        satellite: dict | None = None,
        boundaries: tuple[str, str] | None = None,
        origin: tuple[float, float] | None = None,
        start: Sequence[float] | None = None,
    ) -> None:
        """
        :param width: a constant width [m]. By default a polygon is ``DEFAULT_WIDTH``
            wide, and a track given by its boundaries is as wide as they are apart.
        """
        edges = corners = None
        self.origin = origin  # latitude and longitude of the point (0, 0), if known
        if boundaries is None:
            width = DEFAULT_WIDTH if width is None else width
            vertices = scale * np.asarray(vertices, dtype=float)
            radii = scale * np.broadcast_to(np.asarray(radii, float), len(vertices))
            self.lanes, corners = _rounded_polygon(vertices, radii, width)
        else:
            edges = [_load_boundary(name, origin) for name in boundaries]
            self.lanes, edges = _between(*edges, np.asarray(start, dtype=float))
        lengths = np.array([lane.length for lane in self.lanes])
        self.starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
        self.length = float(lengths.sum())
        self.curvatures = np.array([_curvature(lane) for lane in self.lanes])
        # Half the width of the track, every metre along the centerline
        self._grid = np.arange(0.0, self.length, 1.0)
        if width is not None:
            self.half_widths = np.full(len(self._grid), width / 2)
        else:
            points = np.array([self.position(s) for s in self._grid])
            apart = [
                _distance(points, edge, self._grid / self.length) for edge in edges
            ]
            self.half_widths = (apart[0] + apart[1]) / 2
        self.min_width = 2 * float(self.half_widths.min())
        self._bounds: tuple[np.ndarray, np.ndarray] | None = None
        # Aerial image, as (file name, top-left corner [m], bottom-right corner [m])
        self.satellite = None
        if satellite is not None:
            bounds = scale * np.asarray(satellite["bounds"], dtype=float)
            self.satellite = (satellite["image"], bounds[:2], bounds[2:])
        # Named corners, as (label, arc length of the corner)
        if corners is not None:
            self.turns = [
                (label, float(self.starts[corners[i]] + lengths[corners[i]] / 2))
                for i, label in (turns or {}).items()
            ]
        else:
            self.turns = [(label, self._apex(p)) for label, p in (turns or {}).items()]

    @classmethod
    def from_name(cls, name: str, width: float | None = None) -> Track:
        """
        A track of ``TRACKS``. It is built once and then shared: do not modify it.

        :param width: a constant width [m], instead of the track's own
        """
        if name not in TRACKS:
            raise ValueError(f"Unknown track {name!r}. Available: {sorted(TRACKS)}")
        return _from_name(name, width)

    def to_network(self) -> RoadNetwork:
        """
        A HighwayEnv road network. HighwayEnv wants one and looks up the lane nearest
        to the car at every step, but nothing here uses the answer, so the network
        holds a single segment rather than the hundreds a track can have.
        """
        net = RoadNetwork()
        net.add_lane("0", "1", self.lanes[0])
        return net

    def metres(self, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        """World positions [m] of points given by their latitude and longitude."""
        if self.origin is None:
            raise ValueError("This track is not placed on the map: it has no origin.")
        return _metres(np.asarray(latitude), np.asarray(longitude), self.origin)

    def half_width(self, s: float | np.ndarray) -> float | np.ndarray:
        """Half the width of the track [m] at the arc length(s) ``s``."""
        return np.interp(s, self._grid, self.half_widths, period=self.length)

    def _apex(self, point: Sequence[float]) -> float:
        """Arc length of the sharpest part of the corner near a point."""
        point = np.asarray(point, dtype=float)
        nearest = min(
            range(len(self.lanes)), key=lambda i: self.lanes[i].distance(point)
        )
        around = self.starts[nearest] + np.arange(-40.0, 41.0)
        return float(around[np.argmax(np.abs(self.curvature(around)))] % self.length)

    def segment_at(self, s: float | np.ndarray) -> np.ndarray:
        """Index of the segment containing the arc length(s) ``s``."""
        return np.searchsorted(self.starts, np.mod(s, self.length), side="right") - 1

    def position(self, s: float, lateral: float = 0.0) -> np.ndarray:
        i = int(self.segment_at(s))
        return self.lanes[i].position(s % self.length - self.starts[i], lateral)

    def heading(self, s: float) -> float:
        i = int(self.segment_at(s))
        return self.lanes[i].heading_at(s % self.length - self.starts[i])

    def curvature(self, s: float | np.ndarray) -> np.ndarray:
        """Signed centerline curvature [1/m] at the arc length(s) ``s``."""
        return self.curvatures[self.segment_at(s)]

    def locate(self, position: np.ndarray, hint: int) -> tuple[int, float, float]:
        """
        Project a world position onto the centerline.

        Only the segments around ``hint`` are considered (the one behind it, and those
        within 25 m ahead), so that a car cannot "teleport" to another part of the
        track by cutting across the infield.

        :param position: a world position [m]
        :param hint: index of the segment the car was last on
        :return: (segment index, arc length s along the lap [m], lateral offset [m])
        """
        n = len(self.lanes)
        candidates = [(hint - 1) % n, hint]
        ahead = 0.0
        while ahead < 25.0 and len(candidates) < n:
            candidates.append((candidates[-1] + 1) % n)
            ahead += self.lanes[candidates[-1]].length
        i = min(candidates, key=lambda j: self.lanes[j].distance(position))
        local_s, lateral = self.lanes[i].local_coordinates(position)
        s = self.starts[i] + float(np.clip(local_s, 0.0, self.lanes[i].length))
        return i, s % self.length, float(lateral)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box (min corner, max corner) of the track surface."""
        if self._bounds is None:
            s = np.arange(0.0, self.length, 2.0)
            points = np.array([self.position(si) for si in s])
            half_width = self.half_widths.max()
            self._bounds = (
                points.min(axis=0) - half_width,
                points.max(axis=0) + half_width,
            )
        return self._bounds


@cache
def _from_name(name: str, width: float | None) -> Track:
    return Track(**TRACKS[name], width=width)


def _load_boundary(name: str, origin: tuple[float, float]) -> np.ndarray:
    """One edge of a track, from a file of latitudes and longitudes, in metres."""
    path = resources.files("racer_rl") / "assets" / "tracks" / name
    with resources.as_file(path) as file:
        latitude, longitude = np.loadtxt(
            file, delimiter=",", skiprows=1, usecols=(0, 1)
        ).T
    return _metres(latitude, longitude, origin)


def _metres(
    latitude: np.ndarray, longitude: np.ndarray, origin: tuple[float, float]
) -> np.ndarray:
    """
    Positions east and south of an origin (y points down), from latitudes and
    longitudes. Good to a few centimetres over a few kilometres.
    """
    x = (longitude - origin[1]) * 111_320 * math.cos(math.radians(origin[0]))
    y = -(latitude - origin[0]) * 110_540
    return np.stack([x, y], axis=-1)


def _between(
    inner: np.ndarray, outer: np.ndarray, start: np.ndarray
) -> tuple[list[AbstractLane], list[np.ndarray]]:
    """
    The centerline midway between the two edges of a track, as a chain of arcs.

    :param inner: points of one edge, going around the lap
    :param outer: points of the other edge, point i facing point i of ``inner``
    :param start: the lap starts where the centerline passes closest to this point
    :return: the lanes, and the two edges starting from the start/finish line
    """
    if np.allclose(inner[0], inner[-1]):  # closed loops: drop the repeated point
        inner, outer = inner[:-1], outer[:-1]
    first = np.argmin(np.linalg.norm((inner + outer) / 2 - start, axis=1))
    inner, outer = np.roll(inner, -first, axis=0), np.roll(outer, -first, axis=0)

    # The middle line, resampled evenly and smoothed a little: the arcs take their
    # directions from it, and noise in the points would become noise in the curvature
    spacing = 0.5  # [m]
    middle = (inner + outer) / 2
    closed = np.vstack([middle, middle[:1]])
    along = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    )
    count = round(along[-1] / spacing)
    even = np.arange(count) * along[-1] / count
    path = np.column_stack([np.interp(even, along, closed[:, k]) for k in (0, 1)])
    path = _smooth(path, 1.0 / spacing)
    directions = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
    directions /= np.linalg.norm(directions, axis=1)[:, None]

    # A pair of arcs (a biarc) from each node to the next: they pass through the nodes
    # in the direction of the middle line there, so the chain has no kink
    every = round(8.0 / spacing)  # a node every 8 m
    nodes = np.arange(0, count - every // 2, every)
    lanes: list[AbstractLane] = []
    for i, j in zip(nodes, np.roll(nodes, -1)):
        lanes += _biarc(path[i], directions[i], path[j], directions[j])
    return lanes, [inner, outer]


def _smooth(points: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of an evenly sampled closed curve; ``sigma`` in samples."""
    reach = math.ceil(4 * sigma)
    weights = np.exp(-0.5 * (np.arange(-reach, reach + 1) / sigma) ** 2)
    weights /= weights.sum()
    wrapped = np.vstack([points[-reach:], points, points[:reach]])
    return np.column_stack(
        [np.convolve(wrapped[:, k], weights, mode="valid") for k in (0, 1)]
    )


def _biarc(
    start: np.ndarray, direction: np.ndarray, end: np.ndarray, end_direction: np.ndarray
) -> list[AbstractLane]:
    """Two arcs that leave ``start`` and reach ``end`` in the given unit directions."""
    chord = end - start
    both = direction + end_direction
    spread = 2 * (1 - direction @ end_direction)
    if spread > 1e-12:
        reach = (
            -(chord @ both) + math.sqrt((chord @ both) ** 2 + spread * (chord @ chord))
        ) / spread
    else:  # same direction at both ends
        reach = (chord @ chord) / (4 * (chord @ end_direction))
    joint = (start + reach * direction + end - reach * end_direction) / 2
    first, joint_direction = _arc(start, direction, joint)
    second, _ = _arc(joint, joint_direction, end)
    return [first, second]


def _arc(
    start: np.ndarray, direction: np.ndarray, end: np.ndarray
) -> tuple[AbstractLane, np.ndarray]:
    """
    The arc that leaves ``start`` in a unit direction and passes through ``end``.

    :return: the lane, and the direction in which it arrives at ``end``
    """
    lines = (LineType.CONTINUOUS, LineType.CONTINUOUS)
    chord = end - start
    turn = 2 * math.atan2(
        direction[0] * chord[1] - direction[1] * chord[0], direction @ chord
    )
    if abs(turn) < 1e-7:
        lane = StraightLane(start, end, width=DEFAULT_WIDTH, line_types=lines)
        return lane, direction
    radius = float(np.linalg.norm(chord)) / (2 * abs(math.sin(turn / 2)))
    sign = math.copysign(1.0, turn)
    center = start + sign * radius * np.array([-direction[1], direction[0]])
    phase = math.atan2(direction[1], direction[0]) - sign * math.pi / 2
    lane = CircularLane(
        center,
        radius,
        phase,
        phase + turn,
        clockwise=sign > 0,
        width=DEFAULT_WIDTH,
        line_types=lines,
    )
    cos, sin = math.cos(turn), math.sin(turn)
    arrival = np.array(
        [
            cos * direction[0] - sin * direction[1],
            sin * direction[0] + cos * direction[1],
        ]
    )
    return lane, arrival


def _distance(points: np.ndarray, edge: np.ndarray, lap: np.ndarray) -> np.ndarray:
    """
    Distance from points of the centerline to an edge of the track.

    :param points: the points
    :param edge: the points of the edge, starting from the start/finish line
    :param lap: how far around the lap each point is, from 0 to 1. Only the part of
        the edge about as far around is searched, which is all that can be nearest.
    """
    reach = max(50, len(edge) // 50)
    middle = np.round(lap * len(edge)).astype(int)[:, None]
    index = (middle + np.arange(-reach, reach)) % len(edge)
    a, b = edge[index], edge[(index + 1) % len(edge)]
    ab = b - a
    to_points = points[:, None, :] - a
    along = np.clip((to_points * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
    return np.linalg.norm(to_points - along[..., None] * ab, axis=-1).min(axis=1)


def _curvature(lane: AbstractLane) -> float:
    if isinstance(lane, CircularLane):
        return lane.direction / lane.radius
    return 0.0


def _rounded_polygon(
    vertices: np.ndarray, radii: np.ndarray, width: float
) -> tuple[list[AbstractLane], list[int]]:
    """
    Build the centerline of a closed polygon whose corners are rounded with arcs.

    The start/finish line is the midpoint of the edge joining the last vertex to the
    first one, and the track runs through the vertices in the given order.

    :return: the lanes, and the index of the lane rounding each vertex
    """
    n = len(vertices)
    lines = (LineType.CONTINUOUS, LineType.CONTINUOUS)

    # For each corner: the points where the arc leaves the incoming edge (entry) and
    # joins the outgoing edge (exit), and the arc itself.
    entries, exits, arcs = [], [], []
    for i in range(n):
        corner = vertices[i]
        d_in = _unit(corner - vertices[i - 1])
        d_out = _unit(vertices[(i + 1) % n] - corner)
        turn = np.arctan2(d_in[0] * d_out[1] - d_in[1] * d_out[0], d_in @ d_out)
        if abs(turn) < 1e-6:
            raise ValueError(f"Vertex {i} is not a corner (edges are collinear).")
        sign = np.sign(turn)
        tangent = radii[i] * np.tan(abs(turn) / 2)
        entry, exit_ = corner - tangent * d_in, corner + tangent * d_out
        center = entry + sign * radii[i] * np.array([-d_in[1], d_in[0]])
        start_phase = np.arctan2(d_in[1], d_in[0]) - sign * np.pi / 2
        arcs.append(
            CircularLane(
                center,
                radii[i],
                start_phase,
                start_phase + turn,
                clockwise=bool(sign > 0),
                width=width,
                line_types=lines,
            )
        )
        entries.append(entry)
        exits.append(exit_)

    def straight(start: np.ndarray, end: np.ndarray, edge: int) -> list[AbstractLane]:
        direction = _unit(vertices[edge] - vertices[edge - 1])
        length = (end - start) @ direction
        if length < -1e-6:
            raise ValueError(
                f"The corners at both ends of the edge leading to vertex {edge} "
                "overlap: reduce their radii or move the vertices apart."
            )
        if length < 1e-6:
            return []
        return [StraightLane(start, end, width=width, line_types=lines)]

    # Split the last-to-first edge at its midpoint to place the start/finish line.
    start_line = (exits[-1] + entries[0]) / 2
    if (entries[0] - exits[-1]) @ _unit(vertices[0] - vertices[-1]) < 1.0:
        raise ValueError("The edge holding the start/finish line needs a straight.")
    lanes: list[AbstractLane] = straight(start_line, entries[0], 0)
    corners = []
    for i in range(n):
        if i > 0:
            lanes += straight(exits[i - 1], entries[i], i)
        corners.append(len(lanes))
        lanes.append(arcs[i])
    lanes += straight(exits[-1], start_line, 0)
    return lanes, corners


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)
