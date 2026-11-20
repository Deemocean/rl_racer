"""Closed race tracks: a centerline of straights and arcs, and a width.

The centerline is closed and has no kink. There are two ways to describe a track:

- a closed polygon whose corners are rounded with circular arcs, and a constant width:
  a new track takes a few vertices;
- its two edges, as files of points. The centerline then runs midway between them.

Conventions:
- world coordinates are in metres, and the y axis points *down* on screen;
- ``s`` is the arc length along the centerline, starting at the start/finish line;
- the lateral offset is positive on the side towards which heading increases, which
  is also the inside of a positive-curvature corner. Positive steering turns the car
  towards positive lateral offsets.
"""

from __future__ import annotations

import math
from functools import cache
from importlib import resources

import jax
import jax.numpy as jnp
import numpy as np

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
    # The same lap through the pit lane: it leaves the track before Turn 11, passes
    # the garages along the main straight, and rejoins after Turn 2.
    "laguna_seca_pit": {
        "boundaries": ("ls_wall_pit_inner.csv", "ls_wall_pit_outer.csv"),
        "origin": (36.584, -121.753),
        "start": (-257.1, -365.2),
        "satellite": {
            "image": "laguna_seca.jpg",
            "bounds": (-855.7, -710.0, 785.7, 650.0),
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
    """
    A closed track, see ``TRACKS`` for the arguments.

    Its queries are JAX functions of arrays: they take one value or many, and can be
    compiled into a larger computation.
    """

    def __init__(
        self,
        *,
        vertices=None,
        radii=None,
        width: float | None = None,
        scale: float = 1.0,
        turns: dict | None = None,
        satellite: dict | None = None,
        boundaries: tuple[str, str] | None = None,
        origin: tuple[float, float] | None = None,
        start=None,
    ) -> None:
        if (vertices is None) == (boundaries is None):
            raise ValueError("a track is given by its vertices or by its boundaries")
        edges = corners = None
        if boundaries is None:
            # A polygon is DEFAULT_WIDTH wide, unless told otherwise
            width = DEFAULT_WIDTH if width is None else width
            vertices = scale * np.asarray(vertices, dtype=float)
            radii = scale * np.broadcast_to(np.asarray(radii, float), len(vertices))
            segments, corners = _rounded_polygon(vertices, radii)
        else:
            edges = [_load_boundary(name, origin) for name in boundaries]
            segments, edges = _between(*edges, np.asarray(start, dtype=float))

        # Each segment: where it starts, its heading there, its curvature, its length
        points, headings, curvatures, lengths = np.split(
            np.asarray(segments), [2, 3, 4], 1
        )
        headings, curvatures, lengths = headings[:, 0], curvatures[:, 0], lengths[:, 0]
        self.origin = origin  # latitude and longitude of the point (0, 0), if known
        self.segments = len(lengths)
        ends = np.cumsum(lengths)
        self.length = float(ends[-1])
        self.points = jnp.asarray(points)
        self.headings = jnp.asarray(headings)
        self.curvatures = jnp.asarray(curvatures)
        self.lengths = jnp.asarray(lengths)
        self.starts = jnp.asarray(np.concatenate([[0.0], ends[:-1]]))
        # What an arc turns around, at which signed radius and from which angle
        radii = 1 / np.where(curvatures == 0, 1.0, curvatures)
        self._tangents = jnp.asarray(np.stack([np.cos(headings), np.sin(headings)], 1))
        self._normals = jnp.asarray(np.stack([-np.sin(headings), np.cos(headings)], 1))
        self._radii = jnp.asarray(radii)
        self._centers = self.points + self._radii[:, None] * self._normals
        self._phases = jnp.asarray(headings - np.sign(radii) * np.pi / 2)
        # How many segments `locate` looks at from each one: the segment behind, the
        # segment itself, and those within 25 m ahead
        reach = []
        for hint in range(self.segments):
            count, ahead = 2, 0.0
            while ahead < 25.0 and count < self.segments:
                ahead += lengths[(hint - 1 + count) % self.segments]
                count += 1
            reach.append(count)
        self._reach = jnp.asarray(reach)
        self._window = max(reach)

        # Half the width of the track, every metre along the centerline
        grid = np.arange(0.0, self.length, 1.0)
        if width is not None:
            half_widths = np.full(len(grid), width / 2)
        else:
            centerline = np.asarray(self.position(grid))
            apart = [_distance(centerline, edge, grid / self.length) for edge in edges]
            half_widths = (apart[0] + apart[1]) / 2
        self.half_widths = jnp.asarray(half_widths)
        self.min_width = 2 * float(half_widths.min())
        self._grid = jnp.asarray(np.append(grid, self.length))  # closed into a loop
        self._widths = jnp.asarray(np.append(half_widths, half_widths[0]))
        self._bounds: tuple[np.ndarray, np.ndarray] | None = None

        # Aerial image, as (file name, top-left corner [m], bottom-right corner [m])
        self.satellite = None
        if satellite is not None:
            bounds = scale * np.asarray(satellite["bounds"], dtype=float)
            self.satellite = (satellite["image"], bounds[:2], bounds[2:])
        # Named corners, as (label, arc length of the corner)
        turns = turns or {}
        if corners is None:
            self.turns = [(label, self._apex(near)) for label, near in turns.items()]
        else:
            middles = np.asarray(self.starts) + lengths / 2
            self.turns = [
                (label, float(middles[corners[i]])) for i, label in turns.items()
            ]

    @classmethod
    def from_name(cls, name: str, width: float | None = None) -> Track:
        """
        A track of ``TRACKS``. It is built once and then shared: do not modify it.

        :param width: a constant width [m], instead of the track's own
        """
        if name not in TRACKS:
            raise ValueError(f"Unknown track {name!r}. Available: {sorted(TRACKS)}")
        return _from_name(name, width)

    def metres(self, latitude, longitude) -> np.ndarray:
        """World positions [m] of points given by their latitude and longitude."""
        if self.origin is None:
            raise ValueError("This track is not placed on the map: it has no origin.")
        return _metres(np.asarray(latitude), np.asarray(longitude), self.origin)

    def segment_at(self, s) -> jax.Array:
        return jnp.searchsorted(self.starts, jnp.mod(s, self.length), side="right") - 1

    def position(self, s, lateral=0.0) -> jax.Array:
        """World position [m], ``s`` along the centerline and ``lateral`` across it."""
        s, lateral = jnp.broadcast_arrays(jnp.asarray(s, float), jnp.asarray(lateral))
        i = self.segment_at(s)
        along = jnp.mod(s, self.length) - self.starts[i]
        heading = self.headings[i] + self.curvatures[i] * along
        normal = jnp.stack([-jnp.sin(heading), jnp.cos(heading)], axis=-1)
        straight = self.points[i] + along[..., None] * self._tangents[i]
        arc = self._centers[i] - self._radii[i][..., None] * normal
        curved = (self.curvatures[i] != 0)[..., None]
        return jnp.where(curved, arc, straight) + lateral[..., None] * normal

    def heading(self, s) -> jax.Array:
        """Direction of the centerline [rad]."""
        i = self.segment_at(s)
        along = jnp.mod(s, self.length) - self.starts[i]
        return self.headings[i] + self.curvatures[i] * along

    def curvature(self, s) -> jax.Array:
        """Signed curvature of the centerline [1/m]."""
        return self.curvatures[self.segment_at(s)]

    def half_width(self, s) -> jax.Array:
        """Half the width of the track [m]."""
        return jnp.interp(jnp.mod(s, self.length), self._grid, self._widths)

    def locate(self, position, hint) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Project a world position onto the centerline.

        Only the segments around ``hint`` are considered (the one behind it, and those
        within 25 m ahead), so that a car cannot "teleport" to another part of the
        track by cutting across the infield.

        :param hint: index of the segment the car was last on
        :return: (segment index, arc length s along the lap [m], lateral offset [m])
        """
        nearby = jnp.arange(self._window, dtype=jnp.int32)
        index = (jnp.asarray(hint, dtype=jnp.int32) - 1 + nearby) % self.segments
        along, across, distance = self._coordinates(position, index)
        distance = jnp.where(nearby < self._reach[hint], distance, jnp.inf)
        nearest = jnp.argmin(distance)
        segment = index[nearest]
        # Where two segments meet, a position may be found on either: within a
        # nanometre of their joint, it is at the joint
        length = self.lengths[segment]
        along = jnp.clip(along[nearest], 0.0, length)
        along = jnp.where(
            along < 1e-9, 0.0, jnp.where(length - along < 1e-9, length, along)
        )
        return segment, (self.starts[segment] + along) % self.length, across[nearest]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box (min corner, max corner) of the track surface."""
        if self._bounds is None:
            points = np.asarray(self.position(jnp.arange(0.0, self.length, 2.0)))
            margin = float(self.half_widths.max())
            self._bounds = points.min(axis=0) - margin, points.max(axis=0) + margin
        return self._bounds

    def _coordinates(self, position, index):
        """Where a position is along and across some segments, and how far from each."""
        offset = position - self.points[index]
        on_straight = (offset * self._tangents[index]).sum(-1)
        off_straight = (offset * self._normals[index]).sum(-1)
        offset = position - self._centers[index]
        radius, turn = jnp.abs(self._radii[index]), jnp.sign(self._radii[index])
        angle = jnp.arctan2(offset[..., 1], offset[..., 0]) - self._phases[index]
        on_arc = turn * wrap_to_pi(angle) * radius
        off_arc = turn * (radius - jnp.linalg.norm(offset, axis=-1))

        curved = self.curvatures[index] != 0
        along = jnp.where(curved, on_arc, on_straight)
        across = jnp.where(curved, off_arc, off_straight)
        beyond = jnp.maximum(along - self.lengths[index], 0) + jnp.maximum(-along, 0)
        return along, across, jnp.abs(across) + beyond

    def _apex(self, point) -> float:
        """Arc length of the sharpest part of the corner near a point."""
        everywhere = jnp.arange(self.segments)
        *_, distance = self._coordinates(jnp.asarray(point, dtype=float), everywhere)
        around = self.starts[jnp.argmin(distance)] + jnp.arange(-40.0, 41.0)
        sharpest = jnp.argmax(jnp.abs(self.curvature(around)))
        return float(around[sharpest] % self.length)


@cache
def _from_name(name: str, width: float | None) -> Track:
    return Track(**TRACKS[name], width=width)


def wrap_to_pi(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


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
) -> tuple[list, list[np.ndarray]]:
    """
    The centerline midway between the two edges of a track, as a chain of arcs.

    :param inner: points of one edge, going around the lap
    :param outer: points of the other edge, point i facing point i of ``inner``
    :param start: the lap starts where the centerline passes closest to this point
    :return: the segments, and the two edges starting from the start/finish line
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
    segments = []
    for i, j in zip(nodes, np.roll(nodes, -1)):
        segments += _biarc(path[i], directions[i], path[j], directions[j])
    return segments, [inner, outer]


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
) -> list:
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
) -> tuple[list, np.ndarray]:
    """
    The arc that leaves ``start`` in a unit direction and passes through ``end``.

    :return: the segment, and the direction in which it arrives at ``end``
    """
    chord = end - start
    turn = 2 * math.atan2(
        direction[0] * chord[1] - direction[1] * chord[0], direction @ chord
    )
    if abs(turn) < 1e-7:
        return _straight(start, end), direction
    radius = float(np.linalg.norm(chord)) / (2 * abs(math.sin(turn / 2)))
    cos, sin = math.cos(turn), math.sin(turn)
    arrival = np.array(
        [
            cos * direction[0] - sin * direction[1],
            sin * direction[0] + cos * direction[1],
        ]
    )
    return _curve(start, direction, math.copysign(radius, turn), abs(turn)), arrival


def _straight(start: np.ndarray, end: np.ndarray) -> list:
    """A segment: (x, y, heading, curvature, length)."""
    along = end - start
    return [*start, math.atan2(along[1], along[0]), 0.0, float(np.linalg.norm(along))]


def _curve(
    start: np.ndarray, direction: np.ndarray, radius: float, turn: float
) -> list:
    """An arc that turns by ``turn`` [rad], towards positive headings if ``radius`` > 0."""
    heading = math.atan2(direction[1], direction[0])
    return [*start, heading, 1 / radius, abs(radius) * turn]


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


def _rounded_polygon(vertices: np.ndarray, radii: np.ndarray) -> tuple[list, list[int]]:
    """
    The centerline of a closed polygon whose corners are rounded with arcs.

    The start/finish line is the midpoint of the edge joining the last vertex to the
    first one, and the track runs through the vertices in the given order.

    :return: the segments, and the index of the segment rounding each vertex
    """
    n = len(vertices)
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
        tangent = radii[i] * np.tan(abs(turn) / 2)
        entry, exit_ = corner - tangent * d_in, corner + tangent * d_out
        arcs.append(_curve(entry, d_in, math.copysign(radii[i], turn), abs(turn)))
        entries.append(entry)
        exits.append(exit_)

    def straight(start: np.ndarray, end: np.ndarray, edge: int) -> list:
        direction = _unit(vertices[edge] - vertices[edge - 1])
        length = (end - start) @ direction
        if length < -1e-6:
            raise ValueError(
                f"The corners at both ends of the edge leading to vertex {edge} "
                "overlap: reduce their radii or move the vertices apart."
            )
        return [] if length < 1e-6 else [_straight(start, end)]

    # Split the last-to-first edge at its midpoint to place the start/finish line.
    start_line = (exits[-1] + entries[0]) / 2
    if (entries[0] - exits[-1]) @ _unit(vertices[0] - vertices[-1]) < 1.0:
        raise ValueError("The edge holding the start/finish line needs a straight.")
    segments = straight(start_line, entries[0], 0)
    corners = []
    for i in range(n):
        if i > 0:
            segments += straight(exits[i - 1], entries[i], i)
        corners.append(len(segments))
        segments.append(arcs[i])
    segments += straight(exits[-1], start_line, 0)
    return segments, corners


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)
