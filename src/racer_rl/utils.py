"""Pictures of a tree search on a track, for the notebooks.

A ``Scene`` draws a stretch of a track from above, with the car and where it should
stop. The other
functions turn the tree of a Bellmax search into something to draw: the value of
each action it tried, the value of each state it reached, and the tree as it grew.

They expect the search to have a method ``path(state, action)``, which returns the
states that an action leads through.

``draw_flow_chart`` draws the search itself, as the steps it repeats.
"""

from __future__ import annotations

from functools import cache, partial
from importlib import resources
from pathlib import Path
from typing import NamedTuple

import imageio_ffmpeg
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from PIL import Image

from racer_rl.evaluation import show_video
from racer_rl.rendering import COCKPIT, THEMES, car_parts
from racer_rl.track import Track
from racer_rl.vehicle import HEADING, RaceCar

CAR_SCALE = 1.6  # the car and its box are drawn this many times their size


class Values(NamedTuple):
    """What a search tree says, one entry per state of the tree. The first is the root."""

    positions: np.ndarray  # where each state is [m]
    paths: np.ndarray  # the way into each state from its parent [m]
    q: np.ndarray  # value of the action that leads into each state
    v: np.ndarray  # value of the best action tried from each state


def tree_values(search, tree) -> Values:
    count = int(tree.size)
    parents = jnp.maximum(tree.parent[:count], 0)
    before = jax.tree.map(lambda x: x[parents], tree.state)
    paths = _paths(search, before, tree.action[:count])
    positions = tree.state.car[:count, :HEADING]
    paths = jnp.concatenate([positions[parents][:, None], paths], axis=1)

    q = np.array(tree.value_sum[:count] / jnp.maximum(tree.visits[:count], 1))
    q[0] = np.nan
    children = np.asarray(tree.children[:count])
    below = np.where(children >= 0, q[np.maximum(children, 0)], -np.inf).max(axis=1)
    v = np.where(np.isfinite(below), below, np.nan)
    return Values(np.asarray(positions), np.asarray(paths), q, v)


def value_scale(*values: Values) -> Normalize:
    """One scale for the values of several trees, from the worst to the best."""
    q = np.concatenate([each.q[np.isfinite(each.q)] for each in values])
    return Normalize(vmin=q.min(), vmax=q.max())


@partial(jax.jit, static_argnums=(0, 4))
def search_snapshots(search, key, state, args: tuple, every: int):
    """The tree of a search after every ``every`` simulations, stacked."""
    tree = search._init_tree(state, search.actions(state))

    def record(tree, first):
        def simulate(index, tree):
            return search._simulate(jax.random.fold_in(key, index), tree, args)

        last = jnp.minimum(first + every, search.num_sim)
        tree = jax.lax.fori_loop(first, last, simulate, tree)
        return tree, tree

    return jax.lax.scan(record, tree, jnp.arange(0, search.num_sim, every))[1]


def draw_flow_chart(ax) -> None:
    """One search of ``bellmax.MCTS`` as a flow chart, with the methods it calls."""
    ink, middle, side = "0.15", 4.5, 9.7

    def write(x, y, title, text):
        if title:
            ax.text(x, y + 0.42, title, ha="center", va="center", weight="bold")
        ax.text(x, y - (0.2 if title else 0), text, ha="center", va="center")

    def step(x, y, width, height, title, text, rounding=0.12):
        style = f"round,pad=0,rounding_size={rounding}"
        corner = (x - width / 2, y - height / 2)
        box = FancyBboxPatch(corner, width, height, boxstyle=style, fc="white", ec=ink)
        ax.add_patch(box)
        write(x, y, title, text)

    def question(x, y, width, height, text):
        w, h = width / 2, height / 2
        corners = [(x - w, y), (x, y + h), (x + w, y), (x, y - h)]
        ax.add_patch(Polygon(corners, fc="0.93", ec=ink))
        write(x, y, None, text)

    def arrow(*points, label=None, at=None):
        ax.plot(*np.array(points[:-1]).T, color=ink, linewidth=1.0)
        head = {"arrowstyle": "-|>", "color": ink, "shrinkA": 0, "shrinkB": 0}
        ax.annotate("", points[-1], points[-2], arrowprops=head)
        if label:
            ax.text(*at, label, ha="center", va="center", style="italic")

    step(middle, 14.0, 5.6, 1.0, None, "Start at the root: the car as it is now", 0.5)
    question(middle, 11.9, 5.6, 2.0, "Has every action been\ntried from this node?")
    step(
        side, 11.9, 3.8, 1.6, "Select",
        "go to the child\n" r"that $\mathtt{select}$ picks",
    )  # fmt: skip
    step(
        middle, 9.2, 6.4, 1.7, "Expand",
        r"try a new action: $\mathtt{transition}$, $\mathtt{reward}$" "\n"
        "add where it leads to the tree",
    )  # fmt: skip
    step(
        middle, 6.8, 6.4, 1.7, "Roll out",
        r"play on at random until $\mathtt{max\_depth}$ or $\mathtt{terminal}$" "\n"
        r"add up each $\mathtt{reward}$ on the way",
    )  # fmt: skip
    step(
        middle, 4.4, 6.4, 1.7, "Back up",
        "at every node on the way back to the root:\n"
        "one more visit, and the rewards from there on",
    )  # fmt: skip
    question(middle, 2.0, 5.6, 1.8, r"$\mathtt{num\_sim}$ simulations done?")
    step(
        middle, -0.2, 6.4, 1.0, None,
        "Take the action tried the most at the root", 0.5,
    )  # fmt: skip

    arrow((middle, 13.5), (middle, 12.9))
    arrow((middle + 2.8, 11.9), (side - 1.9, 11.9), label="yes", at=(7.55, 12.2))
    arrow((side, 12.7), (side, 13.2), (middle, 13.2))
    arrow((middle, 10.9), (middle, 10.05), label="no", at=(middle + 0.45, 10.5))
    arrow((middle, 8.35), (middle, 7.65))
    arrow((middle, 5.95), (middle, 5.25))
    arrow((middle, 3.55), (middle, 2.9))
    arrow((middle, 1.1), (middle, 0.3), label="yes", at=(middle + 0.5, 0.72))
    arrow((middle - 2.8, 2.0), (0.6, 2.0), (0.6, 14.0), (middle - 2.8, 14.0))
    ax.text(1.15, 2.3, "no", ha="center", va="center", style="italic")
    ax.text(0.3, 8.0, "next simulation", ha="center", va="center", rotation=90)
    ax.set(xlim=(0, 12), ylim=(-1, 14.8), aspect="equal")
    ax.axis("off")


class Scene:
    """A stretch of a track seen from above, on which to draw a car, a box and a tree."""

    def __init__(self, track: Track, *, first: float, last: float, box, margin=20.0):
        """
        :param first: arc length where the stretch begins [m]
        :param last: arc length where it ends [m], possibly past the start/finish line
        :param box: where the car should stop, to mark with a box: the arc length
            [m], the lateral offset [m] and the angle to the track [rad]
        """
        self.track = track
        last = last if last > first else last + track.length
        points = np.asarray(track.position(jnp.arange(first, last, 2.0)))
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        reach = np.ptp(points, axis=0).max() / 2 + margin
        self.low, self.high = center - reach, center + reach
        along, across, angle = (float(each) for each in box)
        self.box = np.append(
            track.position(along, across), track.heading(along) + angle
        )

    def draw(self, ax, car=None, driven=None, center=None, reach=None) -> None:
        """
        The track, the box, the way driven so far and the car.

        :param center: look at this point [m] instead of the whole stretch
        :param reach: and this far around it [m]
        """
        track = self.track
        low, high = self.low, self.high
        if center is not None:
            low, high = np.asarray(center) - reach, np.asarray(center) + reach
        if track.satellite is not None:
            picture, extent = _aerial(track, tuple(low), tuple(high))
            ax.imshow(picture, extent=extent, cmap="gray", vmin=0, vmax=255, zorder=0)
        for edge in _edges(track):
            ax.plot(*edge.T, color="0.25", linewidth=1.0, zorder=1)
        _draw_box(ax, self.box)
        if driven is not None and len(driven) > 1:
            ax.plot(*np.asarray(driven).T, color="black", linewidth=2, zorder=5)
        if car is not None:
            _draw_car(ax, np.asarray(car))
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(high[1], low[1])  # y points down, as on screen
        ax.set_aspect("equal")
        ax.tick_params(
            which="both", bottom=False, left=False, labelbottom=False, labelleft=False
        )

    def draw_action_values(self, ax, values: Values, scale, car=None, driven=None):
        """Each action of a tree as the curve it drives. Thicker: higher value."""
        widths = 0.5 + 4 * scale(values.q[1:])
        curves = LineCollection(
            values.paths[1:], colors="C1", linewidths=widths, alpha=0.5, zorder=3
        )
        ax.add_collection(curves)
        ax.scatter(
            *values.positions[1:].T,
            s=8,
            facecolors="none",
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
        )
        self.draw(ax, car, driven)

    def draw_state_values(self, ax, values: Values, scale, car=None, driven=None):
        """Each state of a tree, coloured by the value of the best action from it."""
        known = np.isfinite(values.v)
        ax.scatter(
            *values.positions[known].T,
            c=values.v[known],
            s=40,
            cmap="Blues",
            norm=scale,
            edgecolors="none",
            zorder=3,
        )
        self.draw(ax, car, driven)

    def animate(self, frames, labels, file: str, interval: float = 150):
        """
        A video of trees, as Q-values and as state values side by side.

        :param frames: for each picture, (the state of the car, the ``Values`` of
            the tree, the way driven so far)
        :param labels: the title of each picture
        :param file: where to save the video (.mp4)
        :param interval: time for which each picture is shown [ms]
        :return: the video, to show in a notebook
        """
        scale = value_scale(*(values for _, values, _ in frames))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), dpi=110, layout="constrained")

        def update(index):
            car, values, driven = frames[index]
            for ax in axes:
                ax.clear()
            self.draw_action_values(axes[0], values, scale, car, driven)
            self.draw_state_values(axes[1], values, scale, car, driven)
            axes[0].set_title("Q-values", fontsize=14)
            axes[1].set_title("State values", fontsize=14)
            fig.suptitle(labels[index], fontsize=15)

        Path(file).parent.mkdir(parents=True, exist_ok=True)
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        animation = FuncAnimation(fig, update, frames=len(frames))
        animation.save(file, writer=FFMpegWriter(fps=1000 / interval))
        plt.close(fig)
        return show_video(file, width=800)


@partial(jax.jit, static_argnums=0)
def _paths(search, states, actions):
    return jax.vmap(search.path)(states, actions).car[..., :HEADING]


@cache
def _edges(track: Track) -> list[np.ndarray]:
    s = jnp.arange(0.0, track.length + 1.0)
    return [
        np.asarray(track.position(s, side * track.half_width(s))) for side in (1, -1)
    ]


@cache
def _aerial(track: Track, low: tuple, high: tuple) -> tuple[np.ndarray, tuple]:
    """The part of the aerial image of a track in a view, pale and grey, and its extent."""
    name, corner, opposite = track.satellite
    path = resources.files("racer_rl") / "assets" / "tracks" / name
    with resources.as_file(path) as file:
        grey = np.asarray(Image.open(file).convert("L"), dtype=float)
    per_metre = np.array(grey.shape[::-1]) / (opposite - corner)
    first = np.clip(np.floor((np.array(low) - corner) * per_metre), 0, None).astype(int)
    last = np.ceil((np.array(high) - corner) * per_metre).astype(int)
    pale = 255 - 0.45 * (255 - grey[first[1] : last[1], first[0] : last[0]])
    x0, y0 = corner + first / per_metre
    x1, y1 = corner + np.minimum(last, grey.shape[::-1]) / per_metre
    return pale, (x0, x1, y1, y0)


def _draw_box(ax, box: np.ndarray) -> None:
    """Where the car should stop and which way it should point: a box that fits it."""
    x, y, heading = box
    place = Affine2D().scale(CAR_SCALE).rotate(heading).translate(x, y) + ax.transData
    length, width = 1.15 * RaceCar.LENGTH, 1.3 * RaceCar.WIDTH
    outline = Rectangle((-length / 2, -width / 2), length, width, transform=place)
    outline.set(facecolor="gold", edgecolor="black", linewidth=1.0, zorder=7)
    ax.add_patch(outline)


def _draw_car(ax, car: np.ndarray) -> None:
    """The car of the race pictures, larger than life so that it can be seen."""
    x, y, heading = car[:3]
    place = Affine2D().scale(CAR_SCALE).rotate(heading).translate(x, y) + ax.transData
    canvas, ink = (np.array(THEMES["light"][name]) / 255 for name in ("canvas", "text"))
    parts = car_parts()
    for part in parts:  # the outline: every part again, thicker, underneath
        outline = Polygon(part, color=canvas, linewidth=1.5, transform=place, zorder=8)
        ax.add_patch(outline)
    for part in parts:
        ax.add_patch(Polygon(part, color=ink, linewidth=0, transform=place, zorder=9))
    middle, long, wide = COCKPIT
    cockpit = Ellipse((middle, 0.0), long, wide, color=canvas, linewidth=0)
    cockpit.set(transform=place, zorder=10)
    ax.add_patch(cockpit)
