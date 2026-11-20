import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from racer_rl import (
    TRACKS,
    RaceCar,
    Step,
    Track,
    make_env,
    observe,
    show_video,
    start,
    time_trial,
    transition,
)
from racer_rl.env import CURVATURE_SCALE, DT, FRAMES, LOOKAHEAD, simulate


def follow_centerline(obs: np.ndarray) -> np.ndarray:
    """A simple driver: stay on the centerline, and slow down for the corners ahead."""
    # The speed that each point ahead allows, and the speed here from which it can
    # still be reached by braking. A corner first seen at one point may start right
    # after the point before it.
    corner = np.sqrt(10.0 * CURVATURE_SCALE / np.maximum(np.abs(obs[7:]), 1e-4))
    before = np.concatenate([[0.0], LOOKAHEAD[:-1]])
    speed = min(80.0, np.sqrt(corner**2 + 2 * 8.0 * before).min())
    pedal = 0.5 * (speed - 80.0 * obs[0])
    steering = 0.24 * obs[8] - 0.4 * obs[1] - 2.5 * obs[2]
    return np.clip([pedal, steering], -1, 1)


@pytest.mark.parametrize("name", sorted(TRACKS))
def test_track_is_closed_and_smooth(name):
    track = Track.from_name(name)
    s = np.arange(0.0, track.length + 0.1, 0.1)
    points = np.asarray(track.position(s))
    headings = np.unwrap(track.heading(s))
    # No jump in position or heading anywhere, including across the finish line
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).max() < 0.11
    assert np.abs(np.diff(headings)).max() < 0.1 * np.abs(track.curvatures).max() + 1e-6
    # One full turn per lap
    assert abs(headings[-1] - headings[0]) == pytest.approx(2 * np.pi, abs=1e-2)


@pytest.mark.parametrize("name", sorted(TRACKS))
def test_locate_inverts_position(name):
    track = Track.from_name(name)
    rng = np.random.default_rng(0)
    s = rng.uniform(0, track.length, 200)
    lateral = rng.uniform(-1, 1, 200) * track.half_width(s)
    points, hints = track.position(s, lateral), track.segment_at(s)
    _, s_found, lateral_found = jax.vmap(track.locate)(points, hints)
    np.testing.assert_allclose(s_found, s, atol=1e-6)
    np.testing.assert_allclose(lateral_found, lateral, atol=1e-6)


def test_laguna_seca_is_as_wide_as_its_boundaries():
    from racer_rl.track import _load_boundary

    track = Track.from_name("laguna_seca")
    assert track.min_width == pytest.approx(7.4, abs=0.1)
    assert 2 * track.half_widths.max() == pytest.approx(17.6, abs=0.1)
    # Both edges of the track are on the points of the boundary files
    entry = TRACKS["laguna_seca"]
    given = np.vstack([_load_boundary(f, entry["origin"]) for f in entry["boundaries"]])
    s = np.arange(0.0, track.length, 25.0)
    for side in (1, -1):
        for edge in np.asarray(track.position(s, side * track.half_width(s))):
            assert np.linalg.norm(given - edge, axis=1).min() < 1.0
    # A constant width can be asked for instead
    assert Track.from_name("laguna_seca", 12.0).half_width(1000.0) == 6.0
    # The car can turn tighter than the tightest corner
    lock = RaceCar.STEERING_LOCK
    tightest_turn = (RaceCar.WHEELBASE / 2) / np.sin(np.arctan(np.tan(lock) / 2))
    assert tightest_turn < 1 / np.abs(track.curvatures).max()


def test_env_follows_gymnasium_api():
    env = make_env()
    assert env.observation_space.shape == (19,)
    check_env(env.unwrapped, skip_render_check=True)


def test_env_refuses_an_option_it_does_not_have():
    with pytest.raises(TypeError, match="dynamics"):
        make_env(dynamics="kinematic")


def test_reset_is_reproducible():
    env = make_env()
    obs_a, _ = env.reset(seed=3)
    obs_b, _ = env.reset(seed=3)
    np.testing.assert_array_equal(obs_a, obs_b)


def test_reward_is_zero_without_a_reward_function():
    env = make_env()
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(np.array([1.0, 0.0]))
    assert reward == 0.0


def test_reward_function_receives_each_step():
    steps: list[Step] = []

    def reward(step: Step) -> float:
        steps.append(step)
        return 2.0 * step.progress

    env = make_env("oval", reward=reward, random_start=False)
    env.reset()
    _, first_reward, _, _, _ = env.step(np.array([1.0, 0.5]))
    _, _, _, _, info = env.step(np.array([-0.5, -1.0]))

    first, second = steps
    assert first_reward == 2.0 * first.progress
    assert (first.pedal, first.steering) == pytest.approx((1.0, 0.5))
    assert (first.previous_pedal, first.previous_steering) == (0.0, 0.0)
    assert (second.pedal, second.steering) == pytest.approx((-0.5, -1.0))
    assert (second.previous_pedal, second.previous_steering) == pytest.approx(
        (1.0, 0.5)
    )
    # Full throttle from rest for one step, then half braking
    assert first.longitudinal_acceleration == pytest.approx(RaceCar.MAX_ACCELERATION)
    assert second.longitudinal_acceleration == pytest.approx(-RaceCar.MAX_BRAKING / 2)
    assert first.progress + second.progress == pytest.approx(info["distance"])
    assert second.speed == info["speed"] and second.on_track
    assert second.lap_time is None
    assert second.time == pytest.approx(2 * second.dt)


def test_steering_follows_the_command_at_a_limited_rate():
    steps: list[Step] = []
    env = make_env(
        "oval", reward=lambda step: steps.append(step) or 0.0, random_start=False
    )
    env.reset()
    per_step = RaceCar.STEERING_RATE * DT / RaceCar.STEERING_LOCK  # share of full lock
    assert 0 < per_step < 1
    obs, *_ = env.step(np.array([0.0, 1.0]))  # ask for full lock to the right
    assert steps[-1].steering == pytest.approx(1.0)
    assert steps[-1].steering_actual == pytest.approx(per_step)
    assert obs[4] == pytest.approx(per_step)  # the policy can see where the wheels are
    obs, *_ = env.step(np.array([0.0, -1.0]))  # then for full lock to the left
    assert steps[-1].steering_actual == pytest.approx(0.0, abs=1e-9)
    for _ in range(int(2 / per_step)):
        obs, *_ = env.step(np.array([0.0, -1.0]))
    assert steps[-1].steering_actual == pytest.approx(-1.0)  # it gets there, and stops


def test_leaving_the_track_terminates():
    steps: list[Step] = []
    env = make_env(reward=lambda step: steps.append(step) or 0.0, random_start=False)
    env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        # Full throttle, full steering lock
        _, _, terminated, truncated, info = env.step(np.array([1.0, 1.0]))
    assert terminated and not info["on_track"]
    assert not steps[-1].on_track and all(step.on_track for step in steps[:-1])
    assert abs(steps[-1].lateral) > steps[-1].half_width


@pytest.mark.parametrize("name", sorted(TRACKS))
def test_a_simple_controller_completes_the_time_trial(name):
    lap_times = []

    def reward(step: Step) -> float:
        if step.lap_time is not None:
            lap_times.append(step.lap_time)
        return 0.0

    result = time_trial(follow_centerline, name, laps=2, reward=reward)
    assert result.completed and not result.off_track
    assert result.laps_covered == 2
    assert result.total_time == pytest.approx(sum(result.lap_times))
    # The flying lap is faster than the one from a standing start
    assert result.lap_times[1] < result.lap_times[0]
    # The reward function is told about each completed lap
    assert lap_times == result.lap_times


def test_time_trial_gives_a_planner_the_state_of_the_race():
    track = Track.from_name("oval")
    step = jax.jit(lambda state, action: transition(track, state, action))
    expected = [start(track)]

    def planner(obs, state):
        np.testing.assert_allclose(state.car, expected[0].car, atol=1e-9)
        np.testing.assert_allclose(observe(track, state), obs, atol=1e-6)
        action = follow_centerline(obs)
        expected[0] = step(state, action)  # where the planner expects to be next
        return action

    result = time_trial(planner, "oval", with_state=True)
    assert result.completed


def test_time_trial_reports_an_incomplete_run():
    result = time_trial(lambda obs: np.array([1.0, 1.0]), "oval")
    assert not result.completed and result.off_track
    assert result.total_time is None and result.laps_covered < 1


@pytest.mark.parametrize("camera", ["both", "track", "follow"])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_render_returns_an_image(camera, theme):
    env = make_env(render_mode="rgb_array", camera=camera, theme=theme)
    env.reset(seed=0)
    for _ in range(3):
        env.step(env.action_space.sample())
    image = env.render()
    env.close()
    assert image.shape == (1600, 2400, 3)  # 1200 x 800 points, 2 pixels each
    assert len(np.unique(image.reshape(-1, 3), axis=0)) > 10  # not a blank frame


def test_time_trial_saves_a_video(tmp_path):
    result = time_trial(
        follow_centerline, "oval", laps=1, video=str(tmp_path), pixel_ratio=1
    )
    assert result.completed
    assert result.video == str(tmp_path / "oval.mp4")
    assert (tmp_path / "oval.mp4").stat().st_size > 100_000
    # A notebook is told to load a different address when the file is written again
    first = show_video(result.video)._repr_html_()
    assert f'src="{result.video}?v=' in first
    time_trial(
        lambda obs: np.array([1.0, 0.0]), "oval", video=str(tmp_path), pixel_ratio=1
    )
    assert show_video(result.video)._repr_html_() != first


def test_time_trial_saves_the_whole_trial(tmp_path):
    result = time_trial(
        follow_centerline,
        "oval",
        laps=2,
        save_trial=str(tmp_path / "trials" / "someone"),
    )
    assert result.saved == str(tmp_path / "trials" / "someone.npz")
    trial = np.load(result.saved)
    assert list(trial["lap_times"]) == result.lap_times
    assert trial["laps"] == 2 and trial["completed"] and not trial["off_track"]
    assert str(trial["track"]) == "oval" and str(trial["dynamics"]) == "dynamic"
    # From the standing start to the end of the last lap
    assert len(trial["position"]) == len(trial["actions"]) + 1
    assert trial["time"][0] == 0.0 and trial["speed"][0] == 0.0
    duration = trial["time"][-1]
    assert sum(result.lap_times) <= duration < sum(result.lap_times) + trial["dt"]

    # The actions, applied to a car in the first state, give the same trial again
    car = RaceCar.state(trial["position"][0], trial["heading"][0])
    _, cars = jax.lax.scan(
        lambda car, action: (simulate(car, action)[-1],) * 2, car, trial["actions"]
    )
    np.testing.assert_allclose(cars[:, :2], trial["position"][1:], atol=1e-6)


def test_time_trial_saves_a_trial_that_leaves_the_track(tmp_path):
    result = time_trial(
        lambda obs: np.array([1.0, 1.0]), "oval", save_trial=str(tmp_path / "x")
    )
    trial = np.load(result.saved)
    assert trial["off_track"] and not trial["completed"]
    assert len(trial["lap_times"]) == 0 and len(trial["actions"]) > 0


def cornering(speed: float, seconds: float, pedal: float = 0.0) -> float:
    """Cornering acceleration after holding the wheels at a small angle, at speed."""
    dt = DT / FRAMES
    angle = 0.05  # [rad], as if the wheels were already there
    action = jnp.array([pedal, angle / RaceCar.STEERING_LOCK])
    car = RaceCar.state(jnp.zeros(2), 0.0, speed).at[6].set(angle)
    step = jax.jit(RaceCar.step)
    for _ in range(round(seconds / dt)):
        car = step(car, action, dt).at[3].set(speed)
    return float(RaceCar.lateral_acceleration(car, action))


def test_cornering_builds_up_to_the_grip_limit():
    grip = RaceCar.GRIP
    # At 60 m/s this steering angle asks for more than the tyres can give. The front
    # tyres bite at once, but the rear ones only push once the car has started to
    # rotate
    at_once = cornering(60.0, 0.05)
    settled = cornering(60.0, 2.0)
    assert at_once < 0.6 * grip
    assert 0.7 * grip < settled < 1.1 * grip
    # Braking leaves less grip for turning
    assert abs(cornering(60.0, 2.0, pedal=-1.0)) < 0.5 * settled


def test_car_goes_straight_without_steering():
    dt = DT / FRAMES
    car = RaceCar.state(jnp.zeros(2), 0.3, 40.0)
    step = jax.jit(RaceCar.step)
    for _ in range(round(2.0 / dt)):
        car = step(car, jnp.array([1.0, 0.0]), dt)
    x, y, heading, speed, lateral_speed, yaw_rate, wheels = car
    assert heading == pytest.approx(0.3)
    assert lateral_speed == 0.0 and yaw_rate == 0.0 and wheels == 0.0
    assert speed == pytest.approx(40.0 + 2.0 * RaceCar.MAX_ACCELERATION)
    assert np.hypot(x, y) == pytest.approx(40.0 * 2.0 + RaceCar.MAX_ACCELERATION * 2.0)


def test_transition_is_the_step_of_the_environment():
    env = make_env("hairpins", random_start=False)
    obs, _ = env.reset()
    track = env.unwrapped.track
    state = start(track)
    step = jax.jit(lambda state, action: transition(track, state, action))
    for _ in range(100):
        np.testing.assert_allclose(observe(track, state), obs, atol=1e-6)
        action = follow_centerline(obs)
        obs, *_ = env.step(action)
        state = step(state, action)
        for mine, theirs in zip(state, env.unwrapped.state):
            np.testing.assert_allclose(mine, theirs, atol=1e-9)


def test_many_futures_are_computed_at_once():
    track = Track.from_name("oval")

    def drive(steering):
        action = jnp.stack([jnp.ones(()), steering])
        return jax.lax.scan(
            lambda state, _: (transition(track, state, action), state.on_track),
            start(track),
            length=40,
        )

    final, on_track = jax.jit(jax.vmap(drive))(jnp.array([-0.3, 0.0, 0.3]))
    # Four seconds of full throttle, 80 m: only the car going straight is still racing
    assert list(final.on_track) == [False, True, False]
    assert float(final.lateral[1]) == pytest.approx(0.0, abs=1e-9)
    # The other two stopped where they left the track, and stayed there
    left = int(np.argmin(on_track[0]))
    assert 0 < left < 39 and not on_track[0, left:].any()
    assert float(final.distance[0]) < 0.5 * float(final.distance[1])


def test_bellmax_searches_the_race():
    from racer_rl import bellmax

    track = Track.from_name("oval")
    choices = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0], [-1.0, 0.0]])

    class Search(bellmax.MCTS):
        def actions(self, state):
            return choices

        def transition(self, state, action):
            return transition(track, state, action)

        def reward(self, state, action, next_state, *args):
            covered = (next_state.distance - state.distance) / 8.0
            return (covered - ~next_state.on_track).astype(jnp.float32)

        def terminal(self, state, *args):
            return ~state.on_track

        def path(self, state, action):
            return jax.tree.map(lambda x: x[None], self.transition(state, action))

    search = Search(max_depth=20, num_sim=200)
    key, state = jax.random.key(0), start(track, speed=30.0)
    result = search.search(key, state)
    assert int(result.action_visits.sum()) == search.num_sim
    # On a straight at 30 m/s: full throttle, and no steering
    assert int(jnp.argmax(result.action_visits)) == 0

    # The pictures of that search
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from racer_rl import utils

    values = utils.tree_values(search, result.tree)
    assert len(values.positions) == len(values.q) == search.num_sim + 1
    assert values.paths.shape == (search.num_sim + 1, 2, 2)
    assert np.isnan(values.q[0]) and np.isfinite(values.q[1:]).all()
    assert values.v[0] == np.nanmax(values.q[np.asarray(result.tree.children[0])])
    snapshots = utils.search_snapshots(search, key, state, (), 50)
    assert list(snapshots.size) == [51, 101, 151, 201]
    scene = utils.Scene(track, first=0.0, last=200.0, box=(150.0, 2.0, 0.0))
    fig, axes = plt.subplots(1, 2)
    scene.draw_action_values(axes[0], values, utils.value_scale(values), state.car)
    scene.draw_state_values(axes[1], values, utils.value_scale(values), state.car)
    utils.draw_flow_chart(axes[0])
    plt.close(fig)


def test_car_survives_wild_driving():
    env = make_env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        action = rng.choice([-1.0, 1.0], size=2)  # pedals and steering slammed about
        obs, _, terminated, truncated, _ = env.step(action)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            env.reset()
