import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from racer_rl import TRACKS, RaceCar, Step, Track, make_env, show_video, time_trial
from racer_rl.env import CARS
from racer_rl.observation import CURVATURE_SCALE, LOOKAHEAD
from racer_rl.vehicle import DynamicRaceCar


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
    points = np.array([track.position(x) for x in s])
    headings = np.unwrap([track.heading(x) for x in s])
    # No jump in position or heading anywhere, including across the finish line
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).max() < 0.11
    assert np.abs(np.diff(headings)).max() < 0.1 * np.abs(track.curvatures).max() + 1e-6
    # One full turn per lap
    assert abs(headings[-1] - headings[0]) == pytest.approx(2 * np.pi, abs=1e-2)


@pytest.mark.parametrize("name", sorted(TRACKS))
def test_locate_inverts_position(name):
    track = Track.from_name(name)
    rng = np.random.default_rng(0)
    for s in rng.uniform(0, track.length, 200):
        lateral = rng.uniform(-1, 1) * track.half_width(s)
        hint = int(track.segment_at(s))
        _, s_found, lateral_found = track.locate(track.position(s, lateral), hint)
        assert s_found == pytest.approx(s, abs=1e-6)
        assert lateral_found == pytest.approx(lateral, abs=1e-6)


def test_laguna_seca_is_as_wide_as_its_boundaries():
    from racer_rl.track import _load_boundary

    track = Track.from_name("laguna_seca")
    assert track.min_width == pytest.approx(7.4, abs=0.1)
    assert 2 * track.half_widths.max() == pytest.approx(17.6, abs=0.1)
    # Both edges of the track are on the points of the boundary files
    entry = TRACKS["laguna_seca"]
    given = np.vstack([_load_boundary(f, entry["origin"]) for f in entry["boundaries"]])
    for side in (1, -1):
        for s in np.arange(0.0, track.length, 25.0):
            edge = track.position(s, side * track.half_width(s))
            assert np.linalg.norm(given - edge, axis=1).min() < 1.0
    # A constant width can be asked for instead
    assert Track.from_name("laguna_seca", 12.0).half_width(1000.0) == 6.0
    # The car can turn tighter than the tightest corner
    tightest_turn = (RaceCar.WHEELBASE / 2) / np.sin(np.arctan(np.tan(0.25) / 2))
    assert tightest_turn < 1 / np.abs(track.curvatures).max()


@pytest.mark.parametrize("dynamics", ["kinematic", "dynamic"])
def test_env_follows_gymnasium_api(dynamics):
    env = make_env(dynamics=dynamics)
    assert env.observation_space.shape == (19,)
    check_env(env.unwrapped, skip_render_check=True)


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
    full_lock = env.unwrapped.action_type.steering_range[1]
    per_step = RaceCar.STEERING_RATE * steps_dt(env) / full_lock  # share of full lock
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


def steps_dt(env) -> float:
    return 1 / env.unwrapped.config["policy_frequency"]


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


def test_grip_limits_cornering():
    env = make_env(random_start=False)
    env.reset()
    car = env.unwrapped.vehicle
    car.speed = 25.0
    for _ in range(5):
        env.step(np.array([0.0, 1.0]))  # full steering lock, no pedal
        assert abs(car.lateral_acceleration) <= RaceCar.GRIP + 1e-9
    # Braking leaves less grip for turning
    car.speed = 25.0
    env.step(np.array([-1.0, 1.0]))
    available = np.sqrt(RaceCar.GRIP**2 - RaceCar.MAX_BRAKING**2)
    assert abs(car.lateral_acceleration) <= available + 1e-9


@pytest.mark.parametrize("dynamics", ["kinematic", "dynamic"])
@pytest.mark.parametrize("name", sorted(TRACKS))
def test_a_simple_controller_completes_the_time_trial(name, dynamics):
    lap_times = []

    def reward(step: Step) -> float:
        if step.lap_time is not None:
            lap_times.append(step.lap_time)
        return 0.0

    result = time_trial(
        follow_centerline, name, laps=2, reward=reward, dynamics=dynamics
    )
    assert result.completed and not result.off_track
    assert result.laps_covered == 2
    assert result.total_time == pytest.approx(sum(result.lap_times))
    # The flying lap is faster than the one from a standing start
    assert result.lap_times[1] < result.lap_times[0]
    # The reward function is told about each completed lap
    assert lap_times == result.lap_times


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


@pytest.mark.parametrize("dynamics", ["kinematic", "dynamic"])
def test_time_trial_saves_the_whole_trial(tmp_path, dynamics):
    result = time_trial(
        follow_centerline,
        "oval",
        laps=2,
        save_trial=str(tmp_path / "trials" / "someone"),
        dynamics=dynamics,
    )
    assert result.saved == str(tmp_path / "trials" / "someone.npz")
    trial = np.load(result.saved)
    assert list(trial["lap_times"]) == result.lap_times
    assert trial["laps"] == 2 and trial["completed"] and not trial["off_track"]
    assert str(trial["track"]) == "oval" and str(trial["dynamics"]) == dynamics
    # From the standing start to the end of the last lap
    assert len(trial["position"]) == len(trial["actions"]) + 1
    assert trial["time"][0] == 0.0 and trial["speed"][0] == 0.0
    duration = trial["time"][-1]
    assert sum(result.lap_times) <= duration < sum(result.lap_times) + trial["dt"]

    # The actions, applied to a car in the first state, give the same trial again
    car = CARS[dynamics](None, trial["position"][0], trial["heading"][0], 0.0)
    for action, position in zip(trial["actions"], trial["position"][1:]):
        car.act({"acceleration": action[0], "steering": 0.25 * action[1]})
        for _ in range(2):  # two simulation steps for each decision
            car.step(0.05)
        assert car.position == pytest.approx(position, abs=1e-6)


def test_time_trial_saves_a_trial_that_leaves_the_track(tmp_path):
    result = time_trial(
        lambda obs: np.array([1.0, 1.0]), "oval", save_trial=str(tmp_path / "x")
    )
    trial = np.load(result.saved)
    assert trial["off_track"] and not trial["completed"]
    assert len(trial["lap_times"]) == 0 and len(trial["actions"]) > 0


def cornering(car_class, speed: float, seconds: float) -> float:
    """Cornering acceleration after holding the wheels at a small angle, at speed."""
    car = car_class(None, np.zeros(2), heading=0.0, speed=speed)
    car.steering_angle = 0.05  # [rad], as if the wheels were already there
    car.act({"steering": 0.05, "acceleration": 0.0})
    for _ in range(round(seconds / 0.05)):
        car.step(0.05)
        car.speed = speed
    return car.lateral_acceleration


def test_kinematic_car_corners_at_once_and_dynamic_car_builds_up():
    grip = RaceCar.GRIP
    # At 60 m/s this steering angle asks for more than the tyres can give
    assert cornering(RaceCar, 60.0, 0.05) == pytest.approx(grip)
    # In the dynamic car the front tyres bite at once, but the rear ones only push
    # once the car has started to rotate
    at_once = cornering(DynamicRaceCar, 60.0, 0.05)
    settled = cornering(DynamicRaceCar, 60.0, 2.0)
    assert at_once < 0.6 * grip
    assert 0.7 * grip < settled < 1.1 * grip


def test_dynamic_car_goes_straight_without_steering():
    car = DynamicRaceCar(None, np.zeros(2), heading=0.3, speed=40.0)
    car.act({"steering": 0.0, "acceleration": 1.0})
    for _ in range(40):  # 2 s
        car.step(0.05)
    assert car.heading == pytest.approx(0.3)
    assert car.sideslip == 0.0 and car.yaw_rate == 0.0
    assert car.speed == pytest.approx(40.0 + 2.0 * RaceCar.MAX_ACCELERATION)


def test_dynamic_car_survives_wild_driving():
    env = make_env(dynamics="dynamic")
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        action = rng.choice([-1.0, 1.0], size=2)  # pedals and steering slammed about
        obs, _, terminated, truncated, _ = env.step(action)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            env.reset()
