"""
Bar tracking (§18: a bar-path trace drawn from per-frame bar positions).

The synthetic clips here carry the failure modes found on real footage: a head
is also round and fast, a plate parked on a stand is round and big, and a
barbell has a second plate at the far end that a trail can hop onto and invent a
sideways move that never happened.
"""
import numpy as np  # type: ignore
import pytest

from server.utils.bar_track import (
    MIN_COVERAGE,
    PLATE_DIAMETER_M,
    BarPath,
    BarPoint,
    track_bar_path,
)

WIDTH, HEIGHT, FPS = 320, 480, 24
PLATE_R = 26


def write_clip(
    path: str,
    *,
    plate_track,
    duration: float = 3.0,
    distractors=(),
    plate_colour=(60, 60, 200),
    noise: bool = True,
) -> str:
    """
    A plate travelling along `plate_track(fraction) -> (x, y)`, plus any static
    or moving distractor circles, on a textured background.
    """
    import cv2  # type: ignore

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    rng = np.random.default_rng(7)
    backdrop = np.full((HEIGHT, WIDTH, 3), 70, dtype=np.uint8)
    # Some structure so the frame is not uniformly flat.
    cv2.rectangle(backdrop, (0, HEIGHT - 90), (WIDTH, HEIGHT), (48, 48, 48), -1)
    try:
        total = int(duration * FPS)
        for index in range(total):
            fraction = index / max(1, total - 1)
            frame = backdrop.copy()
            if noise:
                frame = np.clip(frame + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)
            for draw in distractors:
                draw(frame, fraction)
            x, y = plate_track(fraction)
            cv2.circle(frame, (int(x), int(y)), PLATE_R, plate_colour, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), PLATE_R, (230, 230, 230), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return path


def straight_lift(fraction):
    """Bar rises from the floor to overhead, dead vertical."""
    return 160, HEIGHT - 70 - fraction * (HEIGHT - 200)


def looping_lift(fraction):
    """Bar rises but loops forward in the middle — the fault we want to show."""
    x = 160 + 46 * np.sin(np.pi * fraction)
    return x, HEIGHT - 70 - fraction * (HEIGHT - 200)


def static_plate_on_a_stand(frame, _fraction):
    import cv2  # type: ignore

    cv2.circle(frame, (285, HEIGHT - 60), PLATE_R + 6, (190, 120, 40), -1, cv2.LINE_AA)
    cv2.circle(frame, (285, HEIGHT - 60), PLATE_R + 6, (230, 230, 230), 2, cv2.LINE_AA)


def bobbing_head(frame, fraction):
    """Round, high contrast, moves fast — but travels much less than the bar."""
    import cv2  # type: ignore

    y = 150 + 18 * np.sin(2 * np.pi * fraction)
    cv2.circle(frame, (70, int(y)), 17, (150, 170, 190), -1, cv2.LINE_AA)


# ---------- geometry on a known path ----------


def path_of(points, radius=0.08, width=320, height=480) -> BarPath:
    return BarPath(
        points=[BarPoint(t=t, x=x, y=y) for t, x, y in points],
        plate_radius=radius,
        coverage=1.0,
        sampled=len(points),
        frame_width=width,
        frame_height=height,
    )


def test_reference_x_is_where_the_bar_started():
    path = path_of([(i / 10, 0.30 + 0.02 * i, 0.9 - 0.05 * i) for i in range(12)])
    assert path.reference_x == pytest.approx(0.30, abs=0.03)


def test_max_drift_finds_the_worst_departure():
    points = [(0.0, 0.30, 0.9), (0.1, 0.30, 0.8), (0.2, 0.44, 0.6), (0.3, 0.31, 0.4)]
    path = path_of(points + [(0.4, 0.30, 0.2)])
    drift, worst = path.max_drift()
    assert worst.t == pytest.approx(0.2)
    assert drift == pytest.approx(0.14, abs=0.02)


def test_drift_in_cm_is_scaled_by_the_plate():
    # Drift of exactly one plate diameter must read as one plate diameter.
    path = path_of([(0.0, 0.30, 0.9), (0.1, 0.30 + 0.16, 0.5), (0.2, 0.30, 0.2)], radius=0.08)
    assert path.drift_cm() == pytest.approx(PLATE_DIAMETER_M * 100, rel=0.02)


def test_drift_in_cm_is_none_without_a_plate_size():
    assert path_of([(0.0, 0.3, 0.9)], radius=0.0).drift_cm() is None


def test_vertical_travel():
    assert path_of([(0.0, 0.3, 0.9), (0.1, 0.3, 0.3)]).vertical_travel == pytest.approx(0.6)


def test_an_empty_path_is_not_tracked():
    empty = BarPath()
    assert not empty.tracked
    assert empty.max_drift() == (0.0, None)
    assert empty.drift_cm() is None


# ---------- tracking real-ish footage ----------


def test_follows_a_plate_up_the_frame(tmp_path):
    clip = write_clip(str(tmp_path / "lift.mp4"), plate_track=straight_lift)
    path = track_bar_path(clip, start_sec=0.0, end_sec=3.0)
    assert path.tracked, path.reason
    assert path.coverage >= MIN_COVERAGE
    # It went up, and it did not wander sideways.
    assert path.vertical_travel > 0.4
    assert path.max_drift()[0] < 0.04
    # plate_radius is a fraction of the frame width, so compare it as one.
    assert path.plate_radius == pytest.approx(PLATE_R / WIDTH, rel=0.35)


def test_measures_a_forward_loop(tmp_path):
    clip = write_clip(str(tmp_path / "loop.mp4"), plate_track=looping_lift)
    path = track_bar_path(clip, start_sec=0.0, end_sec=3.0)
    assert path.tracked, path.reason
    drift, worst = path.max_drift()
    # The loop peaks at mid-lift and is ~46px of a 320px frame.
    assert drift == pytest.approx(46 / WIDTH, abs=0.05)
    assert 0.3 < worst.t / 3.0 < 0.8


def test_ignores_a_plate_parked_on_a_stand(tmp_path):
    # The first real failure: a big static plate at the frame edge won the seed.
    clip = write_clip(
        str(tmp_path / "stand.mp4"), plate_track=straight_lift, distractors=(static_plate_on_a_stand,)
    )
    path = track_bar_path(clip, start_sec=0.0, end_sec=3.0)
    assert path.tracked, path.reason
    assert path.vertical_travel > 0.4
    # Nowhere near the stand at x=285/320.
    assert max(p.x for p in path.points) < 0.75


def test_prefers_the_bar_over_a_head(tmp_path):
    # The second real failure: a head is round, fast and high-contrast, so the
    # fastest circle is not the bar. The bar is the thing that travels.
    clip = write_clip(
        str(tmp_path / "head.mp4"), plate_track=straight_lift, distractors=(bobbing_head,)
    )
    path = track_bar_path(clip, start_sec=0.0, end_sec=3.0)
    assert path.tracked, path.reason
    assert path.vertical_travel > 0.4
    assert min(p.x for p in path.points) > 0.3  # the head sits at x=70/320


def test_a_still_clip_is_not_tracked(tmp_path):
    clip = write_clip(str(tmp_path / "still.mp4"), plate_track=lambda _f: (160, 300))
    path = track_bar_path(clip, start_sec=0.0, end_sec=3.0)
    assert not path.tracked
    assert path.reason


def test_a_missing_file_degrades(tmp_path):
    path = track_bar_path(str(tmp_path / "nope.mp4"), start_sec=0.0, end_sec=2.0)
    assert not path.tracked
    assert "frames" in path.reason
