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


# ---------- lifts as rises (segmentation) ----------


def rising(t0, t1, y0, y1, x=0.5, fps=12):
    """Points for the bar going from y0 to y1 between t0 and t1."""
    n = max(2, int((t1 - t0) * fps))
    return [BarPoint(t=t0 + (t1 - t0) * k / (n - 1), x=x, y=y0 + (y1 - y0) * k / (n - 1)) for k in range(n)]


def holding(t0, t1, y, x=0.5, fps=12):
    n = max(2, int((t1 - t0) * fps))
    return [BarPoint(t=t0 + (t1 - t0) * k / (n - 1), x=x, y=y) for k in range(n)]


def test_rise_episodes_finds_a_single_lift():
    from server.utils.bar_track import rise_episodes

    points = holding(0, 1, 0.85) + rising(1, 3, 0.85, 0.45) + holding(3, 4, 0.45)
    episodes = rise_episodes(points)
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.start_sec == pytest.approx(1.0, abs=0.15)
    assert episode.end_sec == pytest.approx(3.0, abs=0.15)
    assert episode.rise == pytest.approx(0.40, abs=0.02)
    assert episode.credible


def test_rise_episodes_ignores_a_bar_being_lowered():
    from server.utils.bar_track import rise_episodes

    points = holding(0, 1, 0.3) + rising(1, 3, 0.3, 0.85)  # y increasing: going down
    assert rise_episodes(points) == []


def test_rise_episodes_separates_two_attempts():
    from server.utils.bar_track import rise_episodes

    points = rising(0, 2, 0.85, 0.4) + holding(2, 3, 0.4) + rising(3, 4, 0.4, 0.85) + holding(4, 6, 0.85) + rising(6, 8, 0.85, 0.4)
    episodes = rise_episodes(points)
    assert [round(e.start_sec) for e in episodes] == [0, 6]


def test_rise_episodes_flags_a_hop():
    from server.utils.bar_track import MAX_STEP_RADII_PER_FRAME, rise_episodes

    points = rising(0, 2, 0.85, 0.4)
    # One step of 5 radii, as when the trail jumps from a plate on a stand to the bar.
    steps = [0.3] * len(points)
    steps[len(points) // 2] = MAX_STEP_RADII_PER_FRAME * 3
    episode = rise_episodes(points, step_radii=steps)[0]
    assert episode.fastest_step > MAX_STEP_RADII_PER_FRAME
    assert not episode.credible


def test_rise_episodes_flags_a_wandering_trail():
    from server.utils.bar_track import rise_episodes

    # Rises 0.2 but swings 0.4 sideways: a trail hopping between things.
    points = [BarPoint(t=k / 12, x=0.3 + 0.4 * (k % 2), y=0.8 - 0.2 * k / 23) for k in range(24)]
    episode = rise_episodes(points)[0]
    assert episode.wander > 1.0 and not episode.credible


def test_chunks_cover_the_clip_with_overlap():
    from server.utils.bar_track import CHUNK_OVERLAP, CHUNK_SECONDS, _chunks

    assert _chunks(0.0, 7.0) == [(0.0, 7.0)]
    chunks = _chunks(0.0, 29.3)
    assert chunks[0][0] == 0.0 and chunks[-1][1] == 29.3
    for (a0, a1), (b0, b1) in zip(chunks, chunks[1:]):
        assert a1 - b0 == pytest.approx(CHUNK_OVERLAP)
        assert a1 - a0 <= CHUNK_SECONDS + 1e-9


def test_merge_windows_joins_phases_and_drops_weak_rises():
    from server.utils.bar_track import LiftWindow, STRONG_RISE, _merge_windows

    windows = [
        LiftWindow(1.0, 4.0, 0.30, 3.0, radius=40),  # the pull
        LiftWindow(4.5, 6.0, 0.20, 5.5, radius=40),  # the recovery, 0.5s later
        LiftWindow(8.0, 9.0, STRONG_RISE - 0.01, 8.8, radius=40),  # fidget: weak, ignored
        LiftWindow(12.0, 15.0, 0.25, 14.0, radius=40),  # next attempt
    ]
    merged = _merge_windows(windows)
    assert [(w.start_sec, w.end_sec) for w in merged] == [(1.0, 6.0), (12.0, 15.0)]
    assert merged[0].rise == 0.30


def test_merge_windows_never_makes_one_longer_than_a_lift():
    from server.utils.bar_track import LiftWindow, MAX_WINDOW_SECONDS, _merge_windows

    windows = [LiftWindow(0.0, 7.0, 0.3, 6.0, radius=40), LiftWindow(7.5, 13.0, 0.3, 12.0, radius=40)]
    merged = _merge_windows(windows)
    assert len(merged) == 2
    assert all(w.duration_sec <= MAX_WINDOW_SECONDS for w in merged)


def test_same_plate_only_rejects_a_differently_sized_circle():
    from server.utils.bar_track import LiftWindow, _same_plate_only

    windows = [
        LiftWindow(1.0, 4.0, 0.40, 3.0, radius=40),  # the bar
        LiftWindow(6.0, 8.0, 0.20, 7.0, radius=74),  # the athlete's torso
        LiftWindow(9.0, 11.0, 0.22, 10.0, radius=44),  # the other plate, slightly nearer
    ]
    kept = _same_plate_only(windows)
    assert [w.radius for w in kept] == [40, 44]


def test_detect_lift_windows_on_a_synthetic_clip(tmp_path):
    from server.utils.bar_track import detect_lift_windows

    def one_lift(fraction):
        # Still for the first 30%, rises over the next 40%, holds overhead.
        if fraction < 0.3:
            return 160, HEIGHT - 70
        if fraction < 0.7:
            return 160, HEIGHT - 70 - (fraction - 0.3) / 0.4 * (HEIGHT - 200)
        return 160, 130

    clip = write_clip(
        str(tmp_path / "lift.mp4"), plate_track=one_lift, duration=5.0, distractors=(static_plate_on_a_stand,)
    )
    windows = detect_lift_windows(clip, duration_sec=5.0)
    assert len(windows) == 1, windows
    window = windows[0]
    # Padded around 1.5s-3.5s.
    assert window.start_sec < 1.6 and window.end_sec > 3.4
    assert window.rise > 0.3


def test_detect_lift_windows_finds_two_attempts(tmp_path):
    from server.utils.bar_track import detect_lift_windows

    def two_lifts(fraction):
        # Two attempts 8s apart in a 16s clip: lift at 1.5-3s, again at 9.5-11s.
        # Closer than ~3.5s and they would merge as the phases of one lift,
        # which is what a clean followed by its jerk looks like.
        phase = (fraction * 16) % 8
        if phase < 1.5:
            return 160, HEIGHT - 70
        if phase < 3.0:
            return 160, HEIGHT - 70 - (phase - 1.5) / 1.5 * (HEIGHT - 200)
        if phase < 4.0:
            return 160, 130
        return 160, HEIGHT - 70  # lowered again

    clip = write_clip(str(tmp_path / "two.mp4"), plate_track=two_lifts, duration=16.0)
    windows = detect_lift_windows(clip, duration_sec=16.0)
    assert len(windows) == 2, windows
    assert windows[0].end_sec < windows[1].start_sec


def test_detect_lift_windows_on_a_still_clip(tmp_path):
    from server.utils.bar_track import detect_lift_windows

    clip = write_clip(str(tmp_path / "still.mp4"), plate_track=lambda _f: (160, 300))
    assert detect_lift_windows(clip, duration_sec=3.0) == []


def test_rise_episode_records_when_the_bar_comes_down():
    from server.utils.bar_track import rise_episodes

    points = rising(0, 2, 0.85, 0.4) + holding(2, 3, 0.4) + rising(3, 4, 0.4, 0.85)
    episode = rise_episodes(points)[0]
    assert episode.fall_sec is not None
    assert 3.0 < episode.fall_sec < 3.6  # soon after the lowering begins at 3s


def test_rise_episode_without_a_fall():
    from server.utils.bar_track import rise_episodes

    episode = rise_episodes(rising(0, 2, 0.85, 0.4) + holding(2, 4, 0.4))[0]
    assert episode.fall_sec is None


def test_detect_lift_windows_ends_before_the_drop(tmp_path):
    from server.utils.bar_track import detect_lift_windows

    def lift_then_drop(fraction):
        # Rise 1-2.5s, hold until 3s, drop to the floor by 3.3s, lie there.
        if fraction < 0.2:
            return 160, HEIGHT - 70
        if fraction < 0.5:
            return 160, HEIGHT - 70 - (fraction - 0.2) / 0.3 * (HEIGHT - 200)
        if fraction < 0.6:
            return 160, 130
        if fraction < 0.66:
            return 160, 130 + (fraction - 0.6) / 0.06 * (HEIGHT - 200)
        return 160, HEIGHT - 70

    clip = write_clip(str(tmp_path / "drop.mp4"), plate_track=lift_then_drop, duration=5.0)
    windows = detect_lift_windows(clip, duration_sec=5.0)
    assert len(windows) == 1
    # The hold ends at 3.0s; the drop is over by 3.3s. The window must not run
    # through it to the clip's end.
    assert windows[0].end_sec < 3.6, windows


def test_lift_window_carries_its_trail(tmp_path):
    from server.utils.bar_track import detect_lift_windows

    def one_lift(fraction):
        if fraction < 0.3:
            return 160, HEIGHT - 70
        if fraction < 0.7:
            return 160, HEIGHT - 70 - (fraction - 0.3) / 0.4 * (HEIGHT - 200)
        return 160, 130

    clip = write_clip(str(tmp_path / "lift.mp4"), plate_track=one_lift, duration=5.0)
    window = detect_lift_windows(clip, duration_sec=5.0)[0]
    assert window.points and window.frame_width > 0
    path = window.bar_path()
    assert path.tracked, path.reason
    assert path.vertical_travel > 0.3
    # Cut to a sub-range still yields the right piece.
    part = window.bar_path(window.start_sec, window.peak_sec)
    assert part.points and part.points[-1].t <= window.peak_sec + 1e-6


def test_tracking_is_not_sensitive_to_the_window_start(tmp_path):
    # A window starting a couple of frames later samples different frames; the
    # same plate must still be found.
    from server.utils.bar_track import track_bar_path

    clip = write_clip(str(tmp_path / "phase.mp4"), plate_track=looping_lift, distractors=(bobbing_head,))
    for offset in (0.0, 0.03, 0.06, 0.1, 0.2):
        path = track_bar_path(clip, start_sec=offset, end_sec=3.0)
        assert path.tracked, f"offset {offset}: {path.reason}"
        assert min(p.x for p in path.points) > 0.3, f"offset {offset}: followed the head"
