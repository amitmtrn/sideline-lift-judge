"""
The near plate on a clip filmed from an angle (server/utils/plate_oblique), and
the plate gate on every other tracker's trace.

Synthetic clips only, never committed footage: a dark ellipse (the near plate
seen at ~45°), a bright textured patch moving behind it (foliage in a window),
a smaller round far plate, and — in one clip — a dark ring hanging above the
lifter that never moves.
"""
import time

import numpy as np  # type: ignore
import pytest

from server.utils.bar_track import BarPath
from server.utils.segmentation import CandidateSegment
from tests.conftest import upload

SNATCH_PATH = ["floor", "knees", "overhead", "overhead", "overhead", "overhead"]

WIDTH, HEIGHT, FPS = 360, 640, 30
PLATE_R = 34  # vertical semi-axis of the near plate
ASPECT = 0.6  # its horizontal / vertical axis: ~50° off the side
RISE_SHARE = 0.4  # how far up the frame it rises
FAR_R = 22
RING = (190, 70)


def plate_at(fraction):
    """Still for the first fifth, then up 40 % of the frame, looping forward a little."""
    rise = min(1.0, max(0.0, (fraction - 0.2) / 0.6))
    x = 150 + 25 * np.sin(np.pi * rise)
    y = HEIGHT - 110 - rise * RISE_SHARE * HEIGHT
    return x, y


def _leaves(frame, rng, centre, radius):
    """Bright foliage: a ragged patch of small leaves, no clean outline."""
    import cv2  # type: ignore

    cx, cy = centre
    for _ in range(140):
        angle = rng.uniform(0, 2 * np.pi)
        reach = radius * np.sqrt(rng.uniform(0, 1))
        at = (int(cx + reach * np.cos(angle)), int(cy + reach * np.sin(angle)))
        colour = tuple(int(v) for v in rng.integers(130, 255, 3))
        axes = (int(rng.integers(3, 9)), int(rng.integers(2, 5)))
        cv2.ellipse(frame, at, axes, float(rng.integers(0, 180)), 0, 360, colour, -1)


def write_oblique_clip(path, *, plate=True, ring=False, duration=3.0):
    """
    Returns (path, truth) with truth = [(t, x, y)] of the near plate's centre.
    """
    import cv2  # type: ignore

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    rng = np.random.default_rng(3)
    backdrop = np.clip(rng.normal(95, 6, (HEIGHT, WIDTH, 3)), 0, 255).astype(np.uint8)
    total = int(duration * FPS)
    truth = []
    for index in range(total):
        fraction = index / (total - 1)
        frame = backdrop.copy()
        # Foliage drifting across and up behind the lifter.
        leaves = np.random.default_rng(11)  # the same leaves every frame, moved
        _leaves(frame, leaves, (int(110 + 120 * fraction), int(HEIGHT - 300 - 120 * np.sin(np.pi * fraction))), 42)
        if ring:
            cv2.circle(frame, RING, 30, (30, 30, 30), 7, cv2.LINE_AA)
        x, y = plate_at(fraction)
        if plate:
            # The far plate: smaller, round, further right, rising less (perspective).
            far = (int(x + 95), int(HEIGHT - 110 - (HEIGHT - 110 - y) * 0.7))
            cv2.circle(frame, far, FAR_R, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.circle(frame, far, FAR_R, (120, 120, 120), 2, cv2.LINE_AA)
            # The near plate: a dark ellipse, taller than wide, with a hub.
            centre = (int(round(x)), int(round(y)))
            axes = (int(round(PLATE_R * ASPECT)), PLATE_R)
            cv2.ellipse(frame, centre, axes, 0, 0, 360, (28, 28, 28), -1, cv2.LINE_AA)
            cv2.ellipse(frame, centre, axes, 0, 0, 360, (140, 140, 140), 2, cv2.LINE_AA)
            cv2.circle(frame, centre, 5, (170, 170, 170), -1, cv2.LINE_AA)
        truth.append((index / FPS, x, y))
        writer.write(frame)
    writer.release()
    return path, truth


def _errors(path: BarPath, truth):
    times = [q[0] for q in truth]
    out = []
    for p in path.points:
        tx = np.interp(p.t, times, [q[1] for q in truth])
        ty = np.interp(p.t, times, [q[2] for q in truth])
        out.append(np.hypot(p.x * WIDTH - tx, p.y * HEIGHT - ty))
    return np.array(out)


@pytest.fixture(scope="module")
def oblique_clip(tmp_path_factory):
    return write_oblique_clip(str(tmp_path_factory.mktemp("oblique") / "oblique.mp4"))


@pytest.fixture(scope="module")
def foliage_clip(tmp_path_factory):
    return write_oblique_clip(str(tmp_path_factory.mktemp("foliage") / "foliage.mp4"), plate=False)[0]


@pytest.fixture(scope="module")
def ring_clip(tmp_path_factory):
    return write_oblique_clip(str(tmp_path_factory.mktemp("ring") / "ring.mp4"), ring=True)


def test_the_oblique_lift_is_followed_on_the_near_plate(oblique_clip):
    from server.services.confidence import CAP_OFF_AXIS, assess_clip_quality
    from server.services.measurement import measure
    from server.utils.plate_oblique import OBLIQUE_RATIO, follow_oblique

    clip, truth = oblique_clip
    path = follow_oblique(clip, 0.0, 3.0)
    assert path.tracked, path.reason
    errors = _errors(path, truth)
    assert np.mean(errors <= 0.5 * PLATE_R) >= 0.9, f"{np.mean(errors <= 0.5 * PLATE_R):.0%} within half a radius; worst {errors.max():.0f} px"
    # The near plate, not the far one (95 px to the right) and not the foliage.
    assert abs(path.plate_radius * WIDTH - PLATE_R) <= 0.25 * PLATE_R
    # The plate's shape says the camera is off to the side…
    assert path.plate_axis_ratio is not None and path.plate_axis_ratio < OBLIQUE_RATIO, path.plate_axis_ratio
    assert abs(path.plate_axis_ratio - ASPECT) < 0.15
    # …and the angle cap fires on it.
    quality = assess_clip_quality(measurement=measure(path))
    assert CAP_OFF_AXIS in quality.caps
    assert any("Film square from the side" in warning for warning in quality.warnings)


def test_foliage_alone_is_not_a_bar(foliage_clip):
    from server.services.pipeline import _trace_for
    from server.utils.bar_track import track_bar_path
    from server.utils.plate_oblique import NO_TRACE_REASON, follow_oblique, looks_like_a_plate

    path = follow_oblique(foliage_clip, 0.0, 3.0)
    assert not path.tracked and path.reason
    # The legacy tracker will happily follow a bright round texture; the
    # gate is what stops it being drawn.
    legacy = track_bar_path(foliage_clip, start_sec=0.0, end_sec=3.0)
    if legacy.tracked:
        ok, why = looks_like_a_plate(foliage_clip, legacy)
        assert not ok and "does not look like a plate" in why
    traced = _trace_for(foliage_clip, CandidateSegment(index=1, start_sec=0.0, end_sec=3.0), None, "foliage")
    assert not traced.tracked
    assert traced.reason == NO_TRACE_REASON
    assert "near plate" in traced.reason


def test_a_hanging_ring_is_never_the_top_of_the_lift(ring_clip):
    from server.utils.plate_oblique import follow_oblique

    clip, truth = ring_clip
    path = follow_oblique(clip, 0.0, 3.0)
    assert path.tracked, path.reason
    top = min(path.points, key=lambda p: p.y)
    plate_top = plate_at(1.0)
    assert np.hypot(top.x * WIDTH - plate_top[0], top.y * HEIGHT - plate_top[1]) <= 0.5 * PLATE_R
    assert np.hypot(top.x * WIDTH - RING[0], top.y * HEIGHT - RING[1]) > 3 * PLATE_R
    errors = _errors(path, truth)
    assert np.mean(errors <= 0.5 * PLATE_R) >= 0.9


def test_the_pipeline_takes_the_oblique_trace_on_an_oblique_clip(oblique_clip):
    from server.services.pipeline import _trace_for
    from server.utils.plate_oblique import OBLIQUE_RATIO

    clip, truth = oblique_clip
    path = _trace_for(clip, CandidateSegment(index=1, start_sec=0.0, end_sec=3.0), None, "oblique")
    assert path.tracked, path.reason
    assert path.plate_axis_ratio is not None and path.plate_axis_ratio < OBLIQUE_RATIO
    assert np.mean(_errors(path, truth) <= 0.5 * PLATE_R) >= 0.9


def test_the_tap_marks_the_plate_it_is_nearest(oblique_clip):
    """The tap is a preference between things that rose, not an answer."""
    from server.utils.plate_oblique import follow_oblique

    clip, truth = oblique_clip
    x0, y0 = plate_at(0.0)
    on_plate = follow_oblique(clip, 0.0, 3.0, hint=(0.0, x0 / WIDTH, y0 / HEIGHT))
    assert on_plate.tracked and np.mean(_errors(on_plate, truth) <= 0.5 * PLATE_R) >= 0.9
    assert on_plate.tapped is True
    # A tap on the foliage, which never rises like a bar: the near plate still
    # wins, and the result does not claim to be following the tap.
    off = follow_oblique(clip, 0.0, 3.0, hint=(0.0, 110 / WIDTH, (HEIGHT - 300) / HEIGHT))
    assert off.tracked and np.mean(_errors(off, truth) <= 0.5 * PLATE_R) >= 0.9
    assert off.tapped is False
    # No tap: no claim either.
    assert follow_oblique(clip, 0.0, 3.0).tapped is False


def test_a_tap_made_later_in_the_window_seeds_that_frame(oblique_clip):
    """
    The tap is on the first *trimmed* frame, which can be later than the lift
    window's start; it names the plate where it was at that moment.
    """
    from server.utils.plate_oblique import follow_oblique, read_frames_for, tap_frame

    clip, truth = oblique_clip
    frames = read_frames_for(clip, 0.0, 3.0)
    assert tap_frame(frames, 1.0) == min(range(len(frames)), key=lambda i: abs(frames[i][0] - 1.0))
    assert tap_frame(frames, 0.0) == 0
    assert tap_frame(frames, 9.0) == 0  # outside the window: the first frame
    x1, y1 = plate_at(1.0 / 3.0)  # the plate at t = 1.0 s, part way up
    path = follow_oblique(clip, 0.0, 3.0, hint=(1.0, x1 / WIDTH, y1 / HEIGHT))
    assert path.tracked and path.tapped is True
    assert np.mean(_errors(path, truth) <= 0.5 * PLATE_R) >= 0.9


def test_a_tap_hint_also_marks_the_side_on_followers_plate(tmp_path):
    from tests.test_plate_follow import HEIGHT as H, PLATE_R as R, WIDTH as W, bar_at, write_lift
    from server.utils.plate_follow import follow_bar_path

    clip = write_lift(str(tmp_path / "lift.mp4"))
    x0, y0 = bar_at(0.0)
    tapped = follow_bar_path(clip, 0.0, 2.5, tap=(0.0, x0 / W, y0 / H))
    assert tapped.tracked and tapped.tapped is True
    assert abs(tapped.plate_radius * W - R) <= 0.2 * R
    # A tap on the plate parked on the stand, which never rises: the bar still wins, untapped.
    parked = follow_bar_path(clip, 0.0, 2.5, tap=(0.0, 300 / W, (H - 70) / H))
    assert parked.tracked and parked.tapped is False and parked.vertical_travel > 0.5


def test_the_work_is_strided_and_cached(oblique_clip):
    """
    A 4 s window costs one full scan every third frame plus the opening
    frames; the walks look in a patch around their prediction on the rest.
    The analysis itself (frames already decoded) stays well inside the
    6 s budget — here a fraction of a second.
    """
    from server.utils import plate_oblique as po

    clip, _ = oblique_clip
    frames = po.read_frames_for(clip, 0.0, 3.0)
    assert len(frames) <= int(3.0 * po.ANALYSIS_FPS) + 2
    assert max(frames[0][1].shape) <= po.ANALYSIS_LONG_SIDE
    scans = []
    original = po._Detections

    class Counting(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            scans.append(self)

    po._Detections = Counting
    try:
        started = time.perf_counter()
        path = po.follow_oblique(None, frames=frames)
        elapsed = time.perf_counter() - started
    finally:
        po._Detections = original
    assert path.tracked, path.reason
    assert scans and scans[0].scans <= len(frames) // po.KEY_EVERY + po.FLOOR_FRAMES + 2
    assert elapsed < 6.0, f"{elapsed:.1f} s for {len(frames)} frames"


def test_a_tap_in_display_coordinates_lands_on_the_decoded_frame():
    """
    OpenCV hands back a rotated phone clip either already turned (most builds)
    or as coded; the tap, made on the turned frame, must land on the same
    pixel either way.
    """
    import cv2  # type: ignore

    from server.services.pipeline import _tap_in_decoded_frame
    from server.utils.image_ops import rotate_by_degrees
    from server.utils.video_meta import VideoInfo

    decoded = np.zeros((90, 160), np.uint8)  # coded: landscape
    cv2.circle(decoded, (120, 30), 3, 255, -1)  # the plate, at (0.75, 0.33) of the coded frame
    for rotation in (90, 270):
        info = VideoInfo(width=160, height=90, rotation=rotation, probed=True)
        shown = rotate_by_degrees(decoded, rotation)  # what the phone shows: portrait
        ys, xs = np.where(shown > 0)
        tap = (1.25, float(xs.mean()) / shown.shape[1], float(ys.mean()) / shown.shape[0])
        # Frames at the display size: the tap is used as is.
        assert _tap_in_decoded_frame(tap, info, shown.shape[1], shown.shape[0]) == tap
        # Frames at the coded size: turned back onto the plate; the time is kept.
        t, x, y = _tap_in_decoded_frame(tap, info, decoded.shape[1], decoded.shape[0])
        assert t == 1.25
        assert abs(x * 160 - 120) < 2 and abs(y * 90 - 30) < 2, (rotation, x, y)
    assert _tap_in_decoded_frame(None, None, 160, 90) is None
    assert _tap_in_decoded_frame((0.0, 0.2, 0.4), VideoInfo(), 160, 90) == (0.0, 0.2, 0.4)


# ---------- the form field ----------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("0.40,0.4130,0.8256", (0.40, 0.4130, 0.8256)),
        (" 2 , 0.5 , 0.5 ", (2.0, 0.5, 0.5)),
        ("a,b", None),
        ("2,1.5,0.5", None),
        ("0.5,0.5", None),
        ("-1,0.5,0.5", None),
        ("nan,0.5,0.5", None),
        ("1,2,3,4", None),
    ],
)
def test_the_plate_tap_field_is_parsed_or_ignored_never_refused(value, expected):
    from server.main import _parse_plate_tap

    assert _parse_plate_tap(value) == expected


def test_a_malformed_plate_tap_still_scores_the_clip(client, clip):
    from tests.conftest import judge_reply, observation_reply, triage_reply

    for bad in ("a,b", "2,1.5,0.5"):
        response, payload = upload(
            client,
            clip,
            [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)],
            plate_tap=bad,
        )
        assert response.status_code == 202, (bad, response.text)
        assert payload["status"] == "done", (bad, payload)
        assert payload["lifts"][0]["bar_path"]["tracked"] in (True, False)


def test_a_good_plate_tap_reaches_the_followers_and_is_reported(client, clip, monkeypatch):
    from server.services import pipeline
    from tests.conftest import judge_reply, observation_reply, triage_reply

    seen = {}
    real = pipeline.follow_oblique

    def spy(*args, **kwargs):
        seen["hint"] = kwargs.get("hint")
        path = real(*args, **kwargs)
        return path

    monkeypatch.setattr(pipeline, "follow_oblique", spy)
    _, payload = upload(
        client,
        clip,
        [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)],
        plate_tap="1.000,0.4000,0.7000",
    )
    assert payload["status"] == "done"
    assert seen["hint"] == pytest.approx((1.0, 0.4, 0.7))
    bar_path = payload["lifts"][0]["bar_path"]
    if bar_path["tracked"]:
        assert bar_path["tapped"] in (True, False)
    else:
        assert "tapped" not in bar_path


def test_a_bright_plate_with_a_full_rim_still_passes_the_gate(tmp_path):
    """
    White and yellow competition bumpers read 160-185 inside: too bright for
    the darkness rule, kept on the completeness of their rim. (The ones on
    the labelled competition clips read 0.75-0.97 around.)
    """
    import cv2  # type: ignore

    from server.utils.plate_oblique import looks_like_a_plate
    from server.utils.bar_track import BarPoint

    clip = str(tmp_path / "bright.mp4")
    writer = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    points = []
    for index in range(45):
        frame = np.full((HEIGHT, WIDTH, 3), 60, np.uint8)
        y = HEIGHT - 100 - index * 6
        cv2.circle(frame, (150, y), 30, (210, 220, 230), -1, cv2.LINE_AA)
        cv2.circle(frame, (150, y), 30, (20, 20, 20), 3, cv2.LINE_AA)
        points.append(BarPoint(t=index / FPS, x=150 / WIDTH, y=y / HEIGHT))
        writer.write(frame)
    writer.release()
    path = BarPath(points=points, plate_radius=30 / WIDTH, coverage=1.0, sampled=45, frame_width=WIDTH, frame_height=HEIGHT)
    ok, why = looks_like_a_plate(clip, path)
    assert ok, why


def test_a_side_on_trace_that_passes_the_gate_is_kept_unless_the_oblique_plate_is_another_bigger_one():
    """
    On a side-on clip both followers find the same plate and the side-on
    follower's finer trace stands, whatever the oblique follower made of the
    plate's shape. It gives way only to an oblique reading of a different,
    clearly bigger plate — the near plate it could not see.
    """
    from server.services.pipeline import _prefer_oblique
    from server.utils.bar_track import BarPoint

    def trace(x, radius, ratio=None, coverage=1.0):
        points = [BarPoint(t=i / 10, x=x, y=0.8 - 0.05 * i) for i in range(10)]
        return BarPath(points=points, plate_radius=radius, coverage=coverage, sampled=10, frame_width=640, frame_height=360, plate_axis_ratio=ratio)

    side = trace(0.40, 0.06)
    assert not _prefer_oblique(None, side)
    assert not _prefer_oblique(BarPath(), side)
    # The same plate, read as oblique: the side-on trace stands.
    assert not _prefer_oblique(trace(0.41, 0.065, ratio=0.7), side)
    # A different plate, bigger, but read in the side-on band (0.88-0.94 on
    # the labelled side-on clips): stands too.
    assert not _prefer_oblique(trace(0.70, 0.09, ratio=0.97), side)
    assert not _prefer_oblique(trace(0.70, 0.09, ratio=0.9), side)
    # A different, bigger plate read as clearly oblique: the near plate.
    assert _prefer_oblique(trace(0.70, 0.09, ratio=0.7), side)
    # Bigger only on paper (within the tolerance): no.
    assert not _prefer_oblique(trace(0.70, 0.07, ratio=0.7), side)
    # Found in too few frames to be trusted over a trace that is on a plate.
    assert not _prefer_oblique(trace(0.70, 0.09, ratio=0.7, coverage=0.6), side)
