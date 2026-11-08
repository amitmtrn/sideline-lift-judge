"""
Continuity: once a follower is on the plate it never switches to something
else. The live bug: on an angled phone clip of a snatch, the trace followed
the tapped near plate through the pull, then jumped to the gymnastics ring
hanging at the top-right of the frame during the turnover — a false 41 cm
drift, a "bar left the frame" warning, and still "following the plate you
tapped".

Synthetic clips only: a filled dark plate that rises and stops (the catch),
a thick dark ring of the same size hanging a couple of radii from the top
of the pull with the wall showing through it, and a bright arm that hides
the plate for a moment at the turnover and sweeps toward the ring, which is
how the optical-flow follower was carried there. The real clip runs only
where SIDELINE_OBLIQUE_CLIP and SIDELINE_OBLIQUE_TRUTH point at it (see
test_amits_clip_stays_on_the_tapped_plate_through_the_catch).
"""
import json
import os
from pathlib import Path

import numpy as np  # type: ignore
import pytest

from server.utils.bar_track import BarPath
from server.utils.segmentation import CandidateSegment

WIDTH, HEIGHT, FPS = 360, 640, 30
PLATE_R = 34
DURATION = 3.0
# The ring: a plate's radius, hanging up and to the right of where the pull
# tops out — within the old walk's reach after a short gap, outside a
# plate's reach at a plate's speed.
RING_DX, RING_DY = 2.5 * PLATE_R, -1.0 * PLATE_R
WALL = 205  # what shows through the ring
# The plate is hidden from HIDE_FROM for HIDE_FOR seconds at the top of the
# pull: shorter than the bridge in the main clips, longer in the lost one.
HIDE_FROM = 1.95
BRIDGE_TEST_HIDE = 0.2
LOST_TEST_HIDE = 0.7


def plate_at(fraction):
    """Still, then floor to the top of the pull by 0.65 of the clip, then still: the catch."""
    rise = min(1.0, max(0.0, (fraction - 0.2) / 0.45))
    x = 150 + 20 * np.sin(np.pi * rise)
    y = HEIGHT - 110 - rise * 0.45 * HEIGHT
    return x, y


def ring_at():
    x, y = plate_at(1.0)
    return int(x + RING_DX), int(y + RING_DY)


def write_ring_clip(path, *, hide_for=BRIDGE_TEST_HIDE, oblique=False, ring_inside=WALL):
    """
    Returns (path, truth) with truth = [(t, x, y)] of the plate's centre.
    `oblique` draws the plate as a 0.6 ellipse for the oblique follower;
    `ring_inside` is the grey inside the ring (the wall, or something dark
    so that only the motion rules can tell the ring from a plate).
    """
    import cv2  # type: ignore

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    rng = np.random.default_rng(7)
    backdrop = np.clip(rng.normal(95, 6, (HEIGHT, WIDTH, 3)), 0, 255).astype(np.uint8)
    # The arm: a sleeve, bright and faintly textured, wide enough to hide the plate.
    arm = np.clip(rng.normal(190, 12, (int(2.8 * PLATE_R), int(2.4 * PLATE_R), 3)), 0, 255).astype(np.uint8)
    for k in range(0, arm.shape[0], 9):
        arm[k : k + 3, :] = 150
    total = int(DURATION * FPS)
    rx, ry = ring_at()
    truth = []
    try:
        for index in range(total):
            t = index / FPS
            fraction = index / (total - 1)
            frame = backdrop.copy()
            # The ring, hollow, hanging in front of a wall that shows through it.
            wall = int(1.6 * PLATE_R)
            cv2.rectangle(frame, (rx - wall, ry - wall), (rx + wall, ry + wall), (ring_inside, ring_inside, ring_inside), -1)
            cv2.circle(frame, (rx, ry), PLATE_R, (30, 30, 30), 7, cv2.LINE_AA)
            x, y = plate_at(fraction)
            centre = (int(round(x)), int(round(y)))
            axes = (int(round(PLATE_R * (0.6 if oblique else 1.0))), PLATE_R)
            cv2.ellipse(frame, centre, axes, 0, 0, 360, (40, 40, 40), -1, cv2.LINE_AA)
            cv2.ellipse(frame, centre, axes, 0, 0, 360, (150, 150, 150), 2, cv2.LINE_AA)
            cv2.circle(frame, centre, 5, (170, 170, 170), -1, cv2.LINE_AA)
            for angle in (0.4, 2.0, 3.6, 5.1):
                mark = (int(centre[0] + 0.55 * axes[0] * np.cos(angle)), int(centre[1] + 0.55 * axes[1] * np.sin(angle)))
                cv2.rectangle(frame, (mark[0] - 3, mark[1] - 3), (mark[0] + 3, mark[1] + 3), (120, 120, 120), -1)
            # The arm: sits over the plate for `hide_for`, then sweeps to
            # the ring in a tenth of a second and is gone.
            if HIDE_FROM <= t < HIDE_FROM + hide_for + 0.1:
                sweep = min(1.0, max(0.0, (t - HIDE_FROM - hide_for) / 0.1))
                ax, ay = int(x + sweep * (rx - x)), int(y + sweep * (ry - y))
                h, w = arm.shape[:2]
                top, left = max(0, ay - h // 2), max(0, ax - w // 2)
                patch = arm[: HEIGHT - top, : WIDTH - left]
                frame[top : top + patch.shape[0], left : left + patch.shape[1]] = patch
            truth.append((t, x, y))
            writer.write(frame)
    finally:
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


def _nearest_to_ring(path: BarPath) -> float:
    rx, ry = ring_at()
    return min(np.hypot(p.x * WIDTH - rx, p.y * HEIGHT - ry) for p in path.points)


@pytest.fixture(scope="module")
def side_on_ring_clip(tmp_path_factory):
    return write_ring_clip(str(tmp_path_factory.mktemp("ring") / "side.mp4"))


@pytest.fixture(scope="module")
def oblique_ring_clip(tmp_path_factory):
    # Dark inside the ring: the darkness gate alone cannot reject it.
    return write_ring_clip(str(tmp_path_factory.mktemp("ring") / "oblique.mp4"), oblique=True, ring_inside=60)


@pytest.fixture(scope="module")
def lost_clip(tmp_path_factory):
    return write_ring_clip(str(tmp_path_factory.mktemp("lost") / "lost.mp4"), hide_for=LOST_TEST_HIDE)


def _on_the_plate(path: BarPath, truth, *, through_the_catch: bool = True) -> None:
    assert path.tracked, path.reason
    errors = _errors(path, truth)
    assert np.mean(errors <= 0.5 * PLATE_R) >= 0.9, f"off the plate: {np.mean(errors <= 0.5 * PLATE_R):.0%} within half a radius"
    assert _nearest_to_ring(path) > 1.5 * PLATE_R, "the trace went to the ring"
    if through_the_catch:
        assert path.points[-1].t >= DURATION - 0.3, f"the trace ends at {path.points[-1].t:.2f} s, before the catch"


def test_the_side_on_follower_bridges_the_hidden_plate_and_never_takes_the_ring(side_on_ring_clip):
    from server.utils.plate_follow import follow_bar_path

    clip, truth = side_on_ring_clip
    _on_the_plate(follow_bar_path(clip, 0.0, DURATION), truth)
    x0, y0 = plate_at(0.0)
    tapped = follow_bar_path(clip, 0.0, DURATION, tap=(0.0, x0 / WIDTH, y0 / HEIGHT))
    _on_the_plate(tapped, truth)
    assert tapped.tapped is True


def test_the_oblique_follower_bridges_the_hidden_plate_and_never_takes_the_ring(oblique_ring_clip):
    from server.utils.plate_oblique import follow_oblique

    clip, truth = oblique_ring_clip
    _on_the_plate(follow_oblique(clip, 0.0, DURATION), truth)
    x0, y0 = plate_at(0.0)
    tapped = follow_oblique(clip, 0.0, DURATION, hint=(0.0, x0 / WIDTH, y0 / HEIGHT))
    _on_the_plate(tapped, truth)
    assert tapped.tapped is True


def test_the_pipeline_trace_stays_on_the_tapped_plate_through_the_catch(side_on_ring_clip):
    from server.services.measurement import measure
    from server.services.pipeline import _trace_for

    clip, truth = side_on_ring_clip
    x0, y0 = plate_at(0.0)
    window = CandidateSegment(index=1, start_sec=0.0, end_sec=DURATION)
    path = _trace_for(clip, window, None, "ring", tap=(0.0, x0 / WIDTH, y0 / HEIGHT))
    _on_the_plate(path, truth)
    assert path.tapped is True and not path.tap_reason
    measurement = measure(path)
    assert measurement.tapped and not measurement.near_frame_edge
    # The drift is the plate's own loop forward, not the ring's two and a half radii.
    assert measurement.max_drift_diameters is not None and measurement.max_drift_diameters < 0.5
    assert "tap_ignored" not in measurement.as_dict()


def test_a_plate_lost_for_longer_than_the_bridge_ends_the_path_there(lost_clip):
    """
    Hidden for longer than the bridge, the plate is lost: the path ends
    where it was last seen, coverage says the catch was not seen (which
    caps the confidence), and nothing is drawn on the ring.
    """
    from server.utils.plate_follow import follow_bar_path
    from server.utils.plate_oblique import follow_oblique

    clip, truth = lost_clip
    for path in (follow_bar_path(clip, 0.0, DURATION), follow_oblique(clip, 0.0, DURATION)):
        assert path.tracked, path.reason
        errors = _errors(path, truth)
        assert np.mean(errors <= 0.5 * PLATE_R) >= 0.9
        assert _nearest_to_ring(path) > 1.5 * PLATE_R
        assert path.points[-1].t <= HIDE_FROM + 0.45, f"the path went on to {path.points[-1].t:.2f} s with the plate hidden"
        assert path.coverage < 0.9


def test_the_tap_lands_on_the_frame_nearest_its_time_or_the_first(side_on_ring_clip):
    """
    `t` is seconds into the file that went up: the trim start when the
    original went up, 0 when the browser re-encoded the trim. A lift window
    that opens later than the tap uses its first frame; one that contains
    the tap's moment uses that frame.
    """
    from server.utils.plate_follow import read_gray_frames
    from server.utils.plate_oblique import tap_frame

    clip, _ = side_on_ring_clip
    frames = read_gray_frames(clip, 0.8, DURATION)
    assert frames[0][0] >= 0.75
    assert tap_frame(frames, 0.0) == 0  # the tap was before the window: its first frame
    assert tap_frame(frames, 1.2) == min(range(len(frames)), key=lambda i: abs(frames[i][0] - 1.2))
    assert tap_frame(frames, 99.0) == 0


def test_a_tap_on_nothing_is_reported_as_not_used(side_on_ring_clip):
    from server.services.measurement import measure
    from server.services.pipeline import _trace_for

    clip, truth = side_on_ring_clip
    window = CandidateSegment(index=1, start_sec=0.0, end_sec=DURATION)
    path = _trace_for(clip, window, None, "ring", tap=(0.0, 0.92, 0.08))  # the empty top-right corner
    _on_the_plate(path, truth)
    assert path.tapped is False
    assert path.tap_reason, "a tap that was not followed must say why"
    body = measure(path).as_dict()
    assert body["tap_ignored"] is True and body["tap_reason"] == path.tap_reason and body["tapped"] is False


# ---------- the real clip, where it is ----------

CLIP = os.getenv("SIDELINE_OBLIQUE_CLIP") or ""
TRUTH = os.getenv("SIDELINE_OBLIQUE_TRUTH") or ""
needs_the_clip = pytest.mark.skipif(
    not (CLIP and TRUTH and Path(CLIP).is_file() and Path(TRUTH).is_file()),
    reason="set SIDELINE_OBLIQUE_CLIP and SIDELINE_OBLIQUE_TRUTH to run on the real clip (never committed)",
)


@needs_the_clip
def test_amits_clip_stays_on_the_tapped_plate_through_the_catch(tmp_path):
    """
    SIDELINE_OBLIQUE_TRUTH is a JSON file:
        {"trims": [[18.3, 21.6], [17.6, 22.0]],   # seconds into the clip
         "tap": [0.41, 0.83],                       # the near plate on the first trimmed frame, 0..1 of the displayed frame
         "max_drift_cm": 15,
         "ring": [0.9, 0.1]}                        # optional: where the ring is, 0..1 of the displayed frame
    With SIDELINE_OBLIQUE_OUT set, the annotated replay of each trim is
    written there for a look.
    """
    from server.services.measurement import measure
    from server.services.pipeline import _tap_in_decoded_frame, _trace_for
    from server.utils.bar_track import detect_lift_windows
    from server.utils.video_meta import probe_video

    truth = json.loads(Path(TRUTH).read_text())
    info = probe_video(CLIP)
    tap_x, tap_y = truth["tap"]
    out_dir = os.getenv("SIDELINE_OBLIQUE_OUT") or ""
    for start, end in truth["trims"]:
        windows = detect_lift_windows(CLIP, start_sec=start, end_sec=end)
        assert windows, f"no lift window in {start}-{end} s"
        window = max(windows, key=lambda w: w.end_sec - w.start_sec)
        segment = CandidateSegment(index=1, start_sec=window.start_sec, end_sec=window.end_sec, peak_sec=window.peak_sec)
        path = _trace_for(CLIP, segment, window, f"trim {start}-{end}", tap=(start, tap_x, tap_y), info=info)
        assert path.tracked, f"{start}-{end}: {path.reason}"
        assert path.tapped, f"{start}-{end}: the tap was not followed ({path.tap_reason})"
        measurement = measure(path)
        assert measurement.max_drift_cm is not None and measurement.max_drift_cm < truth.get("max_drift_cm", 15)
        assert not measurement.near_frame_edge, "the trace went to the edge of the frame"
        if truth.get("ring"):
            ring = _tap_in_decoded_frame((0.0, truth["ring"][0], truth["ring"][1]), info, path.frame_width, path.frame_height)
            nearest = min(np.hypot((p.x - ring[1]) * path.frame_width, (p.y - ring[2]) * path.frame_height) for p in path.points)
            assert nearest > 1.5 * path.plate_radius * path.frame_width, "the trace went to the ring"
        if out_dir:
            from server.services.replay import write_trace_overlay

            Path(out_dir).mkdir(parents=True, exist_ok=True)
            write_trace_overlay(path, path.frame_width, path.frame_height, str(Path(out_dir) / f"trace_{start}_{end}.png"))
