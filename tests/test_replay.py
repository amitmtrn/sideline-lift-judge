"""
FR-10 / AC-6: the annotated replay must be an MP4 iOS Safari will play, and
every overlay (score, cue, trace, drift) must be visibly burnt in — including
on a dyno whose only ffmpeg is the imageio-ffmpeg wheel's bundled binary,
which carries no `drawtext` filter. The tests here work the renderer under
that constraint on purpose, and compare pixels so a silently-plain fallback
cannot pass.
"""
import json
import shutil
import subprocess

import pytest

from server.schemas.judgment import LiftJudgment
from server.services.replay import (
    output_size,
    MAX_LONG_SIDE,
    _scale_filter,
    build_full_overlay,
    render_annotated_replay,
    wrap_text,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def judgment(**overrides) -> LiftJudgment:
    body = dict(
        lift="snatch",
        score=4,
        band="poor",
        max_horizontal_drift="bar loops forward off the floor",
        cues=["keep the bar close to the body"],
        confidence=0.45,
    )
    body.update(overrides)
    return LiftJudgment(**body)


def test_wrap_text_breaks_on_words():
    lines = wrap_text("keep the bar close to the body", 12)
    assert all(len(line) <= 12 for line in lines)
    assert " ".join(lines) == "keep the bar close to the body"


def test_wrap_text_handles_empty_and_long_words():
    assert wrap_text("", 10) == []
    assert wrap_text("supercalifragilistic", 5) == ["supercalifragilistic"]


def test_output_size_caps_the_long_side_and_stays_even():
    # Explicit sizes, not ffmpeg's -2, so the trace overlay can be built at
    # exactly the encoded size and composited at 0:0.
    assert output_size(1920, 1080) == (MAX_LONG_SIDE, 404)
    assert output_size(1080, 1920) == (404, MAX_LONG_SIDE)
    for width, height in [(1920, 1080), (1080, 1920), (1366, 768), (640, 361), (101, 203)]:
        out_w, out_h = output_size(width, height)
        assert out_w % 2 == 0 and out_h % 2 == 0, (width, height)
        assert max(out_w, out_h) <= MAX_LONG_SIDE
        # Aspect ratio preserved to within the even-rounding.
        assert out_w / out_h == pytest.approx(width / height, rel=0.02)


def test_output_size_leaves_a_small_clip_alone():
    assert output_size(320, 240) == (320, 240)


def test_output_size_without_dimensions():
    assert output_size(0, 0) is None


def test_scale_filter_uses_the_computed_size():
    assert _scale_filter(1920, 1080) == "scale=720:404"
    assert _scale_filter(0, 0) == "scale=trunc(iw/2)*2:trunc(ih/2)*2"


def test_full_overlay_draws_header_and_footer_regions():
    """
    The pre-rendered overlay must put opaque pixels in the header strip (top)
    and the footer strip (bottom when a cue is present). The middle stays
    mostly transparent so the picture reads through.
    """
    import numpy as np  # type: ignore

    overlay = build_full_overlay(judgment(), bar_path=None, width=400, height=720)
    assert overlay is not None
    assert overlay.shape == (720, 400, 4)
    alpha = overlay[:, :, 3]
    # Top 10% must have overlay pixels; so must the bottom 10%.
    assert alpha[:72].max() > 0, "header strip never drew"
    assert alpha[-72:].max() > 0, "footer strip never drew"
    # Middle band should be mostly transparent — a bar-less overlay only has
    # the two banners.
    assert alpha[200:400].max() == 0


def test_full_overlay_omits_footer_when_cues_are_empty():
    """With no cue, the footer banner must not be drawn — only the header is."""
    from server.schemas.judgment import LiftType

    # LiftJudgment forbids an empty `cues` list at the schema; use
    # model_construct to bypass validation and exercise the defensive branch.
    cueless = LiftJudgment.model_construct(
        lift=LiftType.snatch, score=4, band="poor", max_horizontal_drift="forward loop",
        cues=[], confidence=0.45,
    )
    overlay = build_full_overlay(cueless, bar_path=None, width=400, height=720)
    assert overlay is not None
    alpha = overlay[:, :, 3]
    assert alpha[:72].max() > 0
    assert alpha[-72:].max() == 0


@needs_ffmpeg
def test_render_produces_an_ios_safe_mp4(lift_video, tmp_path):
    out = tmp_path / "replay.mp4"
    written, annotated = render_annotated_replay(
        lift_video, judgment(), str(out), start_sec=1.0, end_sec=3.0
    )
    assert written is not None, "ffmpeg produced no replay"
    assert annotated is True, "the full overlay should have been drawn"
    assert out.exists() and out.stat().st_size > 1024

    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height:format=duration,format_name",
                "-of",
                "json",
                str(out),
            ]
        )
    )
    video = next(stream for stream in probe["streams"] if stream.get("codec_name"))
    assert video["codec_name"] == "h264"
    assert video["pix_fmt"] == "yuv420p"
    # Odd dimensions cannot be encoded as yuv420p; the scaler must have fixed them.
    assert video["width"] % 2 == 0 and video["height"] % 2 == 0
    assert "mp4" in probe["format"]["format_name"]
    # Only the requested window was encoded.
    assert float(probe["format"]["duration"]) < 2.6


def test_render_without_ffmpeg_returns_none(lift_video, tmp_path, monkeypatch):
    monkeypatch.setattr("server.services.replay.ffmpeg_exe", lambda: None)
    path, annotated = render_annotated_replay(lift_video, judgment(), str(tmp_path / "none.mp4"))
    assert path is None and annotated is False


def test_render_never_builds_a_drawtext_command(lift_video, tmp_path, monkeypatch):
    """
    The imageio-ffmpeg bundled binary has no `drawtext` filter, and the
    static ffmpeg on Heroku's apt buildpack sometimes does not either. Any
    command we build that mentions drawtext is a silent-fallback trap — the
    render module must not emit one, in any attempt, in any clip.
    """
    recorded = []

    def _run(args, timeout):  # noqa: ARG001
        recorded.append(list(args))
        return False

    original_popen = subprocess.Popen

    def tracking_popen(args, *rest, **kw):
        recorded.append(list(args))
        return original_popen(args, *rest, **kw)

    monkeypatch.setattr("server.services.replay._run", _run)
    monkeypatch.setattr("server.services.replay.subprocess.Popen", tracking_popen)
    render_annotated_replay(lift_video, judgment(), str(tmp_path / "nope.mp4"))
    assert recorded, "render should have attempted at least one encode"
    for args in recorded:
        flat = " ".join(args)
        assert "drawtext" not in flat, args


def test_render_uses_imageio_ffmpeg_and_burns_the_overlay(tmp_path, monkeypatch):
    """
    Hide the system ffmpeg so only imageio-ffmpeg answers. The written clip
    must be portrait (rotation respected), play as H.264 + yuv420p + faststart,
    and visibly carry the overlay — the mean pixel of the header strip after
    the render must be far from the solid gym-floor grey of the source. A
    drawtext-free overlay that still works is the whole point of this fix.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    from tests.conftest import write_lift_video
    from server.utils import ffmpeg_binary

    source = write_lift_video(str(tmp_path / "portrait.mp4"), width=96, height=160, duration=2.0)
    ffmpeg_binary.reset_cache_for_tests()
    monkeypatch.setattr("server.utils.ffmpeg_binary.shutil.which", lambda name: None)
    assert ffmpeg_binary.ffmpeg_exe() is not None, "imageio-ffmpeg must provide a binary"

    out = tmp_path / "replay.mp4"
    written, annotated = render_annotated_replay(
        source, judgment(), str(out), start_sec=0.0, end_sec=1.5
    )
    assert written is not None, "imageio-ffmpeg should have produced a replay"
    assert annotated is True, "overlay annotations should have been burnt in"
    assert out.exists() and out.stat().st_size > 1024

    # Decode the first frame of the source and the replay, scaled to the same
    # output size, so the overlay's pixel diff is isolated to the overlay bands.
    source_cap = cv2.VideoCapture(source)
    reply_cap = cv2.VideoCapture(str(out))
    try:
        _, source_frame = source_cap.read()
        _, reply_frame = reply_cap.read()
    finally:
        source_cap.release()
        reply_cap.release()
    assert source_frame is not None and reply_frame is not None

    out_h, out_w = reply_frame.shape[:2]
    assert out_h > out_w, (out_w, out_h)  # portrait preserved
    source_resized = cv2.resize(source_frame, (out_w, out_h))

    header = reply_frame[: max(6, out_h // 15)]
    header_src = source_resized[: max(6, out_h // 15)]
    diff = float(np.mean(np.abs(header.astype(np.int16) - header_src.astype(np.int16))))
    assert diff > 20, f"header overlay was not visibly burnt in (mean abs diff {diff:.1f})"
    ffmpeg_binary.reset_cache_for_tests()


# ---------- the bar-path trace (§18) ----------


def test_furthest_marker_appears_only_once_the_bar_gets_there():
    """
    The orange furthest-point marker used to be drawn from frame 0, giving
    the plot's answer away before the lift started. It must appear at the
    moment the bar reaches that point and stay from then on.
    """
    import numpy as np  # type: ignore
    from server.services.replay import DRIFT_ORANGE, _draw_progressive_trace

    path = tracked_path(drift=0.2)
    _drift, worst = path.max_drift()
    assert worst is not None and worst.t == 0.5

    def orange_pixels(at_t: float) -> int:
        frame = np.full((720, 404, 3), 128, dtype=np.uint8)
        _draw_progressive_trace(frame, path, at_t, 404, 720)
        hit = np.all(frame == np.array(DRIFT_ORANGE[:3], dtype=np.uint8), axis=-1)
        return int(hit.sum())

    assert orange_pixels(0.3) == 0, "marker drawn before the bar reached its furthest point"
    assert orange_pixels(0.5) > 0
    assert orange_pixels(0.9) > 0


def tracked_path(drift: float = 0.12):
    """A short rising path that loops sideways by `drift` of the frame width."""
    from server.utils.bar_track import BarPath, BarPoint

    points = [
        BarPoint(t=i / 10, x=0.30 + drift * (1 if i == 5 else 0), y=0.90 - 0.07 * i)
        for i in range(10)
    ]
    return BarPath(
        points=points, plate_radius=0.08, coverage=1.0, sampled=10,
        frame_width=404, frame_height=720,
    )


def test_trace_overlay_draws_only_when_tracked():
    from server.services.replay import build_trace_overlay
    from server.utils.bar_track import BarPath

    assert build_trace_overlay(BarPath(), 404, 720) is None
    assert build_trace_overlay(tracked_path(), 0, 0) is None
    assert build_trace_overlay(tracked_path(), 404, 720) is not None


def test_trace_overlay_is_transparent_except_where_it_draws():
    import numpy as np  # type: ignore
    from server.services.replay import build_trace_overlay

    layer = build_trace_overlay(tracked_path(), 404, 720)
    assert layer.shape == (720, 404, 4)
    alpha = layer[:, :, 3]
    painted = float((alpha > 0).mean())
    # Enough to see, nowhere near a full-frame wash that would hide the lift.
    assert 0.001 < painted < 0.2


def test_trace_overlay_marks_the_worst_drift():
    import numpy as np  # type: ignore
    from server.services.replay import DRIFT_ORANGE, build_trace_overlay

    layer = build_trace_overlay(tracked_path(drift=0.2), 404, 720)
    orange = np.all(layer[:, :, :3] == np.array(DRIFT_ORANGE[:3]), axis=-1) & (layer[:, :, 3] > 0)
    assert orange.any(), "the max-drift marker was not drawn"
    # It sits out at the loop, not back on the starting line.
    xs = np.where(orange.any(axis=0))[0]
    assert xs.max() > 0.40 * 404


@needs_ffmpeg
def test_render_burns_the_trace_into_the_video(lift_video, tmp_path):
    import numpy as np  # type: ignore

    plain_path, _ = render_annotated_replay(
        lift_video, judgment(), str(tmp_path / "plain.mp4")
    )
    traced_path, _ = render_annotated_replay(
        lift_video, judgment(), str(tmp_path / "traced.mp4"), bar_path=tracked_path()
    )
    assert plain_path is not None and traced_path is not None

    def mid_frame(path):
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(path))
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
            ok, frame = capture.read()
            assert ok
            return frame
        finally:
            capture.release()

    before, after = mid_frame(plain_path), mid_frame(traced_path)
    assert before.shape == after.shape
    # The trace must have changed pixels in a region neither banner covers.
    middle = slice(after.shape[0] // 4, 3 * after.shape[0] // 4)
    assert not np.array_equal(before[middle], after[middle]), "no trace in the frame"


@needs_ffmpeg
def test_render_cleans_up_the_overlay_pngs(lift_video, tmp_path):
    out = tmp_path / "replay.mp4"
    written, _ = render_annotated_replay(
        lift_video, judgment(), str(out), bar_path=tracked_path()
    )
    assert written is not None
    assert list(tmp_path.glob("*.png")) == [], "temporary overlay PNG was left behind"


# ---------- the evidence strip ----------


def evidenced_cue(t_start: float = 0.2, t_end: float = 0.8, value=12.0):
    from server.schemas.judgment import CueEvidence, EvidencedCue

    return EvidencedCue(
        id="c1",
        text="keep the bar close",
        phase="second_pull",
        issue="bar_forward_drift",
        evidence=CueEvidence(t_start_s=t_start, t_end_s=t_end, metric="drift_cm", value=value),
    )


def test_strip_times_span_the_window_and_include_the_key_moment():
    from server.services.replay import _strip_times

    times = _strip_times(0.2, 0.8, 0.47, 5)
    assert len(times) == 5 and times == sorted(times)
    assert times[0] == 0.2 and times[-1] == 0.8
    assert 0.47 in times
    # A key moment outside the window is not forced in.
    assert _strip_times(0.2, 0.8, 3.0, 5) == _strip_times(0.2, 0.8, None, 5)


def test_strip_is_a_row_of_frames_with_the_key_one_outlined(lift_video, tmp_path):
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from server.services.replay import STRIP_RED, STRIP_TILE_HEIGHT, STRIP_TILES, render_cue_strip

    out = render_cue_strip(lift_video, evidenced_cue(), tracked_path(), str(tmp_path / "strip.jpg"))
    assert out is not None
    strip = cv2.imread(out)
    assert strip.shape[0] == STRIP_TILE_HEIGHT
    tile_w = (strip.shape[1] - 4 * (STRIP_TILES - 1)) / STRIP_TILES
    assert tile_w == int(tile_w) and tile_w > 0
    # The worst drift is at t=0.5 — the third of five frames over 0.2-0.8 s —
    # and that tile, alone, carries the red border.
    def border_is_red(k):
        left = int(k * (tile_w + 4))
        pixel = strip[STRIP_TILE_HEIGHT // 2, left + 1].astype(int)
        return abs(pixel - np.array(STRIP_RED)).max() < 60
    assert [border_is_red(k) for k in range(STRIP_TILES)] == [False, False, True, False, False]


def test_strip_without_a_trace_still_shows_the_frames(lift_video, tmp_path):
    import cv2  # type: ignore
    from server.services.replay import STRIP_TILE_HEIGHT, render_cue_strip
    from server.utils.bar_track import BarPath

    for path in (None, BarPath()):
        out = render_cue_strip(lift_video, evidenced_cue(value=None), path, str(tmp_path / "bare.jpg"))
        assert out is not None
        assert cv2.imread(out).shape[0] == STRIP_TILE_HEIGHT


def test_strip_needs_a_readable_video_and_a_window(tmp_path, lift_video):
    from server.services.replay import render_cue_strip
    from server.schemas.judgment import EvidencedCue

    missing = str(tmp_path / "nope.mp4")
    assert render_cue_strip(missing, evidenced_cue(), tracked_path(), str(tmp_path / "a.jpg")) is None
    bare = EvidencedCue(id="c1", text="x", phase="pull", issue="unknown")
    assert render_cue_strip(lift_video, bare, tracked_path(), str(tmp_path / "b.jpg")) is None


def test_evidence_uses_the_strip_and_says_so(lift_video, tmp_path):
    from server.services.measurement import measure
    from server.services.pipeline import _build_evidence

    path = tracked_path()
    built = _build_evidence(
        ["keep the bar close"], measure(path), path, "snatch", 0.0, 0.9,
        lift_video, str(tmp_path), 1,
    )
    assert built, "the cue was dropped"
    for cue, image in built:
        assert image is not None and image.endswith(".jpg")
        assert cue.media is not None and cue.media.layout == "strip"
