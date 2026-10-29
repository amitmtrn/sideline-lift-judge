"""FR-10 / AC-6: the annotated replay must be an MP4 iOS Safari will play."""
import json
import shutil
import subprocess

import pytest

from server.schemas.judgment import LiftJudgment
from server.services.replay import (
    output_size,
    MAX_LONG_SIDE,
    _scale_filter,
    build_overlay_filters,
    escape_drawtext,
    find_font,
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


def test_escape_drawtext_neutralises_filtergraph_syntax():
    escaped = escape_drawtext("score: 4/10, 90% 'close' [x]")
    for char in (":", ",", "'", "[", "]"):
        assert f"\\{char}" in escaped
    assert "\n" not in escape_drawtext("two\nlines")


def test_escape_drawtext_leaves_percent_alone():
    # Escaping % makes drawtext's expander complain about a stray %; the filters
    # set expansion=none instead, which is what keeps it literal.
    assert escape_drawtext("confidence 72%") == "confidence 72%"


def test_every_drawtext_disables_expansion():
    filters = build_overlay_filters(judgment(), font="/tmp/font.ttf", display_width=640, display_height=360)
    drawtexts = [item for item in filters if item.startswith("drawtext")]
    assert drawtexts
    assert all("expansion=none" in item for item in drawtexts)


def test_escape_drawtext_escapes_backslashes_first():
    # A naive implementation turns "a\b" into "a\\\\b" by re-escaping its own
    # escapes; the backslash must be doubled exactly once.
    assert escape_drawtext("a\\b") == "a\\\\b"


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


def test_overlay_mentions_score_band_and_cue():
    filters = build_overlay_filters(judgment(), font="/tmp/font.ttf", display_width=640, display_height=360)
    chain = " ".join(filters)
    assert "SNATCH" in chain and "4/10" in chain and "POOR" in chain
    assert "confidence" in chain
    assert "close" in chain  # the cue made it into the footer
    assert chain.count("drawbox") == 2


def test_overlay_without_cues_has_no_footer():
    filters = build_overlay_filters(
        judgment(band="unknown", score=None, cues=[]),
        font="/tmp/font.ttf",
        display_width=640,
        display_height=360,
    )
    assert sum(1 for item in filters if item.startswith("drawbox")) == 1


@needs_ffmpeg
def test_render_produces_an_ios_safe_mp4(lift_video, tmp_path):
    out = tmp_path / "replay.mp4"
    written = render_annotated_replay(lift_video, judgment(), str(out), start_sec=1.0, end_sec=3.0)
    assert written is not None, "ffmpeg produced no replay"
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


@needs_ffmpeg
@pytest.mark.skipif(find_font() is None, reason="no font available for drawtext")
def test_ffmpeg_accepts_the_overlay_filtergraph(tmp_path):
    """
    The renderer falls back to an un-annotated clip when the filtergraph is bad,
    so a syntax error shows up only as a silently plain replay. Run the real
    graph and insist ffmpeg is happy with it.
    """
    chain = ",".join(
        [
            _scale_filter(640, 360),
            *build_overlay_filters(
                judgment(cues=["bar close into the hips; contact then explode up"]),
                font=find_font(),
                display_width=640,
                display_height=360,
            ),
        ]
    )
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=gray:s=640x360:d=1",
            "-vf", chain, "-frames:v", "1", str(tmp_path / "frame.png"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ffmpeg rejected the overlay filtergraph:\n{result.stderr}"
    assert result.stderr.strip() == ""
    assert (tmp_path / "frame.png").exists()


@needs_ffmpeg
@pytest.mark.skipif(find_font() is None, reason="no font available for drawtext")
def test_rendered_replay_actually_carries_the_overlay(lift_video, tmp_path):
    """The header strip must darken the top of the frame — proof it was drawn."""
    import numpy as np  # type: ignore

    plain = tmp_path / "plain.mp4"
    annotated = tmp_path / "annotated.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", lift_video, "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-an", str(plain)],
        check=True,
    )
    assert render_annotated_replay(lift_video, judgment(), str(annotated)) is not None

    def first_frame(path):
        import cv2  # type: ignore

        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
            assert ok
            return frame
        finally:
            capture.release()

    before, after = first_frame(plain), first_frame(annotated)
    strip = slice(0, max(4, after.shape[0] // 12))
    assert float(np.mean(after[strip])) < float(np.mean(before[strip])), "header strip was not drawn"


@needs_ffmpeg
def test_render_falls_back_when_the_font_is_missing(lift_video, tmp_path, monkeypatch):
    monkeypatch.setattr("server.services.replay.find_font", lambda: None)
    out = tmp_path / "plain.mp4"
    assert render_annotated_replay(lift_video, judgment(), str(out)) is not None
    assert out.exists()


def test_render_without_ffmpeg_returns_none(lift_video, tmp_path, monkeypatch):
    monkeypatch.setattr("server.services.replay.ffmpeg_exe", lambda: None)
    assert render_annotated_replay(lift_video, judgment(), str(tmp_path / "none.mp4")) is None


def test_render_uses_imageio_ffmpeg_when_path_has_none(tmp_path, monkeypatch):
    """
    The pip fallback carries the deploy on Heroku's Python buildpack by itself:
    without a system ffmpeg, the imageio-ffmpeg wheel's bundled binary should
    be picked up and produce a playable MP4. The clip is portrait so the test
    also pins the "respect rotation metadata" guarantee: a portrait source goes
    in, a portrait replay comes out.
    """
    from tests.conftest import write_lift_video
    from server.utils import ffmpeg_binary

    portrait = write_lift_video(str(tmp_path / "portrait.mp4"), width=96, height=160, duration=2.0)
    # Hide the system ffmpeg from both the binary resolver and anywhere else
    # the module might still call shutil.which. The imageio wheel's bundled
    # binary stays reachable because it does not go through PATH.
    ffmpeg_binary.reset_cache_for_tests()
    monkeypatch.setattr("server.utils.ffmpeg_binary.shutil.which", lambda name: None)
    assert ffmpeg_binary.ffmpeg_exe() is not None, "imageio-ffmpeg must provide a binary"

    out = tmp_path / "replay.mp4"
    written = render_annotated_replay(portrait, judgment(), str(out), start_sec=0.0, end_sec=1.5)
    assert written is not None, "imageio-ffmpeg should have produced a replay"
    assert out.exists() and out.stat().st_size > 1024

    # The first bytes of an iOS-safe MP4 are an `ftyp` box with a brand that
    # the Safari player recognises. mp42/isom are the usual ones ffmpeg writes.
    head = out.read_bytes()[:32]
    assert b"ftyp" in head, head

    # Replay should respect portrait orientation: the output is taller than wide.
    # Decode the first frame with OpenCV; if ffprobe is on PATH, double-check it.
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(out))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    assert height > width, (width, height)
    ffmpeg_binary.reset_cache_for_tests()


def test_find_font_returns_an_existing_file_or_none():
    font = find_font()
    assert font is None or font.endswith((".ttf", ".ttc"))


# ---------- the bar-path trace (§18) ----------


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
@pytest.mark.skipif(find_font() is None, reason="no font available for drawtext")
def test_render_burns_the_trace_into_the_video(lift_video, tmp_path):
    import numpy as np  # type: ignore

    plain = tmp_path / "plain.mp4"
    traced = tmp_path / "traced.mp4"
    assert render_annotated_replay(lift_video, judgment(), str(plain)) is not None
    assert render_annotated_replay(lift_video, judgment(), str(traced), bar_path=tracked_path()) is not None

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

    before, after = mid_frame(plain), mid_frame(traced)
    assert before.shape == after.shape
    # The trace must have changed pixels the text overlay does not touch.
    middle = slice(after.shape[0] // 4, 3 * after.shape[0] // 4)
    assert not np.array_equal(before[middle], after[middle]), "no trace in the frame"


@needs_ffmpeg
def test_render_cleans_up_the_trace_png(lift_video, tmp_path):
    out = tmp_path / "replay.mp4"
    assert render_annotated_replay(lift_video, judgment(), str(out), bar_path=tracked_path()) is not None
    assert list(tmp_path.glob("*.png")) == [], "temporary trace overlay was left behind"


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
