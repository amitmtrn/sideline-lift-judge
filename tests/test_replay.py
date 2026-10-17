"""FR-10 / AC-6: the annotated replay must be an MP4 iOS Safari will play."""
import json
import shutil
import subprocess

import pytest

from app.schemas.judgment import LiftJudgment
from app.services.replay import (
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


def test_scale_filter_keeps_dimensions_even():
    assert _scale_filter(1920, 1080) == f"scale={MAX_LONG_SIDE}:-2"
    assert _scale_filter(1080, 1920) == f"scale=-2:{MAX_LONG_SIDE}"
    # Already small: only force even dimensions.
    assert "trunc" in _scale_filter(320, 240)


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
    monkeypatch.setattr("app.services.replay.find_font", lambda: None)
    out = tmp_path / "plain.mp4"
    assert render_annotated_replay(lift_video, judgment(), str(out)) is not None
    assert out.exists()


def test_render_without_ffmpeg_returns_none(lift_video, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.replay.shutil.which", lambda name: None)
    assert render_annotated_replay(lift_video, judgment(), str(tmp_path / "none.mp4")) is None


def test_find_font_returns_an_existing_file_or_none():
    font = find_font()
    assert font is None or font.endswith((".ttf", ".ttc"))
