"""
Rotation metadata (AC-5).

ffprobe spells the side-data type "Display Matrix". Comparing it against
"displaymatrix" silently matched nothing, so every rotated phone clip reported
0°, `is_portrait` was always False and FR-8's portrait penalty never fired.
"""
import shutil

import pytest

from server.utils.video_meta import (
    VideoInfo,
    _is_display_matrix,
    _rotation_from_stream,
    normalize_rotation,
    probe_video,
    rotation_still_needed,
)

needs_ffprobe = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is not installed")


@pytest.mark.parametrize(
    "value",
    ["Display Matrix", "displaymatrix", "DISPLAY MATRIX", "display_matrix", "Display  Matrix"],
)
def test_display_matrix_spellings_all_match(value):
    assert _is_display_matrix(value)


@pytest.mark.parametrize("value", ["Stereo 3D", "", None, 42, "matrix"])
def test_other_side_data_does_not_match(value):
    assert not _is_display_matrix(value)


def test_rotation_read_from_display_matrix():
    # Exactly the shape ffprobe returns for an iPhone portrait clip.
    stream = {
        "side_data_list": [
            {"side_data_type": "Display Matrix", "displaymatrix": "...", "rotation": -90}
        ]
    }
    assert _rotation_from_stream(stream) == 270


def test_rotate_tag_still_wins_when_present():
    stream = {"tags": {"rotate": "180"}, "side_data_list": []}
    assert _rotation_from_stream(stream) == 180


def test_no_rotation_metadata_is_zero():
    assert _rotation_from_stream({}) == 0
    assert _rotation_from_stream({"side_data_list": [{"side_data_type": "Stereo 3D"}]}) == 0


@pytest.mark.parametrize("deg,expected", [(-90, 270), (90, 90), (270, 270), (360, 0), (-180, 180)])
def test_normalize_rotation(deg, expected):
    assert normalize_rotation(deg) == expected


def test_portrait_is_decided_after_rotation():
    # Coded landscape, quarter-turned for display: that is a portrait clip.
    info = VideoInfo(width=1920, height=1080, rotation=270, probed=True)
    assert (info.display_width, info.display_height) == (1080, 1920)
    assert info.is_portrait

    upright = VideoInfo(width=1920, height=1080, rotation=0, probed=True)
    assert not upright.is_portrait

    square = VideoInfo(width=1080, height=1080, rotation=0, probed=True)
    assert not square.is_portrait


class TestRotationStillNeeded:
    """Guards against rotating a frame the decoder already rotated."""

    info = VideoInfo(width=1920, height=1080, rotation=270, probed=True)

    def test_decoder_already_applied_it(self):
        # OpenCV handed back display-sized frames: do nothing more.
        assert rotation_still_needed(self.info, 1080, 1920) == 0

    def test_decoder_left_it_raw(self):
        assert rotation_still_needed(self.info, 1920, 1080) == 270

    def test_unexpected_size_is_left_alone(self):
        # Better an unrotated frame than one rotated twice.
        assert rotation_still_needed(self.info, 640, 480) == 0

    def test_no_rotation_metadata(self):
        upright = VideoInfo(width=1920, height=1080, rotation=0, probed=True)
        assert rotation_still_needed(upright, 1920, 1080) == 0

    def test_unprobed_info(self):
        assert rotation_still_needed(VideoInfo(), 1080, 1920) == 0


@needs_ffprobe
def test_probe_reads_a_real_file(lift_video):
    info = probe_video(lift_video)
    assert info.probed
    assert info.width == 96 and info.height == 72
    assert info.duration_sec > 0
    assert not info.is_portrait


def test_duration_and_fps_are_read_off_the_decoded_frames(tmp_path):
    """
    A browser MediaRecorder upload carries no trustworthy duration or frame
    rate in its container. The decoder's own presentation times are the
    clock everything else runs on, so that is what the probe reports.
    """
    from tests.conftest import write_lift_video
    from server.utils.video_meta import _measure_by_decoding, _probe_with_opencv

    clip = write_lift_video(str(tmp_path / "sixty.mp4"), duration=2.0, fps=60)
    measured = _measure_by_decoding(clip)
    assert measured is not None
    duration, fps, frames = measured
    assert abs(fps - 60.0) < 3.0, fps
    assert abs(duration - 2.0) < 0.1, duration
    assert frames == 120

    info = _probe_with_opencv(clip)
    assert info.probed
    assert abs(info.fps - 60.0) < 3.0
    assert abs(info.duration_sec - 2.0) < 0.1


def test_probe_of_a_missing_file_degrades(tmp_path):
    info = probe_video(str(tmp_path / "nope.mp4"))
    assert not info.probed
    assert info.rotation == 0 and info.duration_sec == 0.0
    assert not info.is_portrait
