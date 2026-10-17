"""FR-8 / AC-4: a bad clip must come back with visibly lower confidence."""
import pytest

from server.schemas.judgment import LiftJudgment
from server.services.confidence import (
    CAP_MULTIPLE_LIFTS,
    CAP_OFF_AXIS,
    CAP_PORTRAIT,
    ClipQuality,
    apply_quality,
    assess_clip_quality,
    merge_notes,
)
from server.utils.video_meta import VideoInfo


def judgment(confidence=0.85, notes="Bar drifts forward.") -> LiftJudgment:
    return LiftJudgment(
        lift="clean",
        score=3,
        band="poor",
        max_horizontal_drift="forward at the catch",
        cues=["keep the bar close to the body"],
        confidence=confidence,
        notes=notes,
    )


def portrait() -> VideoInfo:
    return VideoInfo(width=1080, height=1920, rotation=0, duration_sec=6.0, probed=True)


def landscape() -> VideoInfo:
    return VideoInfo(width=1920, height=1080, rotation=0, duration_sec=6.0, probed=True)


def test_landscape_side_view_clip_is_not_penalised():
    quality = assess_clip_quality(
        observations="Bar stays close to the body through all phases; catch over mid-foot.",
        info=landscape(),
        frame_count=10,
        duration_sec=6.0,
    )
    assert quality.confidence_cap is None
    assert quality.warnings == []
    assert apply_quality(judgment(0.8), quality).confidence == 0.8


def test_mixed_clean_and_jerk_caps_hard():
    # The spec's §10 regression: 0.85 on a clip that held both lifts.
    quality = assess_clip_quality(
        observations="catch overhead, legs split",
        info=landscape(),
        frame_count=10,
        duration_sec=6.0,
        lifts_in_window=2,
        was_split=False,
    )
    assert quality.confidence_cap == CAP_MULTIPLE_LIFTS
    result = apply_quality(judgment(0.85), quality)
    assert result.confidence == CAP_MULTIPLE_LIFTS
    assert result.confidence < 0.5
    assert any("more than one lift" in warning for warning in result.warnings)
    assert "more than one lift" in (result.notes or "")


def test_portrait_video_lowers_confidence_and_warns():
    quality = assess_clip_quality(observations="Bar close.", info=portrait(), frame_count=10, duration_sec=6.0)
    assert quality.confidence_cap == CAP_PORTRAIT
    result = apply_quality(judgment(0.9), quality)
    assert result.confidence == CAP_PORTRAIT
    assert any("portrait" in warning.lower() for warning in result.warnings)


def test_oblique_camera_angle_lowers_confidence():
    quality = assess_clip_quality(
        observations="The camera is at an oblique angle to the platform.",
        info=landscape(),
        frame_count=10,
        duration_sec=6.0,
    )
    assert quality.confidence_cap == CAP_OFF_AXIS
    assert any("square to the side" in warning for warning in quality.warnings)


def test_worst_problem_wins_when_several_apply():
    quality = assess_clip_quality(
        observations="Oblique view, the bar is not visible in some frames.",
        info=portrait(),
        frame_count=10,
        duration_sec=6.0,
        lifts_in_window=2,
    )
    assert quality.confidence_cap == CAP_MULTIPLE_LIFTS
    assert len(quality.warnings) >= 3


def test_split_lift_is_noted_but_not_penalised():
    quality = assess_clip_quality(
        observations="Bar racked then driven overhead.",
        info=landscape(),
        frame_count=10,
        duration_sec=4.0,
        lifts_in_window=2,
        was_split=True,
    )
    assert quality.confidence_cap is None
    assert any("scored on its own" in warning for warning in quality.warnings)


def test_cap_never_raises_an_already_low_confidence():
    quality = assess_clip_quality(observations="", info=portrait(), frame_count=10, duration_sec=6.0)
    assert apply_quality(judgment(0.2), quality).confidence == 0.2


def test_cap_applies_when_the_judge_gave_no_confidence():
    quality = assess_clip_quality(observations="", info=portrait(), frame_count=10, duration_sec=6.0)
    assert apply_quality(judgment(None), quality).confidence == CAP_PORTRAIT


@pytest.mark.parametrize(
    "observations,expected_fragment",
    [
        ("The bar is not visible for most of the clip.", "not visible"),
        ("The platform is dark and the lifter is blurry.", "occluded"),
        ("The phases are unclear in these frames.", "not clearly separable"),
    ],
)
def test_observation_phrases_produce_warnings(observations, expected_fragment):
    quality = assess_clip_quality(observations=observations, info=landscape(), frame_count=10, duration_sec=5.0)
    assert quality.confidence_cap is not None
    assert any(expected_fragment in warning for warning in quality.warnings)


def test_short_clip_and_few_frames_are_flagged():
    quality = assess_clip_quality(observations="", info=landscape(), frame_count=3, duration_sec=0.4)
    assert any("too short" in warning for warning in quality.warnings)
    assert any("few frames" in warning for warning in quality.warnings)


def test_merge_notes_does_not_repeat_what_the_judge_said():
    notes = "Filmed in portrait."
    assert merge_notes(notes, ["Filmed in portrait."]) == notes
    merged = merge_notes("Bar drifts.", ["Camera is oblique"])
    assert merged == "Bar drifts. Camera is oblique."


def test_clip_quality_deduplicates_warnings():
    quality = ClipQuality()
    quality.add("same", 0.5)
    quality.add("same", 0.4)
    assert quality.warnings == ["same"]
    assert quality.confidence_cap == 0.4
