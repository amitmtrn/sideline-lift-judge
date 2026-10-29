from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from server.utils.ffmpeg_binary import ffprobe_exe


@dataclass
class VideoInfo:
    """
    What we can learn about a clip without decoding it, via ffprobe.

    `width`/`height` are as coded; `display_width`/`display_height` are what a
    viewer sees once rotation metadata is applied, which is what decides whether
    a phone clip is portrait.
    """

    width: int = 0
    height: int = 0
    rotation: int = 0
    duration_sec: float = 0.0
    fps: float = 0.0
    has_audio: bool = False
    probed: bool = False

    @property
    def display_width(self) -> int:
        return self.height if self.rotation in (90, 270) else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.rotation in (90, 270) else self.height

    @property
    def is_portrait(self) -> bool:
        """Taller than it is wide once rotated. Square counts as neither."""
        if not (self.display_width and self.display_height):
            return False
        return self.display_height > self.display_width


def probe_video(video_path: str) -> VideoInfo:
    """
    Read size, rotation, duration and fps with ffprobe. Falls back to OpenCV
    for size/duration/fps when ffprobe is absent (Heroku without the apt
    buildpack) or returns nothing useful, so the pipeline still has a real
    window to score. Rotation metadata is only read from ffprobe; the OpenCV
    fallback already hands back display-oriented frames, so leaving rotation
    at 0 is the right thing. Returns an unpopulated VideoInfo (probed=False)
    only when the file itself is unreadable.
    """
    ffprobe = ffprobe_exe()
    if ffprobe:
        try:
            # -show_streams rather than a -show_entries selector: asking for
            # `side_data_list` by name returns the section with its fields
            # stripped ([{}]), which is how the display-matrix rotation went
            # missing.
            out = subprocess.check_output(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    video_path,
                ],
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            data = json.loads(out.decode("utf-8", errors="ignore"))
        except Exception:
            data = None
        if data is not None:
            info = VideoInfo(probed=True)
            for stream in data.get("streams", []) or []:
                kind = (stream.get("codec_type") or "").lower()
                if kind == "audio":
                    info.has_audio = True
                    continue
                if kind != "video" or info.width:
                    continue
                info.width = int(stream.get("width") or 0)
                info.height = int(stream.get("height") or 0)
                info.rotation = _rotation_from_stream(stream)
                info.fps = _parse_frame_rate(stream.get("avg_frame_rate"))
            try:
                info.duration_sec = max(0.0, float((data.get("format") or {}).get("duration") or 0.0))
            except (TypeError, ValueError):
                info.duration_sec = 0.0
            format_name = str((data.get("format") or {}).get("format_name") or "").lower()
            if info.duration_sec > 0 or info.width:
                # A MediaRecorder file (webm/matroska) carries no reliable
                # duration or frame rate in its header; a container that
                # claims none at all is in the same boat. Read the clock off
                # the decoded frames in those cases.
                suspect = (
                    info.duration_sec <= 0
                    or info.fps <= 0
                    or info.fps > 240
                    or "webm" in format_name
                    or "matroska" in format_name
                )
                if suspect:
                    _reconcile_with_decoded(info, video_path)
                return info
    return _probe_with_opencv(video_path)


def _measure_by_decoding(video_path: str, max_seconds: float = 60.0) -> Optional[tuple]:
    """
    Walk the frames and return (duration_sec, fps, frames) from the decoder's
    own presentation times — the clock the tracker, the trim and the replay
    actually run on. Container metadata is what a browser's MediaRecorder
    gets wrong (webm with no duration, a nominal 30 fps on a 60 fps capture);
    the frames themselves cannot lie about when they were shown. Bounded at
    `max_seconds` of media so a long upload does not stall the request.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    first_t: Optional[float] = None
    last_t = 0.0
    count = 0
    try:
        while cap.grab():
            t = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 6)
            if first_t is None:
                first_t = t
            # A decoder that reports no clock at all (0.0 on every frame)
            # cannot be measured this way; fall back to what we had.
            if count and t <= last_t and t == 0.0:
                return None
            last_t = max(last_t, t)
            count += 1
            if first_t is not None and last_t - first_t > max_seconds:
                break
    finally:
        cap.release()
    if count < 2 or first_t is None:
        return None
    span = last_t - first_t
    if span <= 0:
        return None
    fps = (count - 1) / span
    duration = last_t + 1.0 / fps
    return max(0.0, duration), max(0.0, fps), count


def _reconcile_with_decoded(info: "VideoInfo", video_path: str) -> None:
    """Replace a suspect container duration/fps with what the frames say."""
    measured = _measure_by_decoding(video_path)
    if measured is None:
        return
    duration, fps, _frames = measured
    if duration > 0:
        info.duration_sec = duration
    if fps > 0:
        info.fps = fps


def _probe_with_opencv(video_path: str) -> VideoInfo:
    """
    Fallback probe used when ffprobe is not on the PATH. OpenCV's decoder
    applies the display matrix itself on the builds we ship, so width/height
    are what a viewer sees and rotation stays 0.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return VideoInfo()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return VideoInfo()
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    if not (width and height):
        return VideoInfo()
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    info = VideoInfo(
        width=width,
        height=height,
        rotation=0,
        duration_sec=max(0.0, duration),
        fps=max(0.0, fps),
        probed=True,
    )
    # Without ffprobe (the Heroku dyno) this is the only probe, and the
    # container's frame count and rate are exactly what a MediaRecorder
    # upload gets wrong. Always read the clock off the decoded frames here;
    # the clip-length cap keeps the pass short.
    _reconcile_with_decoded(info, video_path)
    return info


def _is_display_matrix(side_data_type: object) -> bool:
    """
    ffprobe spells it "Display Matrix" (with a space, capitalised). Comparing
    against "displaymatrix" silently never matched, so every rotated phone clip
    reported 0° — fold case and drop non-letters instead.
    """
    if not isinstance(side_data_type, str):
        return False
    return "".join(ch for ch in side_data_type.lower() if ch.isalpha()) == "displaymatrix"


def _rotation_from_stream(stream: dict) -> int:
    rotate_tag = (stream.get("tags") or {}).get("rotate")
    if rotate_tag is not None:
        try:
            return normalize_rotation(int(str(rotate_tag).strip()))
        except ValueError:
            pass
    for side_data in stream.get("side_data_list") or []:
        if _is_display_matrix(side_data.get("side_data_type")):
            rotation = side_data.get("rotation")
            if rotation is not None:
                try:
                    return normalize_rotation(int(float(rotation)))
                except (TypeError, ValueError):
                    continue
    return 0


def rotation_still_needed(info: "VideoInfo", frame_width: int, frame_height: int) -> int:
    """
    How much we must rotate a frame OpenCV just handed us.

    OpenCV's FFmpeg backend applies the display matrix itself on most builds, so
    rotating by `info.rotation` on top of that turns a portrait clip on its side.
    Decide from what actually came back: frames already at the display size need
    nothing, frames at the coded size need the full rotation, and anything else
    is left alone rather than risk rotating twice.
    """
    if not info.probed or not info.rotation:
        return 0
    if (frame_width, frame_height) == (info.display_width, info.display_height):
        return 0
    if (frame_width, frame_height) == (info.width, info.height):
        return info.rotation
    return 0


def _parse_frame_rate(value: object) -> float:
    """avg_frame_rate arrives as a rational string like '30000/1001'."""
    if not isinstance(value, str) or "/" not in value:
        try:
            return max(0.0, float(value))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
    numerator, _, denominator = value.partition("/")
    try:
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 0.0
        return max(0.0, float(numerator) / denominator_value)
    except ValueError:
        return 0.0


def get_rotation_degrees(video_path: str) -> int:
    """
    Rotation metadata in clockwise degrees, one of {0, 90, 180, 270}; 0 when
    ffprobe is missing or the file carries none. Delegates to probe_video so the
    display-matrix parsing lives in exactly one place.

    Note this is the rotation *the file declares*, which is not necessarily the
    rotation a decoded frame still needs — see `rotation_still_needed`.
    """
    return probe_video(video_path).rotation


def normalize_rotation(deg: int) -> int:
    # Map negative and arbitrary values to {0,90,180,270}
    deg = deg % 360
    if deg < 0:
        deg += 360
    # Snap to nearest right angle
    candidates = [0, 90, 180, 270]
    best = min(candidates, key=lambda x: abs(x - deg))
    return best

