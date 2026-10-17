from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


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
    Read size, rotation, duration and fps with ffprobe. Returns an unpopulated
    VideoInfo (probed=False) when ffprobe is missing or the file is unreadable,
    so callers degrade instead of failing.
    """
    if not shutil.which("ffprobe"):
        return VideoInfo()
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height,avg_frame_rate,side_data_list:stream_tags=rotate:format=duration",
                "-of",
                "json",
                video_path,
            ],
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        data = json.loads(out.decode("utf-8", errors="ignore"))
    except Exception:
        return VideoInfo()

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
    return info


def _rotation_from_stream(stream: dict) -> int:
    rotate_tag = (stream.get("tags") or {}).get("rotate")
    if rotate_tag is not None:
        try:
            return normalize_rotation(int(str(rotate_tag).strip()))
        except ValueError:
            pass
    for side_data in stream.get("side_data_list") or []:
        if (side_data.get("side_data_type") or "").lower() == "displaymatrix":
            rotation = side_data.get("rotation")
            if rotation is not None:
                try:
                    return normalize_rotation(int(float(rotation)))
                except (TypeError, ValueError):
                    continue
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
    Best-effort detection of rotation metadata using ffprobe if available.
    Returns one of {0, 90, 180, 270} (clockwise degrees). Missing ffprobe → 0.
    """
    if not shutil.which("ffprobe"):
        return 0
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=side_data_list:stream_tags=rotate",
            "-of",
            "json",
            video_path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", errors="ignore"))
        streams = data.get("streams", [])
        if not streams:
            return 0
        s0 = streams[0]
        # Tag-based rotate (common)
        tags = s0.get("tags", {}) or {}
        rotate_val = tags.get("rotate")
        if rotate_val is not None:
            deg = int(str(rotate_val).strip())
            return normalize_rotation(deg)
        # Display matrix side data
        for sd in s0.get("side_data_list", []) or []:
            if sd.get("side_data_type", "").lower() == "displaymatrix":
                # Some ffprobe builds include rotation in degrees
                deg = sd.get("rotation")
                if deg is not None:
                    return normalize_rotation(int(deg))
        return 0
    except Exception:
        return 0


def normalize_rotation(deg: int) -> int:
    # Map negative and arbitrary values to {0,90,180,270}
    deg = deg % 360
    if deg < 0:
        deg += 360
    # Snap to nearest right angle
    candidates = [0, 90, 180, 270]
    best = min(candidates, key=lambda x: abs(x - deg))
    return best

