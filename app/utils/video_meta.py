from __future__ import annotations

import json
import shutil
import subprocess
from typing import Optional


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

