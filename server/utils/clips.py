from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

from server.utils.ffmpeg_binary import ffmpeg_exe


def has_ffmpeg() -> bool:
    return ffmpeg_exe() is not None


def cut_segment(
    video_path: str,
    *,
    start_sec: float,
    end_sec: float,
    out_path: str,
    reencode: bool = False,
) -> Optional[str]:
    """
    Write [start_sec, end_sec] of `video_path` to `out_path` with ffmpeg.

    Stream-copy first (fast, no quality loss) and fall back to re-encoding when
    the copy produces nothing usable — a copy can only cut on a keyframe, so a
    short window between keyframes sometimes comes out empty. Returns the written
    path, or None when ffmpeg is missing or both attempts fail.
    """
    ffmpeg = ffmpeg_exe()
    if not ffmpeg:
        return None
    start = max(0.0, float(start_sec))
    duration = max(0.05, float(end_sec) - start)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    attempts: List[List[str]] = []
    copy_args = ["-c", "copy"]
    encode_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac"]
    for codec_args in ([encode_args] if reencode else [copy_args, encode_args]):
        attempts.append(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{duration:.3f}",
                *codec_args,
                "-movflags",
                "+faststart",
                str(destination),
            ]
        )

    for args in attempts:
        try:
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, OSError):
            continue
        if destination.exists() and destination.stat().st_size > 1024:
            return str(destination)
    return None


def write_contact_sheet(frames_bgr: List["np.ndarray"], out_dir: str, *, prefix: str = "frame") -> Optional[str]:
    """
    Save the frames that were sent to the vision model as numbered JPEGs, so a
    bad score can be traced back to what the model actually saw. Returns the
    directory, or None when nothing was written.
    """
    import cv2  # type: ignore

    if not frames_bgr:
        return None
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for position, frame in enumerate(frames_bgr, start=1):
        path = directory / f"{prefix}_{position:02d}.jpg"
        if cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
            written += 1
    return str(directory) if written else None


def format_timestamp(seconds: float) -> str:
    """mm:ss.s — short enough for a table, precise enough to scrub to."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return f"{minutes:d}:{seconds - minutes * 60:04.1f}"
