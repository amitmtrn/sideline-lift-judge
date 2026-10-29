"""
Where to find ffmpeg and ffprobe at run time.

On a workstation or a Docker image we installed, both are on PATH. On
Heroku's buildpack path without the apt buildpack, neither is — and the
Aptfile does nothing. The pip package `imageio-ffmpeg` ships a static
ffmpeg binary for every platform the deploy might run on, so we use that
as a last resort for `ffmpeg` itself. There is no equivalent for `ffprobe`;
`probe_video` already falls back to OpenCV for size and duration when
ffprobe is missing (`server.utils.video_meta`).
"""
from __future__ import annotations

import logging
import shutil
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def ffmpeg_exe() -> Optional[str]:
    """
    Path to an ffmpeg binary, or None when nothing can be found. PATH wins
    because a system install usually matches whatever libraries the host has;
    the imageio-ffmpeg wheel is only reached for when the host has none.
    """
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg  # type: ignore
    except Exception as exc:
        logger.debug("imageio-ffmpeg not importable: %s", exc)
        return None
    try:
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        logger.warning("imageio-ffmpeg could not resolve its bundled binary: %s", exc)
        return None
    return bundled or None


@lru_cache(maxsize=1)
def ffprobe_exe() -> Optional[str]:
    """
    Path to an ffprobe binary, or None. imageio-ffmpeg does not ship ffprobe,
    so this is PATH only; callers degrade to OpenCV for metadata when None.
    """
    return shutil.which("ffprobe")


def log_binaries() -> None:
    """One line per binary at startup, so a deploy's logs name what it will run."""
    ffmpeg = ffmpeg_exe()
    ffprobe = ffprobe_exe()
    logger.info("ffmpeg: %s", ffmpeg or "NOT FOUND (replays disabled)")
    logger.info("ffprobe: %s", ffprobe or "not on PATH (OpenCV reads metadata instead)")


def reset_cache_for_tests() -> None:
    """Tests that monkeypatch PATH or imageio_ffmpeg need the lookup to re-run."""
    ffmpeg_exe.cache_clear()
    ffprobe_exe.cache_clear()
