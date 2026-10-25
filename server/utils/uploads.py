from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

DEFAULT_MAX_UPLOAD_MB = 200
READ_CHUNK_BYTES = 1024 * 1024


class UnsupportedUpload(ValueError):
    """The file is not something we can treat as a video clip."""


class UploadTooLarge(ValueError):
    """The file is larger than the configured upload limit."""


# Phones are inconsistent about the type they attach: iOS Safari sends
# video/quicktime or video/mp4, Android Chrome can send video/3gpp or video/webm,
# and either may fall back to a generic binary type. So accept any video/* and
# resolve the generic types by file extension instead of rejecting the upload.
GENERIC_CONTENT_TYPES = frozenset(
    {
        "",
        "application/octet-stream",
        "binary/octet-stream",
        "application/mp4",
        "application/x-mpegurl",
    }
)

VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".m4v", ".mov", ".qt", ".3gp", ".3g2", ".webm", ".mkv", ".avi", ".mpeg", ".mpg", ".hevc", ".h265"}
)


def normalize_content_type(content_type: Optional[str]) -> str:
    """Lowercase the media type and drop any parameters (e.g. '; codecs=hvc1')."""
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def has_video_extension(filename: Optional[str]) -> bool:
    if not filename:
        return False
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS


def is_supported_video_upload(content_type: Optional[str], filename: Optional[str] = None) -> bool:
    """
    True when the upload looks like a video clip. Permissive on purpose: a clip we
    cannot decode is caught later with a clear message, whereas a wrongly rejected
    upload just looks broken to someone standing at the platform.
    """
    media_type = normalize_content_type(content_type)
    if media_type.startswith("video/"):
        return True
    if media_type in GENERIC_CONTENT_TYPES:
        return has_video_extension(filename)
    return False


def max_upload_bytes() -> int:
    """Upload ceiling in bytes, from MAX_UPLOAD_MB (0 or negative disables it)."""
    raw = os.getenv("MAX_UPLOAD_MB")
    if raw is None or not raw.strip():
        megabytes = DEFAULT_MAX_UPLOAD_MB
    else:
        try:
            megabytes = int(float(raw))
        except ValueError:
            megabytes = DEFAULT_MAX_UPLOAD_MB
    if megabytes <= 0:
        return 0
    return megabytes * 1024 * 1024


async def save_upload_to_temp(
    upload,
    *,
    max_bytes: Optional[int] = None,
    suffix: Optional[str] = None,
    dir: Optional[str] = None,
) -> Path:
    """
    Stream a Starlette UploadFile to a temp file, enforcing the size limit as we
    go so an oversized clip never has to fit in memory. The partial file is
    removed before UploadTooLarge is raised. Caller owns the returned path.
    """
    if max_bytes is None:
        max_bytes = max_upload_bytes()
    if suffix is None:
        suffix = Path(getattr(upload, "filename", "") or "clip").suffix or ".mp4"

    written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=dir) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await upload.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if max_bytes and written > max_bytes:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise UploadTooLarge(
                    f"Clips need to be under {max_bytes // (1024 * 1024)} MB. "
                    "Trim this one to the lift in your gallery app and try again."
                )
            tmp.write(chunk)

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise UnsupportedUpload("The uploaded file was empty.")
    return tmp_path
