import asyncio

import pytest

from server.utils.uploads import (
    DEFAULT_MAX_UPLOAD_MB,
    UploadTooLarge,
    UnsupportedUpload,
    is_supported_video_upload,
    max_upload_bytes,
    normalize_content_type,
    save_upload_to_temp,
)


class FakeUpload:
    """Minimal stand-in for Starlette's UploadFile.read(size)."""

    def __init__(self, data: bytes, filename: str = "clip.mp4") -> None:
        self._data = data
        self._pos = 0
        self.filename = filename

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._data[self._pos :]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


def test_normalize_content_type_drops_parameters():
    assert normalize_content_type("VIDEO/MP4; codecs=avc1") == "video/mp4"
    assert normalize_content_type(None) == ""


@pytest.mark.parametrize(
    "content_type,filename",
    [
        ("video/mp4", "lift.mp4"),
        ("video/quicktime", "IMG_0001.MOV"),  # iOS Safari
        ("video/3gpp", "clip.3gp"),  # Android Chrome
        ("video/webm", "clip.webm"),
        ("video/mp4; codecs=hvc1", "hevc.mp4"),  # iPhone HEVC
        ("application/octet-stream", "clip.mov"),  # generic type, video extension
        ("", "clip.mp4"),
    ],
)
def test_accepts_phone_video_uploads(content_type, filename):
    assert is_supported_video_upload(content_type, filename)


@pytest.mark.parametrize(
    "content_type,filename",
    [
        ("image/jpeg", "frame.jpg"),
        ("text/plain", "notes.txt"),
        ("application/pdf", "course.pdf"),
        ("application/octet-stream", "archive.zip"),  # generic type, not a video
        ("application/octet-stream", None),
    ],
)
def test_rejects_non_video_uploads(content_type, filename):
    assert not is_supported_video_upload(content_type, filename)


def test_max_upload_bytes_env(monkeypatch):
    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
    assert max_upload_bytes() == DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    monkeypatch.setenv("MAX_UPLOAD_MB", "5")
    assert max_upload_bytes() == 5 * 1024 * 1024
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")  # documented way to disable the cap
    assert max_upload_bytes() == 0
    monkeypatch.setenv("MAX_UPLOAD_MB", "nonsense")
    assert max_upload_bytes() == DEFAULT_MAX_UPLOAD_MB * 1024 * 1024


def test_save_upload_writes_file_under_limit():
    upload = FakeUpload(b"x" * 2048)
    path = asyncio.run(save_upload_to_temp(upload, max_bytes=1024 * 1024))
    try:
        assert path.exists()
        assert path.read_bytes() == b"x" * 2048
        assert path.suffix == ".mp4"
    finally:
        path.unlink(missing_ok=True)


def test_save_upload_rejects_oversized_clip_and_cleans_up(tmp_path):
    upload = FakeUpload(b"y" * 4096)
    with pytest.raises(UploadTooLarge):
        asyncio.run(save_upload_to_temp(upload, max_bytes=1024, dir=str(tmp_path)))
    # The partial file must not be left behind.
    assert list(tmp_path.iterdir()) == []


def test_save_upload_rejects_empty_file(tmp_path):
    upload = FakeUpload(b"")
    with pytest.raises(UnsupportedUpload):
        asyncio.run(save_upload_to_temp(upload, max_bytes=1024, dir=str(tmp_path)))
    assert list(tmp_path.iterdir()) == []
