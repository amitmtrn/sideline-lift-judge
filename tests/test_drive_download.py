"""
Google Drive serves large public files behind a "cannot scan for viruses"
HTML interstitial that the browser POSTs through. The downloader must follow
it, cache the result on disk, and not re-fetch a cached clip on the next
call. No real network: every test mocks `requests.Session.get`.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from server.utils import drive_download


VIDEO_BYTES = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 2048


class _FakeResponse:
    def __init__(self, *, status: int = 200, content_type: str = "video/mp4", chunks=(VIDEO_BYTES,), text: str = ""):
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self._chunks = list(chunks)
        self._text = text

    def iter_content(self, chunk_size=65536, decode_unicode=False):  # noqa: ARG002
        yield from self._chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    """Records calls so a test can assert "downloaded once"."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self._responses:
            raise RuntimeError("no more fake responses")
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _reset_locks():
    drive_download.reset_locks_for_tests()
    yield
    drive_download.reset_locks_for_tests()


def test_direct_video_response_is_streamed_to_disk(tmp_path):
    session = _FakeSession([_FakeResponse()])
    out = tmp_path / "clip.mp4"
    drive_download.download_drive_file("https://drive.google.com/uc?id=abc", out, session=session)
    assert out.exists()
    assert out.read_bytes().startswith(b"\x00\x00\x00\x18ftyp")
    assert len(session.calls) == 1


def test_virus_scan_confirm_page_is_followed(tmp_path):
    confirm_html = (
        '<html><body><form id="download-form" action="/uc?export=download&amp;confirm=t-123&amp;id=abc" method="get">'
        '<input type="hidden" name="confirm" value="t-123">'
        '<input type="hidden" name="id" value="abc">'
        "</form></body></html>"
    )
    session = _FakeSession(
        [
            _FakeResponse(content_type="text/html; charset=utf-8", chunks=(confirm_html.encode("utf-8"),)),
            _FakeResponse(),
        ]
    )
    out = tmp_path / "clip.mp4"
    drive_download.download_drive_file(
        "https://drive.google.com/uc?export=download&id=abc", out, session=session
    )
    assert out.exists() and out.read_bytes().startswith(b"\x00\x00\x00\x18ftyp")
    assert len(session.calls) == 2
    # The second request POSTed to the confirm action with the token fields.
    _, kwargs = session.calls[1]
    assert kwargs["params"] == {"confirm": "t-123", "id": "abc"}


def test_html_response_without_a_form_is_an_error(tmp_path):
    session = _FakeSession(
        [_FakeResponse(content_type="text/html", chunks=(b"<html>nothing here</html>",))]
    )
    with pytest.raises(RuntimeError, match="confirm form"):
        drive_download.download_drive_file("https://drive.google.com/uc?id=x", tmp_path / "x.mp4", session=session)


def test_empty_response_is_an_error(tmp_path):
    session = _FakeSession([_FakeResponse(chunks=())])
    with pytest.raises(RuntimeError, match="empty"):
        drive_download.download_drive_file(
            "https://drive.google.com/uc?id=x", tmp_path / "x.mp4", session=session
        )


def test_oversize_response_is_refused(tmp_path):
    big = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 200_000
    session = _FakeSession([_FakeResponse(chunks=(big,))])
    with pytest.raises(RuntimeError, match="exceeds"):
        drive_download.download_drive_file(
            "https://drive.google.com/uc?id=x",
            tmp_path / "x.mp4",
            session=session,
            max_bytes=100_000,
        )
    assert not (tmp_path / "x.mp4").exists()


def test_ensure_cached_clip_only_downloads_once(tmp_path, monkeypatch):
    session = _FakeSession([_FakeResponse(), _FakeResponse()])
    # Skip the real OpenCV verification: our fake bytes are not a real video.
    monkeypatch.setattr(drive_download, "verify_video_file", lambda path: True)
    with patch.object(drive_download, "requests") as fake_requests:
        fake_requests.Session.return_value = session

        destination = tmp_path / "clip.mp4"
        a = drive_download.ensure_cached_clip(
            "https://drive.google.com/uc?id=abc", destination, session=session
        )
        b = drive_download.ensure_cached_clip(
            "https://drive.google.com/uc?id=abc", destination, session=session
        )
    assert a == destination and b == destination
    assert len(session.calls) == 1


def test_ensure_cached_clip_returns_none_when_download_fails(tmp_path):
    session = _FakeSession([_FakeResponse(status=500)])
    result = drive_download.ensure_cached_clip(
        "https://drive.google.com/uc?id=abc", tmp_path / "clip.mp4", session=session
    )
    assert result is None
    assert not (tmp_path / "clip.mp4").exists()


def test_ensure_cached_clip_rejects_non_video_payload(tmp_path, monkeypatch):
    session = _FakeSession([_FakeResponse()])
    monkeypatch.setattr(drive_download, "verify_video_file", lambda path: False)
    result = drive_download.ensure_cached_clip(
        "https://drive.google.com/uc?id=abc", tmp_path / "clip.mp4", session=session
    )
    assert result is None
    assert not (tmp_path / "clip.mp4").exists()
