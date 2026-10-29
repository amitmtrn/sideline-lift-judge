"""
Download a Google Drive "anyone-with-the-link" clip, with large-file virus-scan
confirmation handled, and cache it on local disk. We never hand the Drive URL
to the browser: iOS Safari needs HTTP Range support and the right MIME, which
our own file endpoint provides and Drive's redirect chain does not reliably.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

CHUNK_BYTES = 64 * 1024
# Hard ceiling on a cached clip. A sample is a few seconds long; anything in
# this ballpark is a configuration mistake we should refuse rather than stream
# to disk for ten minutes.
DEFAULT_MAX_BYTES = 200 * 1024 * 1024

# One lock per URL + destination, so parallel requests for the same sample do
# not both download. A plain module-level dict suffices on a single-worker dyno.
_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def _is_video_response(response: requests.Response) -> bool:
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "video/" in content_type:
        return True
    # Drive serves video binaries as application/octet-stream too.
    if "application/octet-stream" in content_type:
        return True
    return False


_FORM_RE = re.compile(r'<form[^>]+action="([^"]+)"[^>]*>(.*?)</form>', re.S | re.I)
_INPUT_RE = re.compile(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', re.I)


def _parse_confirm_form(html: str) -> Optional[Dict[str, Any]]:
    """
    Google Drive's "cannot scan for viruses" page is a short HTML form the
    browser POSTs to continue the download. Extract the form so our client
    can follow it in the same session.
    """
    match = _FORM_RE.search(html)
    if not match:
        return None
    action = match.group(1).replace("&amp;", "&")
    body = match.group(2)
    fields = dict(_INPUT_RE.findall(body))
    return {"action": action, "fields": fields}


def _absolute(url: str, action: str) -> str:
    if action.startswith("http://") or action.startswith("https://"):
        return action
    if action.startswith("/"):
        return "https://drive.google.com" + action
    # Rare: a relative form on the same path. Keep it simple and resolve.
    base = url.rsplit("/", 1)[0]
    return f"{base}/{action}"


def download_drive_file(
    url: str,
    destination: Path,
    *,
    timeout: float = 60.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    session: Optional[requests.Session] = None,
) -> Path:
    """
    Download `url` to `destination` and return the path. Follows Drive's
    virus-scan confirm page when it appears. Raises `RuntimeError` on any
    non-video response, HTTP error, or file-size violation.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    sess = session or requests.Session()
    response = sess.get(url, stream=True, timeout=timeout, allow_redirects=True)
    response.raise_for_status()

    if not _is_video_response(response):
        # Big files hit a confirm page; small files stream directly. Read a
        # bounded amount of HTML to find the form, so a huge garbage response
        # does not pin memory.
        html = ""
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES, decode_unicode=False):
            if not chunk:
                continue
            html += chunk.decode("utf-8", errors="ignore")
            if len(html) > 2 * 1024 * 1024:
                break
        form = _parse_confirm_form(html)
        if form is None:
            raise RuntimeError(
                f"Google Drive did not return a video or a confirm form for {url} "
                f"(Content-Type: {response.headers.get('Content-Type')!r})"
            )
        confirm_url = _absolute(url, form["action"])
        logger.info("Google Drive virus-scan confirm; following %s", confirm_url)
        response = sess.get(
            confirm_url,
            params=form["fields"],
            stream=True,
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
        if not _is_video_response(response):
            raise RuntimeError(
                f"Drive confirm did not return a video: {response.headers.get('Content-Type')!r}"
            )

    bytes_written = 0
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            if not chunk:
                continue
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                handle.close()
                destination.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Clip at {url} exceeds the {max_bytes // 1024 // 1024} MB cap"
                )
            handle.write(chunk)
    if bytes_written == 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Clip at {url} was empty")
    return destination


def verify_video_file(path: Path) -> bool:
    """
    Open with OpenCV to confirm the file is a readable video. Guards against a
    hoster returning HTML with a video MIME when the file is actually broken.
    """
    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        return capture.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    finally:
        capture.release()


def ensure_cached_clip(
    url: str,
    destination: Path,
    *,
    timeout: float = 60.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    session: Optional[requests.Session] = None,
    verify: bool = True,
) -> Optional[Path]:
    """
    Download `url` to `destination` the first time, then reuse the cached file.
    Returns the local path on success, or None when the download failed. One
    lock per destination keeps parallel requests from fetching twice.
    """
    key = str(destination)
    with _lock_for(key):
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        try:
            download_drive_file(
                url,
                destination,
                timeout=timeout,
                max_bytes=max_bytes,
                session=session,
            )
        except Exception as exc:
            logger.warning("Could not download %s: %s", url, exc)
            destination.unlink(missing_ok=True)
            return None
        if verify and not verify_video_file(destination):
            logger.warning("Downloaded %s is not a readable video; discarding", url)
            destination.unlink(missing_ok=True)
            return None
    return destination


def reset_locks_for_tests() -> None:
    with _LOCKS_GUARD:
        _LOCKS.clear()
