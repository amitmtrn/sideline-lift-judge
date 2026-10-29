"""
Where ffmpeg comes from on a dyno without the apt buildpack.

A system install on PATH is preferred; the imageio-ffmpeg wheel is the
fallback that keeps replays working on Heroku's Python buildpack by itself.
ffprobe has no bundled binary and so is PATH-only — the metadata path falls
back to OpenCV elsewhere.
"""
from __future__ import annotations

from server.utils import ffmpeg_binary


def test_ffmpeg_prefers_path(monkeypatch):
    ffmpeg_binary.reset_cache_for_tests()
    monkeypatch.setattr("server.utils.ffmpeg_binary.shutil.which", lambda name: "/usr/bin/" + name)
    assert ffmpeg_binary.ffmpeg_exe() == "/usr/bin/ffmpeg"
    assert ffmpeg_binary.ffprobe_exe() == "/usr/bin/ffprobe"


def test_ffmpeg_falls_back_to_imageio(monkeypatch):
    ffmpeg_binary.reset_cache_for_tests()
    monkeypatch.setattr("server.utils.ffmpeg_binary.shutil.which", lambda name: None)
    # imageio-ffmpeg ships a static ffmpeg for the test platform.
    import imageio_ffmpeg  # type: ignore

    path = ffmpeg_binary.ffmpeg_exe()
    assert path is not None
    assert path == imageio_ffmpeg.get_ffmpeg_exe()


def test_ffprobe_is_path_only(monkeypatch):
    ffmpeg_binary.reset_cache_for_tests()
    monkeypatch.setattr("server.utils.ffmpeg_binary.shutil.which", lambda name: None)
    # imageio-ffmpeg does not provide ffprobe; expect None here even though
    # the bundled ffmpeg comes back from the test above.
    assert ffmpeg_binary.ffprobe_exe() is None


def test_ffmpeg_returns_none_when_imageio_errors(monkeypatch):
    """A dependency that raises mid-lookup should degrade silently, not crash."""
    ffmpeg_binary.reset_cache_for_tests()
    monkeypatch.setattr("server.utils.ffmpeg_binary.shutil.which", lambda name: None)
    import sys
    import types

    broken = types.ModuleType("imageio_ffmpeg")

    def _raise():
        raise RuntimeError("imageio-ffmpeg download failed")

    broken.get_ffmpeg_exe = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", broken)
    assert ffmpeg_binary.ffmpeg_exe() is None


def teardown_module(_module):
    ffmpeg_binary.reset_cache_for_tests()
