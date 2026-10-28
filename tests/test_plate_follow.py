"""
Following the plate frame by frame (plate_follow), and the clock every frame
reader shares (frames.read_frames).

Phone footage is variable frame rate. On it, frame number / fps is not the
time a frame is shown, and seeking by frame number lands far from the frame
asked for — the trace, the replay and the evidence stills drifted apart by
up to two seconds before the readers were moved onto presentation time.
"""
import shutil
import subprocess

import numpy as np  # type: ignore
import pytest

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")

WIDTH, HEIGHT, FPS = 360, 640, 60
PLATE_R = 34


def _plate(frame, x, y, radius, colour):
    """A plate as a camera sees one: a rim, a hub and markings to follow."""
    import cv2  # type: ignore

    centre = (int(round(x)), int(round(y)))
    cv2.circle(frame, centre, radius, colour, -1, cv2.LINE_AA)
    cv2.circle(frame, centre, radius, (230, 230, 230), 3, cv2.LINE_AA)
    cv2.circle(frame, centre, radius // 4, (200, 200, 200), -1, cv2.LINE_AA)
    for angle in (0.3, 1.9, 3.4, 4.9):
        mark = (int(centre[0] + 0.6 * radius * np.cos(angle)), int(centre[1] + 0.6 * radius * np.sin(angle)))
        cv2.rectangle(frame, (mark[0] - 4, mark[1] - 3), (mark[0] + 4, mark[1] + 3), (20, 20, 20), -1)


def bar_at(fraction):
    """Still for the first fifth, then floor to overhead, looping forward on the way."""
    rise = min(1.0, max(0.0, (fraction - 0.2) / 0.6))
    x = 150 + 40 * np.sin(np.pi * rise)
    y = HEIGHT - 90 - rise * (HEIGHT - 260)
    return x, y


def write_lift(path, duration=2.5):
    """A rising plate, a plate parked on a stand, and a head that rises less."""
    import cv2  # type: ignore

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    rng = np.random.default_rng(3)
    backdrop = np.clip(rng.normal(80, 6, (HEIGHT, WIDTH, 3)), 0, 255).astype(np.uint8)
    try:
        total = int(duration * FPS)
        for index in range(total):
            fraction = index / (total - 1)
            frame = backdrop.copy()
            _plate(frame, 300, HEIGHT - 70, PLATE_R + 4, (150, 90, 30))  # parked on a stand
            x, y = bar_at(fraction)
            head_y = HEIGHT - 330 - 0.25 * min(1.0, max(0.0, (fraction - 0.2) / 0.6)) * 200
            _plate(frame, 210, head_y, 20, (120, 150, 200))  # head: smaller, rises less
            _plate(frame, x, y, PLATE_R, (60, 60, 200))
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_the_follower_stays_on_the_lifted_plate(tmp_path):
    from server.utils.plate_follow import follow_bar_path

    path = follow_bar_path(write_lift(str(tmp_path / "lift.mp4")), 0.0, 2.5)
    assert path.tracked, path.reason
    assert abs(path.plate_radius * WIDTH - PLATE_R) <= 0.2 * PLATE_R
    errors = []
    for point in path.points:
        x, y = bar_at(min(1.0, point.t / 2.5 * 150 / 149))
        errors.append(np.hypot(point.x * WIDTH - x, point.y * HEIGHT - y))
    assert np.median(errors) < 4, f"median {np.median(errors):.1f} px off the plate"
    assert max(errors) < 12, f"worst {max(errors):.1f} px off the plate"
    # The loop forward is there to be measured: 40 px is 40 / 68 of a plate.
    drift_cm = path.drift_cm()
    assert drift_cm is not None and abs(drift_cm - 40 / (2 * PLATE_R) * 45) < 4


def test_the_follower_is_seeded_by_a_hint_it_does_not_have_to_trust(tmp_path):
    from server.utils.plate_follow import follow_bar_path

    clip = write_lift(str(tmp_path / "lift.mp4"))
    # A hint on the parked plate: one more seed, which does not rise and loses.
    hint = (0.0, 300 / WIDTH, (HEIGHT - 70) / HEIGHT, float(PLATE_R + 4))
    path = follow_bar_path(clip, 0.0, 2.5, hint=hint)
    assert path.tracked and path.vertical_travel > 0.5


def test_nothing_rising_is_not_called_a_bar(tmp_path):
    import cv2  # type: ignore
    from server.utils.plate_follow import follow_bar_path

    clip = str(tmp_path / "still.mp4")
    writer = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    frame = np.full((HEIGHT, WIDTH, 3), 80, np.uint8)
    _plate(frame, 150, HEIGHT - 90, PLATE_R, (60, 60, 200))
    for _ in range(90):
        writer.write(frame)
    writer.release()
    path = follow_bar_path(clip, 0.0, 1.5)
    assert not path.tracked and path.reason


# ---------- one clock for every reader ----------


@pytest.fixture
def vfr_clip(tmp_path):
    """
    60 frames whose brightness is their number: the first 20 shown at 20 fps,
    the rest at 40 fps — so frame number / average fps is wrong by a third
    of a second by the end, as on phone footage.
    """
    import cv2  # type: ignore

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    plain = str(tmp_path / "plain.mp4")
    writer = cv2.VideoWriter(plain, cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 64))
    for index in range(60):
        writer.write(np.full((64, 64, 3), index * 4, np.uint8))
    writer.release()
    out = str(tmp_path / "vfr.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", plain,
         "-vf", "settb=1/1000,setpts='if(lt(N,20),N/20,1+(N-20)/40)/TB'",
         "-fps_mode", "passthrough", "-enc_time_base", "1/1000",
         "-c:v", "mpeg4", "-q:v", "1", "-video_track_timescale", "1000", out],
        check=True,
    )
    # Each frame's brightness as decoded, in order — the n-th frame decoded
    # from the top is frame n — so a frame read any other way can be named.
    capture = cv2.VideoCapture(out)
    BRIGHTNESS.clear()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        BRIGHTNESS.append(float(frame.mean()))
    capture.release()
    assert len(BRIGHTNESS) == 60
    return out


BRIGHTNESS = []


def _number(frame):
    return int(np.argmin(np.abs(np.array(BRIGHTNESS) - float(frame.mean()))))


def _shown_at(t):
    """Which frame of vfr_clip is on screen at t seconds. The container keeps
    whole milliseconds (1.275 s is stored as 1.274), hence the 2 ms slack."""
    t += 0.002
    return int(t * 20) if t < 1.0 else 20 + int((t - 1.0) * 40)


@needs_ffmpeg
def test_frames_carry_the_time_they_are_shown(vfr_clip):
    from server.utils.frames import read_frames

    got = list(read_frames(vfr_clip, 1.2, 1.5))
    assert got, "nothing read"
    for t, frame in got:
        assert _number(frame) == _shown_at(t), f"frame {_number(frame)} stamped {t:.3f} s"
    assert abs(got[0][0] - 1.2) < 0.03 and abs(got[-1][0] - 1.5) < 0.03


@needs_ffmpeg
def test_thinning_is_by_time_not_by_frame_count(vfr_clip):
    from server.utils.frames import read_frames

    times = [t for t, _f in read_frames(vfr_clip, 0.0, 1.45, max_fps=10)]
    gaps = np.diff(times)
    # 10 a second across both halves, though the frame rate doubles at 1 s.
    assert len(times) in (14, 15) and gaps.min() > 0.09 and gaps.max() < 0.11


@needs_ffmpeg
def test_frames_at_times_are_the_ones_on_screen(vfr_clip):
    from server.utils.frames import read_frames_at, sample_frames_in_range_bgr

    wanted = [0.5, 1.25, 1.3, 0.1]
    frames = read_frames_at(vfr_clip, wanted)
    assert [_number(f) for f in frames] == [_shown_at(t) for t in wanted]
    sampled = sample_frames_in_range_bgr(vfr_clip, start_sec=1.0, end_sec=1.4, target_count=5)
    assert [_number(f) for f in sampled] == [20, 24, 28, 32, 36]
