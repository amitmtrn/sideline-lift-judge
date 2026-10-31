#!/usr/bin/env python3
"""
How long the oblique follower takes on a 4 s, 1080p window.

    python tools/bench_oblique.py            # synthetic 1080x1920 @ 30 fps clip
    python tools/bench_oblique.py clip.mp4 3.0 7.0

Prints wall and CPU time for (a) decoding the window the way the pipeline
does (once, shared by every follower) and (b) the follower's own analysis on
those frames. CPU is measured with OpenCV on one thread: with its default
thread pool the parallel Hough and blur calls spin every core and the CPU
figure reads several times the wall time for the same work.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_synthetic(path: str, *, width: int = 1080, height: int = 1920, fps: int = 30, duration: float = 4.0) -> str:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    rng = np.random.default_rng(5)
    backdrop = np.clip(rng.normal(100, 8, (height, width, 3)), 0, 255).astype(np.uint8)
    radius = int(0.07 * width)
    total = int(duration * fps)
    for index in range(total):
        fraction = index / (total - 1)
        rise = min(1.0, max(0.0, (fraction - 0.2) / 0.6))
        frame = backdrop.copy()
        x = int(width * 0.4 + 0.05 * width * np.sin(np.pi * rise))
        y = int(height - 0.18 * height - rise * 0.4 * height)
        cv2.ellipse(frame, (x, y), (int(radius * 0.6), radius), 0, 0, 360, (28, 28, 28), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (x, y), (int(radius * 0.6), radius), 0, 0, 360, (140, 140, 140), 4, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    return path


def main(argv) -> None:
    import cv2  # type: ignore

    from server.utils.plate_follow import read_gray_frames
    from server.utils.plate_oblique import follow_oblique

    if len(argv) >= 2:
        clip, start, end = argv[1], float(argv[2]) if len(argv) > 2 else 0.0, float(argv[3]) if len(argv) > 3 else 4.0
    else:
        import tempfile

        clip = write_synthetic(str(Path(tempfile.mkdtemp()) / "bench.mp4"))
        start, end = 0.0, 4.0
        print(f"synthetic 1080x1920 @ 30 fps, {end - start:.0f} s: {clip}")

    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        wall, cpu = time.perf_counter(), time.process_time()
        frames = read_gray_frames(clip, start, end)
        print(f"decode  {len(frames):4d} frames at 640 wide: wall {time.perf_counter() - wall:5.2f} s  cpu {time.process_time() - cpu:5.2f} s")
        wall, cpu = time.perf_counter(), time.process_time()
        path = follow_oblique(None, frames=frames)
        print(f"analyse (frames shared):          wall {time.perf_counter() - wall:5.2f} s  cpu {time.process_time() - cpu:5.2f} s")
        wall, cpu = time.perf_counter(), time.process_time()
        alone = follow_oblique(clip, start, end)
        print(f"standalone (decode + analyse):    wall {time.perf_counter() - wall:5.2f} s  cpu {time.process_time() - cpu:5.2f} s")
    finally:
        cv2.setNumThreads(threads)
    print(f"tracked {path.tracked} ratio {path.plate_axis_ratio} points {len(path.points)} coverage {path.coverage:.2f}; standalone tracked {alone.tracked}")


if __name__ == "__main__":
    main(sys.argv)
