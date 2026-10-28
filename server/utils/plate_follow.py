"""
Follow the plate through one lift, frame by frame, at the camera's own rate.

bar_track finds *when* the lifts are by looking for circles every twelfth of a
second and chaining them; that is good enough to find a rise and poor at
saying where the bar went, because each sample is a fresh guess among a dozen
circles (hub and rim of one plate, the other plate, a head, wall graphics).
This module does the second job: it picks the plate once, at the start of the
window while the bar is still, and then follows it with pyramidal
Lucas-Kanade optical flow on corners inside the plate — the method validated
against motion capture for barbell tracking (Nagao & Yamashita 2022) — nudged
back onto a circle of the plate's size whenever one is found where the flow
says the plate is.

Which circle is the plate is settled by following all of them: the plate is
the one that rises furthest while still looking like a plate (a circle of its
size keeps turning up where the track is). Wall graphics do not rise; a head
rises less than the bar and is a different size.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from server.utils.bar_track import (
    ANALYSIS_WIDTH,
    MAX_RADIUS_RATIO,
    MIN_RADIUS_RATIO,
    MIN_VERTICAL_TRAVEL,
    BarPath,
    BarPoint,
    _circles,
    _smooth,
)

logger = logging.getLogger(__name__)

# Frames are processed at up to this rate; faster footage is thinned to it.
# Flow needs small steps between frames — at 12 fps a pulled bar moves more
# than a plate radius per frame, at 60 it moves a few pixels.
MAX_FPS = 60.0
# Seeds come from circles found in these frames of the window. The window
# opens before the bar leaves the floor (or rack), so this is the setup.
SEED_FRAMES = (0, 4, 8, 12)
MAX_SEEDS = 24
# Corners followed per plate, and how often they are re-found from where the
# plate is now — so the set neither drains away nor drifts onto the body.
FEATURES = 30
REFRESH_EVERY = 3
# Forward-backward check: a corner that does not come back within this many
# pixels was not followed.
ROUND_TRIP_PX = 1.0
LK_PARAMS = dict(winSize=(21, 21), maxLevel=4)
# Re-centring: every REFRESH_EVERY frames a circle within SNAP_RADIUS of the
# plate's size and SNAP_DISTANCE radii of the track takes over from the flow.
SNAP_RADIUS = 0.12
SNAP_DISTANCE = 0.35
# What it takes to be the plate: this share of those checks found a circle of
# the plate's size on the track, and the track rose this far up the frame.
MIN_SUPPORT = 0.3
MIN_RISE = MIN_VERTICAL_TRAVEL
# Points handed on, at most this many per second: the trace and the evidence
# need nothing finer, and a 60 fps trace is 5x the points to draw.
OUTPUT_FPS = 30.0


@dataclass
class _Seed:
    frame: int  # index into the window's frames where the seed was found
    x: float
    y: float
    radius: float
    track: List[Optional[Tuple[float, float]]] = field(default_factory=list)
    checks: int = 0
    found: int = 0

    @property
    def support(self) -> float:
        return self.found / self.checks if self.checks else 0.0


def _read_gray(video_path: str, start_sec: float, end_sec: float, width: int = ANALYSIS_WIDTH):
    """(t, gray frame at `width`) for each processed frame in the span, t in presentation time."""
    import cv2  # type: ignore

    from server.utils.frames import read_frames

    out = []
    for t, frame in read_frames(video_path, start_sec, end_sec, max_fps=MAX_FPS):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > width:
            gray = cv2.resize(gray, (width, int(round(gray.shape[0] * width / gray.shape[1]))), interpolation=cv2.INTER_AREA)
        out.append((t, gray))
    return out


def _seeds(frames, hint: Optional[Tuple[float, float, float, float]]) -> List[_Seed]:
    """Circles in the opening frames, one seed per object, plus the hint."""
    import cv2  # type: ignore

    height, width = frames[0][1].shape[:2]
    shorter = min(width, height)
    low = max(6, int(shorter * MIN_RADIUS_RATIO))
    high = max(low + 4, int(shorter * MAX_RADIUS_RATIO))
    seeds: List[_Seed] = []

    def add(frame: int, x: float, y: float, r: float) -> None:
        for other in seeds:
            if (x - other.x) ** 2 + (y - other.y) ** 2 < (0.5 * other.radius) ** 2 and abs(r - other.radius) < 0.3 * other.radius:
                return
        seeds.append(_Seed(frame, float(x), float(y), float(r)))

    if hint is not None:
        t, hx, hy, hr = hint
        at = min(range(len(frames)), key=lambda i: abs(frames[i][0] - t))
        add(at, hx * width, hy * height, hr)
    for at in SEED_FRAMES:
        if at >= len(frames):
            break
        for x, y, r in _circles(cv2.GaussianBlur(frames[at][1], (5, 5), 0), low, high):
            add(at, x, y, r)
    return seeds[:MAX_SEEDS]


def _features(gray, x: float, y: float, radius: float):
    """Corners inside the plate, found in a patch around it rather than the whole frame."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    height, width = gray.shape[:2]
    reach = int(radius) + 2
    left, top = max(0, int(x) - reach), max(0, int(y) - reach)
    patch = gray[top : int(y) + reach, left : int(x) + reach]
    if patch.size == 0:
        return np.empty((0, 1, 2), np.float32)
    mask = np.zeros_like(patch)
    cv2.circle(mask, (int(x) - left, int(y) - top), int(radius * 0.9), 255, -1)
    found = cv2.goodFeaturesToTrack(patch, FEATURES, 0.01, 4, mask=mask)
    if found is None:
        return np.empty((0, 1, 2), np.float32)
    return found + np.array([left, top], np.float32)


def _snap(gray, x: float, y: float, radius: float) -> Optional[Tuple[float, float]]:
    """A circle of the plate's size close to the track, if there is one."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    reach = int(radius * 1.6)
    top, left = max(0, int(y) - reach), max(0, int(x) - reach)
    patch = gray[top : int(y) + reach, left : int(x) + reach]
    if patch.shape[0] < 8 or patch.shape[1] < 8:
        return None
    found = cv2.HoughCircles(
        cv2.medianBlur(cv2.GaussianBlur(patch, (5, 5), 0), 5), cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(4.0, radius), param1=120, param2=30,
        minRadius=int(radius * (1 - SNAP_RADIUS)), maxRadius=int(radius * (1 + SNAP_RADIUS)) + 1,
    )
    if found is None:
        return None
    best = min(found[0], key=lambda c: np.hypot(c[0] + left - x, c[1] + top - y))
    bx, by = float(best[0] + left), float(best[1] + top)
    return (bx, by) if np.hypot(bx - x, by - y) <= SNAP_DISTANCE * radius else None


def _follow(frames, order: Sequence[int], seeds: List[_Seed]) -> None:
    """
    Every seed through `order` (frame indices, forwards or backwards) at once:
    one flow call per frame for all of their corners, each seed moving by the
    median of its own. A seed joins when `order` reaches its frame, and
    fills seed.track at each frame it is followed through.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    active: List[int] = []
    positions = {}
    corners = {}
    previous = None
    for step, index in enumerate(order):
        gray = frames[index][1]
        for k, seed in enumerate(seeds):
            if seed.frame == index and k not in positions:
                positions[k] = (seed.x, seed.y)
                corners[k] = _features(gray, seed.x, seed.y, seed.radius)
                active.append(k)
                seed.track[index] = (seed.x, seed.y)
        moving = [k for k in active if previous is not None and len(corners[k]) >= 5]
        if moving:
            stacked = np.concatenate([corners[k] for k in moving]).astype(np.float32)
            moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, stacked, None, **LK_PARAMS)
            back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, previous, moved, None, **LK_PARAMS)
            error = np.linalg.norm((back - stacked).reshape(-1, 2), axis=1)
            good = (status.ravel() == 1) & (status_back.ravel() == 1) & (error < ROUND_TRIP_PX)
            shift = (moved - stacked).reshape(-1, 2)
            at = 0
            for k in moving:
                count = len(corners[k])
                mine = good[at : at + count]
                if mine.sum() >= 5:
                    dx, dy = np.median(shift[at : at + count][mine], axis=0)
                    x, y = positions[k]
                    positions[k] = (x + float(dx), y + float(dy))
                    corners[k] = moved[at : at + count][mine].reshape(-1, 1, 2)
                at += count
        refresh = step % REFRESH_EVERY == 0
        for k in active:
            seed = seeds[k]
            x, y = positions[k]
            if refresh and seed.frame != index:
                seed.checks += 1
                rim = _snap(gray, x, y, seed.radius)
                if rim is not None:
                    seed.found += 1
                    x, y = rim
                    positions[k] = rim
            if refresh or len(corners[k]) < 10:
                corners[k] = _features(gray, x, y, seed.radius)
            seed.track[index] = (x, y)
        previous = gray


def _rise(track: List[Optional[Tuple[float, float]]]) -> float:
    """The biggest climb in pixels: from a low point to a later high one."""
    lowest = None
    best = 0.0
    for point in track:
        if point is None:
            continue
        y = point[1]
        lowest = y if lowest is None else max(lowest, y)
        best = max(best, lowest - y)
    return best


def follow_bar_path(
    video_path: str,
    start_sec: float,
    end_sec: float,
    *,
    hint: Optional[Tuple[float, float, float, float]] = None,
) -> BarPath:
    """
    The plate's path through [start_sec, end_sec], as a BarPath in the same
    normalised coordinates as bar_track's. `hint` is (t, x, y, radius px at
    640 wide) for where another method believes the plate was; it becomes one
    more seed, not an answer. Check `.tracked` before believing the result.
    """
    frames = _read_gray(video_path, start_sec, end_sec)
    if len(frames) < 6:
        return BarPath(reason="not enough frames to follow the plate")
    height, width = frames[0][1].shape[:2]
    seeds = _seeds(frames, hint)
    if not seeds:
        return BarPath(sampled=len(frames), frame_width=width, frame_height=height, reason="no circle to start from")
    for seed in seeds:
        seed.track = [None] * len(frames)
    _follow(frames, range(len(frames)), seeds)
    late = [s for s in seeds if s.frame > 0]
    if late:
        # Back from where they were found to the start of the window.
        _follow(frames, range(max(s.frame for s in late), -1, -1), late)

    def score(seed: _Seed) -> Tuple[float, float]:
        rise = _rise(seed.track) / height
        return (rise * seed.support if seed.support >= MIN_SUPPORT and rise >= MIN_RISE else 0.0, seed.radius)

    best = max(seeds, key=score)
    if score(best)[0] <= 0.0:
        rises = sorted(((_rise(s.track) / height, s.support) for s in seeds), reverse=True)[:3]
        logger.info("No plate followed: best (rise, support) %s", [(round(r, 2), round(s, 2)) for r, s in rises])
        return BarPath(
            sampled=len(frames), frame_width=width, frame_height=height,
            reason="nothing followed through the window rose like a bar and stayed round like a plate",
        )
    keep_every = max(1, int(round((len(frames) / max(1e-6, frames[-1][0] - frames[0][0])) / OUTPUT_FPS)))
    points = [
        BarPoint(t=frames[i][0], x=p[0] / width, y=p[1] / height)
        for i, p in enumerate(best.track)
        if p is not None and i % keep_every == 0
    ]
    followed = sum(1 for p in best.track if p is not None)
    return BarPath(
        points=_smooth(points),
        plate_radius=best.radius / width,
        coverage=followed / len(frames),
        sampled=len(frames),
        frame_width=width,
        frame_height=height,
    )
