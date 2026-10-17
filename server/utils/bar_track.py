"""
Follow the barbell through a lift, locally, with no model call.

The plate is the one big circle in the frame that moves, so: find the moving
region across the window, look for circles each frame, and keep the one that is
consistent with where the plate was a moment ago. Everything is returned in
normalised coordinates (0..1 of the frame) so a trace drawn from it lands in the
right place whatever resolution the replay is encoded at.

This is deliberately conservative. A trace drawn through the wrong object would
be worse than no trace, so `BarPath.tracked` is False unless the path covers most
of the window and actually goes up and down like a lift.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# A competition bumper plate is 450 mm across. That is what turns a pixel
# measurement into centimetres; training plates are smaller, so the figure is an
# estimate and is presented as one.
PLATE_DIAMETER_M = 0.45

ANALYSIS_WIDTH = 640
SAMPLE_FPS = 12.0
# Plate radius as a fraction of the frame's shorter side. Wide enough for a
# close crop and for a lifter filmed from across the gym.
MIN_RADIUS_RATIO = 0.03
MAX_RADIUS_RATIO = 0.13
# How far the plate may move between samples, in plate radii. A bar at peak
# speed covers well under half its own diameter per frame at these sample rates,
# so a generous 1.4 radii still rules out hopping to an unrelated circle. Tying
# this to the frame instead of the plate let trails jump between wall posters.
MAX_STEP_RADII = 0.9
# How far a candidate's mean colour may sit from the anchor plate's, in 0-255
# BGR distance. A barbell has a plate at each end, often different colours and,
# on a slightly off-axis camera, a long way apart horizontally — hopping from
# one to the other invents a huge sideways move that never happened.
MAX_COLOUR_DISTANCE = 70.0
# Mean motion energy along a trail. A circle in a wall banner scores ~0; the bar
# does not. This is what separates a real lift from a static pattern.
MIN_TRAIL_MOTION = 3.0
# Mean 8-bit motion energy the anchor plate must show. Below this nothing in the
# window is really moving and anything we locked onto is scenery.
MIN_ANCHOR_MOTION = 6.0
# Hypotheses tracked in parallel before picking the one that behaves like a bar.
MAX_SEEDS = 14
# What it takes to believe the result. A lift moves the bar a long way up the
# frame; a tracked object that barely rises is furniture, not a barbell.
MIN_COVERAGE = 0.55
MIN_POINTS = 6
MIN_VERTICAL_TRAVEL = 0.10


@dataclass
class BarPoint:
    t: float
    x: float
    y: float


@dataclass
class BarPath:
    points: List[BarPoint] = field(default_factory=list)
    plate_radius: float = 0.0  # normalised to frame width
    coverage: float = 0.0
    sampled: int = 0
    frame_width: int = 0
    frame_height: int = 0
    reason: str = ""

    @property
    def tracked(self) -> bool:
        return (
            len(self.points) >= MIN_POINTS
            and self.coverage >= MIN_COVERAGE
            and self.vertical_travel >= MIN_VERTICAL_TRAVEL
        )

    @property
    def vertical_travel(self) -> float:
        if not self.points:
            return 0.0
        ys = [p.y for p in self.points]
        return max(ys) - min(ys)

    @property
    def reference_x(self) -> float:
        """
        Where the bar started, which stands in for the line over mid-foot that a
        coach eyeballs. Median of the opening points, so one bad detection in the
        setup does not move it.
        """
        if not self.points:
            return 0.0
        head = sorted(p.x for p in self.points[: max(1, len(self.points) // 6)])
        return head[len(head) // 2]

    def max_drift(self) -> Tuple[float, Optional[BarPoint]]:
        """Largest horizontal departure from the starting line, normalised."""
        if not self.points:
            return 0.0, None
        reference = self.reference_x
        worst = max(self.points, key=lambda p: abs(p.x - reference))
        return abs(worst.x - reference), worst

    def drift_cm(self) -> Optional[float]:
        """
        Max drift in centimetres, scaled by the plate's apparent diameter.
        None when we have no plate size to scale by.
        """
        if not self.plate_radius or not self.frame_width:
            return None
        drift_norm, _ = self.max_drift()
        drift_px = drift_norm * self.frame_width
        plate_px = 2 * self.plate_radius * self.frame_width
        if plate_px <= 0:
            return None
        return (drift_px / plate_px) * PLATE_DIAMETER_M * 100.0


def _frame_motion(grays):
    """
    Per-frame difference maps. A window-wide mask is not enough on its own: it is
    dilated, so a plate parked on a stand next to a moving athlete falls inside
    it and makes a very attractive big circle. Scoring each candidate against the
    motion *at its own frame* is what separates the bar from the furniture.
    """
    import cv2  # type: ignore

    diffs = [cv2.absdiff(first, second) for first, second in zip(grays, grays[1:])]
    if not diffs:
        return [grays[0]]
    # Frame 0 has no predecessor; reuse the first difference for it.
    return [diffs[0]] + diffs


def _mean_colour(frame, circle, radius_scale: float = 0.7):
    """Mean BGR inside the circle — used to keep hold of the same plate."""
    import numpy as np  # type: ignore

    x, y, r = circle
    r = max(2, int(r * radius_scale))
    height, width = frame.shape[:2]
    top, bottom = max(0, y - r), min(height, y + r)
    left, right = max(0, x - r), min(width, x + r)
    if bottom <= top or right <= left:
        return np.zeros(3, dtype=np.float32)
    return frame[top:bottom, left:right].reshape(-1, 3).mean(axis=0).astype(np.float32)


def _motion_score(motion, circle, radius_scale: float = 0.8) -> float:
    """Mean motion energy inside the circle — how fast this thing is moving."""
    import numpy as np  # type: ignore

    x, y, r = circle
    r = max(2, int(r * radius_scale))
    height, width = motion.shape[:2]
    top, bottom = max(0, y - r), min(height, y + r)
    left, right = max(0, x - r), min(width, x + r)
    if bottom <= top or right <= left:
        return 0.0
    patch = motion[top:bottom, left:right]
    return float(np.mean(patch))


def _circles(gray, min_radius: int, max_radius: int):
    import cv2  # type: ignore

    found = cv2.HoughCircles(
        cv2.medianBlur(gray, 5),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_radius * 2),
        param1=120,
        param2=38,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if found is None:
        return []
    return [(int(c[0]), int(c[1]), int(c[2])) for c in found[0]]


def _read_window(video_path: str, start_sec: float, end_sec: float, sample_fps: float, analysis_width: int):
    import cv2  # type: ignore

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return 0.0, []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (0 < fps <= 480):
            fps = 30.0
        step = max(1, int(round(fps / max(1.0, sample_fps))))
        first = max(0, int(round(max(0.0, start_sec) * fps)))
        last = int(round(max(start_sec, end_sec) * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)

        samples = []
        index = first
        while index <= last:
            ok, frame = capture.read()
            if not ok:
                break
            if (index - first) % step == 0:
                # Downscale only: blowing a small clip up costs time and adds no
                # detail for Hough to find.
                if frame.shape[1] > analysis_width:
                    scale = analysis_width / frame.shape[1]
                    frame = cv2.resize(
                        frame, (analysis_width, max(1, int(round(frame.shape[0] * scale))))
                    )
                samples.append((index / fps, frame))
            index += 1
        return fps, samples
    finally:
        capture.release()


def track_bar_path(
    video_path: str,
    *,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    sample_fps: float = SAMPLE_FPS,
    analysis_width: int = ANALYSIS_WIDTH,
) -> BarPath:
    """
    Follow the plate through [start_sec, end_sec]. Always returns a BarPath;
    check `.tracked` before believing it.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if end_sec is None:
        end_sec = start_sec + 10.0
    _, samples = _read_window(video_path, start_sec, end_sec, sample_fps, analysis_width)
    if len(samples) < MIN_POINTS:
        return BarPath(reason="not enough frames to track")

    height, width = samples[0][1].shape[:2]
    grays = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for _, f in samples]
    motions = _frame_motion(grays)

    shorter = min(width, height)
    min_radius = max(6, int(shorter * MIN_RADIUS_RATIO))
    max_radius = max(min_radius + 4, int(shorter * MAX_RADIUS_RATIO))
    per_frame = [_circles(gray, min_radius, max_radius) for gray in grays]
    colours = [
        {circle: _mean_colour(frame, circle) for circle in candidates}
        for (_t, frame), candidates in zip(samples, per_frame)
    ]

    seeds = _seed_candidates(per_frame, motions, shorter)
    if not seeds:
        return BarPath(
            sampled=len(samples),
            frame_width=width,
            frame_height=height,
            reason="nothing round was moving enough to be the bar",
        )

    # A head is also round, high-contrast and fast, so the fastest circle is not
    # reliably the bar. Track every plausible seed instead and keep the path that
    # behaves like a barbell: the lifted thing travels far up the frame, while a
    # head or a plate on a stand does not.
    best: Optional[Tuple[float, List[Tuple[int, Tuple[int, int, int]]]]] = None
    for _score, anchor_index, anchor in seeds:
        trail = _walk_from(anchor_index, anchor, per_frame, colours)
        if len(trail) < MIN_POINTS:
            continue
        coverage = len(trail) / len(samples)
        if coverage < MIN_COVERAGE:
            continue
        moving = sum(_motion_score(motions[index], circle) for index, circle in trail) / len(trail)
        if moving < MIN_TRAIL_MOTION:
            continue  # a static pattern that happens to be round
        ys = [circle[1] / height for _, circle in trail]
        travel = max(ys) - min(ys)
        rank = travel * (0.5 + coverage)
        if best is None or rank > best[0]:
            best = (rank, trail)

    if best is None:
        # Fall back to the strongest seed so the caller still gets a diagnosis.
        _score, anchor_index, anchor = seeds[0]
        trail = _walk_from(anchor_index, anchor, per_frame, colours)
    else:
        trail = best[1]

    points: List[BarPoint] = []
    radii: List[int] = []
    for index, (x, y, r) in trail:
        radii.append(r)
        points.append(BarPoint(t=samples[index][0], x=x / width, y=y / height))

    path = BarPath(
        points=_smooth(points),
        plate_radius=(float(np.median(radii)) / width) if radii else 0.0,
        coverage=len(points) / len(samples),
        sampled=len(samples),
        frame_width=width,
        frame_height=height,
    )
    if not path.tracked:
        path.reason = _why_not(path)
        logger.info("Bar tracking rejected: %s", path.reason)
    return path


def _seed_candidates(per_frame, motions, shorter: int, limit: int = MAX_SEEDS):
    """
    The most clearly moving circles in the window, spread out so we do not spend
    every hypothesis on one object. Returns (motion score, frame index, circle).
    """
    scored = []
    for index, candidates in enumerate(per_frame):
        for circle in candidates:
            score = _motion_score(motions[index], circle)
            if score >= MIN_ANCHOR_MOTION:
                scored.append((score, index, circle))
    scored.sort(key=lambda item: -item[0])

    kept: List = []
    spacing = shorter * 0.08
    for score, index, circle in scored:
        if any(
            abs(index - other_index) < 4
            and abs(circle[0] - other[0]) < spacing
            and abs(circle[1] - other[1]) < spacing
            for _s, other_index, other in kept
        ):
            continue
        kept.append((score, index, circle))
        if len(kept) >= limit:
            break
    return kept


def _walk_from(anchor_index: int, anchor, per_frame, colours):
    """
    Follow one object forwards from its anchor frame, then backwards into the
    setup — where the bar rests on the floor and has no motion to be found by.
    Radius is held near the anchor's, which stops the trail hopping to a
    differently sized circle.
    """
    import numpy as np  # type: ignore

    radius_tolerance = max(4.0, anchor[2] * 0.35)
    max_step = max(8.0, anchor[2] * MAX_STEP_RADII)
    anchor_colour = colours[anchor_index].get(anchor)

    def walk(indices):
        found = []
        previous = anchor
        for index in indices:
            candidates = [
                circle
                for circle in per_frame[index]
                if abs(circle[2] - anchor[2]) <= radius_tolerance
                and _colour_distance(colours[index].get(circle), anchor_colour) <= MAX_COLOUR_DISTANCE
            ]
            if not candidates:
                continue
            pick = min(
                candidates,
                key=lambda c: np.hypot(c[0] - previous[0], c[1] - previous[1]) + 3 * abs(c[2] - previous[2]),
            )
            if np.hypot(pick[0] - previous[0], pick[1] - previous[1]) > max_step:
                continue  # too far to be the same plate; wait for it to reappear
            previous = pick
            found.append((index, pick))
        return found

    tracked = dict(walk(range(anchor_index, len(per_frame))))
    tracked.update(dict(walk(range(anchor_index, -1, -1))))
    tracked[anchor_index] = anchor
    return [(index, tracked[index]) for index in sorted(tracked)]


def _colour_distance(colour, reference) -> float:
    import numpy as np  # type: ignore

    if colour is None or reference is None:
        return 0.0
    return float(np.linalg.norm(colour - reference))


def _smooth(points: List[BarPoint], window: int = 3) -> List[BarPoint]:
    """Take the jitter out of the detections without flattening the loop."""
    if len(points) < window:
        return points
    half = window // 2
    smoothed: List[BarPoint] = []
    for index, point in enumerate(points):
        low = max(0, index - half)
        high = min(len(points), index + half + 1)
        neighbours = points[low:high]
        smoothed.append(
            BarPoint(
                t=point.t,
                x=sum(p.x for p in neighbours) / len(neighbours),
                y=sum(p.y for p in neighbours) / len(neighbours),
            )
        )
    return smoothed


def _why_not(path: BarPath) -> str:
    if len(path.points) < MIN_POINTS:
        return f"only {len(path.points)} detections"
    if path.coverage < MIN_COVERAGE:
        return f"the bar was found in only {path.coverage:.0%} of frames"
    if path.vertical_travel < MIN_VERTICAL_TRAVEL:
        return "whatever was tracked barely moved vertically, so it is probably not the bar"
    return "unknown"
