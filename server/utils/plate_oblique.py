"""
Follow the near plate through a clip filmed from an angle, and say whether a
trace from any tracker is on a plate at all.

From a phone held ~45° off the side, the bumper plate nearest the camera is an
ellipse — taller than wide, with an axis ratio around 0.6 — and the circle-only
Hough in bar_track and plate_follow never finds it. Those trackers then settle
on whatever *is* round and moving: the smaller far plate, a window with foliage
behind it, and `BarPath.tracked` cannot tell. This module:

* runs Hough on the frame stretched sideways by 1.0 / 1.3 / 1.6 / 2.0, so an
  ellipse with that axis ratio becomes a circle at one of the stretches, and
  measures the plate's axis ratio as 1 / the stretch that fitted best;
* keeps only candidates whose inside is dark enough to be a plate (a window,
  foliage and a lit wall are not);
* walks forward from the plate on the floor at the start of the window, and
  backward from every plate-like thing found high in the frame — a backward
  walk counts only if it lands back on the floor plate, which a ring hanging
  from the ceiling or the far plate never does — and keeps the trail that
  rose furthest in pixels (the near plate rises more than the far one);
* costs one full-frame scan every third frame and a small patch around the
  prediction on the others, with every scan cached per frame.

`looks_like_a_plate` is the same darkness-and-rim test applied to a finished
trace, so the legacy trackers' fallbacks are checked before they are drawn.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from server.utils.bar_track import (
    MAX_RADIUS_RATIO,
    MIN_RADIUS_RATIO,
    MIN_VERTICAL_TRAVEL,
    BarPath,
    BarPoint,
    _smooth,
)

logger = logging.getLogger(__name__)

# Frames are analysed at this rate and size. Twelve a second is enough for a
# walk that predicts where the plate goes next (a bar at full speed moves well
# under a plate radius between frames), and a 480 px long side keeps a plate
# at 15-60 px, where Hough is quick and still exact to a pixel or two.
ANALYSIS_FPS = 12.0
ANALYSIS_LONG_SIDE = 480
# Sideways stretches tried per frame. An ellipse of axis ratio q (minor/major,
# horizontal/vertical) is a circle once the frame is stretched by 1/q, so these
# cover ratios 1.0, 0.77, 0.63 and 0.5 — square to the side down to ~60° off.
STRETCHES = (1.0, 1.3, 1.6, 2.0)
# The inside of a plate, as mean 8-bit grey over the inner 60 % of its disc.
# The oblique clip that motivated this gate had a black bumper under 90, but
# on 522 samples from 44 hand-labelled clips (tests/data/*bar_path_truth.json)
# real plates read a median of 100, p90 128 and p95 137: coloured bumpers,
# hub lettering and gym lighting. 130 keeps 91 % of those samples and still
# rejects a lit window, foliage and a white wall (all 150+).
PLATE_MAX_GREY = 130.0
# Share of points around the rim where the image has an edge. A plate has a
# rim all the way round; a texture that happens to be round does not. On the
# labelled clips a plate's rim reads 0.5-0.75 at its centre (p25-p75), 0.36
# at p10. A rim this complete (RIM_ALONE) is a disc whatever its colour: the
# white and yellow competition bumpers read 160-185 inside and 0.75-0.97
# around, and `looks_like_a_plate` keeps them on the rim alone.
MIN_RIM_SUPPORT = 0.35
RIM_ALONE = 0.75
RIM_SAMPLES = 36
RIM_EDGE = 10.0  # grey levels of gradient (3x3 Sobel / 4) that count as an edge
# Hough: a stretched ellipse is only nearly a circle, so the accumulator
# threshold sits under bar_track's 38; the strongest few circles per stretch
# are plenty, the gate and the walks do the choosing.
HOUGH_PARAM2 = 30
MAX_PER_STRETCH = 12
# A full-frame scan (all stretches) on every KEY_EVERY-th frame and on the
# opening FLOOR_FRAMES; the walks look in a patch around their prediction on
# the frames between.
KEY_EVERY = 3
FLOOR_FRAMES = 4
# The walk: the plate may move this many of its own radii per frame of gap,
# the gap is capped, and after this many missed frames the walk stops.
MAX_STEP_RADII = 1.5
MAX_GAP = 4
LOST_AFTER = 10
RADIUS_TOLERANCE = 0.3
# A plate's inside does not change brightness much from one found frame to the
# next; a step of this many grey levels is a hop onto something else.
GREY_STEP = 60.0
# Two trails on the same object agree (within a plate radius) on this share
# of the frames they both saw; only the better of the two is kept.
SAME_OBJECT_SHARE = 0.8
# A backward walk from something high in the frame must end this close (in
# plate radii) to a plate found on the floor at the start of the window.
LAND_RADII = 1.5
MAX_TOPS = 6
# What a trail needs before it is believed: found in this share of the
# frames it spans, and risen this much of the frame (bar_track's rule).
MIN_FOUND_SHARE = 0.5
MIN_RISE = MIN_VERTICAL_TRAVEL
# How far (as a share of the frame's long side) a tap may be from the plate
# it meant.
TAP_REACH = 0.08
# Below this measured axis ratio the view is oblique: the follower here is
# trusted over the side-on follower, and confidence.py caps confidence.
OBLIQUE_RATIO = 0.93

NO_TRACE_REASON = "Couldn't track the bar — film from the side with the near plate visible"


@dataclass
class Plate:
    """One plate-like detection: an ellipse with vertical semi-axis `r`."""

    x: float
    y: float
    r: float  # vertical semi-axis, analysis pixels
    stretch: float  # the sideways stretch it was a circle at
    grey: float  # mean grey of the inner disc
    support: float  # share of the rim that has an edge

    @property
    def rx(self) -> float:
        """Horizontal semi-axis."""
        return self.r / self.stretch

    @property
    def ratio(self) -> float:
        """Minor/major axis ratio: 1 side-on, smaller the more oblique."""
        return 1.0 / self.stretch

    def distance(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


# ---------- frames ----------


def _prepare(frames: Sequence[Tuple[float, "np.ndarray"]]):
    """
    Thin `frames` ((t, gray or BGR) in time order, any rate and size) to
    ANALYSIS_FPS and ANALYSIS_LONG_SIDE. Downscale only: a small clip is not
    blown up.
    """
    import cv2  # type: ignore

    kept = []
    due = None
    for t, frame in frames:
        if due is not None and t < due:
            continue
        due = t + 0.98 / ANALYSIS_FPS
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        long_side = max(height, width)
        if long_side > ANALYSIS_LONG_SIDE:
            scale = ANALYSIS_LONG_SIDE / long_side
            frame = cv2.resize(
                frame, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA
            )
        kept.append((t, frame))
    return kept


def read_frames_for(video_path: str, start_sec: float, end_sec: float):
    """(t, gray) at the analysis rate and size, straight from the clip."""
    from server.utils.frames import read_frames

    return _prepare(read_frames(video_path, start_sec, end_sec, max_fps=ANALYSIS_FPS))


def _gradient(gray):
    """Gradient magnitude in grey levels per pixel (3x3 Sobel, scaled back)."""
    import cv2  # type: ignore

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy) * 0.25


# ---------- one frame's plate-like things ----------


def _inner_grey(gray, x: float, y: float, rx: float, ry: float) -> float:
    """Mean grey over the inner 60 % of the ellipse."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    ax, ay = max(1, int(rx * 0.6)), max(1, int(ry * 0.6))
    height, width = gray.shape[:2]
    left, top = max(0, int(x) - ax - 1), max(0, int(y) - ay - 1)
    right, bottom = min(width, int(x) + ax + 2), min(height, int(y) + ay + 2)
    if right - left < 3 or bottom - top < 3:
        return 255.0
    patch = gray[top:bottom, left:right]
    mask = np.zeros(patch.shape, np.uint8)
    cv2.ellipse(mask, (int(x) - left, int(y) - top), (ax, ay), 0, 0, 360, 255, -1)
    values = patch[mask > 0]
    return float(values.mean()) if values.size else 255.0


def _rim_support(grad, x: float, y: float, rx: float, ry: float, samples: int = RIM_SAMPLES) -> float:
    """
    Share of directions in which the ellipse's rim sits on an edge. Each
    direction looks at a short radial band (0.9-1.1 of the axes) so a radius
    a pixel or two off, or a rim drawn as a thick ring, still counts.
    """
    import numpy as np  # type: ignore

    height, width = grad.shape[:2]
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    best = np.zeros(samples, np.float32)
    seen = np.zeros(samples, bool)
    for scale in (0.9, 1.0, 1.1):
        xs = np.rint(x + scale * rx * np.cos(angles)).astype(int)
        ys = np.rint(y + scale * ry * np.sin(angles)).astype(int)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        best[inside] = np.maximum(best[inside], grad[ys[inside], xs[inside]])
        seen |= inside
    if seen.sum() < samples // 2:
        return 0.0
    return float(np.mean(best[seen] >= RIM_EDGE))


def _hough(gray, low: int, high: int):
    import cv2  # type: ignore

    found = cv2.HoughCircles(
        cv2.medianBlur(gray, 5),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(4.0, float(low)),
        param1=120,
        param2=HOUGH_PARAM2,
        minRadius=low,
        maxRadius=high,
    )
    if found is None:
        return []
    return [(float(c[0]), float(c[1]), float(c[2])) for c in found[0][:MAX_PER_STRETCH]]


def _detect(
    gray,
    grad,
    low: int,
    high: int,
    stretches: Sequence[float] = STRETCHES,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> List[Plate]:
    """
    Plate-like ellipses in one frame (or in its `roi` = (left, top, right,
    bottom)), at every stretch asked for, gated by darkness and rim, with the
    duplicates one object produces at neighbouring stretches collapsed onto
    the best-supported one.
    """
    import cv2  # type: ignore

    height, width = gray.shape[:2]
    if roi is None:
        left, top, right, bottom = 0, 0, width, height
    else:
        left, top = max(0, roi[0]), max(0, roi[1])
        right, bottom = min(width, roi[2]), min(height, roi[3])
    if right - left < 2 * low + 4 or bottom - top < 2 * low + 4:
        return []
    patch = gray[top:bottom, left:right]
    found: List[Plate] = []
    for stretch in stretches:
        if stretch == 1.0:
            image = patch
        else:
            image = cv2.resize(patch, (int(round(patch.shape[1] * stretch)), patch.shape[0]), interpolation=cv2.INTER_LINEAR)
        for cx, cy, r in _hough(image, low, high):
            x, y = left + cx / stretch, top + cy
            rx = r / stretch
            grey = _inner_grey(gray, x, y, rx, r)
            if grey > PLATE_MAX_GREY:
                continue
            support = _rim_support(grad, x, y, rx, r)
            if support < MIN_RIM_SUPPORT:
                continue
            found.append(Plate(x, y, r, stretch, grey, support))
    return _dedupe(found)


def _dedupe(plates: List[Plate]) -> List[Plate]:
    """One detection per object: the best-supported, the least stretched on a tie."""
    kept: List[Plate] = []
    for plate in sorted(plates, key=lambda p: (-p.support, p.stretch)):
        if any(
            plate.distance(other.x, other.y) < 0.5 * other.r and abs(plate.r - other.r) <= RADIUS_TOLERANCE * other.r
            for other in kept
        ):
            continue
        kept.append(plate)
    return kept


class _Detections:
    """Per-frame detections, computed on demand and remembered."""

    def __init__(self, frames, low: int, high: int) -> None:
        self.frames = frames
        self.low, self.high = low, high
        self._grad: Dict[int, "np.ndarray"] = {}
        self._full: Dict[int, List[Plate]] = {}
        self.scans = 0  # full-frame scans, for the tests

    def grad(self, index: int):
        if index not in self._grad:
            self._grad[index] = _gradient(self.frames[index][1])
        return self._grad[index]

    def full(self, index: int) -> List[Plate]:
        if index not in self._full:
            self._full[index] = _detect(self.frames[index][1], self.grad(index), self.low, self.high)
            self.scans += 1
        return self._full[index]

    def is_key(self, index: int) -> bool:
        return index < FLOOR_FRAMES or index % KEY_EVERY == 0 or index == len(self.frames) - 1

    def near(self, index: int, x: float, y: float, reach: float, like: Plate) -> List[Plate]:
        """
        Plates within `reach` of (x, y) at frame `index`: from the full scan
        when there is one, else from a patch around the point at the stretch
        the plate was last seen at and its neighbours.
        """
        if index in self._full or self.is_key(index):
            plates = self.full(index)
        else:
            margin = int(reach + like.r * 1.5) + 4
            roi = (int(x) - margin, int(y) - margin, int(x) + margin, int(y) + margin)
            position = STRETCHES.index(like.stretch) if like.stretch in STRETCHES else 0
            stretches = STRETCHES[max(0, position - 1) : position + 2]
            plates = _detect(self.frames[index][1], self.grad(index), self.low, self.high, stretches, roi)
        return [p for p in plates if p.distance(x, y) <= reach + p.r]


# ---------- walks ----------


def _walk(detections: _Detections, start_index: int, start: Plate, step: int) -> Dict[int, Plate]:
    """
    Follow one plate from `start` frame by frame (step +1 forwards, -1 back),
    predicting the next position from the last two, surviving frames where
    Hough misses it, and refusing to hop to anything too far, too different
    in size or too much brighter to be the same plate.
    """
    found = {start_index: start}
    previous, previous_index = start, start_index
    velocity = (0.0, 0.0)
    last = len(detections.frames) if step > 0 else -1
    for index in range(start_index + step, last, step):
        gap = abs(index - previous_index)
        if gap > LOST_AFTER:
            break
        predicted = (previous.x + velocity[0] * gap, previous.y + velocity[1] * gap)
        reach = MAX_STEP_RADII * start.r * min(gap, MAX_GAP)
        candidates = [
            plate
            for plate in detections.near(index, predicted[0], predicted[1], reach, previous)
            if abs(plate.r - start.r) <= RADIUS_TOLERANCE * start.r and abs(plate.grey - previous.grey) <= GREY_STEP
        ]
        if not candidates:
            continue

        def cost(plate: Plate) -> float:
            return (
                min(plate.distance(*predicted), plate.distance(previous.x, previous.y))
                + 2.0 * abs(plate.r - previous.r)
                + 0.3 * abs(plate.grey - previous.grey)
                - 10.0 * plate.support
            )

        pick = min(candidates, key=cost)
        if min(pick.distance(*predicted), pick.distance(previous.x, previous.y)) > reach:
            continue
        if gap <= MAX_GAP:
            velocity = ((pick.x - previous.x) / gap, (pick.y - previous.y) / gap)
        else:
            velocity = (0.0, 0.0)
        previous, previous_index = pick, index
        found[index] = pick
    return found


def tap_frame(frames, t: float) -> int:
    """
    The frame the athlete's tap was made on: the one nearest `t` when `t`
    falls inside the window (half a second of slack for the trim's own
    rounding), else the first frame — the tap was on the first trimmed frame
    and the window begins there or later.
    """
    if not frames:
        return 0
    if frames[0][0] - 0.5 <= t <= frames[-1][0] + 0.5:
        return min(range(len(frames)), key=lambda i: abs(frames[i][0] - t))
    return 0


def _floor_seeds(detections: _Detections, tap: Optional[Tuple[float, float, float]]) -> List[Tuple[int, Plate, bool]]:
    """
    Plates on the floor at the start of the window: (frame, plate, tapped).
    One per object across the opening frames, plus the frame the athlete
    tapped on when that is later; `tapped` marks the one nearest the tap, if
    a plate was found within reach of it.
    """
    seeds: List[Tuple[int, Plate, bool]] = []
    indices = list(range(min(FLOOR_FRAMES, len(detections.frames))))
    tapped_index = tap_frame(detections.frames, tap[0]) if tap is not None else 0
    if tapped_index not in indices:
        indices.append(tapped_index)
    for index in indices:
        for plate in detections.full(index):
            if any(plate.distance(s.x, s.y) < 0.5 * s.r and abs(plate.r - s.r) <= RADIUS_TOLERANCE * s.r for _, s, _ in seeds):
                continue
            seeds.append((index, plate, False))
    if tap is not None and seeds:
        height, width = detections.frames[0][1].shape[:2]
        tx, ty = tap[1] * width, tap[2] * height
        reach = TAP_REACH * max(width, height)
        # Prefer a plate seen on the tapped frame itself; any other frame's
        # plate at the same spot stands in for it (the bar is still there).
        on_frame = [k for k in range(len(seeds)) if seeds[k][0] == tapped_index]
        pool = on_frame or list(range(len(seeds)))
        nearest = min(pool, key=lambda k: seeds[k][1].distance(tx, ty))
        if seeds[nearest][1].distance(tx, ty) <= reach + seeds[nearest][1].r:
            index, plate, _ = seeds[nearest]
            seeds[nearest] = (index, plate, True)
    return seeds


def _top_candidates(detections: _Detections, floor_y: float, height: int) -> List[Tuple[int, Plate]]:
    """
    Plate-like things found well above the floor on the key frames, highest
    first, one per object. Each is a hypothesis for where the bar got to.
    """
    tops: List[Tuple[int, Plate]] = []
    limit = floor_y - MIN_RISE * height
    for index in range(len(detections.frames)):
        if not detections.is_key(index):
            continue
        for plate in detections.full(index):
            if plate.y > limit:
                continue
            if any(plate.distance(p.x, p.y) < 1.0 * p.r and abs(plate.r - p.r) <= RADIUS_TOLERANCE * p.r for _, p in tops):
                continue
            tops.append((index, plate))
    tops.sort(key=lambda item: item[1].y)
    return tops[:MAX_TOPS]


def _rise_px(trail: Dict[int, Plate]) -> float:
    """The biggest climb: from a low point to a later high one, in pixels."""
    lowest = None
    best = 0.0
    for index in sorted(trail):
        y = trail[index].y
        lowest = y if lowest is None else max(lowest, y)
        best = max(best, lowest - y)
    return best


def _median_radius(trail: Dict[int, Plate]) -> float:
    radii = sorted(p.r for p in trail.values())
    return radii[len(radii) // 2] if radii else 0.0


def _travel(trail: Dict[int, Plate]) -> float:
    """Total path length: of two trails on one object the jittery one is longer."""
    order = sorted(trail)
    return sum(trail[a].distance(trail[b].x, trail[b].y) for a, b in zip(order, order[1:]))


def _same_object(a: Dict[int, Plate], b: Dict[int, Plate]) -> bool:
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return False
    radius = max(1.0, _median_radius(a))
    agreed = sum(1 for i in common if a[i].distance(b[i].x, b[i].y) <= radius)
    return agreed / len(common) >= SAME_OBJECT_SHARE


def _consensus(trails: List[Tuple[Dict[int, Plate], bool]]) -> List[Tuple[Dict[int, Plate], bool]]:
    """
    One trail per object. A floor seed's forward walk and a top candidate's
    backward walk usually meet on the same plate; of the trails that agree
    the one found in the most frames with the least jitter stands for it,
    and a stray detection at one end of a duplicate cannot outvote it.
    """
    ordered = sorted(trails, key=lambda item: (-len(item[0]), _travel(item[0])))
    kept: List[Tuple[Dict[int, Plate], bool]] = []
    for trail, tapped in ordered:
        for position, (other, other_tapped) in enumerate(kept):
            if _same_object(other, trail):
                if tapped and not other_tapped:
                    kept[position] = (other, True)
                break
        else:
            kept.append((trail, tapped))
    return kept


def _axis_ratio(trail: Dict[int, Plate], frames) -> Optional[float]:
    """
    The plate's minor/major axis ratio over the trail. The stretch that
    fitted each frame gives it in four coarse steps (1, 0.77, 0.63, 0.5); an
    ellipse fitted to the rim where the plate was found gives it finely, so
    the median of those fits is used wherever enough of them succeed, and
    the median of 1/stretch otherwise.
    """
    import numpy as np  # type: ignore

    from server.utils.plate_follow import _ellipse_centre

    order = sorted(trail)
    if not order:
        return None
    fitted = []
    for index in order:
        plate = trail[index]
        fit = _ellipse_centre(frames[index][1], plate.x, plate.y, plate.r)
        if fit is not None:
            fitted.append(fit[1])
    if len(fitted) >= max(3, len(order) // 2):
        return float(np.median(fitted))
    return float(np.median([trail[i].ratio for i in order]))


def _trail_path(trail: Dict[int, Plate], frames, reason: str = "", tapped: bool = False) -> BarPath:
    """A trail as a BarPath in normalised coordinates."""
    import numpy as np  # type: ignore

    height, width = frames[0][1].shape[:2]
    order = sorted(trail)
    points = [BarPoint(t=frames[i][0], x=trail[i].x / width, y=trail[i].y / height) for i in order]
    if order:
        span = order[-1] - order[0] + 1
    else:
        span = len(frames)
    radii = [trail[i].r for i in order]
    return BarPath(
        points=_smooth(points),
        plate_radius=(float(np.median(radii)) / width) if radii else 0.0,
        coverage=len(order) / max(1, span),
        sampled=len(frames),
        frame_width=width,
        frame_height=height,
        plate_axis_ratio=_axis_ratio(trail, frames),
        tapped=tapped,
        reason=reason,
    )


def follow_oblique(
    video_path: Optional[str],
    start_sec: float = 0.0,
    end_sec: float = 0.0,
    *,
    hint: Optional[Tuple[float, float, float]] = None,
    frames=None,
) -> BarPath:
    """
    The near plate's path through [start_sec, end_sec] of `video_path`, or
    through `frames` ((t, gray or BGR) already decoded, any rate and size —
    the pipeline decodes a window once and hands it to every follower).
    `hint` is (t, x, y): when the athlete tapped the near plate, in seconds
    of the clip, and where, as fractions of the frame. It decides between
    plates that both rise, never whether one did; `.tapped` on the result
    says whether the plate followed is the one they meant. Check `.tracked`
    before believing the result; `.reason` says why not.
    """
    frames = _prepare(frames) if frames is not None else read_frames_for(video_path, start_sec, end_sec)
    if len(frames) < 6:
        return BarPath(sampled=len(frames), reason="not enough frames to follow the plate")
    height, width = frames[0][1].shape[:2]
    shorter = min(width, height)
    low = max(6, int(shorter * MIN_RADIUS_RATIO))
    high = max(low + 4, int(shorter * MAX_RADIUS_RATIO))
    detections = _Detections(frames, low, high)

    seeds = _floor_seeds(detections, hint)
    trails: List[Tuple[Dict[int, Plate], bool]] = []
    for index, plate, tapped in seeds:
        trail = _walk(detections, index, plate, +1)
        if index > 0:
            trail.update(_walk(detections, index, plate, -1))
        trails.append((trail, tapped))

    floor_y = max((plate.y for _, plate, _ in seeds), default=float(height))
    for index, plate in _top_candidates(detections, floor_y, height):
        back = _walk(detections, index, plate, -1)
        landed_at = min(back)
        if landed_at > FLOOR_FRAMES:
            continue  # lost before it got back to the start of the window
        landing = back[landed_at]
        if seeds:
            near = [s for _, s, _ in seeds if landing.distance(s.x, s.y) <= LAND_RADII * s.r]
            if not near:
                continue  # did not come from the floor plate: a ring, a logo, the far plate
            tapped = any(t and landing.distance(s.x, s.y) <= LAND_RADII * s.r for _, s, t in seeds)
        else:
            tapped = False
        trail = dict(back)
        trail.update(_walk(detections, index, plate, +1))
        trails.append((trail, tapped))

    if not trails:
        return BarPath(sampled=len(frames), frame_width=width, frame_height=height, reason="no plate-like ellipse to start from")
    trails = _consensus(trails)
    largest = max(_median_radius(trail) for trail, _ in trails) or 1.0

    def score(item: Tuple[Dict[int, Plate], bool]) -> float:
        """
        Pixel rise first — the near plate rises more than the far one, a
        parked plate or a ring not at all — weighted by how much of its span
        the plate was found in, by size (the near plate is the larger), and
        a little by the athlete's tap.
        """
        trail, tapped = item
        order = sorted(trail)
        share = len(order) / max(1, order[-1] - order[0] + 1)
        rise = _rise_px(trail)
        if share < MIN_FOUND_SHARE or rise < MIN_RISE * height:
            return 0.0
        return rise * (0.5 + 0.5 * share) * (0.5 + 0.5 * _median_radius(trail) / largest) * (1.3 if tapped else 1.0)

    best, tapped = max(trails, key=score)
    if score((best, tapped)) <= 0.0:
        rises = sorted((round(_rise_px(t) / height, 2) for t, _ in trails), reverse=True)[:3]
        logger.info("Oblique follower: nothing dark and round rose like a bar (best rises %s)", rises)
        return BarPath(
            sampled=len(frames),
            frame_width=width,
            frame_height=height,
            reason="nothing dark and round rose like a bar",
        )
    path = _trail_path(best, frames, tapped=tapped)
    if not path.tracked:
        path.reason = "the plate was followed through too little of the window"
    return path


# ---------- is a trace on a plate at all? ----------

GATE_SAMPLES = 6
# A trace passes when this share of its sampled points look like a plate.
GATE_SHARE = 0.5


def looks_like_a_plate(video_path: Optional[str], path: BarPath, *, frames=None) -> Tuple[bool, str]:
    """
    Is this trace on a plate — dark inside, with a rim — rather than on
    foliage, a window or a head? Samples GATE_SAMPLES points along the trace
    and reads the clip there (or `frames`, (t, image) already decoded).
    Returns (verdict, why).
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if not path.points or not path.plate_radius or not path.frame_width:
        return False, "no trace to check"
    picks = [path.points[int(k)] for k in np.linspace(0, len(path.points) - 1, min(GATE_SAMPLES, len(path.points)))]
    if frames is not None:
        images = [min(frames, key=lambda item: abs(item[0] - p.t))[1] if frames else None for p in picks]
    else:
        from server.utils.frames import read_frames_at

        images = read_frames_at(video_path, [p.t for p in picks])
    passed = 0
    checked = 0
    for point, image in zip(picks, images):
        if image is None:
            continue
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if width != path.frame_width:
            scale = path.frame_width / width
            gray = cv2.resize(gray, (path.frame_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
            height, width = gray.shape[:2]
        x, y = point.x * width, point.y * height
        r = path.plate_radius * width
        grey = _inner_grey(gray, x, y, r, r)
        support = _rim_support(_gradient(gray), x, y, r, r)
        checked += 1
        if (grey <= PLATE_MAX_GREY and support >= MIN_RIM_SUPPORT) or support >= RIM_ALONE:
            passed += 1
    if not checked:
        return False, "no frames to check the trace against"
    if passed / checked >= GATE_SHARE:
        return True, ""
    return False, f"what was followed does not look like a plate ({passed} of {checked} samples dark or rimmed)"
