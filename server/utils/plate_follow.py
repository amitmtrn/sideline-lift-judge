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
    # Minor/major axis ratio from each successful ellipse fit along the
    # trail. A plate filmed square to the side has values close to 1; an
    # oblique view flattens the ellipse so the median drops.
    axis_ratios: List[float] = field(default_factory=list)
    # How many of the opening frames found this same circle. A plate on the
    # bar turns up in every seed frame; a one-off detection (a hub, a rim of
    # the far plate catching the light) does not.
    sightings: int = 1
    checks: int = 0
    found: int = 0
    # The athlete tapped this one as the near plate.
    tapped: bool = False

    @property
    def support(self) -> float:
        return self.found / self.checks if self.checks else 0.0


def read_gray_frames(video_path: str, start_sec: float, end_sec: float, width: int = ANALYSIS_WIDTH):
    """
    (t, gray frame at `width`) for each processed frame in the span, t in
    presentation time — the frames every follower works from. The pipeline
    reads a window once with this and hands the list to each of them.
    """
    import cv2  # type: ignore

    from server.utils.frames import read_frames

    out = []
    for t, frame in read_frames(video_path, start_sec, end_sec, max_fps=MAX_FPS):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > width:
            gray = cv2.resize(gray, (width, int(round(gray.shape[0] * width / gray.shape[1]))), interpolation=cv2.INTER_AREA)
        out.append((t, gray))
    return out


_read_gray = read_gray_frames

# How far (as a share of the frame's long side) the athlete's tap may be from
# the circle it meant.
TAP_REACH = 0.08


def _seeds(
    frames,
    hint: Optional[Tuple[float, float, float, float]],
    tap: Optional[Tuple[float, float]] = None,
) -> List[_Seed]:
    """
    Circles in the opening frames, one seed per object, plus the hint. The
    tap marks the seed nearest it (or becomes a seed of its own when no circle
    was found near it): one more candidate with a little more weight, not an
    answer.
    """
    import cv2  # type: ignore

    height, width = frames[0][1].shape[:2]
    shorter = min(width, height)
    low = max(6, int(shorter * MIN_RADIUS_RATIO))
    high = max(low + 4, int(shorter * MAX_RADIUS_RATIO))
    seeds: List[_Seed] = []

    def add(frame: int, x: float, y: float, r: float) -> None:
        for other in seeds:
            if (x - other.x) ** 2 + (y - other.y) ** 2 < (0.5 * other.radius) ** 2 and abs(r - other.radius) < 0.3 * other.radius:
                other.sightings += 1
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
    seeds = seeds[:MAX_SEEDS]
    if tap is not None:
        tx, ty = tap[0] * width, tap[1] * height
        reach = TAP_REACH * max(width, height)
        nearest = min(seeds, key=lambda s: (s.x - tx) ** 2 + (s.y - ty) ** 2, default=None)
        if nearest is not None and ((nearest.x - tx) ** 2 + (nearest.y - ty) ** 2) ** 0.5 <= reach + nearest.radius:
            nearest.tapped = True
        else:
            radii = sorted(s.radius for s in seeds)
            radius = radii[len(radii) // 2] if radii else (low + high) / 2.0
            seeds.append(_Seed(0, float(tx), float(ty), float(radius), tapped=True))
    return seeds


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


def _snap(gray, x: float, y: float, radius: float) -> Optional[Tuple[float, float, Optional[float]]]:
    """
    A circle of the plate's size close to the track, if there is one, then
    an ellipse fit on the rim pixels around it so the returned centre lands
    on the sleeve rather than wherever Hough fitted its circle. The ellipse
    refinement matters for oblique views where the plate projects to an
    ellipse — a side-on circle degenerates to the Hough centre, which is
    the same point.

    Returns `(x, y, axis_ratio)`: `axis_ratio` is the ellipse's minor/major
    (close to 1 side-on, below 1 the more the camera tilts). None when no
    ellipse fit was healthy enough to measure it.
    """
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
    if np.hypot(bx - x, by - y) > SNAP_DISTANCE * radius:
        return None
    refined = _ellipse_centre(gray, bx, by, radius)
    if refined is not None:
        (ex, ey), ratio = refined
        if np.hypot(ex - bx, ey - by) <= 0.3 * radius:
            return ex, ey, ratio
    return bx, by, None


def _ellipse_centre(
    gray, cx: float, cy: float, radius: float
) -> Optional[Tuple[Tuple[float, float], float]]:
    """
    Fit an ellipse to the plate's rim and return `((cx, cy), axis_ratio)`.
    `axis_ratio` is minor/major, close to 1 on a side-on plate and below 1
    the more the camera tilts. Returns None when the fit is unreliable
    (not enough edge pixels, bad aspect, far from the Hough centre).
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    reach = int(radius * 1.5) + 2
    height, width = gray.shape[:2]
    top = max(0, int(round(cy)) - reach)
    bottom = min(height, int(round(cy)) + reach)
    left = max(0, int(round(cx)) - reach)
    right = min(width, int(round(cx)) + reach)
    patch = gray[top:bottom, left:right]
    if patch.shape[0] < 10 or patch.shape[1] < 10:
        return None

    blurred = cv2.GaussianBlur(patch, (3, 3), 0.8)
    edges = cv2.Canny(blurred, 60, 160)
    # Mask to an annulus around the Hough circle so letters inside the plate
    # or background edges outside do not pull the ellipse off.
    local = (int(cx) - left, int(cy) - top)
    mask = np.zeros_like(edges)
    # Wider annulus than Hough's radius so an oblique-view ellipse (minor
    # axis well inside the Hough radius; major axis a little outside) keeps
    # both ends of its rim in the fit. The inner radius still clears the
    # plate's hub lettering (which lives inside ~0.4 r).
    cv2.circle(mask, local, int(radius * 1.3), 255, -1)
    cv2.circle(mask, local, int(radius * 0.5), 0, -1)
    edges = cv2.bitwise_and(edges, mask)

    ys, xs = np.where(edges > 0)
    if len(xs) < 10:
        return None
    points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    try:
        (ex, ey), (axis_a, axis_b), _angle = cv2.fitEllipse(points)
    except cv2.error:
        return None
    major = max(axis_a, axis_b)
    minor = min(axis_a, axis_b)
    expected_dia = 2.0 * radius
    # Guard against pathological fits: a very elongated ellipse is a letter
    # or a chain of unrelated edges, not a plate.
    if major <= 0 or minor / major < 0.4:
        return None
    if not (expected_dia * 0.7 <= major <= expected_dia * 1.4):
        return None
    return (float(ex + left), float(ey + top)), float(minor / major)


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
                    x, y, ratio = rim
                    positions[k] = (x, y)
                    if ratio is not None:
                        seed.axis_ratios.append(ratio)
                else:
                    # Hough failed — still measure the plate's axis ratio at
                    # the flow-tracked position so an oblique view (where
                    # Hough struggles for exactly this reason) is detected.
                    fit = _ellipse_centre(gray, x, y, seed.radius)
                    if fit is not None:
                        seed.axis_ratios.append(fit[1])
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


# A plate cannot physically translate more than ~2 radii between consecutive
# processed frames (we're at <= 60 fps). A gap larger than that is the track
# hopping to another object — usually a static plate on a stand next to the
# bar — so we treat it as a segment break and discard everything up to and
# including the last jump. `BarPath.reference_x` is the median of the first
# sixth of the surviving points, so a trail that was briefly on another
# object before snapping onto the bar no longer anchors the dashed reference
# line at the wrong x. The surviving segment usually begins with the bar on
# the floor (if the hop happened in the setup) or right at lift-off, so the
# reference line lands where a coach would eyeball it either way.
MAX_JUMP_RADII = 2.0


def _trim_pre_liftoff(
    track: List[Optional[Tuple[float, float]]],
    radius: float,
) -> List[Optional[Tuple[float, float]]]:
    """
    Return the track with any leading points before the last big
    consecutive-frame jump masked to None. See the MAX_JUMP_RADII comment
    above for why. A clean trail with no jumps is returned unchanged.
    """
    valid = [(index, point) for index, point in enumerate(track) if point is not None]
    if len(valid) < 4 or radius <= 0:
        return track

    max_jump_sq = (MAX_JUMP_RADII * radius) ** 2
    start_pos = 0
    for pos in range(1, len(valid)):
        _, prev = valid[pos - 1]
        _, curr = valid[pos]
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        if dx * dx + dy * dy > max_jump_sq:
            start_pos = pos
    if start_pos == 0:
        return track

    liftoff_index = valid[start_pos][0]
    cleaned: List[Optional[Tuple[float, float]]] = [None] * len(track)
    for index, point in valid:
        if index >= liftoff_index:
            cleaned[index] = point
    return cleaned


def follow_bar_path(
    video_path: Optional[str],
    start_sec: float,
    end_sec: float,
    *,
    hint: Optional[Tuple[float, float, float, float]] = None,
    tap: Optional[Tuple[float, float]] = None,
    frames=None,
) -> BarPath:
    """
    The plate's path through [start_sec, end_sec], as a BarPath in the same
    normalised coordinates as bar_track's. `hint` is (t, x, y, radius px at
    640 wide) for where another method believes the plate was; it becomes one
    more seed, not an answer. `tap` is (x, y), normalised, where the athlete
    tapped the near plate. `frames` is the window already read with
    `read_gray_frames`, so the pipeline decodes it once for every follower.
    Check `.tracked` before believing the result.
    """
    if frames is None:
        frames = read_gray_frames(video_path, start_sec, end_sec)
    if len(frames) < 6:
        return BarPath(reason="not enough frames to follow the plate")
    height, width = frames[0][1].shape[:2]
    seeds = _seeds(frames, hint, tap)
    if not seeds:
        return BarPath(sampled=len(frames), frame_width=width, frame_height=height, reason="no circle to start from")
    for seed in seeds:
        seed.track = [None] * len(frames)
    _follow(frames, range(len(frames)), seeds)
    late = [s for s in seeds if s.frame > 0]
    if late:
        # Back from where they were found to the start of the window.
        _follow(frames, range(max(s.frame for s in late), -1, -1), late)

    # Both plates on the bar rise together, so rise alone cannot tell the
    # near plate from the far one — and on a downscaled upload the far
    # (smaller) plate sometimes snaps to Hough circles a little more often
    # and used to win on support. Weight by radius so the nearest, largest
    # plate wins when rises are comparable; a plate on a stand or a head
    # still scores zero because it does not rise.
    largest = max(seed.radius for seed in seeds) or 1.0

    def score(seed: _Seed) -> Tuple[float, int, float]:
        rise = _rise(seed.track) / height
        if seed.support < MIN_SUPPORT or rise < MIN_RISE:
            return (0.0, seed.sightings, seed.radius)
        size_weight = 0.5 + 0.5 * (seed.radius / largest)
        # The tap decides between two plates that both rose; it does not
        # make something that did not rise the bar.
        tap_weight = 1.5 if seed.tapped else 1.0
        return (rise * seed.support * size_weight * tap_weight, seed.sightings, seed.radius)

    best = max(seeds, key=score)
    if score(best)[0] <= 0.0:
        rises = sorted(((_rise(s.track) / height, s.support) for s in seeds), reverse=True)[:3]
        logger.info("No plate followed: best (rise, support) %s", [(round(r, 2), round(s, 2)) for r, s in rises])
        return BarPath(
            sampled=len(frames), frame_width=width, frame_height=height,
            reason="nothing followed through the window rose like a bar and stayed round like a plate",
        )
    # Drop the pre-lift-off tail: a hop from a static plate onto the bar
    # leaves a leading segment at the stand's x which is where the dashed
    # reference line would anchor. The lift-off trim pins it at the bar's
    # actual starting line instead.
    best.track = _trim_pre_liftoff(best.track, best.radius)
    keep_every = max(1, int(round((len(frames) / max(1e-6, frames[-1][0] - frames[0][0])) / OUTPUT_FPS)))
    points = [
        BarPoint(t=frames[i][0], x=p[0] / width, y=p[1] / height)
        for i, p in enumerate(best.track)
        if p is not None and i % keep_every == 0
    ]
    followed = sum(1 for p in best.track if p is not None)
    # Coverage over the lift window the trail covers, not the whole analysis
    # span — a trail trimmed to post-lift-off must not be penalised for the
    # setup frames it deliberately left out.
    if points:
        span_frames = sum(1 for ft, _ in frames if points[0].t <= ft <= points[-1].t)
    else:
        span_frames = len(frames)
    axis_ratio: Optional[float] = None
    if best.axis_ratios:
        import numpy as np  # type: ignore

        # The MINIMUM across frames rather than the median: a seed that
        # settled on the plate's minor-axis circle contributes near-circular
        # fits across most frames (its patch does not contain the full
        # ellipse), while frames where the fit captured the major axis as
        # well give the real ratio. The minimum is the clearest look at the
        # ellipse the camera saw.
        axis_ratio = float(np.min(best.axis_ratios))
    return BarPath(
        points=_smooth(points),
        plate_radius=best.radius / width,
        coverage=followed / max(1, span_frames),
        sampled=len(frames),
        frame_width=width,
        frame_height=height,
        plate_axis_ratio=axis_ratio,
    )
