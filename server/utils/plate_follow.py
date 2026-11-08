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
# plate's size and SNAP_DISTANCE radii of the track takes over from the flow;
# while the plate is being bridged (hidden behind a hand, blurred through the
# turnover) the search widens to SNAP_DISTANCE_BRIDGE radii around where the
# track coasted to, so the plate is picked up again where it reappears.
SNAP_RADIUS = 0.12
SNAP_DISTANCE = 0.35
SNAP_DISTANCE_BRIDGE = 1.5
# What it takes to be the plate: this share of those checks found a circle of
# the plate's size on the track, and the track rose this far up the frame.
MIN_SUPPORT = 0.3
MIN_RISE = MIN_VERTICAL_TRAVEL
# Points handed on, at most this many per second: the trace and the evidence
# need nothing finer, and a 60 fps trace is 5x the points to draw.
OUTPUT_FPS = 30.0
# Continuity, in the clip's own seconds (the same physics as plate_oblique's
# walk): a flow step that would move the plate faster than
# MAX_SPEED_RADII_PER_SEC of its radii a second is not the plate moving but
# the corners landing on something else, so it is refused and the position
# coasts on the last velocity for up to COAST_SECONDS instead. At every
# re-centring the image under the track is read: dark like the plate was
# when it was seeded (within GREY_STEP) with some rim, or a rim all the way
# round, means the track is still on the plate; anything else for longer
# than BRIDGE_SECONDS means the plate was lost, and the path ends at the
# last place it was seen rather than go on with an arm, a ring or a wall.
MAX_SPEED_RADII_PER_SEC = 18.0
MIN_STEP_RADII = 0.3
COAST_SECONDS = 0.17
BRIDGE_SECONDS = 0.35
GREY_STEP = 60.0
ON_PLATE_RIM = 0.2
# A diagnostic hook: (seed index, frame index, t, event, payload) at every
# refused step, coast, re-centring and loss. None in production.
_debug = None


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
    # Mean grey inside the plate where it was seeded: what "still on the
    # same plate" is measured against. Read when the seed joins the follow.
    grey: Optional[float] = None
    # Set when the plate was lost for longer than the bridge: the track ends
    # at the last place it was seen.
    lost: bool = False

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
    tap: Optional[Tuple[float, float, float]] = None,
) -> List[_Seed]:
    """
    Circles in the opening frames, one seed per object, plus the hint. The
    tap (t, x, y) marks the seed nearest it, on the frame it was made on;
    when no circle was found near it, a closer look there (every stretch,
    a lower Hough bar) sizes a seed of its own, and failing that the tap
    itself becomes one at the typical radius. One more candidate with a
    little more weight, not an answer.
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
        from server.utils.plate_oblique import HOUGH_PARAM2_TAP, STRETCHES, _detect, _gradient, tap_frame

        t, tx, ty = tap[0], tap[1] * width, tap[2] * height
        at = tap_frame(frames, t)
        if at >= len(frames) - 1:
            at = 0
        reach = TAP_REACH * max(width, height)
        nearest = min(seeds, key=lambda s: (s.x - tx) ** 2 + (s.y - ty) ** 2, default=None)
        if nearest is not None and ((nearest.x - tx) ** 2 + (nearest.y - ty) ** 2) ** 0.5 <= reach + nearest.radius:
            nearest.tapped = True
            logger.info(
                "Side-on follower: tap at %.2f s applied to frame %d (t=%.2f s), the circle at (%d, %d) r=%d",
                t, at, frames[at][0], nearest.x, nearest.y, nearest.radius,
            )
            return seeds
        margin = int(reach + high) + 4
        roi = (int(tx) - margin, int(ty) - margin, int(tx) + margin, int(ty) + margin)
        gray = frames[at][1]
        plates = [
            p
            for p in _detect(gray, _gradient(gray), low, high, STRETCHES, roi, HOUGH_PARAM2_TAP)
            if p.distance(tx, ty) <= reach + p.r
        ]
        if plates:
            plate = min(plates, key=lambda p: p.distance(tx, ty))
            seeds.append(_Seed(at, float(plate.x), float(plate.y), float(plate.r), tapped=True))
            logger.info(
                "Side-on follower: tap at %.2f s applied to frame %d (t=%.2f s), a closer look found the plate at (%d, %d) r=%d",
                t, at, frames[at][0], plate.x, plate.y, plate.r,
            )
        else:
            radii = sorted(s.radius for s in seeds)
            radius = radii[len(radii) // 2] if radii else (low + high) / 2.0
            seeds.append(_Seed(at, float(tx), float(ty), float(radius), tapped=True))
            logger.info(
                "Side-on follower: tap at %.2f s applied to frame %d (t=%.2f s) as a seed of its own at (%d, %d), r=%d assumed",
                t, at, frames[at][0], tx, ty, radius,
            )
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


# What the image under a track says: ON, the plate (its shade, with some
# rim); DARK, its shade but too little rim — a track that has slipped
# half a radius off-centre reads this, and so would a sleeve of the same
# colour, so the flow goes on but nothing is confirmed; OFF, not the plate
# at all (another shade, or a hollow ring), when the flow must not steer.
ON, DARK, OFF = 2, 1, 0


def _on_plate(gray, x: float, y: float, radius: float, grey0: Optional[float]) -> Tuple[int, float, float]:
    """
    ON, DARK or OFF for the image under (x, y) against the plate last seen
    at grey `grey0`, with the grey and the rim support read. A rim on its
    own is not enough here — a noisy patch has "edges" in every direction
    too, and a plate's own inside reads as much texture as noise does. The
    caller lets its reference shade follow the plate from the floor into
    the light: on the labelled clips a plate's inside changes by 80 grey
    levels over a clean, a step or two of 20 at a time.
    """
    from server.utils.plate_oblique import plate_reading

    reading = plate_reading(gray, x, y, radius, radius)
    grey = reading.grey
    if reading.hollow or (grey0 is not None and abs(grey - grey0) > GREY_STEP):
        return OFF, grey, reading.support
    return (ON if reading.support >= ON_PLATE_RIM else DARK), grey, reading.support


# A circle picked up again after the plate was hidden must be unmistakably
# a plate: this much of its rim on an edge (the labelled plates' median).
BRIDGE_RIM = 0.5


def _snap(
    gray,
    x: float,
    y: float,
    radius: float,
    grey0: Optional[float] = None,
    distance: float = SNAP_DISTANCE,
    *,
    strict: bool = False,
) -> Optional[Tuple[float, float, Optional[float], float]]:
    """
    A circle of the plate's size within `distance` radii of the track, if
    there is one and it is the plate (its shade, with a rim — the hollow
    ring hanging in a gym is not; `strict`, for a plate picked up again
    after a gap, wants most of the rim), then an ellipse fit on the rim
    pixels around it so the returned centre lands on the sleeve rather
    than wherever Hough fitted its circle. The ellipse refinement matters
    for oblique views where the plate projects to an ellipse — a side-on
    circle degenerates to the Hough centre, which is the same point.

    Returns `(x, y, axis_ratio, grey)`: `axis_ratio` is the ellipse's
    minor/major (close to 1 side-on, below 1 the more the camera tilts),
    None when no ellipse fit was healthy enough to measure it; `grey` is
    the inside of the plate as read there.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    reach = int(radius * (distance + 1.25))
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
    if np.hypot(bx - x, by - y) > distance * radius:
        return None
    state, grey, support = _on_plate(gray, bx, by, radius, grey0)
    if state != ON or (strict and support < BRIDGE_RIM):
        return None
    refined = _ellipse_centre(gray, bx, by, radius)
    if refined is not None:
        (ex, ey), ratio = refined
        if np.hypot(ex - bx, ey - by) <= 0.3 * radius:
            return ex, ey, ratio, grey
    return bx, by, None, grey


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

    Continuity rules the flow: a step too fast for a plate is refused and
    the position coasts on its velocity instead; at every re-centring the
    image under the track must still look like the plate it was seeded on,
    and when it has not for longer than the bridge the seed is lost — its
    track is cut back to the last place the plate was seen and it is
    followed no further.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    from server.utils.plate_oblique import plate_reading

    active: List[int] = []
    positions = {}
    corners = {}
    velocity = {}  # pixels per second
    confirmed = {}  # (time, step) the track was last ON the plate
    seen = {}  # (time, step) the plate's shade was last under the track (ON or DARK)
    off = {}  # the last re-centring found the track OFF the plate: bridging
    last_time = None
    previous = None
    for step, index in enumerate(order):
        t, gray = frames[index]
        dt = abs(t - last_time) if last_time is not None else 0.0
        if dt <= 0.0:
            dt = 1.0 / MAX_FPS
        for k, seed in enumerate(seeds):
            if seed.frame == index and k not in positions:
                positions[k] = (seed.x, seed.y)
                corners[k] = _features(gray, seed.x, seed.y, seed.radius)
                velocity[k] = (0.0, 0.0)
                confirmed[k] = seen[k] = (t, step)
                off[k] = False
                if seed.grey is None:
                    seed.grey = plate_reading(gray, seed.x, seed.y, seed.radius, seed.radius).grey
                active.append(k)
                seed.track[index] = (seed.x, seed.y)
        live = [k for k in active if not seeds[k].lost]
        # While the track is off the plate the flow is whatever hid it (a
        # hand, an arm) and must not steer; the position coasts instead.
        moving = [k for k in live if previous is not None and len(corners[k]) >= 5 and not off[k]]
        stepped: set = set()
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
                    dx, dy = (float(v) for v in np.median(shift[at : at + count][mine], axis=0))
                    radius = seeds[k].radius
                    limit = max(MIN_STEP_RADII * radius, MAX_SPEED_RADII_PER_SEC * radius * dt)
                    if np.hypot(dx, dy) <= limit:
                        x, y = positions[k]
                        positions[k] = (x + dx, y + dy)
                        corners[k] = moved[at : at + count][mine].reshape(-1, 1, 2)
                        vx, vy = velocity[k]
                        velocity[k] = (0.5 * vx + 0.5 * dx / dt, 0.5 * vy + 0.5 * dy / dt)
                        stepped.add(k)
                    elif _debug is not None:
                        _debug(k, index, t, "refused", (dx, dy, int(mine.sum())))
                elif _debug is not None:
                    _debug(k, index, t, "few corners", (int(mine.sum()), count))
                at += count
        for k in live:
            if k in stepped or previous is None or seeds[k].frame == index:
                continue
            # No usable flow this frame, or a step no plate could make:
            # coast on the last velocity for a moment, then hold.
            if t - seen[k][0] <= COAST_SECONDS:
                x, y = positions[k]
                vx, vy = velocity[k]
                positions[k] = (x + vx * dt, y + vy * dt)
            if _debug is not None:
                _debug(k, index, t, "coast", (positions[k], velocity[k], off[k]))
            corners[k] = corners[k][:0]
        refresh = step % REFRESH_EVERY == 0
        for k in live:
            seed = seeds[k]
            x, y = positions[k]
            state = None
            if refresh and seed.frame != index:
                seed.checks += 1
                was_off = off[k]
                rim = _snap(
                    gray, x, y, seed.radius, seed.grey,
                    SNAP_DISTANCE_BRIDGE if was_off else SNAP_DISTANCE, strict=was_off,
                )
                if rim is not None:
                    seed.found += 1
                    x, y, ratio, grey = rim
                    positions[k] = (x, y)
                    if ratio is not None:
                        seed.axis_ratios.append(ratio)
                    state = ON
                else:
                    state, grey, _ = _on_plate(gray, x, y, seed.radius, seed.grey)
                    if state == ON:
                        # Hough failed — still measure the plate's axis ratio
                        # at the flow-tracked position so an oblique view
                        # (where Hough struggles for exactly this reason) is
                        # detected; and a tapped plate counts as found: the
                        # athlete said it is the plate, it need only still be.
                        fit = _ellipse_centre(gray, x, y, seed.radius)
                        if fit is not None:
                            seed.axis_ratios.append(fit[1])
                        if seed.tapped:
                            seed.found += 1
                off[k] = state == OFF
                if _debug is not None:
                    _debug(k, index, t, "refresh", (positions[k], rim is not None, state, grey, seed.grey))
                if state != OFF:
                    since = seen[k][1]
                    if was_off and step - since > 1 and seed.track[order[since]] is not None:
                        # Bridged: the positions coasted through are
                        # replaced by the straight line from where the plate
                        # was last seen to where it was found again.
                        x0, y0 = seed.track[order[since]]
                        for n, cut in enumerate(order[since + 1 : step], 1):
                            share = n / (step - since)
                            seed.track[cut] = (x0 + share * (x - x0), y0 + share * (y - y0))
                    seen[k] = (t, step)
                if state == ON:
                    confirmed[k] = (t, step)
                    # The reference shade follows the plate into the light.
                    seed.grey = grey if seed.grey is None else 0.7 * seed.grey + 0.3 * grey
                elif state == OFF and t - seen[k][0] > BRIDGE_SECONDS:
                    # Lost: cut the track back to where the plate was last
                    # seen and leave this seed there.
                    seed.lost = True
                    for cut in order[seen[k][1] + 1 : step]:
                        seed.track[cut] = None
                    continue
            if (refresh or len(corners[k]) < 10) and state != OFF:
                corners[k] = _features(gray, x, y, seed.radius)
            seed.track[index] = (x, y)
        previous = gray
        last_time = t


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
    tap: Optional[Tuple[float, float, float]] = None,
    frames=None,
) -> BarPath:
    """
    The plate's path through [start_sec, end_sec], as a BarPath in the same
    normalised coordinates as bar_track's. `hint` is (t, x, y, radius px at
    640 wide) for where another method believes the plate was; it becomes one
    more seed, not an answer. `tap` is (t, x, y): when and where, as fractions
    of the frame, the athlete tapped the near plate. `frames` is the window
    already read with `read_gray_frames`, so the pipeline decodes it once for
    every follower. Check `.tracked` before believing the result; `.tapped`
    says whether the plate followed is the one they tapped.
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
    tap_reason = ""
    if tap is not None and not best.tapped:
        marked = [s for s in seeds if s.tapped]
        if marked:
            mine = max(marked, key=lambda s: _rise(s.track))
            rise = _rise(mine.track) / height
            if mine.support < MIN_SUPPORT:
                tap_reason = f"the plate at the tap was followed through too little of the window ({mine.support:.0%} of the checks found it)"
            else:
                tap_reason = f"the plate at the tap did not rise like a bar (rose {rise:.0%} of the frame)"
        else:
            tap_reason = "the plate at the tap could not be followed"
        logger.info("Side-on follower: tap at %.2f s not followed: %s", tap[0], tap_reason)
    if score(best)[0] <= 0.0:
        rises = sorted(((_rise(s.track) / height, s.support) for s in seeds), reverse=True)[:3]
        logger.info("No plate followed: best (rise, support) %s", [(round(r, 2), round(s, 2)) for r, s in rises])
        return BarPath(
            sampled=len(frames), frame_width=width, frame_height=height,
            reason="nothing followed through the window rose like a bar and stayed round like a plate",
            tap_reason=tap_reason,
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
    # Coverage from the trail's first point to the end of the window, not
    # the whole analysis span — a trail trimmed to post-lift-off must not be
    # penalised for the setup frames it deliberately left out — but a trail
    # that ended early, the plate lost at the turnover, has not seen the
    # catch, and that shows in the number and caps the confidence.
    if points:
        span_frames = sum(1 for ft, _ in frames if points[0].t <= ft)
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
        tapped=best.tapped,
        tap_reason=tap_reason,
    )
