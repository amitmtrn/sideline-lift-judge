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
MAX_COLOUR_DISTANCE = 140.0
# Colour also enters the walk's cost softly, in pixels per unit of BGR distance,
# so the near plate beats the far one when both are in reach.
COLOUR_COST = 0.25
# A trail whose colour jumps this far between consecutive points has moved
# onto a different object — typically from a plate parked on a stand to the
# plate being lifted as they pass each other. It is cut there.
COLOUR_SPLIT = 120.0
# Frames the plate may go undetected before the walk stops trusting its velocity.
MAX_GAP_FRAMES = 4
# Mean motion energy along a trail. A circle in a wall banner scores ~0; the bar
# does not. This is what separates a real lift from a static pattern.
MIN_TRAIL_MOTION = 3.0
# Mean 8-bit motion energy the anchor plate must show. Below this nothing in the
# window is really moving and anything we locked onto is scenery.
MIN_ANCHOR_MOTION = 6.0
# Hypotheses tracked in parallel before picking the one that behaves like a bar.
MAX_SEEDS = 14
# A circle found at the same spot throughout a chunk is scenery — a logo on a
# wall banner, a plate on a stand — and is dropped before tracking. Hough finds
# such things only in some frames, so presence is counted per second, not per
# frame: scenery shows up in most seconds of a chunk, while the bar rests on the
# floor for a few seconds of it and is overhead for fewer.
STATIC_FRACTION = 0.6
STATIC_CELL_PX = 20
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
    """(fps, [(t, frame at analysis width)]) at about sample_fps, t in presentation time."""
    import cv2  # type: ignore

    from server.utils.frames import read_frames

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return 0.0, []
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    samples = []
    for t, frame in read_frames(video_path, start_sec, max(start_sec, end_sec), max_fps=max(1.0, sample_fps)):
        # Downscale only: blowing a small clip up costs time and adds no
        # detail for Hough to find.
        if frame.shape[1] > analysis_width:
            scale = analysis_width / frame.shape[1]
            frame = cv2.resize(frame, (analysis_width, max(1, int(round(frame.shape[0] * scale)))))
        samples.append((t, frame))
    return fps, samples


@dataclass
class _Analysis:
    """Everything the walks need, computed once per window."""

    times: List[float]
    width: int
    height: int
    per_frame: list
    colours: list
    motions: list
    seeds: list


@dataclass
class _Trail:
    circles: List[Tuple[int, Tuple[int, int, int]]]  # (sample index, circle)
    coverage: float
    travel: float  # vertical, normalised to frame height
    motion: float
    radius: int = 0  # median, analysis pixels


def _analyse(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
    analysis_width: int,
    scenery: Optional[set] = None,
) -> Optional[_Analysis]:
    """
    One window's worth of detections, ready to walk. `scenery` is the set of
    static cells to drop (see `_static_cells`); None keeps every circle.
    """
    import cv2  # type: ignore

    _, samples = _read_window(video_path, start_sec, end_sec, sample_fps, analysis_width)
    if len(samples) < MIN_POINTS:
        return None
    height, width = samples[0][1].shape[:2]
    grays = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for _, f in samples]
    motions = _frame_motion(grays)
    shorter = min(width, height)
    min_radius = max(6, int(shorter * MIN_RADIUS_RATIO))
    max_radius = max(min_radius + 4, int(shorter * MAX_RADIUS_RATIO))
    per_frame = [_circles(gray, min_radius, max_radius) for gray in grays]
    if scenery:
        per_frame = [[c for c in cands if _cell(c) not in scenery] for cands in per_frame]
    colours = [
        {circle: _mean_colour(frame, circle) for circle in candidates}
        for (_t, frame), candidates in zip(samples, per_frame)
    ]
    return _Analysis(
        times=[t for t, _ in samples],
        width=width,
        height=height,
        per_frame=per_frame,
        colours=colours,
        motions=motions,
        seeds=_seed_candidates(per_frame, motions, shorter),
    )


def _cell(circle, cell: int = STATIC_CELL_PX):
    return (circle[0] // cell, circle[1] // cell)


def _static_cells(analysis: "_Analysis", fraction: float = STATIC_FRACTION) -> set:
    """
    Cells where a circle keeps turning up through this window. Presence is per
    second so a sparsely detected logo still counts. A bar racked or resting
    for most of one window lands here too — which is why scenery is only what
    is static in more than one window of the clip (see detect_lift_windows).
    """
    seconds_seen = {}
    for timestamp, candidates in zip(analysis.times, analysis.per_frame):
        for circle in candidates:
            seconds_seen.setdefault(_cell(circle), set()).add(int(timestamp))
    total_seconds = int(analysis.times[-1]) - int(analysis.times[0]) + 1
    return {key for key, secs in seconds_seen.items() if len(secs) >= fraction * total_seconds}


def _trails(analysis: _Analysis) -> List[_Trail]:
    """
    One trail per seed, dropping the ones that are too short or belong to
    something static (a round pattern on a wall banner scores ~0 motion).
    A head is also round, high-contrast and fast, so the fastest circle is not
    reliably the bar: callers rank these by how far they travel up the frame.
    """
    trails: List[_Trail] = []
    for _score, anchor_index, anchor in analysis.seeds:
        walked = _walk_from(anchor_index, anchor, analysis.per_frame, analysis.colours)
        for circles in _split_on_colour(walked, analysis.colours):
            if len(circles) < MIN_POINTS:
                continue
            motion = sum(_motion_score(analysis.motions[i], c) for i, c in circles) / len(circles)
            if motion < MIN_TRAIL_MOTION:
                continue
            ys = [c[1] / analysis.height for _, c in circles]
            radii = sorted(c[2] for _, c in circles)
            trails.append(
                _Trail(
                    circles=circles,
                    coverage=len(circles) / len(analysis.times),
                    travel=max(ys) - min(ys),
                    motion=motion,
                    radius=radii[len(radii) // 2],
                )
            )
    return trails


def _step_radii(circles, radius: int) -> List[float]:
    """How far each point moved from the one before it, in radii per frame."""
    import numpy as np  # type: ignore

    radius = max(1, radius)
    out = [0.0]
    for (i0, c0), (i1, c1) in zip(circles, circles[1:]):
        gap = max(1, i1 - i0)
        out.append(float(np.hypot(c1[0] - c0[0], c1[1] - c0[1])) / gap / radius)
    return out


def _split_on_colour(circles, colours):
    """Cut a walked trail wherever it changes colour abruptly — it changed object."""
    if not circles:
        return []
    pieces, current = [], [circles[0]]
    for previous, item in zip(circles, circles[1:]):
        before = colours[previous[0]].get(previous[1])
        after = colours[item[0]].get(item[1])
        if _colour_distance(after, before) > COLOUR_SPLIT:
            pieces.append(current)
            current = []
        current.append(item)
    pieces.append(current)
    return pieces


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
    import numpy as np  # type: ignore

    if end_sec is None:
        end_sec = start_sec + 10.0
    analysis = _analyse(video_path, start_sec, end_sec, sample_fps, analysis_width)
    if analysis is None:
        return BarPath(reason="not enough frames to track")
    if not analysis.seeds:
        return BarPath(
            sampled=len(analysis.times),
            frame_width=analysis.width,
            frame_height=analysis.height,
            reason="nothing round was moving enough to be the bar",
        )

    # The path that behaves like a barbell: the lifted thing travels far up the
    # frame, while a head or a plate on a stand does not.
    trails = _trails(analysis)
    # First choice: among the trails with a credible rise of at least half the
    # biggest, the one the other trails agree with — several seeds on the plate
    # walk onto the same circles, while a trail that hopped from a wall logo to
    # a plate agrees with nobody — then the one with the most of the window.
    rises = [(_main_rise(tr, analysis), tr) for tr in trails]
    rises = [(rise, tr) for rise, tr in rises if rise >= MIN_RISE]
    candidates = [t for t in trails if t.coverage >= MIN_COVERAGE]
    if rises:
        best = max(rise for rise, _ in rises)
        contenders = [tr for rise, tr in rises if rise >= RISE_CONTENDER_SHARE * best]
        as_points = [
            [BarPoint(t=analysis.times[i], x=c[0] / analysis.width, y=c[1] / analysis.height) for i, c in tr.circles]
            for tr in trails
        ]
        contender_points = [as_points[trails.index(tr)] for tr in contenders]

        def rank(position: int):
            tr = contenders[position]
            support = sum(
                1
                for other, points in enumerate(as_points)
                if trails[other] is not tr and _same_plate(contender_points[position], points, tr.radius, analysis.width, analysis.height)
            )
            return _lead_rank(support, _main_rise(tr, analysis), tr.coverage)

        chosen = contenders[max(range(len(contenders)), key=rank)]
        trail = _around_main_rise(chosen, analysis)
    elif candidates:
        chosen = max(candidates, key=lambda t: t.travel * (0.5 + t.coverage))
        trail = _around_main_rise(chosen, analysis)
    else:
        # Fall back to the strongest seed so the caller still gets a diagnosis.
        _score, anchor_index, anchor = analysis.seeds[0]
        trail = _walk_from(anchor_index, anchor, analysis.per_frame, analysis.colours)
    samples = analysis.times
    width, height = analysis.width, analysis.height

    points: List[BarPoint] = []
    radii: List[int] = []
    for index, (x, y, r) in trail:
        radii.append(r)
        points.append(BarPoint(t=samples[index], x=x / width, y=y / height))

    if points:
        span_samples = sum(1 for t_ in samples if points[0].t <= t_ <= points[-1].t)
    else:
        span_samples = len(samples)
    path = BarPath(
        points=_smooth(points),
        plate_radius=(float(np.median(radii)) / width) if radii else 0.0,
        # Over the span the trail covers: a trail trimmed to the lift must not
        # be penalised for the setup and walk-away it deliberately left out.
        coverage=len(points) / max(1, span_samples),
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
        """
        Hough does not find the plate in every frame, so the walk must survive
        gaps: predict where the plate should be from its last two positions,
        and widen the allowed step by how many frames have gone by. Colour is a
        soft cost here rather than a hard cut — a hard cut threw away the plate
        itself whenever motion blur shifted its mean colour.
        """
        found = []
        previous = anchor
        previous_index = anchor_index
        velocity = (0.0, 0.0)
        for index in indices:
            candidates = [
                circle
                for circle in per_frame[index]
                if abs(circle[2] - anchor[2]) <= radius_tolerance
                and _colour_distance(colours[index].get(circle), anchor_colour) <= MAX_COLOUR_DISTANCE
            ]
            if not candidates:
                continue
            gap = max(1, abs(index - previous_index))
            predicted = (previous[0] + velocity[0] * gap, previous[1] + velocity[1] * gap)

            def cost(circle):
                return (
                    np.hypot(circle[0] - predicted[0], circle[1] - predicted[1])
                    + 3 * abs(circle[2] - previous[2])
                    + COLOUR_COST * _colour_distance(colours[index].get(circle), anchor_colour)
                )

            pick = min(candidates, key=cost)
            # Reach is measured from the prediction OR from the last position:
            # at the bottom of a jerk's dip the bar reverses, the constant-
            # velocity prediction points the wrong way, and the true plate
            # would otherwise be rejected as too far from it.
            reach = max_step * min(gap, MAX_GAP_FRAMES)
            from_prediction = np.hypot(pick[0] - predicted[0], pick[1] - predicted[1])
            from_previous = np.hypot(pick[0] - previous[0], pick[1] - previous[1])
            if min(from_prediction, from_previous) > reach:
                continue  # too far to be the same plate; wait for it to reappear
            if gap <= MAX_GAP_FRAMES:
                velocity = ((pick[0] - previous[0]) / gap, (pick[1] - previous[1]) / gap)
            else:
                velocity = (0.0, 0.0)
            previous, previous_index = pick, index
            found.append((index, pick))
        return found

    tracked = dict(walk(range(anchor_index, len(per_frame))))
    tracked.update(dict(walk(range(anchor_index, -1, -1))))
    tracked[anchor_index] = anchor
    return [(index, tracked[index]) for index in sorted(tracked)]


def _main_rise(trail: "_Trail", analysis: "_Analysis") -> float:
    """The biggest credible rise in a trail, or 0."""
    points = [
        BarPoint(t=analysis.times[i], x=c[0] / analysis.width, y=c[1] / analysis.height)
        for i, c in trail.circles
    ]
    episodes = [e for e in rise_episodes(points, step_radii=_step_radii(trail.circles, trail.radius)) if e.credible]
    return max((e.rise for e in episodes), default=0.0)


def _around_main_rise(trail: "_Trail", analysis: "_Analysis"):
    """
    The part of a trail worth drawing: its biggest credible rise with a little
    before and after. A trail that sat on a plate parked on a stand before
    hopping to the bar keeps only the bar.
    """
    points = [
        BarPoint(t=analysis.times[i], x=c[0] / analysis.width, y=c[1] / analysis.height)
        for i, c in trail.circles
    ]
    episodes = [e for e in rise_episodes(points, step_radii=_step_radii(trail.circles, trail.radius)) if e.credible]
    if not episodes:
        return trail.circles
    main = max(episodes, key=lambda e: e.rise)
    low, high = main.start_sec - PRE_PAD, main.end_sec + POST_PAD
    kept = [(i, c) for i, c in trail.circles if low <= analysis.times[i] <= high]
    return kept if len(kept) >= MIN_POINTS else trail.circles


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


# ---------- lifts as rises of the plate ----------

# A lift is the plate going up a long way in a few seconds. Walking in, chalking
# up, re-racking and dropping the bar are not, however much they move the picture.
MIN_RISE = 0.10          # of frame height
MIN_RISE_DURATION = 0.4  # seconds
MAX_RISE_DURATION = 8.0
RISE_PAUSE = 0.7         # seconds without further rise before the episode ends
RISE_BACKSLIDE = 0.06    # a fall of this much (of frame height) ends an episode
# The bar must gain this much (of frame height) to count as still rising. Set
# from measurement, not guesswork: 0.012 looked like a sensible jitter floor but
# cost 3 points of recall on real clips, because a jerk's drive starts slowly
# enough that the episode ended at the rack before the bar took off.
RISE_PROGRESS = 0.004
PRE_PAD = 0.9            # setup before the bar leaves the floor
POST_PAD = 1.0           # catch / lockout after the top of the rise
# Once the bar has come back down this far from the top of its rise it is
# being lowered or dropped, and the lift is over. The window ends there, so the
# judge is not shown — and does not describe — the bar hitting the floor.
FALL_ENDS_LIFT = 0.08
# What a credible rise looks like, learned from real clips. The plate is found
# in nearly every frame of a real lift (density ~1.0) and barely moves sideways
# relative to how far it rises; a junk trail hopping between plates on the floor
# after a drop is sparse (density ~0.5) and wanders across the frame.
MIN_EPISODE_DENSITY = 0.55
MAX_WANDER_RATIO = 0.8   # horizontal range / rise
# Fastest a plate can move between samples, in radii per frame. Measured on 39
# real clips: genuine pulls reach 1.5-2.7 radii per frame, and trails that hop
# between objects are no faster, so this only rejects outright teleports. The
# wander, density and plate-size rules are what catch the hops.
MAX_STEP_RADII_PER_FRAME = 3.0
# A real rise is spread over many frames; a trail that 'rises' by hopping from
# the athlete's shoulder to a logo higher up the wall does it in one step. No
# single step may account for more than this share of the whole rise.
MAX_STEP_SHARE = 0.5
# Phases of one lift (pull, catch, recovery; clean then jerk) sit closer
# together than separate attempts do.
PHASE_GAP = 1.5
# A trail is in the running to be the bar if its biggest credible rise is at
# least this share of the biggest seen in the window; among those, agreement
# with the other trails decides, not the rise itself (see track_bar_path).
RISE_CONTENDER_SHARE = 0.5
# Only a rise this big can start or extend a window. The small rises an athlete
# produces walking in and bending to the bar (0.10-0.14 of the frame, under a
# second long) are real detections of the wrong thing; a jerk on its own is
# about 0.16 and a snatch 0.3+.
STRONG_RISE = 0.15
# No single attempt — clean and jerk included — takes longer than this.
MAX_WINDOW_SECONDS = 10.0


@dataclass
class RiseEpisode:
    start_sec: float
    end_sec: float
    rise: float  # of frame height
    density: float = 1.0  # detections per expected sample inside the episode
    wander: float = 0.0  # horizontal range / rise
    fastest_step: float = 0.0  # radii per frame, gap-normalised
    still_fraction: float = 0.0  # share of steps where the circle did not move
    biggest_step_share: float = 0.0  # largest single upward step / total rise
    fall_sec: Optional[float] = None  # when the bar started coming back down, if seen

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def credible(self) -> bool:
        return (
            self.density >= MIN_EPISODE_DENSITY
            and self.wander <= MAX_WANDER_RATIO
            and self.fastest_step <= MAX_STEP_RADII_PER_FRAME
            and self.biggest_step_share <= MAX_STEP_SHARE
        )


def rise_episodes(
    points: List[BarPoint],
    sample_fps: float = SAMPLE_FPS,
    step_radii: Optional[List[float]] = None,
) -> List[RiseEpisode]:
    """
    Find every stretch where the bar rose by at least MIN_RISE. Greedy: from
    each candidate start, extend while the bar keeps gaining height; stop on a
    pause or a backslide. Overlapping candidates collapse onto the first start.
    Each episode carries its detection density and sideways wander so the
    caller can reject the ones that do not look like a barbell.
    """
    if len(points) < 3:
        return []
    order = sorted(range(len(points)), key=lambda k: points[k].t)
    ordered = [points[k] for k in order]
    steps = [step_radii[k] for k in order] if step_radii else [0.0] * len(ordered)
    episodes: List[RiseEpisode] = []
    i, n = 0, len(ordered)
    while i < n - 1:
        start = i
        best = i
        best_y = ordered[i].y
        last_progress = ordered[i].t
        j = i + 1
        while j < n:
            p = ordered[j]
            if p.y < best_y - RISE_PROGRESS:
                best_y, best, last_progress = p.y, j, p.t
            elif p.y > best_y + RISE_BACKSLIDE or p.t - last_progress > RISE_PAUSE:
                break
            j += 1
        # The true start is the moment the bar leaves its lowest point: the
        # LAST point at (or within a whisker of) the lowest height before the
        # rise, not the first — the bar can sit on the floor for seconds while
        # the athlete sets up, and the window should not begin back there.
        lowest = max(ordered[k].y for k in range(start, best + 1))
        start = max(k for k in range(start, best + 1) if ordered[k].y >= lowest - 0.01)
        rise = ordered[start].y - best_y
        duration = ordered[best].t - ordered[start].t
        if rise >= MIN_RISE and MIN_RISE_DURATION <= duration <= MAX_RISE_DURATION:
            inside = ordered[start : best + 1]
            xs = [p.x for p in inside]
            episodes.append(
                RiseEpisode(
                    ordered[start].t,
                    ordered[best].t,
                    rise,
                    density=len(inside) / max(1.0, duration * sample_fps),
                    wander=(max(xs) - min(xs)) / rise,
                    fastest_step=max(steps[start + 1 : best + 1], default=0.0),
                    still_fraction=_still_fraction(steps[start + 1 : best + 1]),
                    biggest_step_share=max(
                        (ordered[k - 1].y - ordered[k].y for k in range(start + 1, best + 1)), default=0.0
                    ) / rise,
                    fall_sec=_fall_after(ordered, best, best_y),
                )
            )
            i = best + 1
        else:
            i += 1
    return episodes


def _still_fraction(steps: List[float], threshold: float = 0.15) -> float:
    """
    Share of steps in which the circle barely moved. A plate going up moves in
    every frame of the rise; a logo on a wall banner sits still and the trail
    only 'rises' by hopping to another logo higher up.
    """
    if not steps:
        return 0.0
    return sum(1 for s in steps if s < threshold) / len(steps)


def _fall_after(ordered: List[BarPoint], top_index: int, top_y: float) -> Optional[float]:
    """The first moment after the top of a rise that the bar is clearly lower again."""
    for p in ordered[top_index + 1 :]:
        if p.y > top_y + FALL_ENDS_LIFT:
            return p.t
    return None


@dataclass
class LiftWindow:
    start_sec: float
    end_sec: float
    rise: float
    peak_sec: float  # when the bar reached the top of the rise
    radius: int = 0  # of the circle that was followed, analysis pixels
    # The trail that defined this window, normalised. The trace is drawn from
    # it rather than from a second tracking pass, because a re-run samples
    # different frames and does not reliably find the same plate again.
    points: List[BarPoint] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0
    # When the bar was rising inside this window: from the first rise's start
    # to the last rise's top. The padding either side is setup and catch.
    rise_start_sec: float = 0.0
    rise_end_sec: float = 0.0
    # The whole trail the window was cut from, so a merge can extend the window
    # along the same plate instead of stitching on a trail that may have
    # followed the other plate.
    trail: List[BarPoint] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.rise_end_sec:
            self.rise_end_sec = self.peak_sec
        if not self.rise_start_sec:
            self.rise_start_sec = min(self.peak_sec, self.start_sec + PRE_PAD)

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def rise_diameters(self) -> float:
        """
        How far the bar rose, in plate diameters — the one length in every clip
        whose real size is known (a competition bumper is 450 mm). Floor to
        overhead is ~4.4 diameters, floor to the rack ~2.9, rack to overhead
        ~1.5, so this says what a single rise was without a model in the loop.
        """
        if not self.radius or not self.frame_height:
            return 0.0
        return self.rise * self.frame_height / (2.0 * self.radius)

    def bar_path(self, start_sec: Optional[float] = None, end_sec: Optional[float] = None) -> BarPath:
        """The window's own trail as a BarPath, optionally cut to a sub-range."""
        low = self.start_sec if start_sec is None else start_sec
        high = self.end_sec if end_sec is None else end_sec
        points = [p for p in self.points if low <= p.t <= high]
        span = (points[-1].t - points[0].t) if len(points) > 1 else 0.0
        expected = max(1.0, span * SAMPLE_FPS)
        return BarPath(
            points=_smooth(points),
            plate_radius=(self.radius / self.frame_width) if self.frame_width else 0.0,
            coverage=min(1.0, len(points) / expected),
            sampled=int(round(expected)),
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            reason="" if points else "no trail in this range",
        )


# The camera does not move between attempts, so the plate keeps one apparent
# size through a clip. A circle half or twice that size that rises is a torso,
# a head or a banner logo, not the bar.
RADIUS_TOLERANCE = 0.25


def _same_plate_only(windows: List[LiftWindow]) -> List[LiftWindow]:
    """
    The plate's size is the median circle size across every credible rise, not
    the size of the biggest rise: an athlete's head standing up out of a squat
    is a 26 px circle rising a long way, and taking it as the reference threw
    away every real plate window in the clip.
    """
    radii = sorted(w.radius for w in windows if w.radius)
    if len(radii) < 2:
        return windows
    plate = radii[len(radii) // 2]
    return [w for w in windows if not w.radius or abs(w.radius - plate) <= RADIUS_TOLERANCE * plate]


# A window must keep at least this much after its peak so the catch is in it.
MIN_AFTER_PEAK = 0.3
# Two trails are the same plate when, at the moments both saw something, they
# agree to within this many plate radii; or, when they never overlap in time,
# when the ends that face each other are a short hop apart.
SAME_PLATE_RADII = 1.5
JOIN_GAP_SECONDS = 0.5
JOIN_RADII = 2.5


def _join_points(primary: List[BarPoint], secondary: List[BarPoint]) -> List[BarPoint]:
    """
    One trail per moment: the primary's points everywhere it has them, the
    secondary's only where it does not. Two trails that both saw the plate at
    the same sample would otherwise interleave and jitter the trace.
    """
    tolerance = 0.5 / SAMPLE_FPS
    taken = sorted(p.t for p in primary)
    joined = list(primary)
    for point in secondary:
        if not any(abs(point.t - t) <= tolerance for t in taken):
            joined.append(point)
    return sorted(joined, key=lambda q: q.t)


def _radii_apart(a: BarPoint, b: BarPoint, radius: int, width: int, height: int) -> float:
    """Distance between two normalised points, in plate radii."""
    import numpy as np  # type: ignore

    if not (radius and width and height):
        return 0.0
    return float(np.hypot((a.x - b.x) * width / radius, (a.y - b.y) * height / radius))


def _same_plate(
    primary: List[BarPoint], secondary: List[BarPoint], radius: int, width: int, height: int
) -> bool:
    """
    Do two trails look like the same plate? A barbell has a plate at each end
    and, from an off-axis camera, they sit far apart in the frame: stitching a
    trail that followed the far plate onto one that followed the near plate
    draws a sideways leap that never happened. Where both trails saw something
    at the same sample they must agree; trails that never overlap in time must
    meet in a short hop.
    """
    if not primary or not secondary:
        return False
    by_time = {round(q.t, 4): q for q in secondary}
    pairs = [(p, by_time[round(p.t, 4)]) for p in primary if round(p.t, 4) in by_time]
    if pairs:
        mean = sum(_radii_apart(p, q, radius, width, height) for p, q in pairs) / len(pairs)
        return mean <= SAME_PLATE_RADII
    if primary[-1].t <= secondary[0].t:
        before, after = primary[-1], secondary[0]
    elif secondary[-1].t <= primary[0].t:
        before, after = secondary[-1], primary[0]
    else:
        return False
    return after.t - before.t <= JOIN_GAP_SECONDS and _radii_apart(before, after, radius, width, height) <= JOIN_RADII


def _lead_rank(support: int, rise: float, covered: float):
    """
    Order of preference for the trail that stands for the bar: anything at
    least one other trail agrees with beats a loner, then the biggest rise
    over the most of the window. Rise alone picked a trail that hopped from a
    wall logo to the plate overhead; coverage alone picked the far plate.
    """
    return (1 if support > 0 else 0, rise * covered)


def _support(index: int, trails: List[List[BarPoint]], radius: int, width: int, height: int) -> int:
    """
    How many of the other trails agree with this one. Several seeds on the
    plate walk onto the same circles and agree frame for frame; a trail that
    hopped from a wall logo to a plate agrees with nobody.
    """
    mine = trails[index]
    return sum(
        1
        for other, trail in enumerate(trails)
        if other != index and _same_plate(mine, trail, radius, width, height)
    )


@dataclass
class _Group:
    """Windows being merged into one lift, with the span they cover so far."""

    members: List[LiftWindow]
    start_sec: float
    end_sec: float
    rise: float
    rise_start_sec: float
    rise_end_sec: float

    @classmethod
    def of(cls, window: LiftWindow) -> "_Group":
        return cls(
            [window],
            window.start_sec,
            window.end_sec,
            window.rise,
            window.rise_start_sec,
            window.rise_end_sec,
        )

    def add(self, window: LiftWindow) -> None:
        self.members.append(window)
        self.end_sec = max(self.end_sec, window.end_sec)
        self.rise = max(self.rise, window.rise)
        self.rise_start_sec = min(self.rise_start_sec, window.rise_start_sec)
        self.rise_end_sec = max(self.rise_end_sec, window.rise_end_sec)

    def build(self) -> LiftWindow:
        """
        One window along one plate. A trail no other trail agrees with is a
        hop between objects, so it leads only when there is nothing else; among
        the rest, the biggest rise over the most of the span wins — from an
        off-axis camera that is the near plate, which also shows the lift-off
        that the athlete's legs hide on the far one. The other trails are
        stitched on only where the lead has nothing and only if they followed
        the same plate.
        """
        start, end = self.start_sec, self.end_sec
        expected = max(1.0, (end - start) * SAMPLE_FPS)
        trails = [m.trail or m.points for m in self.members]

        def rank(index: int):
            member = self.members[index]
            covered = sum(1 for q in trails[index] if start <= q.t <= end) / expected
            support = _support(index, trails, member.radius, member.frame_width, member.frame_height)
            return _lead_rank(support, member.rise, covered)

        order = sorted(range(len(self.members)), key=rank, reverse=True)
        lead = self.members[order[0]]
        trail = list(trails[order[0]])
        for index in order[1:]:
            other = trails[index]
            if other is not trails[order[0]] and _same_plate(trail, other, lead.radius, lead.frame_width, lead.frame_height):
                trail = _join_points(trail, other)
        return LiftWindow(
            start_sec=start,
            end_sec=end,
            rise=self.rise,
            peak_sec=lead.peak_sec,
            radius=lead.radius,
            points=[q for q in trail if start <= q.t <= end],
            frame_width=lead.frame_width,
            frame_height=lead.frame_height,
            rise_start_sec=self.rise_start_sec,
            rise_end_sec=self.rise_end_sec,
            trail=trail,
        )


def _merge_windows(windows: List[LiftWindow]) -> List[LiftWindow]:
    """
    Collapse windows that cover the same rise seen from different trails, and
    join the phases of one lift — pull, catch, recovery, or a clean and the jerk
    after it — that sit within PHASE_GAP of each other into a single window.

    A joined window never grows past MAX_WINDOW_SECONDS. Two attempts whose
    padding touches but which are too long to be one lift stay separate, with
    the boundary cut between them so neither holds the other's rise. (Merging
    regardless and truncating the result left one window whose own peak lay
    outside it, with the other attempt's trail inside it.)
    """
    groups: List[_Group] = []
    for window in sorted(windows, key=lambda w: w.start_sec):
        if window.rise < STRONG_RISE:
            continue  # a weak rise neither starts nor extends a window
        if groups:
            group = groups[-1]
            gap = window.start_sec - group.end_sec
            joined_end = max(group.end_sec, window.end_sec)
            if gap <= PHASE_GAP and joined_end - group.start_sec <= MAX_WINDOW_SECONDS:
                group.add(window)
                continue
            if gap < 0:
                if window.rise_start_sec < group.rise_end_sec:
                    # Its rise began while the group's bar was still rising:
                    # another trail's view of a rise the group already holds,
                    # which cannot be joined only because the group is full.
                    continue
                # Padded into each other but not one lift: cut where the overlap
                # begins, keeping the catch of the first and the setup of the
                # second as far as possible.
                boundary = max(group.rise_end_sec + MIN_AFTER_PEAK, (group.end_sec + window.start_sec) / 2)
                boundary = min(boundary, window.rise_start_sec, window.end_sec - MIN_AFTER_PEAK)
                boundary = max(boundary, group.start_sec + MIN_AFTER_PEAK)
                group.end_sec = min(group.end_sec, boundary)
                window.start_sec = max(window.start_sec, boundary)
                if window.end_sec - window.start_sec < MIN_RISE_DURATION:
                    continue
        groups.append(_Group.of(window))
    return [group.build() for group in groups]


# Long clips are analysed in overlapping chunks. One walk across a whole clip
# spends its seeds on drops and walking and loses the plate crossing dead time;
# inside a chunk the size of one attempt it finds the plate in nearly every
# frame. The overlap keeps a lift that straddles a boundary whole in one chunk.
CHUNK_SECONDS = 8.0
CHUNK_OVERLAP = 2.5


def _chunks(start_sec: float, end_sec: float) -> List[Tuple[float, float]]:
    span = end_sec - start_sec
    if span <= CHUNK_SECONDS + CHUNK_OVERLAP:
        return [(start_sec, end_sec)]
    stride = CHUNK_SECONDS - CHUNK_OVERLAP
    chunks: List[Tuple[float, float]] = []
    begin = start_sec
    while True:
        finish = min(end_sec, begin + CHUNK_SECONDS)
        chunks.append((begin, finish))
        if finish >= end_sec:
            break
        begin += stride
    return chunks


def detect_lift_windows(
    video_path: str,
    *,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    duration_sec: Optional[float] = None,
    sample_fps: float = SAMPLE_FPS,
    analysis_width: int = ANALYSIS_WIDTH,
    max_windows: int = 8,
) -> List[LiftWindow]:
    """
    Where in the clip the bar actually goes up. Follows every plausible plate
    through each chunk of the clip and returns one padded window per credible
    rise, in clip order. Empty when nothing was tracked — the caller falls back
    to motion-energy segmentation.
    """
    if end_sec is None:
        end_sec = duration_sec if duration_sec else start_sec + 60.0

    chunks = _chunks(start_sec, end_sec)
    analyses = [_analyse(video_path, a, b, sample_fps, analysis_width) for a, b in chunks]

    # Scenery does not move: a cell static in more than one chunk of the clip
    # is a logo or a parked plate, not the bar, and comes out of every chunk.
    # A single-chunk clip cannot tell scenery from a long rack pause, so there
    # the hop and wander rules carry the load alone.
    seen_in = {}
    for index, analysis in enumerate(analyses):
        if analysis is None:
            continue
        for key in _static_cells(analysis):
            seen_in.setdefault(key, set()).add(index)
    # A bar racked for a long pause before the jerk, or resting through a long
    # setup, is static across two ADJACENT chunks (they overlap). Scenery is
    # static everywhere: in three chunks, or in two that do not touch.
    scenery = {
        key
        for key, chunks_seen in seen_in.items()
        if len(chunks_seen) >= 3 or (len(chunks_seen) == 2 and max(chunks_seen) - min(chunks_seen) >= 2)
    }
    if scenery:
        analyses = [_analyse(video_path, a, b, sample_fps, analysis_width, scenery) for a, b in chunks]

    windows: List[LiftWindow] = []
    for analysis in analyses:
        if analysis is None or not analysis.seeds:
            continue
        for trail in _trails(analysis):
            # Raw points, not smoothed: smoothing spreads a one-frame hop over
            # three points and hides it from the hop rule. The drawn trace is
            # smoothed later, in LiftWindow.bar_path.
            points = [
                BarPoint(t=analysis.times[i], x=c[0] / analysis.width, y=c[1] / analysis.height)
                for i, c in trail.circles
            ]
            for episode in rise_episodes(points, sample_fps, _step_radii(trail.circles, trail.radius)):
                if not episode.credible:
                    continue
                window_end = episode.end_sec + POST_PAD
                if episode.fall_sec is not None:
                    window_end = min(window_end, episode.fall_sec)
                w_start = max(start_sec, episode.start_sec - PRE_PAD)
                w_end = min(end_sec, max(window_end, episode.end_sec + 0.3))
                windows.append(
                    LiftWindow(
                        start_sec=w_start,
                        end_sec=w_end,
                        rise=episode.rise,
                        peak_sec=episode.end_sec,
                        radius=trail.radius,
                        points=[q for q in points if w_start <= q.t <= w_end],
                        frame_width=analysis.width,
                        frame_height=analysis.height,
                        rise_start_sec=episode.start_sec,
                        rise_end_sec=episode.end_sec,
                        trail=points,
                    )
                )
    windows = _same_plate_only(windows)
    windows = _merge_windows(windows)
    if len(windows) > max_windows:
        windows = sorted(windows, key=lambda w: -w.rise)[:max_windows]
        windows.sort(key=lambda w: w.start_sec)
    return windows
