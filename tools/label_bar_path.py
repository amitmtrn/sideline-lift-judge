#!/usr/bin/env python3
"""
Build per-frame ground truth for where the bar is, from one seed per plate.

    python tools/label_bar_path.py candidates --out /tmp/sheets
    #   look at the sheets, write tests/data/bar_path_seeds.json
    python tools/label_bar_path.py propagate --out /tmp/check
    #   look at the check strips, add a seed wherever the track left the plate

The seeds file maps each clip to a list of plates, each a list of seeds
{"t": sec, "x": 0..1 of width, "y": 0..1 of height, "r": 0..1 of width} — a
circle off the sheet (candidates.json has them in 640 px), or a later one put
where an earlier seed's track lost the plate.

The lift-window labels say when each lift happens, not where the bar is, so on
their own they cannot tell a trace that follows the plate from one that hops to
a knee. This turns a person's choice of plate — one circle per plate, picked
off a numbered sheet — into a track of the plate's centre at native frame
rate, followed by pyramidal Lucas-Kanade and nudged onto a Hough circle of the
seed's size when one turns up where the track says the plate is.

What this gets right is which plate, and how it moved. What it does not pin
down exactly is the plate's centre: a seed snapped to a circle that took in
some background sits up to half a radius off, and the flow carries that
offset, unchanged, to the end. (Fitting the rim frame by frame was tried and
found a convincing circle at almost any size in a cluttered setup frame.)
Everything the app reports — drift from the starting line, the shape of the
path — is unchanged by a constant offset, so tools/score_bar_path.py compares
shapes. The radius stored is the seed's Hough radius and is approximate.

Both plates are tracked where both can be. Which one is nearer the camera is
not something Hough's radii can settle (one plate reads 39 px in one frame and
71 px in the next), and a trace on either plate is a true bar path; what is
wrong is a trace that leaves the plate it was on. A plate whose track could not
be made to hold is left out for that clip.

The track is truth only once someone has looked at the check strips.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from server.utils.bar_track import ANALYSIS_WIDTH, MAX_RADIUS_RATIO, MIN_RADIUS_RATIO, _circles  # noqa: E402

SAMPLES = ROOT / "samples"
WINDOWS = ROOT / "tests" / "data" / "lift_windows_truth.json"
SEEDS = ROOT / "tests" / "data" / "bar_path_seeds.json"
TRUTH = ROOT / "tests" / "data" / "bar_path_truth.json"


def use_set(name: str) -> None:
    """
    Point the tool at another set of clips: samples/<name>/ for the video,
    tests/data/<name>/ for the labels, seeds and truth. The default set is
    the gym session in samples/ itself.
    """
    global SAMPLES, WINDOWS, SEEDS, TRUTH
    SAMPLES = ROOT / "samples" / name
    data = ROOT / "tests" / "data" / name
    WINDOWS, SEEDS, TRUTH = data / "lift_windows_truth.json", data / "bar_path_seeds.json", data / "bar_path_truth.json"

# The seed sheet is drawn this long before the first lift is labelled as
# leaving the floor — the labels are read off 0.75 s strips, so a shorter lead
# sometimes starts with the bar already at the hips. The track then runs back
# from the seed as far as LOOK_BACK before that label, for lifts that start
# earlier still, and on to LEAD_OUT past the last lift's end.
LEAD_IN = 1.0
LOOK_BACK = 3.0
LEAD_OUT = 0.3
# Every Nth native frame is written out: ~30 Hz on 60 fps phone footage, finer
# than any tracker samples.
KEEP_EVERY = 2
# Following (see follow): every SNAP_EVERY frames the track is nudged onto a
# Hough circle within SNAP_RADIUS of the seed's size and SNAP_DISTANCE radii of
# the track.
SNAP_EVERY = 3
SNAP_RADIUS = 0.12
SNAP_DISTANCE = 0.35
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def clip_names() -> List[str]:
    # clip.mp4 is a byte-identical copy of 20260925_121535.mp4.
    return sorted(name for name in json.loads(WINDOWS.read_text()) if name != "clip.mp4")


def read_span(path: Path, start_sec: float, end_sec: float):
    """
    (t, frame) at analysis width for every native frame in the span, t the
    frame's presentation time (server.utils.frames.frame_time — the app's
    clock), read from the top so no seek can land a few frames off.
    """
    from server.utils.frames import frame_time

    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            if not capture.grab():
                break
            t = frame_time(capture)
            if t < start_sec:
                continue
            if t > end_sec:
                break
            ok, frame = capture.retrieve()
            if ok:
                scale = ANALYSIS_WIDTH / frame.shape[1]
                yield t, cv2.resize(frame, (ANALYSIS_WIDTH, int(round(frame.shape[0] * scale))))
    finally:
        capture.release()


def frame_circles(frame) -> List[Tuple[int, int, int]]:
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    shorter = min(frame.shape[:2])
    min_radius = max(6, int(shorter * MIN_RADIUS_RATIO))
    return _circles(gray, min_radius, max(min_radius + 4, int(shorter * MAX_RADIUS_RATIO)))


# ---------- candidates: numbered sheets to pick the plate from ----------


def candidates(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    windows = json.loads(WINDOWS.read_text())
    listing = {}
    tiles = []
    for name in clip_names():
        t_seed = max(0.0, windows[name][0][0] - LEAD_IN)
        t, frame = next(read_span(SAMPLES / name, t_seed, t_seed + 1.0))
        found = frame_circles(frame)
        listing[name] = {"t": round(t, 4), "height": frame.shape[0], "circles": found}
        for number, (x, y, r) in enumerate(found):
            cv2.circle(frame, (x, y), r, (0, 0, 255), 2)
            cv2.putText(frame, str(number), (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv2.putText(frame, name[9:15], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        tiles.append(cv2.resize(frame, (400, int(frame.shape[0] * 400 / ANALYSIS_WIDTH))))
    for sheet, at in enumerate(range(0, len(tiles), 4)):
        group = tiles[at : at + 4]
        group += [np.zeros_like(tiles[0])] * (4 - len(group))
        cv2.imwrite(str(out / f"sheet_{sheet:02d}.jpg"), np.vstack([np.hstack(group[:2]), np.hstack(group[2:])]))
    (out / "candidates.json").write_text(json.dumps(listing, indent=1))
    print(f"{len(tiles)} clips, {len(range(0, len(tiles), 4))} sheets in {out}")


# ---------- propagate: seeds to a per-frame track ----------


def _features(gray, centre, radius):
    mask = np.zeros_like(gray)
    cv2.circle(mask, (int(centre[0]), int(centre[1])), int(radius * 0.9), 255, -1)
    found = cv2.goodFeaturesToTrack(gray, 80, 0.01, 4, mask=mask)
    return found if found is not None else np.empty((0, 1, 2), np.float32)


def _snap(gray, centre, radius) -> Optional[Tuple[float, float]]:
    """A Hough circle of the plate's size close to the track, if there is one."""
    x, y = centre
    reach = int(radius * 1.6)
    top, left = max(0, int(y) - reach), max(0, int(x) - reach)
    patch = gray[top : int(y) + reach, left : int(x) + reach]
    if patch.size == 0:
        return None
    found = cv2.HoughCircles(
        cv2.medianBlur(cv2.GaussianBlur(patch, (5, 5), 0), 5), cv2.HOUGH_GRADIENT, dp=1.2, minDist=radius,
        param1=120, param2=30, minRadius=int(radius * (1 - SNAP_RADIUS)), maxRadius=int(radius * (1 + SNAP_RADIUS)) + 1,
    )
    if found is None:
        return None
    best = min(found[0], key=lambda c: np.hypot(c[0] + left - x, c[1] + top - y))
    bx, by = float(best[0] + left), float(best[1] + top)
    return (bx, by) if np.hypot(bx - x, by - y) <= SNAP_DISTANCE * radius else None


def follow(frames, seed_xy, radius) -> List[Tuple[float, float, float]]:
    """
    (t, x, y) per frame, in analysis pixels, from a seed on frame 0: the
    median flow of corners inside the plate, nudged onto a Hough circle of
    the seed's size now and then. This is what keeps hold of the right plate;
    its centre can sit a constant few pixels off the plate's (see the module
    docstring).
    """
    cx, cy = seed_xy
    previous = cv2.cvtColor(frames[0][1], cv2.COLOR_BGR2GRAY)
    points = _features(previous, (cx, cy), radius)
    out = [(frames[0][0], cx, cy)]
    for count, (t, frame) in enumerate(frames[1:], 1):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(points) >= 5:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **LK_PARAMS)
            back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, previous, moved, None, **LK_PARAMS)
            error = np.linalg.norm((back - points).reshape(-1, 2), axis=1)
            good = (status.ravel() == 1) & (status_back.ravel() == 1) & (error < 1.0)
            if good.sum() >= 5:
                shift = np.median((moved - points).reshape(-1, 2)[good], axis=0)
                cx, cy = cx + float(shift[0]), cy + float(shift[1])
        if count % SNAP_EVERY == 0:
            rim = _snap(gray, (cx, cy), radius)
            if rim is not None:
                cx, cy = rim
        # Fresh features from where the plate is now, so the set neither
        # drains away nor drifts onto whatever the plate passes.
        points = _features(gray, (cx, cy), radius)
        out.append((t, cx, cy))
        previous = gray
    return out


PLATE_COLOURS = [(255, 0, 255), (255, 255, 0)]


def strip(frames, tracks, lifts, path: Path) -> None:
    """Frames across the clip with each plate marked, for checking by eye."""
    times = [t for t, _f in frames]
    picks = {int(i) for i in np.linspace(0, len(frames) - 1, 8)}
    # Each lift's top is always shown: that is where trackers go wrong.
    for _start, top, _end in lifts:
        picks.add(int(np.argmin([abs(t - top) for t in times])))
    tiles = []
    for i in sorted(picks):
        t, frame = frames[i]
        frame = frame.copy()
        for colour, (tracked, radius) in zip(PLATE_COLOURS, tracks):
            cv2.polylines(frame, [np.array([[x, y] for _t, x, y in tracked], np.int32)], False, colour, 1)
            _t, x, y = tracked[i]
            cv2.circle(frame, (int(x), int(y)), int(radius), colour, 2)
            cv2.circle(frame, (int(x), int(y)), 4, colour, -1)
        cv2.putText(frame, f"{t:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        tiles.append(cv2.resize(frame, (240, int(frame.shape[0] * 240 / ANALYSIS_WIDTH))))
    cv2.imwrite(str(path), np.hstack(tiles))


def track_plate(frames, plate_seeds: List[dict], height: int):
    """
    One plate through every frame, as (t, x, y) in analysis pixels. The first
    seed is followed back to the start of the span and forward; each later
    seed takes over from its own time, for a stretch where the previous one
    had left the plate.
    """
    plate_seeds = sorted(plate_seeds, key=lambda s: s["t"])
    times = [t for t, _f in frames]
    followed: List[Tuple[float, float, float]] = []
    for k, seed in enumerate(plate_seeds):
        radius = seed["r"] * ANALYSIS_WIDTH
        xy = (seed["x"] * ANALYSIS_WIDTH, seed["y"] * height)
        at = int(np.argmin([abs(t - seed["t"]) for t in times]))
        until = (
            int(np.argmin([abs(t - plate_seeds[k + 1]["t"]) for t in times]))
            if k + 1 < len(plate_seeds)
            else len(frames)
        )
        if k == 0 and at > 0:
            followed = follow(frames[at::-1], xy, radius)[1:][::-1]
        followed += follow(frames[at:until], xy, radius)
    return followed, plate_seeds[0]["r"] * ANALYSIS_WIDTH


def _propagate_one(video: Path, plates: List[List[dict]], lifts, out: Path):
    name = video.name
    first = min(s["t"] for plate in plates for s in plate)
    start = max(0.0, min(first, lifts[0][0] - LOOK_BACK))
    frames = list(read_span(video, start, lifts[-1][2] + LEAD_OUT))
    height = frames[0][1].shape[0]
    tracks = [track_plate(frames, plate, height) for plate in plates]
    entry = {
        "aspect": round(height / ANALYSIS_WIDTH, 4),
        "plates": [
            {
                "radius": round(radius / ANALYSIS_WIDTH, 4),
                "points": [
                    [round(t, 3), round(x / ANALYSIS_WIDTH, 4), round(y / height, 4)]
                    for i, (t, x, y) in enumerate(tracked)
                    if i % KEEP_EVERY == 0
                ],
            }
            for tracked, radius in tracks
        ],
    }
    strip(frames, tracks, lifts, out / f"{name[:-4]}.jpg")
    return name, entry, f"{name}: {len(frames)} frames, {len(tracks)} plates"


def propagate(out: Path, only: Optional[List[str]] = None, workers: int = 8) -> None:
    from concurrent.futures import ProcessPoolExecutor

    out.mkdir(parents=True, exist_ok=True)
    windows = json.loads(WINDOWS.read_text())
    seeds: Dict[str, List[List[dict]]] = json.loads(SEEDS.read_text())
    truth = json.loads(TRUTH.read_text()) if TRUTH.is_file() else {}
    names = [n for n in clip_names() if n in seeds and (not only or n in only)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_propagate_one, SAMPLES / n, seeds[n], windows[n], out) for n in names]
        for job in jobs:
            name, entry, message = job.result()
            truth[name] = entry
            print(message, flush=True)
    lines = ",\n".join(f"  {json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}" for k, v in sorted(truth.items()))
    TRUTH.write_text("{\n" + lines + "\n}\n")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["candidates", "propagate"])
    parser.add_argument("--out", type=Path, required=True, help="where the sheets or check strips go")
    parser.add_argument("--only", nargs="*", help="propagate just these clips (file names)")
    parser.add_argument("--set", default="", help="another set of clips, e.g. competition (see use_set)")
    args = parser.parse_args(argv)
    if args.set:
        use_set(args.set)
    if args.step == "candidates":
        candidates(args.out)
    else:
        propagate(args.out, args.only)


if __name__ == "__main__":
    main()
