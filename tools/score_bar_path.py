#!/usr/bin/env python3
"""
Score the drawn bar trace against per-frame truth.

    python tools/score_bar_path.py            # all clips, 8 workers
    python tools/score_bar_path.py --json out.json

For each hand-labelled lift (tests/data/lift_windows_truth.json) this finds the
window detect_lift_windows put around it, takes the trace the pipeline would
draw for that window (pipeline._trace_for), and compares it, point by point,
with the plate tracks in tests/data/bar_path_truth.json (built by
tools/label_bar_path.py).

A trace on either plate is a true bar path, so each trace is scored against
whichever plate it is closer to over the lift. What counts against it is being
somewhere else: on a knee, the banner, or hopping to the other plate part way.

Shapes are compared, not absolute positions: the truth's centre can sit a
constant few pixels off the plate's (see tools/label_bar_path.py), and so can
a trace that follows the plate faithfully, and neither changes anything the
app reports. Each trace's median offset from the truth over the lift is
removed before measuring. Distances are in pixels at the 640 px analysis
width; the truth's radii are too rough to turn them into centimetres.

    on path    share of trace points within ON_PATH_PX of the truth's shape
    median/p90 distance from it
    drift      max sideways departure from the starting line, trace vs truth,
               in px; and that difference in cm at the trace's own scale,
               which is how far off the number the user is shown would be
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "samples"
WINDOWS = ROOT / "tests" / "data" / "lift_windows_truth.json"
TRUTH = ROOT / "tests" / "data" / "bar_path_truth.json"

WIDTH = 640  # analysis width the errors are reported at
# The lift as scored: from a little before the labelled start (labels are read
# off 0.75 s strips) to the labelled end.
LEAD_IN = 0.5
# A point this close to the truth's shape is on the path. Plates here are 35-80
# px in radius at 640 px, so this is under half a plate; any hop to another
# object is further.
ON_PATH_PX = 20.0


def _interpolate(points: List[List[float]], t: float):
    """Truth position at time t, or None outside the track."""
    import bisect

    times = [p[0] for p in points]
    k = bisect.bisect_left(times, t)
    if k == 0 or k >= len(points):
        return None
    (t0, x0, y0), (t1, x1, y1) = points[k - 1], points[k]
    share = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    return x0 + share * (x1 - x0), y0 + share * (y1 - y0)


def score_clip(name: str, samples: Path = SAMPLES, windows_file: Path = WINDOWS, truth_file: Path = TRUTH) -> List[Dict]:
    import numpy as np

    from server.services.pipeline import _trace_for
    from server.utils.bar_track import BarPath, BarPoint, detect_lift_windows
    from server.utils.segmentation import CandidateSegment
    from server.utils.video_meta import probe_video

    lifts = json.loads(windows_file.read_text())[name]
    truth = json.loads(truth_file.read_text())[name]
    aspect = truth["aspect"]
    path = str(samples / name)
    windows = detect_lift_windows(path, duration_sec=probe_video(path).duration_sec)

    rows = []
    for number, (start, top, end) in enumerate(lifts):
        row = {"clip": name, "lift": number, "start": start, "top": top, "end": end}
        rows.append(row)
        found = [w for w in windows if w.start_sec <= top <= w.end_sec]
        if not found:
            row["status"] = "no window"
            continue
        window = found[0]
        segment = CandidateSegment(index=1, start_sec=window.start_sec, end_sec=window.end_sec, peak_sec=window.peak_sec)
        trace = _trace_for(path, segment, window, name)
        if trace is None or not trace.tracked:
            row["status"] = "no trace"
            continue
        points = [p for p in trace.points if start - LEAD_IN <= p.t <= end]
        if len(points) < 3:
            row["status"] = "trace misses lift"
            continue

        best = None
        for plate_index, plate in enumerate(truth["plates"]):
            pairs = [(p, _interpolate(plate["points"], p.t)) for p in points]
            pairs = [(p, q) for p, q in pairs if q is not None]
            if len(pairs) < 3:
                continue
            gaps = np.array([((p.x - q[0]) * WIDTH, (p.y - q[1]) * WIDTH * aspect) for p, q in pairs])
            # Which plate: the nearer one outright. The plates are a bar's
            # length apart, far more than any constant offset.
            if best is None or np.median(np.hypot(*gaps.T)) < np.median(np.hypot(*best[1].T)):
                best = (plate_index, gaps, plate, pairs)
        if best is None:
            row["status"] = "no truth for span"
            continue
        plate_index, gaps, plate, pairs = best
        errors = np.hypot(*(gaps - np.median(gaps, axis=0)).T)

        span = [q for q in plate["points"] if points[0].t <= q[0] <= points[-1].t]
        truth_path = BarPath(points=[BarPoint(t, x, y) for t, x, y in span])
        trace_drift_px = trace.max_drift()[0] * WIDTH
        truth_drift_px = truth_path.max_drift()[0] * WIDTH
        trace_plate_px = 2 * trace.plate_radius * WIDTH
        covered = sum(1 for t in np.arange(start, end, 0.1) if any(abs(p.t - t) <= 0.15 for p in points))
        row.update(
            status="scored",
            plate=plate_index,
            points=len(points),
            coverage=covered / max(1, len(np.arange(start, end, 0.1))),
            on_path=float(np.mean(errors <= ON_PATH_PX)),
            median_px=float(np.median(errors)),
            p90_px=float(np.percentile(errors, 90)),
            offset_px=float(np.hypot(*np.median(gaps, axis=0))),
            drift_px=trace_drift_px,
            truth_drift_px=truth_drift_px,
            drift_cm=trace.drift_cm(),
            drift_error_cm=abs(trace_drift_px - truth_drift_px) / trace_plate_px * 45.0 if trace_plate_px else None,
        )
    return rows


def _short(clip: str) -> str:
    """113132 for the gym's 20260925_113132.mp4; the first six characters otherwise."""
    return Path(clip).stem.split("_")[-1][:6]


def summarise(rows: List[Dict]) -> str:
    import numpy as np

    scored = [r for r in rows if r["status"] == "scored"]
    lines = [
        f"{'clip':>8} {'lift':>4} {'status':>17} {'on path':>7} {'median':>6} {'p90':>6} {'drift px':>9} {'shown':>6} {'off by':>6}",
    ]
    for r in rows:
        if r["status"] != "scored":
            lines.append(f"{_short(r['clip']):>8} {r['lift']:>4} {r['status']:>17}")
            continue
        lines.append(
            f"{_short(r['clip']):>8} {r['lift']:>4} {'scored':>17} {r['on_path']:>7.0%} "
            f"{r['median_px']:>4.0f}px {r['p90_px']:>4.0f}px {r['drift_px']:>4.0f}/{r['truth_drift_px']:<4.0f} "
            f"{r['drift_cm']:>5.1f}cm {r['drift_error_cm']:>4.1f}cm"
        )
    drift_errors = [r["drift_error_cm"] for r in scored if r["drift_error_cm"] is not None]
    clean = sum(1 for r in scored if r["on_path"] >= 0.9)
    lines += [
        "",
        f"lifts {len(rows)}: scored {len(scored)}, "
        + ", ".join(f"{s} {sum(1 for r in rows if r['status'] == s)}" for s in sorted({r['status'] for r in rows}) if s != "scored"),
        f"clean traces (>=90% of points on the path): {clean} of {len(rows)} lifts ({clean / len(rows):.0%})",
        f"on path, mean over scored: {np.mean([r['on_path'] for r in scored]):.0%}",
        f"shape error, median over scored: {np.median([r['median_px'] for r in scored]):.0f} px, "
        f"p90 median over scored: {np.median([r['p90_px'] for r in scored]):.0f} px",
        f"shown drift off by: median {np.median(drift_errors):.1f} cm, "
        f"p90 {np.percentile(drift_errors, 90):.1f} cm, worst {max(drift_errors):.1f} cm",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--json", type=Path, help="also write every row here")
    parser.add_argument("--set", default="", help="another set of clips: samples/<set>/ and tests/data/<set>/")
    args = parser.parse_args(argv)

    samples, windows_file, truth_file = SAMPLES, WINDOWS, TRUTH
    if args.set:
        data = ROOT / "tests" / "data" / args.set
        samples, windows_file, truth_file = SAMPLES / args.set, data / "lift_windows_truth.json", data / "bar_path_truth.json"
    names = sorted(json.loads(truth_file.read_text()))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(score_clip, n, samples, windows_file, truth_file) for n in names]
        rows = [row for job in jobs for row in job.result()]
    print(summarise(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
