"""
Lift-window detection against hand-labelled real footage.

The 39 clips live in samples/ (gitignored, never committed), so this runs only
on a machine that has them and is skipped everywhere else. The labels in
tests/data/lift_windows_truth.json are [start, top, end] per lift, read off
0.75-second frame strips: start = the bar leaves the floor or rack, top = the
lockout or rack is reached, end = the bar starts coming down.
"""
import json
from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
TRUTH = Path(__file__).resolve().parent / "data" / "lift_windows_truth.json"

truth = json.loads(TRUTH.read_text())
have_samples = all((SAMPLES / name).is_file() for name in truth)

pytestmark = pytest.mark.skipif(not have_samples, reason="real sample clips are not present")

# Measured on 2026-10-21 at 85% recall / 78% precision (9 of 62 lifts missed, 11
# junk windows of 51). The 92% / 84% noted on 2026-10-18 no longer reproduced
# on that day's code either (85% / 79%), so these floors are set a little under
# what is measured now; a change that drops below them needs a reason.
MIN_RECALL = 0.80
MIN_PRECISION = 0.75


def _detect(name):
    # Module-level so the process pool can pickle it; a closure cannot be sent
    # to a worker process.
    from server.utils.bar_track import detect_lift_windows
    from server.utils.video_meta import probe_video

    path = str(SAMPLES / name)
    return name, [(w.start_sec, w.end_sec) for w in detect_lift_windows(path, duration_sec=probe_video(path).duration_sec)]


@pytest.mark.slow
def test_lift_windows_match_hand_labels():
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=4) as pool:
        found = dict(pool.map(_detect, sorted(truth)))

    hits = misses = junk = 0
    for name, lifts in truth.items():
        windows = found[name]
        for start, top, _end in lifts:
            if any(a <= top <= b and b >= start for a, b in windows):
                hits += 1
            else:
                misses += 1
        junk += sum(1 for a, b in windows if not any(a <= top <= b for _s, top, _e in lifts))

    recall = hits / (hits + misses)
    precision = (sum(len(v) for v in found.values()) - junk) / max(1, sum(len(v) for v in found.values()))
    assert recall >= MIN_RECALL, f"recall {recall:.0%}: {misses} of {hits + misses} lifts missed"
    assert precision >= MIN_PRECISION, f"precision {precision:.0%}: {junk} junk windows"
