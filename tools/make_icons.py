#!/usr/bin/env python3
"""
Draw the app icons so they can be regenerated instead of being opaque binaries.

    python tools/make_icons.py

A barbell seen from the side with the vertical bar path through it — the one
thing the app measures. Uses OpenCV, which is already a dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore

STATIC_ICONS = Path(__file__).resolve().parent.parent / "app" / "static" / "icons"

# BGR, matching the palette in styles.css.
STEEL = (18, 15, 12)        # #0c0f12 (--steel-900)
PLATE = (48, 194, 242)      # #f2c230 (--plate-yellow)
CHALK = (245, 242, 239)     # #eff2f5 (--chalk)
PATH_GREEN = (106, 174, 47)  # #2fae6a (--plate-green)


def draw_icon(size: int, *, margin_ratio: float) -> "np.ndarray":
    """
    `margin_ratio` keeps the artwork inside the safe area. A maskable icon can be
    cropped to a circle by the launcher, so it needs a wider margin than a plain
    one.
    """
    canvas = np.full((size, size, 3), STEEL, dtype=np.uint8)
    center = size // 2
    margin = round(size * margin_ratio)
    usable = size - 2 * margin

    # Vertical bar path: 7 symmetric rectangular dashes (6 gaps) across the span.
    # The middle dash sits behind the bar so 3 clean dashes show above and 3 below.
    dash_count = 7
    unit = usable / (2 * dash_count - 1)
    path_half_w = max(2, round(usable * 0.028))
    for i in range(dash_count):
        y0 = round(margin + (2 * i) * unit)
        y1 = round(margin + (2 * i + 1) * unit)
        canvas[y0:y1, center - path_half_w : center + path_half_w] = PATH_GREEN

    # Horizontal bar (chalk white) with sleeve tips extending past both plates.
    bar_half_h = max(2, round(usable * 0.04))
    canvas[center - bar_half_h : center + bar_half_h, margin : size - margin] = CHALK

    # Loading plates (yellow), symmetric at each end of the bar.
    sleeve = max(3, round(usable * 0.08))
    plate_w = max(5, round(usable * 0.13))
    plate_half_h = max(8, round(usable * 0.26))
    y0, y1 = center - plate_half_h, center + plate_half_h

    left_x0 = margin + sleeve
    canvas[y0:y1, left_x0 : left_x0 + plate_w] = PLATE

    right_x1 = size - margin - sleeve
    canvas[y0:y1, right_x1 - plate_w : right_x1] = PLATE

    return canvas


def main() -> int:
    STATIC_ICONS.mkdir(parents=True, exist_ok=True)
    targets = [
        ("icon-192.png", 192, 0.15),
        ("icon-512.png", 512, 0.15),
        # Android crops maskable icons to a circle: keep all artwork inside the central 60%.
        ("icon-maskable-512.png", 512, 0.22),
        ("apple-touch-icon.png", 180, 0.15),
    ]
    for name, size, margin_ratio in targets:
        path = STATIC_ICONS / name
        if not cv2.imwrite(str(path), draw_icon(size, margin_ratio=margin_ratio)):
            print(f"ERROR: could not write {path}", file=sys.stderr)
            return 1
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
