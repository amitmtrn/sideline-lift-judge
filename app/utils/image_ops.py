from __future__ import annotations

from typing import Optional


def rotate_by_degrees(frame: "np.ndarray", deg: int) -> "np.ndarray":
    import cv2  # type: ignore

    d = deg % 360
    if d == 0:
        return frame
    if d == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Fallback arbitrary rotation (unlikely with metadata)
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -d, 1.0)  # negative for clockwise
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def resize_long_side(frame: "np.ndarray", long_side: int = 768) -> "np.ndarray":
    import cv2  # type: ignore

    h, w = frame.shape[:2]
    if max(h, w) <= long_side:
        return frame
    if h >= w:
        scale = long_side / float(h)
        new_h = long_side
        new_w = int(round(w * scale))
    else:
        scale = long_side / float(w)
        new_w = long_side
        new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

