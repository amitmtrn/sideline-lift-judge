from __future__ import annotations

import base64
from typing import List, Optional


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


def frames_to_data_urls(frames_bgr: List["np.ndarray"], *, jpeg_quality: int = 85) -> List[str]:
    """
    JPEG-encode BGR frames as data URLs for the OpenAI-compatible image_url part.
    Unencodable frames are skipped.
    """
    import cv2  # type: ignore

    data_urls: List[str] = []
    for frame in frames_bgr:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        data_urls.append(f"data:image/jpeg;base64,{b64}")
    return data_urls


def prepare_frames(frames_bgr: List["np.ndarray"], *, rotation_deg: int = 0, long_side: int = 768) -> List["np.ndarray"]:
    """Apply phone rotation metadata, then downscale, in the order the models expect."""
    return [resize_long_side(rotate_by_degrees(frame, rotation_deg), long_side) for frame in frames_bgr]

