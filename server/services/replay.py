"""
FR-10: render the annotated replay.

The verdict is burnt into the trimmed clip with ffmpeg so the file a coach
downloads or shares over WhatsApp still carries the score — a sidecar caption
would not survive the share sheet. Output is deliberately conservative so iOS
Safari plays it inline: H.264 baseline-friendly yuv420p, even dimensions, AAC or
no audio, and +faststart so playback can begin before the file finishes loading.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from server.utils.bar_track import BarPath
from server.utils.video_meta import probe_video

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from server.schemas.judgment import EvidencedCue, LiftJudgment

# Fonts vary by host; the first that exists wins. Without any of them drawtext
# cannot render, and we fall back to an un-annotated (but still playable) clip.
FONT_CANDIDATES: Sequence[str] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    # Where Heroku's apt buildpack puts the same package (Aptfile).
    "/app/.apt/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
)

# Keep the replay small enough to send over gym 4G; 720 on the long side is
# plenty for a phone screen.
MAX_LONG_SIDE = 720
BAND_COLORS = {
    "elite": "0x3ddc97",
    "solid": "0x5ec0a6",
    "mixed": "0xf2c744",
    "poor": "0xff8a5b",
    "fail": "0xff6b6b",
    "unknown": "0x9fb2c7",
}


def find_font() -> Optional[str]:
    for path in FONT_CANDIDATES:
        if Path(path).is_file():
            return path
    return None


def escape_drawtext(text: str) -> str:
    """
    Escape for drawtext's text= value inside a filtergraph. Backslashes first,
    then the characters that end a value (: and ') and the ones that would be
    read as filtergraph syntax.

    `%` and `{}` are deliberately not escaped: every drawtext below sets
    `expansion=none`, which is what makes them literal. Escaping `%` here
    instead produces "Stray %" from the text expander.
    """
    out = text.replace("\\", "\\\\")
    for char in (":", "'", ",", "[", "]", ";", "="):
        out = out.replace(char, "\\" + char)
    return out.replace("\n", " ")


# The cue and drift text come from a model, so no part of it may be read as a
# drawtext expansion.
NO_EXPANSION = "expansion=none"


# Mean glyph advance of the bold sans fonts we use, as a fraction of font size.
# Only needs to be close: it decides whether a headline is shrunk, not laid out.
CHAR_ADVANCE = 0.58
MIN_TITLE_SIZE = 11


def fitted_font_size(text: str, available_width: int) -> int:
    """Largest font size at which `text` still fits across `available_width`."""
    if not text or available_width <= 0:
        return MIN_TITLE_SIZE
    return max(MIN_TITLE_SIZE, int(available_width / (len(text) * CHAR_ADVANCE)))


def wrap_text(text: str, width: int) -> List[str]:
    """Greedy wrap on word boundaries; a single over-long word is left alone."""
    words = (text or "").split()
    if not words:
        return []
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _even(value: int) -> int:
    """H.264 with yuv420p cannot encode an odd width or height."""
    return max(2, int(value) - (int(value) % 2))


def output_size(width: int, height: int) -> Optional[Tuple[int, int]]:
    """
    The exact size the replay will be encoded at. Computed here rather than left
    to ffmpeg's `-2` so a trace overlay can be drawn at precisely the same size
    and composited at 0:0 with no scaling guesswork.
    """
    if not (width and height):
        return None
    if max(width, height) > MAX_LONG_SIDE:
        if width >= height:
            return _even(MAX_LONG_SIDE), _even(round(height * MAX_LONG_SIDE / width))
        return _even(round(width * MAX_LONG_SIDE / height)), _even(MAX_LONG_SIDE)
    return _even(width), _even(height)


def _scale_filter(width: int, height: int) -> str:
    """
    Downscale the long side to MAX_LONG_SIDE, keeping the aspect ratio and even
    dimensions.
    """
    size = output_size(width, height)
    if size is None:
        return "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    return f"scale={size[0]}:{size[1]}"


def build_overlay_filters(
    judgment: "LiftJudgment",
    *,
    font: str,
    display_width: int,
    display_height: int,
) -> List[str]:
    """
    The overlay: a header strip with lift, score, band and confidence, and a
    footer strip with the first cue. Sizes are derived from frame height so the
    text stays readable whether the clip is portrait or landscape.
    """
    # Sized from the width, because text runs across the frame: a tall portrait
    # clip would otherwise get letters wider than the picture. These are output
    # pixels, so the caller must pass the encoded size, not the source size.
    base = max(display_width, 240)
    title_size = max(14, int(base * 0.062))
    body_size = max(12, int(base * 0.040))
    pad = max(6, int(base * 0.022))
    header_height = title_size + 2 * pad
    color = BAND_COLORS.get(judgment.band, BAND_COLORS["unknown"])

    score = "—" if judgment.score is None else f"{judgment.score}/10"
    headline = f"{judgment.lift.value.upper()}  {score}  {judgment.band.upper()}"
    if judgment.confidence is not None:
        headline += f"   confidence {judgment.confidence:.0%}"

    # Shrink the headline to fit rather than letting it run off the side of a
    # portrait clip. The confidence is part of the honesty contract, so it is
    # the font that gives way, not the text.
    title_size = min(title_size, fitted_font_size(headline, display_width - 2 * pad))
    header_height = title_size + 2 * pad

    filters = [
        f"drawbox=x=0:y=0:w=iw:h={header_height}:color=black@0.62:t=fill",
        (
            f"drawtext=fontfile='{font}':text='{escape_drawtext(headline)}'"
            f":{NO_EXPANSION}:fontcolor={color}:fontsize={title_size}:x={pad}:y={pad}"
        ),
    ]

    cue = judgment.cues[0] if judgment.cues else ""
    if cue:
        # Roughly how many characters of this size fit the frame width.
        columns = max(16, int(display_width / (body_size * 0.55))) if display_width else 32
        lines = wrap_text(cue, columns)[:2]
        footer_height = len(lines) * (body_size + 6) + 2 * pad
        filters.append(f"drawbox=x=0:y=ih-{footer_height}:w=iw:h={footer_height}:color=black@0.62:t=fill")
        for position, line in enumerate(lines):
            offset = footer_height - pad - (len(lines) - position) * (body_size + 6) + 6
            # drawtext has no `ih`: input height is `H` here, unlike in drawbox.
            filters.append(
                f"drawtext=fontfile='{font}':text='{escape_drawtext(line)}'"
                f":{NO_EXPANSION}:fontcolor=white:fontsize={body_size}:x={pad}:y=H-{footer_height - offset}"
            )
    return filters


# Overlay colours, BGRA to match OpenCV. Same palette as the app.
TRACE_GREEN = (106, 174, 47, 255)
TRACE_SHADOW = (0, 0, 0, 170)
REFERENCE_CHALK = (245, 242, 239, 190)
DRIFT_ORANGE = (59, 125, 239, 255)


def build_trace_overlay(path: "BarPath", width: int, height: int):
    """
    A transparent layer carrying the bar-path trace, the line the bar started
    over, and a marker where it strayed furthest from it. Returns a BGRA image,
    or None when there is nothing trustworthy to draw.

    Drawn once for the whole clip rather than animated: a coach pausing on any
    frame — or sharing a still — sees the whole path.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    if not path.tracked or width <= 0 or height <= 0:
        return None

    layer = np.zeros((height, width, 4), dtype=np.uint8)
    points = [(int(p.x * width), int(p.y * height)) for p in path.points]
    thickness = max(2, round(min(width, height) / 220))

    # The line the bar started over — what "keep the bar close" is measured against.
    reference_x = int(path.reference_x * width)
    for y in range(0, height, thickness * 8):
        cv2.line(layer, (reference_x, y), (reference_x, min(height, y + thickness * 4)),
                 REFERENCE_CHALK, max(1, thickness - 1), cv2.LINE_AA)

    # The path itself, dark-edged so it reads over a light gym floor too.
    for width_, colour in ((thickness + 3, TRACE_SHADOW), (thickness, TRACE_GREEN)):
        cv2.polylines(layer, [np.array(points, dtype=np.int32)], False, colour, width_, cv2.LINE_AA)

    # Where the bar was furthest from where it started.
    _drift, worst = path.max_drift()
    if worst is not None:
        wx, wy = int(worst.x * width), int(worst.y * height)
        for x in range(min(reference_x, wx), max(reference_x, wx), thickness * 6):
            cv2.line(layer, (x, wy), (min(x + thickness * 3, max(reference_x, wx)), wy),
                     DRIFT_ORANGE, thickness, cv2.LINE_AA)
        cv2.circle(layer, (wx, wy), thickness * 4, DRIFT_ORANGE, thickness, cv2.LINE_AA)

    # Where it began, so the direction of travel is unambiguous.
    cv2.circle(layer, points[0], thickness * 3, REFERENCE_CHALK, -1, cv2.LINE_AA)
    return layer


def write_trace_overlay(path: "BarPath", width: int, height: int, out_path: str) -> Optional[str]:
    import cv2  # type: ignore

    layer = build_trace_overlay(path, width, height)
    if layer is None:
        return None
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return str(destination) if cv2.imwrite(str(destination), layer) else None


def _run(args: List[str], timeout: float) -> bool:
    try:
        result = subprocess.run(
            args,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            logger.warning("ffmpeg error: %s", exc.stderr.decode(errors="replace").strip()[:400])
        return False
    except (subprocess.TimeoutExpired, OSError):
        return False
    return True


def _read_frame_at(video_path: str, t_sec: float):
    """The BGR frame presented at t_sec, or None on any failure."""
    from server.utils.frames import read_frames_at

    try:
        frame = read_frames_at(str(video_path), [max(0.0, t_sec)])[0]
    except Exception:
        logger.exception("_read_frame_at: unexpected error at t=%.3f in %s", t_sec, video_path)
        return None
    if frame is None:
        logger.warning("_read_frame_at: read failed at t=%.3f in %s", t_sec, video_path)
    return frame


def render_cue_still(
    video_path: str,
    cue: "EvidencedCue",
    bar_path: Optional["BarPath"],
    out_path: str,
    *,
    window_start_sec: float = 0.0,
    window_end_sec: float = 9999.0,
    timeout: float = 15.0,
) -> Optional[str]:
    """
    Render an annotated JPEG for one evidence cue (PRD §5.4).

    The frame is extracted at the key moment (max-drift point in the cue's
    time window). The bar path trace is drawn over it: the full lift faded,
    the window segment bold, a vertical reference line, a red drift arrow
    with a centimetre label, and text banners naming the cue and the phase.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    evidence = cue.evidence
    t_start = evidence.t_start_s if evidence else (window_start_sec + window_end_sec) / 2 - 0.5
    t_end = evidence.t_end_s if evidence else t_start + 1.0
    key_t = (t_start + t_end) / 2.0

    # Find the actual max-drift point within the time window
    key_point = None
    if bar_path is not None and bar_path.tracked and bar_path.points:
        ref = bar_path.reference_x
        candidates = [p for p in bar_path.points if t_start <= p.t <= t_end] or bar_path.points
        key_point = max(candidates, key=lambda p: abs(p.x - ref))
        key_t = key_point.t

    # Extract frame via OpenCV — same coordinate space as the tracker, no
    # subprocess and no rotation correction needed (bar_path coords are in the
    # OpenCV frame's space already).
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = _read_frame_at(video_path, key_t)
    if frame is None:
        # ffmpeg fallback (handles codecs OpenCV can't decode without extra libs)
        tmp_frame = str(destination.with_suffix(".raw_frame.jpg"))
        extract_args = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{max(0.0, key_t):.3f}",
            "-i", str(video_path),
            "-vframes", "1",
            tmp_frame,
        ]
        if shutil.which("ffmpeg") and _run(extract_args, timeout):
            frame = cv2.imread(tmp_frame)
            try:
                Path(tmp_frame).unlink(missing_ok=True)
            except OSError:
                pass
    if frame is None:
        logger.warning("render_cue_still: could not extract frame at t=%.3f from %s", key_t, video_path)
        return None

    h, w = frame.shape[:2]

    # Scale down to max 1080 px tall for storage
    max_h = 1080
    if h > max_h:
        ratio = max_h / h
        frame = cv2.resize(frame, (_even(int(w * ratio)), max_h))
        h, w = frame.shape[:2]

    # --- Bar-path trace overlay ---
    if bar_path is not None and bar_path.tracked and bar_path.points:
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        thickness = max(2, round(min(w, h) / 220))
        ref_x_px = int(bar_path.reference_x * w)

        # Full path, faded (lower alpha)
        all_pts = [(int(p.x * w), int(p.y * h)) for p in bar_path.points]
        FADED_GREEN = (106, 174, 47, 90)
        if len(all_pts) >= 2:
            cv2.polylines(layer, [np.array(all_pts, dtype=np.int32)], False, FADED_GREEN, thickness, cv2.LINE_AA)

        # Window segment, bold
        win_pts_raw = [p for p in bar_path.points if t_start <= p.t <= t_end]
        if win_pts_raw:
            win_pts = [(int(p.x * w), int(p.y * h)) for p in win_pts_raw]
            if len(win_pts) >= 2:
                cv2.polylines(layer, [np.array(win_pts, dtype=np.int32)], False, TRACE_GREEN, thickness + 2, cv2.LINE_AA)

        # Vertical reference line (dashed)
        dash_on, dash_gap = thickness * 4, thickness * 8
        for y_dash in range(0, h, dash_on + dash_gap):
            cv2.line(layer, (ref_x_px, y_dash), (ref_x_px, min(h, y_dash + dash_on)),
                     REFERENCE_CHALK, max(1, thickness - 1), cv2.LINE_AA)

        # Start-of-path marker
        cv2.circle(layer, all_pts[0], thickness * 3, REFERENCE_CHALK, -1, cv2.LINE_AA)

        # Composite trace onto frame
        alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
        frame_f = frame.astype(np.float32)
        trace_rgb = layer[:, :, :3].astype(np.float32)
        frame = np.clip(frame_f * (1 - alpha) + trace_rgb * alpha, 0, 255).astype(np.uint8)

        # Red drift arrow + circle at the key point (drawn directly on frame, no alpha)
        if key_point is not None:
            kx = int(key_point.x * w)
            ky = int(key_point.y * h)
            RED = (0, 0, 220)
            cv2.arrowedLine(frame, (ref_x_px, ky), (kx, ky), RED, max(2, thickness + 1), cv2.LINE_AA, tipLength=0.25)
            cv2.circle(frame, (kx, ky), max(4, thickness * 4), RED, thickness, cv2.LINE_AA)

            # Centimetre label next to the arrowhead
            if evidence and evidence.value is not None:
                label = f"{evidence.value:.0f} cm"
                label_x = max(4, min(kx + 6, w - 60))
                label_y = max(20, ky - 6)
                font_scale = max(0.4, min(w, h) / 800)
                lthick = max(1, thickness - 1)
                cv2.putText(frame, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), lthick + 2, cv2.LINE_AA)
                cv2.putText(frame, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), lthick, cv2.LINE_AA)

    # --- Text banners ---
    font_scale_base = max(0.4, w / 900)
    lthick = max(1, round(w / 500))
    banner_h = max(32, int(h * 0.07))
    pad = max(6, int(w * 0.015))

    # Top banner: cue text (truncated to fit)
    overlay_top = frame.copy()
    cv2.rectangle(overlay_top, (0, 0), (w, banner_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay_top, 0.65, frame, 0.35, 0)
    cue_display = cue.text if len(cue.text) <= 60 else cue.text[:57] + "…"
    cv2.putText(frame, cue_display, (pad, banner_h - pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_base * 0.85, (255, 255, 255), lthick, cv2.LINE_AA)

    # Bottom banner: phase + time
    phase_label = cue.phase.replace("_", " ").title()
    if evidence:
        bottom_text = f"{phase_label}  {evidence.t_start_s:.1f}–{evidence.t_end_s:.1f} s"
    else:
        bottom_text = phase_label
    bottom_y = h - banner_h
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, bottom_y), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay_bot, 0.65, frame, 0.35, 0)
    cv2.putText(frame, bottom_text, (pad, h - pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_base * 0.75, (200, 200, 200), lthick, cv2.LINE_AA)

    if cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        return str(destination)
    return None


# The strip: this many frames across the cue's window, each this tall and
# cropped to this width:height, around the bar path with this many plate
# diameters of margin (or this share of the frame height, whichever is more).
STRIP_TILES = 5
STRIP_TILE_HEIGHT = 360
STRIP_TILE_ASPECT = 0.75
STRIP_MARGIN_PLATES = 1.5
STRIP_MARGIN_FRAME = 0.08
STRIP_RED = (40, 40, 220)


def _strip_times(t_start: float, t_end: float, key_t: Optional[float], count: int) -> List[float]:
    """Evenly spaced times across the window, with the key moment one of them."""
    import numpy as np  # type: ignore

    times = [float(t) for t in np.linspace(t_start, t_end, count)]
    if key_t is not None and t_start <= key_t <= t_end:
        nearest = min(range(count), key=lambda i: abs(times[i] - key_t))
        times[nearest] = key_t
    return sorted(times)


def _strip_crop(bar_path: Optional["BarPath"], width: int, height: int) -> Tuple[int, int, int, int]:
    """
    One crop for every tile, so the frames line up and the bar's movement
    reads across them: the whole path with a margin, grown to the tile's
    shape. The full frame when there is no trace to centre on.
    """
    if bar_path is None or not bar_path.tracked or not bar_path.points:
        crop_w, crop_h = width, height
        if crop_w / crop_h > STRIP_TILE_ASPECT:
            crop_w = int(crop_h * STRIP_TILE_ASPECT)
        else:
            crop_h = int(crop_w / STRIP_TILE_ASPECT)
        return (width - crop_w) // 2, (height - crop_h) // 2, crop_w, crop_h
    xs = [p.x * width for p in bar_path.points] + [bar_path.reference_x * width]
    ys = [p.y * height for p in bar_path.points]
    margin = max(STRIP_MARGIN_PLATES * 2 * bar_path.plate_radius * width, STRIP_MARGIN_FRAME * height)
    left, right = min(xs) - margin, max(xs) + margin
    top, bottom = min(ys) - margin, max(ys) + margin
    crop_w, crop_h = right - left, bottom - top
    if crop_w / crop_h > STRIP_TILE_ASPECT:
        crop_h = crop_w / STRIP_TILE_ASPECT
    else:
        crop_w = crop_h * STRIP_TILE_ASPECT
    # Too big for the frame: as big as the frame allows, at the tile's shape.
    scale = min(1.0, width / crop_w, height / crop_h)
    crop_w, crop_h = int(crop_w * scale), int(crop_h * scale)
    centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
    x0 = int(min(max(0, centre_x - crop_w / 2), width - crop_w))
    y0 = int(min(max(0, centre_y - crop_h / 2), height - crop_h))
    return x0, y0, crop_w, crop_h


def render_cue_strip(
    video_path: str,
    cue: "EvidencedCue",
    bar_path: Optional["BarPath"],
    out_path: str,
    *,
    tiles: int = STRIP_TILES,
) -> Optional[str]:
    """
    The cue's evidence as the lift itself, frame by frame: `tiles` frames
    across the cue's time window, side by side, one of them the moment the
    bar was furthest from where it started. Every frame carries the bar path
    so far, the line the bar started over, and a circle on what the tracker
    followed — so a strip shows plainly when the tracker was on the plate and
    when it was not. The key frame carries the drift arrow and a red border.

    Without a trace the frames are shown bare: they are still the evidence a
    vision-only cue rests on. Returns the written path, or None when the
    frames could not be read.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    from server.utils.frames import read_frames_at

    evidence = cue.evidence
    if evidence is None:
        return None
    t_start, t_end = evidence.t_start_s, evidence.t_end_s
    tracked = bar_path is not None and bar_path.tracked and bool(bar_path.points)
    key_point = None
    if tracked:
        reference = bar_path.reference_x
        inside = [p for p in bar_path.points if t_start <= p.t <= t_end] or bar_path.points
        key_point = max(inside, key=lambda p: abs(p.x - reference))
    times = _strip_times(t_start, t_end, key_point.t if key_point else None, tiles)
    frames = read_frames_at(video_path, times)
    if not any(frame is not None for frame in frames):
        logger.warning("render_cue_strip: no frames read in %.2f-%.2f s of %s", t_start, t_end, video_path)
        return None
    height, width = next(f for f in frames if f is not None).shape[:2]
    x0, y0, crop_w, crop_h = _strip_crop(bar_path, width, height)
    tile_h = STRIP_TILE_HEIGHT
    tile_w = int(round(tile_h * crop_w / crop_h))
    scale = tile_h / crop_h

    def to_tile(p) -> Tuple[int, int]:
        return int((p.x * width - x0) * scale), int((p.y * height - y0) * scale)

    step = 1.5 / 12.0  # a trace point this close in time is this frame's
    out = []
    for t, frame in zip(times, frames):
        if frame is None:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        else:
            tile = cv2.resize(frame[y0 : y0 + crop_h, x0 : x0 + crop_w], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        is_key = key_point is not None and abs(t - key_point.t) < 1e-6
        if tracked:
            ref_x = int((bar_path.reference_x * width - x0) * scale)
            for y in range(0, tile_h, 12):
                cv2.line(tile, (ref_x, y), (ref_x, min(tile_h, y + 6)), REFERENCE_CHALK[:3], 1, cv2.LINE_AA)
            everything = np.array([to_tile(p) for p in bar_path.points], dtype=np.int32)
            faded = tile.copy()
            cv2.polylines(faded, [everything], False, TRACE_GREEN[:3], 2, cv2.LINE_AA)
            tile = cv2.addWeighted(faded, 0.35, tile, 0.65, 0)
            so_far = np.array([to_tile(p) for p in bar_path.points if p.t <= t + 1e-6], dtype=np.int32)
            if len(so_far) >= 2:
                cv2.polylines(tile, [so_far], False, TRACE_SHADOW[:3], 5, cv2.LINE_AA)
                cv2.polylines(tile, [so_far], False, TRACE_GREEN[:3], 3, cv2.LINE_AA)
            here = min(bar_path.points, key=lambda p: abs(p.t - t))
            if abs(here.t - t) <= step:
                radius = max(4, int(bar_path.plate_radius * width * scale))
                cv2.circle(tile, to_tile(here), radius, (255, 255, 255), 2, cv2.LINE_AA)
            if is_key:
                kx, ky = to_tile(key_point)
                cv2.arrowedLine(tile, (ref_x, ky), (kx, ky), STRIP_RED, 3, cv2.LINE_AA, tipLength=0.3)
                if evidence.value is not None:
                    label = f"{evidence.value:.0f} cm"
                    (text_w, _h), _base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 4)
                    lx, ly = max(4, min(kx + 6, tile_w - text_w - 4)), max(40, ky - 8)
                    cv2.putText(tile, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(tile, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        stamp = f"{t:.1f} s"
        cv2.putText(tile, stamp, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, stamp, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        if is_key:
            cv2.rectangle(tile, (1, 1), (tile_w - 2, tile_h - 2), STRIP_RED, 3)
        out.append(tile)
    gap = np.zeros((tile_h, 4, 3), dtype=np.uint8)
    strip = np.hstack([part for tile in out for part in (tile, gap)][:-1])
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(destination), strip, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        return str(destination)
    return None


def render_annotated_replay(
    video_path: str,
    judgment: "LiftJudgment",
    out_path: str,
    *,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    bar_path: Optional["BarPath"] = None,
    timeout: float = 180.0,
) -> Optional[str]:
    """
    Write an iOS-safe MP4 of [start_sec, end_sec] with the verdict burnt in, and
    the bar-path trace drawn over it when `bar_path` was tracked confidently.

    Degrades in steps: trace + text, then text alone, then a plain clip. A replay
    a coach can play is worth more than no replay at all. Returns the written
    path, or None when ffmpeg is unavailable or every attempt failed.
    """
    if not shutil.which("ffmpeg"):
        return None

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    info = probe_video(video_path)
    scale = _scale_filter(info.display_width, info.display_height)

    trim_args: List[str] = []
    if start_sec is not None:
        trim_args += ["-ss", f"{max(0.0, float(start_sec)):.3f}"]
    if end_sec is not None:
        duration = max(0.1, float(end_sec) - max(0.0, float(start_sec or 0.0)))
        trim_args += ["-t", f"{duration:.3f}"]

    # Rotation metadata is baked in by the scaler, so clear it afterwards or a
    # player would rotate the burnt-in text along with the picture.
    audio_args = ["-c:a", "aac", "-b:a", "96k"] if info.has_audio else ["-an"]
    encode_args = [
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "25",
        *audio_args,
        "-movflags",
        "+faststart",
        "-metadata:s:v:0",
        "rotate=0",
    ]

    # Every filter below runs after `scale`, so all sizes are in output pixels.
    size = output_size(info.display_width, info.display_height)
    text_width, text_height = size if size else (info.display_width, info.display_height)

    font = find_font()
    text_filters = (
        build_overlay_filters(
            judgment,
            font=font,
            display_width=text_width,
            display_height=text_height,
        )
        if font
        else []
    )

    # The trace is a transparent PNG built at exactly the encoded size, so it
    # composites at 0:0 with no scaling of its own.
    trace_file = None
    if bar_path is not None and bar_path.tracked and size is not None:
        trace_file = write_trace_overlay(bar_path, size[0], size[1], str(destination.with_suffix(".trace.png")))

    # Best first, each step dropping one layer of annotation.
    attempts: List[Tuple[str, List[str]]] = []
    if trace_file:
        chain = f"[0:v]{scale}[base];[base][1:v]overlay=0:0[traced]"
        if text_filters:
            chain += f";[traced]{','.join(text_filters)}[out]"
        else:
            chain += ";[traced]null[out]"
        attempts.append(
            ("trace+text", ["-i", trace_file, "-filter_complex", chain, "-map", "[out]", "-map", "0:a?"])
        )
    if text_filters:
        attempts.append(("text", ["-vf", ",".join([scale, *text_filters])]))
    attempts.append(("plain", ["-vf", scale]))

    try:
        for attempt, (label, filter_args) in enumerate(attempts):
            args = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                *trim_args,
                "-i",
                str(video_path),
                *filter_args,
                *encode_args,
                str(destination),
            ]
            if _run(args, timeout) and destination.exists() and destination.stat().st_size > 1024:
                if attempt:
                    # The clip still plays, but with less on it than intended —
                    # worth knowing rather than discovering in a shared replay.
                    logger.warning(
                        "Replay fell back to %s; the fuller overlay could not be drawn.", label
                    )
                return str(destination)
        return None
    finally:
        if trace_file:
            Path(trace_file).unlink(missing_ok=True)
