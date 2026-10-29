"""
FR-10: render the annotated replay.

The verdict is burnt into the trimmed clip so the file a coach downloads or
shares over WhatsApp still carries the score — a sidecar caption would not
survive the share sheet. Output is deliberately conservative so iOS Safari
plays it inline: H.264 baseline-friendly yuv420p, even dimensions, AAC or no
audio, and +faststart so playback can begin before the file finishes loading.

Every overlay — header strip, score, confidence, cue footer, bar-path trace,
reference line, drift arrow — is pre-rendered as one transparent BGRA image
in Python with OpenCV and composited by ffmpeg's `overlay` filter. The static
ffmpeg bundled by imageio-ffmpeg has no `drawtext` filter, and we must not
depend on it: a replay that composites but silently loses its text would ship
a verdict with nothing on it. The overlay filter is in every reasonable build.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from server.utils.bar_track import BarPath
from server.utils.ffmpeg_binary import ffmpeg_exe
from server.utils.video_meta import probe_video

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from server.schemas.judgment import EvidencedCue, LiftJudgment

# Keep the replay small enough to send over gym 4G; 720 on the long side is
# plenty for a phone screen.
MAX_LONG_SIDE = 720
# Cap the output framerate so a 60 fps phone clip stays well under the Basic
# dyno's deadline. 30 reads as a video and halves the encode's work.
OUTPUT_FPS = 30
BAND_COLORS = {
    "elite": "0x3ddc97",
    "solid": "0x5ec0a6",
    "mixed": "0xf2c744",
    "poor": "0xff8a5b",
    "fail": "0xff6b6b",
    "unknown": "0x9fb2c7",
}


def _band_color_bgra(band: str) -> Tuple[int, int, int, int]:
    """BAND_COLORS[band] is '0xRRGGBB'. Return BGRA so OpenCV draws it right."""
    hex_rgb = BAND_COLORS.get(band, BAND_COLORS["unknown"])[2:]
    r = int(hex_rgb[0:2], 16)
    g = int(hex_rgb[2:4], 16)
    b = int(hex_rgb[4:6], 16)
    return (b, g, r, 255)


def wrap_text(text: str, width: int) -> List[str]:
    """Greedy wrap on word boundaries by character count. Public helper."""
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


def _fit_scale(
    text: str,
    max_width_px: int,
    *,
    font_face: int,
    start_scale: float,
    thickness: int,
    min_scale: float = 0.35,
) -> float:
    """
    Pick the largest OpenCV font scale at which `text` still fits inside
    `max_width_px`. The score and the confidence are part of the honesty
    contract, so it is the font size that gives way to a portrait clip, not
    the text.
    """
    import cv2  # type: ignore

    scale = max(min_scale, start_scale)
    while scale > min_scale:
        (text_width, _), _ = cv2.getTextSize(text, font_face, scale, thickness)
        if text_width <= max_width_px:
            return scale
        scale -= 0.05
    return min_scale


def _wrap_to_pixels(
    text: str,
    max_width_px: int,
    *,
    font_face: int,
    scale: float,
    thickness: int,
) -> List[str]:
    """Word-wrap by measured pixel width — no character-count guessing."""
    import cv2  # type: ignore

    words = (text or "").split()
    if not words or max_width_px <= 0:
        return []
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        probe = f"{current} {word}"
        (probe_width, _), _ = cv2.getTextSize(probe, font_face, scale, thickness)
        if probe_width <= max_width_px:
            current = probe
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


# OpenCV's Hershey fonts are what render_cue_still already uses; the renderer
# below keeps that look consistent across stills and the replay.
_TITLE_FONT = None  # set lazily so this module imports without OpenCV at test time
_BODY_FONT = None


def _draw_banners(layer, judgment: "LiftJudgment", width: int, height: int) -> None:
    """
    Header strip (lift, score, band, confidence) and footer strip (first cue,
    wrapped to two lines) drawn into `layer` in place. `layer` must be BGRA.
    """
    import cv2  # type: ignore

    pad = max(6, int(width * 0.022))
    thickness = max(1, round(width / 500))

    score = "—" if judgment.score is None else f"{judgment.score}/10"
    headline = f"{judgment.lift.value.upper()}  {score}  {judgment.band.upper()}"
    if judgment.confidence is not None:
        headline += f"   {int(round(judgment.confidence * 100))}%"

    title_face = cv2.FONT_HERSHEY_DUPLEX
    title_scale = _fit_scale(
        headline,
        width - 2 * pad,
        font_face=title_face,
        start_scale=max(0.5, width / 900),
        thickness=thickness + 1,
    )
    (_, title_h), _ = cv2.getTextSize(headline, title_face, title_scale, thickness + 1)
    header_h = title_h + 2 * pad

    # Dark header strip with alpha, then the headline in the band's colour.
    cv2.rectangle(layer, (0, 0), (width, header_h), (0, 0, 0, 160), -1)
    cv2.putText(
        layer,
        headline,
        (pad, header_h - pad),
        title_face,
        title_scale,
        _band_color_bgra(judgment.band),
        thickness + 1,
        cv2.LINE_AA,
    )

    # Footer with the first cue, if there is one.
    cue = (judgment.cues or [""])[0] if judgment.cues else ""
    if cue:
        body_face = cv2.FONT_HERSHEY_SIMPLEX
        body_scale = max(0.4, width / 1100)
        body_thickness = max(1, round(width / 650))
        lines = _wrap_to_pixels(
            cue,
            width - 2 * pad,
            font_face=body_face,
            scale=body_scale,
            thickness=body_thickness,
        )[:2]
        if lines:
            (_, line_h), _ = cv2.getTextSize("Hp", body_face, body_scale, body_thickness)
            gap = max(2, int(line_h * 0.35))
            footer_h = len(lines) * (line_h + gap) + 2 * pad
            footer_y = height - footer_h
            cv2.rectangle(layer, (0, footer_y), (width, height), (0, 0, 0, 160), -1)
            y = footer_y + pad + line_h
            for line in lines:
                cv2.putText(
                    layer,
                    line,
                    (pad, y),
                    body_face,
                    body_scale,
                    (255, 255, 255, 255),
                    body_thickness,
                    cv2.LINE_AA,
                )
                y += line_h + gap


def build_full_overlay(
    judgment: "LiftJudgment",
    bar_path: Optional["BarPath"],
    width: int,
    height: int,
):
    """
    One BGRA image carrying everything the replay wants burned in: bar-path
    trace, reference line, drift marker, header with score/band/confidence,
    and a two-line cue footer. Returns the BGRA array or None when the size
    is invalid. Transparent outside the overlay regions, so compositing it
    with ffmpeg's `overlay` filter leaves the rest of the frame untouched.
    """
    import numpy as np  # type: ignore

    if width <= 0 or height <= 0:
        return None
    layer = None
    if bar_path is not None and bar_path.tracked:
        layer = build_trace_overlay(bar_path, width, height)
    if layer is None:
        layer = np.zeros((height, width, 4), dtype=np.uint8)
    _draw_banners(layer, judgment, width, height)
    return layer


def write_full_overlay(
    judgment: "LiftJudgment",
    bar_path: Optional["BarPath"],
    width: int,
    height: int,
    out_path: str,
) -> Optional[str]:
    """Serialise `build_full_overlay` to disk for ffmpeg to pick up as input."""
    import cv2  # type: ignore

    layer = build_full_overlay(judgment, bar_path, width, height)
    if layer is None:
        return None
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return str(destination) if cv2.imwrite(str(destination), layer) else None


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
        subprocess.run(
            args,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        logger.warning(
            "ffmpeg %s failed (exit %s): %s",
            Path(args[0]).name if args else "ffmpeg",
            exc.returncode,
            stderr[:800] or "no stderr",
        )
        return False
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg %s timed out after %.0fs", Path(args[0]).name if args else "ffmpeg", timeout)
        return False
    except OSError as exc:
        logger.warning("ffmpeg %s could not be spawned: %s", args[0] if args else "ffmpeg", exc)
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
        ffmpeg = ffmpeg_exe()
        if ffmpeg:
            tmp_frame = str(destination.with_suffix(".raw_frame.jpg"))
            extract_args = [
                ffmpeg, "-y", "-v", "error",
                "-ss", f"{max(0.0, key_t):.3f}",
                "-i", str(video_path),
                "-vframes", "1",
                tmp_frame,
            ]
            if _run(extract_args, timeout):
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
) -> Tuple[Optional[str], bool]:
    """
    Write an iOS-safe MP4 of [start_sec, end_sec] with every overlay (score,
    confidence, cue, bar-path trace, drift arrow) burnt in. The overlay is a
    single transparent BGRA PNG built in Python with OpenCV, composited by
    ffmpeg's `overlay` filter — no `drawtext`, which the imageio-ffmpeg
    bundled binary does not carry and would silently drop the text.

    Degrades in steps: full overlay, trace-only overlay, then a plain clip.
    Returns `(path, annotated)`: `annotated` is False when nothing but the
    plain clip could be produced, so the caller can be honest about it. A
    None path means every attempt failed (or no ffmpeg binary is reachable).
    """
    ffmpeg = ffmpeg_exe()
    if not ffmpeg:
        logger.warning("render_annotated_replay: no ffmpeg binary available; skipping replay")
        return None, False

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    info = probe_video(video_path)
    size = output_size(info.display_width, info.display_height)
    if size is None:
        logger.warning("render_annotated_replay: unknown clip size for %s", video_path)
        return None, False
    width, height = size
    scale = f"scale={width}:{height}"

    trim_args: List[str] = []
    if start_sec is not None:
        trim_args += ["-ss", f"{max(0.0, float(start_sec)):.3f}"]
    if end_sec is not None:
        duration = max(0.1, float(end_sec) - max(0.0, float(start_sec or 0.0)))
        trim_args += ["-t", f"{duration:.3f}"]

    # Rotation metadata is baked in by the scaler, so clear it afterwards or a
    # player would rotate the overlay along with the picture.
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
        # Capping the output framerate keeps a 60 fps portrait clip inside the
        # Basic dyno's job deadline. 30 reads as video and halves encode work.
        "-r",
        str(OUTPUT_FPS),
        *audio_args,
        "-movflags",
        "+faststart",
        "-metadata:s:v:0",
        "rotate=0",
    ]

    full_overlay = write_full_overlay(
        judgment, bar_path, width, height, str(destination.with_suffix(".overlay.full.png"))
    )
    trace_overlay = None
    if bar_path is not None and bar_path.tracked:
        trace_overlay = write_trace_overlay(
            bar_path, width, height, str(destination.with_suffix(".overlay.trace.png"))
        )

    # Best first, each step dropping one layer of annotation. `annotated` is
    # true for any overlay path and false for the bare "plain" fallback.
    attempts: List[Tuple[str, Optional[str], bool]] = []
    if full_overlay:
        attempts.append(("full", full_overlay, True))
    if trace_overlay and trace_overlay != full_overlay:
        attempts.append(("trace", trace_overlay, True))
    attempts.append(("plain", None, False))

    try:
        for attempt_index, (label, overlay_path, annotated) in enumerate(attempts):
            if overlay_path:
                chain = f"[0:v]{scale}[base];[base][1:v]overlay=0:0[out]"
                args = [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    *trim_args,
                    "-i",
                    str(video_path),
                    "-i",
                    overlay_path,
                    "-filter_complex",
                    chain,
                    "-map",
                    "[out]",
                    "-map",
                    "0:a?",
                    *encode_args,
                    str(destination),
                ]
            else:
                args = [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    *trim_args,
                    "-i",
                    str(video_path),
                    "-vf",
                    scale,
                    *encode_args,
                    str(destination),
                ]
            if _run(args, timeout) and destination.exists() and destination.stat().st_size > 1024:
                if attempt_index > 0:
                    logger.warning(
                        "Replay fell back to %s; the fuller overlay could not be drawn.", label
                    )
                return str(destination), annotated
        logger.warning(
            "render_annotated_replay: all %d encode attempts failed for %s",
            len(attempts),
            video_path,
        )
        return None, False
    finally:
        for _, path, _ in attempts:
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
