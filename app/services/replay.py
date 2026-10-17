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
from typing import TYPE_CHECKING, List, Optional, Sequence

from app.utils.video_meta import probe_video

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.schemas.judgment import LiftJudgment

# Fonts vary by host; the first that exists wins. Without any of them drawtext
# cannot render, and we fall back to an un-annotated (but still playable) clip.
FONT_CANDIDATES: Sequence[str] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
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


def _scale_filter(width: int, height: int) -> str:
    """
    Downscale the long side to MAX_LONG_SIDE, keep the aspect ratio, and force
    even dimensions — H.264 with yuv420p cannot encode an odd width or height.
    """
    if width and height and max(width, height) > MAX_LONG_SIDE:
        if width >= height:
            return f"scale={MAX_LONG_SIDE}:-2"
        return f"scale=-2:{MAX_LONG_SIDE}"
    return "scale=trunc(iw/2)*2:trunc(ih/2)*2"


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
    base = max(display_height, 240)
    title_size = max(18, int(base * 0.075))
    body_size = max(14, int(base * 0.045))
    pad = max(8, int(base * 0.02))
    header_height = title_size + 2 * pad
    color = BAND_COLORS.get(judgment.band, BAND_COLORS["unknown"])

    score = "—" if judgment.score is None else f"{judgment.score}/10"
    headline = f"{judgment.lift.value.upper()}  {score}  {judgment.band.upper()}"
    if judgment.confidence is not None:
        headline += f"   confidence {judgment.confidence:.0%}"

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


def _run(args: List[str], timeout: float) -> bool:
    try:
        subprocess.run(
            args,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return True


def render_annotated_replay(
    video_path: str,
    judgment: "LiftJudgment",
    out_path: str,
    *,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    timeout: float = 180.0,
) -> Optional[str]:
    """
    Write an iOS-safe MP4 of [start_sec, end_sec] with the verdict burnt in.

    Falls back to the same clip without the overlay if drawtext fails (no font,
    or a filtergraph the local build rejects), because a replay a coach can play
    is worth more than no replay at all. Returns the written path, or None when
    ffmpeg is unavailable or every attempt failed.
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

    font = find_font()
    filter_chains: List[str] = []
    if font:
        overlay = build_overlay_filters(
            judgment,
            font=font,
            display_width=info.display_width,
            display_height=info.display_height,
        )
        filter_chains.append(",".join([scale, *overlay]))
    # Last resort: no overlay, but a clip that plays.
    filter_chains.append(scale)

    for attempt, chain in enumerate(filter_chains):
        args = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            *trim_args,
            "-i",
            str(video_path),
            "-vf",
            chain,
            *encode_args,
            str(destination),
        ]
        if _run(args, timeout) and destination.exists() and destination.stat().st_size > 1024:
            if attempt and font:
                # The clip still plays, but without the verdict burnt in — worth
                # knowing about rather than discovering it in a shared replay.
                logger.warning("Replay overlay could not be drawn; wrote a plain clip instead.")
            return str(destination)
    return None
