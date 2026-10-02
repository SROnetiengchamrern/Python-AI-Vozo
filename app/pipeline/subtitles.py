"""Subtitle helpers — SRT + styled ASS (Battambang, red fill, white outline)."""

from __future__ import annotations

from pathlib import Path

from .fonts import BATTAMBANG_FAMILY, ensure_battambang_fonts
from .transcribe import Transcript


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_ass_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    cs = int(round(seconds * 100))
    hours, rem = divmod(cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, centi = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centi:02d}"


def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def resolve_khmer_font() -> str:
    """Always use Battambang (Bold file provided via fontsdir)."""
    ensure_battambang_fonts()
    return BATTAMBANG_FAMILY


def write_srt(transcript: Transcript, srt_path: Path) -> Path:
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, seg in enumerate(transcript.segments, start=1):
        text = (seg.text or "").strip()
        if not text:
            continue
        lines.append(str(index))
        lines.append(
            f"{_format_srt_timestamp(seg.start)} --> {_format_srt_timestamp(seg.end)}"
        )
        lines.append(text)
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def write_ass(
    transcript: Transcript,
    ass_path: Path,
    *,
    font_name: str | None = None,
    font_size: int = 110,
) -> Path:
    """
    Write styled ASS subtitles matching cinema Khmer look:
    Battambang Bold, red fill, thick white outline, bottom-center.
    """
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_battambang_fonts()
    font = font_name or BATTAMBANG_FAMILY
    size = max(36, min(int(font_size), 160))
    # Scale outline/shadow/margin with font size so big text still looks cinematic.
    outline_w = round(3.5 + size / 28, 1)
    shadow_w = round(1.2 + size / 90, 1)
    margin_v = max(40, int(size * 0.55))

    # ASS colours are &HAABBGGRR (alpha, blue, green, red).
    primary = "&H000000FF"  # opaque red fill
    secondary = "&H000000FF"
    outline = "&H00FFFFFF"  # white border
    back = "&H64000000"  # soft black shadow

    # Bold=-1 selects Battambang-Bold.ttf when fontsdir is set.
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{primary},{secondary},{outline},{back},-1,0,0,0,100,100,0,0,1,{outline_w},{shadow_w},2,50,50,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    for seg in transcript.segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        start = _format_ass_timestamp(seg.start)
        end = _format_ass_timestamp(max(seg.end, seg.start + 0.4))
        events.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{_escape_ass_text(text)}"
        )

    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")
    return ass_path
