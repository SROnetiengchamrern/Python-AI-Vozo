"""Subtitle helpers — SRT + styled ASS (max 2 lines, voice-synced)."""

from __future__ import annotations

import re
from pathlib import Path

from .fonts import BATTAMBANG_FAMILY, ensure_battambang_fonts
from .transcribe import Transcript

Cue = tuple[float, float, str]


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
    ensure_battambang_fonts()
    return BATTAMBANG_FAMILY


def normalize_khmer_text(text: str) -> str:
    """Shared clean for subtitles AND speech (must stay identical)."""
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if re.search(r"[A-Za-z]{4,}", text) and not re.search(r"[\u1780-\u17FF]", text):
        return ""
    text = re.sub(r"[A-Za-z]{3,}(?:\s+[A-Za-z]{3,}){1,}", " ", text)
    text = re.sub(r"[A-Za-z]{10,}", " ", text)
    text = re.sub(r"របស់\s*([។៕!?])", r"\1", text)
    text = re.sub(r"\s+([។៕!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_subtitle_text(text: str) -> str:
    return normalize_khmer_text(text)


def _safe_khmer_break(text: str, approx: int) -> int:
    """Find a break index near approx that does not split Khmer clusters."""
    if approx >= len(text):
        return len(text)
    window = text[max(0, approx - 12) : min(len(text), approx + 12)]
    for mark in ("។", "៕", " ", "!", "?"):
        rel = window.rfind(mark)
        if rel >= 0:
            return max(0, approx - 12) + rel + 1
    i = approx
    while i > 0 and "\u17b6" <= text[i] <= "\u17d3":
        i -= 1
    return max(1, i)


def wrap_two_lines(text: str, max_chars: int = 26) -> str:
    """Force at most 2 display lines (never split Khmer mid-syllable)."""
    text = normalize_khmer_text(text.replace("\n", " "))
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    words = text.split(" ")
    if len(words) == 1:
        mid = _safe_khmer_break(text, min(max_chars, max(8, len(text) // 2)))
        return text[:mid].rstrip() + "\n" + text[mid:].lstrip()

    line1: list[str] = []
    line2: list[str] = []
    size1 = 0
    for word in words:
        add = len(word) + (1 if line1 else 0)
        if not line1:
            line1.append(word)
            size1 = len(word)
            continue
        if not line2 and size1 + add <= max_chars:
            line1.append(word)
            size1 += add
        else:
            line2.append(word)

    l1 = " ".join(line1).strip()
    l2 = " ".join(line2).strip()
    if len(l2) > max_chars + 10:
        cut = _safe_khmer_break(l2, max_chars + 4)
        l2 = l2[:cut].rstrip()
    return f"{l1}\n{l2}" if l2 else l1


def _chunk_into_cues(text: str, max_chars: int = 26) -> list[str]:
    """Split long dialogue into successive cues, each max 2 lines."""
    text = normalize_khmer_text(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[។៕!?])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [text]

    cues: list[str] = []
    buf = ""
    limit = max_chars * 2 + 4
    for sent in sentences:
        candidate = f"{buf} {sent}".strip() if buf else sent
        if len(candidate) <= limit:
            buf = candidate
            continue
        if buf:
            cues.append(wrap_two_lines(buf, max_chars=max_chars))
        if len(sent) <= limit:
            buf = sent
        else:
            i = 0
            while i < len(sent):
                end = _safe_khmer_break(sent, min(len(sent), i + limit))
                if end <= i:
                    end = min(len(sent), i + limit)
                cues.append(wrap_two_lines(sent[i:end], max_chars=max_chars))
                i = end
            buf = ""
    if buf:
        cues.append(wrap_two_lines(buf, max_chars=max_chars))
    return [c for c in cues if c]


def subtitle_max_chars(font_size: int = 64) -> int:
    size = max(40, min(int(font_size), 78))
    if size <= 56:
        return 30
    if size <= 68:
        return 26
    return 22


def build_timed_cues(
    transcript: Transcript,
    *,
    max_chars: int = 26,
) -> list[Cue]:
    """Create non-overlapping 1–2 line cues that follow speech timing."""
    raw: list[Cue] = []
    for seg in transcript.segments:
        text = normalize_khmer_text(seg.text or "")
        if not text:
            continue
        start = float(seg.start)
        end = max(float(seg.end), start + 0.45)
        parts = _chunk_into_cues(text, max_chars=max_chars)
        if not parts:
            continue
        if len(parts) == 1:
            raw.append((start, end, parts[0]))
            continue
        span = end - start
        slot = span / len(parts)
        for i, part in enumerate(parts):
            s = start + i * slot
            e = start + (i + 1) * slot
            raw.append((s, max(e, s + 0.35), part))

    if not raw:
        return []

    raw.sort(key=lambda x: x[0])
    fixed: list[Cue] = []
    for start, end, text in raw:
        if fixed and start < fixed[-1][1]:
            prev_s, prev_e, prev_t = fixed[-1]
            trimmed = max(prev_s + 0.25, start - 0.03)
            fixed[-1] = (prev_s, trimmed, prev_t)
            start = max(start, trimmed + 0.02)
            if end <= start:
                end = start + 0.35
        fixed.append((start, end, text))
    return fixed


def speak_text_from_cue(text: str) -> str:
    """Exact spoken string for a subtitle cue (same words, one line)."""
    return normalize_khmer_text((text or "").replace("\n", " "))


def write_srt_cues(cues: list[Cue], srt_path: Path) -> Path:
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, (start, end, text) in enumerate(cues, start=1):
        lines.append(str(index))
        lines.append(
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}"
        )
        lines.append(text)
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def write_ass_cues(
    cues: list[Cue],
    ass_path: Path,
    *,
    font_name: str | None = None,
    font_size: int = 64,
) -> Path:
    """Cinema Khmer look, bottom-center, max 2 lines per cue."""
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_battambang_fonts()
    font = font_name or BATTAMBANG_FAMILY
    size = max(40, min(int(font_size), 78))
    outline_w = round(2.8 + size / 32, 1)
    shadow_w = round(1.0 + size / 100, 1)
    margin_v = max(48, int(size * 0.7))

    primary = "&H000000FF"
    secondary = "&H000000FF"
    outline = "&H00FFFFFF"
    back = "&H64000000"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{primary},{secondary},{outline},{back},-1,0,0,0,100,100,0,0,1,{outline_w},{shadow_w},2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    for start, end, text in cues:
        events.append(
            "Dialogue: 0,"
            f"{_format_ass_timestamp(start)},{_format_ass_timestamp(end)},"
            f"Default,,0,0,0,,{_escape_ass_text(text)}"
        )

    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8-sig")
    return ass_path


def write_srt(transcript: Transcript, srt_path: Path) -> Path:
    return write_srt_cues(build_timed_cues(transcript), srt_path)


def write_ass(
    transcript: Transcript,
    ass_path: Path,
    *,
    font_name: str | None = None,
    font_size: int = 64,
) -> Path:
    max_chars = subtitle_max_chars(font_size)
    cues = build_timed_cues(transcript, max_chars=max_chars)
    return write_ass_cues(
        cues, ass_path, font_name=font_name, font_size=font_size
    )
