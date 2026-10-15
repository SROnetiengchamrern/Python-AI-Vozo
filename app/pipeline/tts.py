"""Khmer text-to-speech via Microsoft Edge neural voices."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import subprocess
from pathlib import Path

import edge_tts
from edge_tts.exceptions import NoAudioReceived

from .subtitles import normalize_khmer_text

logger = logging.getLogger(__name__)

# voice_label -> (edge_voice, rate, pitch)
# Female voices: slightly slower + clearer (Sreymom reads muddy when rushed).
VOICE_CHOICES = {
    "Khmer Female": ("km-KH-SreymomNeural", "-8%", "+0Hz"),
    "Khmer Female 1": ("km-KH-SreymomNeural", "-8%", "+0Hz"),
    "Khmer Female 2": ("km-KH-SreymomNeural", "-4%", "+2Hz"),
    "Khmer Male": ("km-KH-PisethNeural", "+0%", "+0Hz"),
    "Khmer Male 1": ("km-KH-PisethNeural", "+0%", "+0Hz"),
    "Khmer Male 2": ("km-KH-PisethNeural", "-4%", "-2Hz"),
}

VOICE_STYLES = {
    "Natural": ("+0%", "+0Hz"),
    "Drama": ("+0%", "+0Hz"),
    "Soft": ("-6%", "+1Hz"),
    "Energetic": ("+8%", "+1Hz"),
}

KHMER_VOICE_OPTIONS = [
    "Khmer Female 1",
    "Khmer Female 2",
    "Khmer Male 1",
    "Khmer Male 2",
]

# Mild speed-up only. Long videos lock each line to its cue window (no drift stack).
_MAX_SPEEDUP = 1.28
_MIN_SLOWDOWN = 0.94
# Slightly conservative — female Natural (-8%) reads slower than raw Edge default.
_CHARS_PER_SEC = 10.0
_SPEAKABLE_RE = re.compile(r"[\u1780-\u17B3\u17B6-\u17C8A-Za-z0-9]")


def resolve_voice(voice_label: str, style: str = "Drama") -> tuple[str, str, str]:
    voice, base_rate, base_pitch = VOICE_CHOICES.get(
        voice_label, VOICE_CHOICES["Khmer Female 1"]
    )
    style_rate, style_pitch = VOICE_STYLES.get(style, VOICE_STYLES["Drama"])
    female = "Female" in voice_label

    if style == "Natural":
        return voice, base_rate, base_pitch

    # Female clarity first: keep slower base; never rush women for "Drama".
    if female:
        if style == "Soft":
            return voice, "-12%", "+1Hz"
        if style == "Energetic":
            return voice, "-2%", "+2Hz"
        # Drama / default
        if "Female 2" in voice_label:
            return voice, "-4%", "+2Hz"
        return voice, "-8%", "+0Hz"

    if style == "Drama" and "Male 2" in voice_label:
        return voice, "-2%", "-2Hz"
    if style_rate != "+0%" or style_pitch != "+0Hz":
        return voice, style_rate, style_pitch
    return voice, base_rate, base_pitch


def resolve_voice_id(voice_label: str) -> str:
    return resolve_voice(voice_label)[0]


def _sanitize_tts_text(text: str) -> str:
    """Same words as subtitle cue — only light TTS-safe cleanup."""
    from .numbers_km import khmerize_numbers_in_text

    text = normalize_khmer_text((text or "").replace("\n", " "))
    if not text:
        return ""
    text = khmerize_numbers_in_text(text)
    text = text.replace("&", " ").replace("<", " ").replace(">", " ")
    text = re.sub(r"[#@*_~`|\\/]+", " ", text)
    text = re.sub(r"\s*([។៕!?])\s*", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not _SPEAKABLE_RE.search(text):
        return ""
    return text


def _budget_text_to_window(text: str, available_ms: int) -> str:
    """
    Shorten Khmer so it can be spoken inside the cue window.

    Keeps voice/title/picture aligned on long videos (no stacked delay).
    Subtitles must use this same shortened string.
    """
    text = _sanitize_tts_text(text)
    if not text:
        return ""
    # Allow mild speed-up headroom in the budget.
    max_chars = max(18, int((available_ms / 1000.0) * _CHARS_PER_SEC * _MAX_SPEEDUP))
    if len(text) <= max_chars:
        return text
    # Prefer a break in the last ~45% of the window (don't stop at the first ។).
    lo = max(10, int(max_chars * 0.55))
    cut = -1
    for mark in ("។", "៕", "!", "?"):
        pos = text.rfind(mark, lo, max_chars + 1)
        if pos > cut:
            cut = pos
    if cut >= lo:
        return text[: cut + 1].strip()
    space = text.rfind(" ", lo, max_chars + 1)
    if space >= lo:
        return text[:space].strip()
    cut = max_chars
    while cut > 10 and cut <= len(text) and "\u17b6" <= text[cut - 1] <= "\u17d3":
        cut -= 1
    return text[:cut].strip()


_last_tts_error: str | None = None


def _format_tts_error(err: Exception | None) -> str:
    if err is None:
        return "unknown error"
    msg = f"{type(err).__name__}: {err}"
    low = msg.lower()
    if "403" in msg or "handshake" in low:
        return (
            "Microsoft Edge TTS blocked the request (403).\n"
            "Update edge-tts:  pip install -U \"edge-tts>=7.3.1\"\n"
            "Then restart the app and try again."
        )
    if "getaddrinfo" in low or "cannot connect" in low or "network" in low:
        return "No internet / cannot reach Edge TTS. Check connection and retry."
    return msg


async def _save_with_retry(
    text: str,
    voice: str,
    output_path: Path,
    *,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    retries: int = 5,
) -> bool:
    global _last_tts_error
    delay = 0.8
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(
                text=text, voice=voice, rate=rate, pitch=pitch
            )
            await communicate.save(str(output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                _last_tts_error = None
                return True
            last_err = NoAudioReceived("empty audio file")
        except NoAudioReceived as err:
            last_err = err
        except Exception as err:  # noqa: BLE001
            last_err = err
            # 403 / handshake: wait longer — token/clock may refresh.
            if "403" in str(err) or "Handshake" in type(err).__name__:
                delay = max(delay, 2.0 + attempt)
        await asyncio.sleep(delay)
        delay = min(delay * 1.6, 8.0)
    _last_tts_error = _format_tts_error(last_err)
    logger.warning("TTS skipped: %s (%s)", text[:50], last_err)
    return False


def synthesize_speech(
    text: str,
    output_path: Path,
    *,
    voice_label: str = "Khmer Female 1",
    style: str = "Drama",
) -> Path:
    text = _sanitize_tts_text(text)
    if not text:
        raise ValueError("No text available for TTS.")
    voice, rate, pitch = resolve_voice(voice_label, style)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async def _run() -> None:
        ok = await _save_with_retry(text, voice, output_path, rate=rate, pitch=pitch)
        if not ok:
            raise RuntimeError(f"Edge TTS failed for: {text[:120]}")

    asyncio.run(_run())
    return output_path


async def _synthesize_segmented(
    segments: list[tuple[float, float, str, str]],
    work_dir: Path,
    style: str,
) -> list[Path]:
    paths: list[Path] = []
    for index, (_start, _end, text, voice_label) in enumerate(segments):
        clip = work_dir / f"seg_{index:04d}.mp3"
        clean = _sanitize_tts_text(text)
        if clean:
            voice, rate, pitch = resolve_voice(voice_label, style)
            await _save_with_retry(clean, voice, clip, rate=rate, pitch=pitch)
            await asyncio.sleep(0.08)
        paths.append(clip)
    return paths


def _atempo_chain(ratio: float) -> str:
    ratio = max(0.5, min(ratio, 2.0))
    parts: list[str] = []
    while ratio > 2.0:
        parts.append("atempo=2.0")
        ratio /= 2.0
    while ratio < 0.5:
        parts.append("atempo=0.5")
        ratio /= 0.5
    parts.append(f"atempo={ratio:.4f}")
    return ",".join(parts)


def _speed_clip(clip_path: Path, ratio: float, work_dir: Path) -> "AudioSegment":
    from pydub import AudioSegment

    clip = AudioSegment.from_file(clip_path)
    if abs(ratio - 1.0) < 0.03:
        return clip

    ratio = max(_MIN_SLOWDOWN, min(ratio, _MAX_SPEEDUP))
    out = work_dir / f"spd_{clip_path.stem}_{ratio:.3f}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(clip_path),
        "-af",
        _atempo_chain(ratio),
        "-ar",
        "44100",
        "-ac",
        "2",
        str(out),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
        return AudioSegment.from_file(out)

    if ratio > 1.0:
        sped = clip._spawn(
            clip.raw_data, overrides={"frame_rate": int(clip.frame_rate * ratio)}
        )
        return sped.set_frame_rate(clip.frame_rate)
    return clip


def _fit_natural(
    clip_path: Path,
    *,
    soft_ms: int,
    hard_ms: int,
    work_dir: Path,
) -> "AudioSegment":
    from pydub import AudioSegment

    clip = AudioSegment.from_file(clip_path)
    if len(clip) <= 0:
        return clip

    soft_ms = max(220, soft_ms)
    hard_ms = max(soft_ms, hard_ms)
    cur = len(clip)

    if cur <= soft_ms + 40:
        return clip
    if cur <= hard_ms + 40:
        return clip

    need = cur / float(hard_ms)
    if need <= _MAX_SPEEDUP:
        return _speed_clip(clip_path, need, work_dir)
    return _speed_clip(clip_path, _MAX_SPEEDUP, work_dir)


def _clarity_filter(*, female: bool) -> str:
    """Make dialogue easier to hear (women especially benefit from presence EQ)."""
    if female:
        return (
            "highpass=f=90,"
            "equalizer=f=2500:t=q:w=1.1:g=3.5,"
            "equalizer=f=4500:t=q:w=1.0:g=2.0,"
            "acompressor=threshold=-18dB:ratio=2.5:attack=8:release=80,"
            "loudnorm=I=-14:TP=-1.5:LRA=8,"
            "alimiter=limit=0.95"
        )
    return (
        "highpass=f=80,"
        "equalizer=f=2200:t=q:w=1.0:g=2.0,"
        "acompressor=threshold=-18dB:ratio=2.2:attack=10:release=90,"
        "loudnorm=I=-14:TP=-1.5:LRA=9,"
        "alimiter=limit=0.95"
    )


def synthesize_timed_speech(
    segments: list[tuple[float, float, str]] | list[tuple[float, float, str, str]],
    output_path: Path,
    *,
    voice_label: str = "Khmer Female 1",
    work_dir: Path,
    total_duration_sec: float | None = None,
    style: str = "Drama",
) -> tuple[Path, list[tuple[float, float, str]]]:
    """
    Build timed Khmer dub locked to picture cue times.

    Each line starts at its cue time and must finish before the next cue.
    Text is shortened to fit when needed so spoken words == on-screen title.
    This prevents long-video drift (voice minutes behind picture).
    """
    from pydub import AudioSegment

    raw_items: list[tuple[float, float, str, str]] = []
    for item in segments:
        if len(item) == 4:
            start, end, text, vlabel = item  # type: ignore[misc]
            raw_items.append(
                (float(start), float(end), text, vlabel or voice_label)
            )
        else:
            start, end, text = item  # type: ignore[misc]
            raw_items.append((float(start), float(end), text, voice_label))

    if not raw_items:
        raise ValueError("No timed segments available for TTS.")

    raw_items.sort(key=lambda x: x[0])

    last_end = max(end for _, end, _, _ in raw_items)
    if total_duration_sec and total_duration_sec > last_end:
        total_ms = int(math.ceil(total_duration_sec * 1000)) + 250
    else:
        total_ms = int(math.ceil(last_end * 1000)) + 1000

    # Budget each cue into [start, next_start) BEFORE TTS so speech matches title.
    normalized: list[tuple[float, float, str, str]] = []
    for i, (start, end, text, vlabel) in enumerate(raw_items):
        start_ms = max(0, int(start * 1000))
        soft_ms = max(240, int((end - start) * 1000))
        if i + 1 < len(raw_items):
            next_start_ms = max(0, int(raw_items[i + 1][0] * 1000))
            # Use almost all time until the next line (picture lock).
            available_ms = max(240, next_start_ms - start_ms - 40)
        else:
            available_ms = max(soft_ms, total_ms - start_ms - 80)
        spoken = _budget_text_to_window(text, available_ms)
        if not spoken:
            continue
        # Keep end = start + available so fit uses the hard window.
        normalized.append(
            (start, start + available_ms / 1000.0, spoken, vlabel)
        )

    if not normalized:
        raise ValueError("No timed segments available for TTS.")

    work_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = work_dir / "fitted"
    fit_dir.mkdir(exist_ok=True)
    clip_paths = asyncio.run(_synthesize_segmented(normalized, work_dir, style))

    timeline = AudioSegment.silent(
        duration=max(total_ms, 1000), frame_rate=44100
    ).set_channels(2)
    placements: list[tuple[float, float, str]] = []
    used_female = any("Female" in v for *_, v in normalized)

    for i, ((start, end, text, _v), clip_path) in enumerate(
        zip(normalized, clip_paths)
    ):
        if not clip_path.exists() or clip_path.stat().st_size <= 0:
            continue

        start_ms = max(0, int(start * 1000))
        # Hard window already baked into end from budgeting.
        hard_ms = max(240, int((end - start) * 1000))
        soft_ms = hard_ms

        try:
            clip = _fit_natural(
                clip_path, soft_ms=soft_ms, hard_ms=hard_ms, work_dir=fit_dir
            )
        except Exception:  # noqa: BLE001
            try:
                clip = AudioSegment.from_file(clip_path)
            except Exception:
                continue

        if clip.channels == 1:
            clip = clip.set_channels(2)
        if clip.frame_rate != 44100:
            clip = clip.set_frame_rate(44100)

        # CRITICAL for long videos: always start on cue time — never stack delay.
        position = start_ms
        if len(clip) > hard_ms:
            clip = clip[:hard_ms]

        timeline = timeline.overlay(clip, position=max(0, position))
        end_ms = position + len(clip)
        placements.append((position / 1000.0, end_ms / 1000.0, text))

    if not placements:
        detail = _last_tts_error or "Check internet and try again."
        raise RuntimeError(
            "Edge TTS produced no usable audio clips.\n" + detail
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".wav":
        output_path = output_path.with_suffix(".wav")
    raw_path = work_dir / "timeline_raw.wav"
    timeline.export(str(raw_path), format="wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        "-af",
        _clarity_filter(female=used_female),
        "-ar",
        "44100",
        "-ac",
        "2",
        str(output_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0 or not output_path.exists():
        timeline.export(str(output_path), format="wav")
    return output_path, placements
