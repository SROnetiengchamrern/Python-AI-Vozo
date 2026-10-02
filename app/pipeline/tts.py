"""Khmer text-to-speech via Microsoft Edge neural voices (free, no API key)."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import edge_tts
from edge_tts.exceptions import NoAudioReceived

logger = logging.getLogger(__name__)

VOICE_CHOICES = {
    "Khmer Female": "km-KH-SreymomNeural",
    "Khmer Female 1": "km-KH-SreymomNeural",
    "Khmer Female 2": "km-KH-SreymomNeural",
    "Khmer Male": "km-KH-PisethNeural",
    "Khmer Male 1": "km-KH-PisethNeural",
    "Khmer Male 2": "km-KH-PisethNeural",
}

KHMER_VOICE_OPTIONS = [
    "Khmer Female 1",
    "Khmer Female 2",
    "Khmer Male 1",
    "Khmer Male 2",
]

_SPEAKABLE_RE = re.compile(r"[A-Za-z0-9\u1780-\u17B3\u17B6-\u17C8]")


def resolve_voice_id(voice_label: str) -> str:
    return VOICE_CHOICES.get(voice_label, VOICE_CHOICES["Khmer Female 1"])


def _sanitize_tts_text(text: str) -> str:
    from .numbers_km import khmerize_numbers_in_text

    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text or not _SPEAKABLE_RE.search(text):
        return ""
    text = khmerize_numbers_in_text(text)
    text = text.replace("&", " and ").replace("<", " ").replace(">", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 800:
        text = text[:800].rsplit(" ", 1)[0] or text[:800]
    return text


async def _save_with_retry(
    text: str,
    voice: str,
    output_path: Path,
    *,
    retries: int = 4,
) -> bool:
    delay = 0.8
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice)
            await communicate.save(str(output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                return True
            last_err = NoAudioReceived("empty audio file")
        except NoAudioReceived as err:
            last_err = err
        except Exception as err:  # noqa: BLE001
            last_err = err
        await asyncio.sleep(delay)
        delay = min(delay * 1.8, 8.0)
    logger.warning("TTS skipped after retries: %s (%s)", text[:60], last_err)
    return False


async def _synthesize(text: str, voice: str, output_path: Path) -> None:
    text = _sanitize_tts_text(text)
    if not text:
        raise ValueError("No speakable text available for TTS.")
    ok = await _save_with_retry(text, voice, output_path)
    if not ok:
        raise RuntimeError(
            "Edge TTS returned no audio. Check internet / try again.\n"
            f"Text: {text[:120]}"
        )


def synthesize_speech(
    text: str,
    output_path: Path,
    *,
    voice_label: str = "Khmer Female 1",
) -> Path:
    text = _sanitize_tts_text(text)
    if not text:
        raise ValueError("No text available for TTS.")
    voice = resolve_voice_id(voice_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_synthesize(text, voice, output_path))
    return output_path


async def _synthesize_segmented(
    segments: list[tuple[float, float, str, str]],
    work_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for index, (_start, _end, text, voice_label) in enumerate(segments):
        clip = work_dir / f"seg_{index:04d}.mp3"
        clean = _sanitize_tts_text(text)
        if clean:
            voice = resolve_voice_id(voice_label)
            await _save_with_retry(clean, voice, clip)
            await asyncio.sleep(0.2)
        paths.append(clip)
    return paths


def _fit_clip_to_duration(clip: "AudioSegment", target_ms: int) -> "AudioSegment":
    """Speed-change clip to roughly fit the original speech window."""
    from pydub import AudioSegment

    if target_ms <= 120 or len(clip) <= 0:
        return clip
    # Allow slight overrun so speech doesn't sound rushed.
    soft_target = int(target_ms * 1.08)
    if len(clip) <= soft_target:
        return clip

    # pydub speed change via frame rate (then reset).
    ratio = len(clip) / float(soft_target)
    # Clamp extreme speeds.
    ratio = max(0.85, min(ratio, 1.65))
    new_rate = int(clip.frame_rate * ratio)
    sped = clip._spawn(clip.raw_data, overrides={"frame_rate": new_rate})
    sped = sped.set_frame_rate(clip.frame_rate)
    if len(sped) > soft_target + 80:
        sped = sped[:soft_target]
    return sped


def synthesize_timed_speech(
    segments: list[tuple[float, float, str]] | list[tuple[float, float, str, str]],
    output_path: Path,
    *,
    voice_label: str = "Khmer Female 1",
    work_dir: Path,
    total_duration_sec: float | None = None,
) -> Path:
    """
    Build a timeline audio track aligned to original speech windows.

    Each clip is speed-fitted into its [start, end] window so Khmer voice
    lands on the right lines instead of stacking / drifting.
    """
    from pydub import AudioSegment

    normalized: list[tuple[float, float, str, str]] = []
    for item in segments:
        if len(item) == 4:
            start, end, text, vlabel = item  # type: ignore[misc]
            normalized.append((float(start), float(end), text, vlabel or voice_label))
        else:
            start, end, text = item  # type: ignore[misc]
            normalized.append((float(start), float(end), text, voice_label))

    if not normalized:
        raise ValueError("No timed segments available for TTS.")

    work_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = asyncio.run(_synthesize_segmented(normalized, work_dir))

    last_end = max(end for _, end, _, _ in normalized)
    if total_duration_sec and total_duration_sec > last_end:
        total_ms = int(total_duration_sec * 1000) + 200
    else:
        total_ms = int(last_end * 1000) + 1000

    timeline = AudioSegment.silent(duration=max(total_ms, 1000), frame_rate=44100)
    placed = 0
    cursor_guard = -1

    for (start, end, text, _v), clip_path in zip(normalized, clip_paths):
        if not _sanitize_tts_text(text):
            continue
        if not clip_path.exists() or clip_path.stat().st_size <= 0:
            continue
        try:
            clip = AudioSegment.from_file(clip_path)
        except Exception:  # noqa: BLE001
            continue

        target_ms = max(180, int((end - start) * 1000))
        clip = _fit_clip_to_duration(clip, target_ms)
        position = max(0, int(start * 1000))
        # Avoid heavy overlap from previous overrun.
        if position < cursor_guard:
            position = cursor_guard
        timeline = timeline.overlay(clip, position=position)
        cursor_guard = position + len(clip) - 30
        placed += 1

    if placed == 0:
        raise RuntimeError(
            "Edge TTS produced no usable audio clips.\n"
            "Check internet connection and try KH Translate & Dub again."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".wav":
        output_path = output_path.with_suffix(".wav")
    timeline.export(str(output_path), format="wav")
    return output_path
