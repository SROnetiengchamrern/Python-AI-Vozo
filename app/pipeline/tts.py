"""Khmer text-to-speech via Microsoft Edge neural voices (free, no API key)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts

# Voices shown in the GUI dropdown.
VOICE_CHOICES = {
    "Khmer Female": "km-KH-SreymomNeural",
    "Khmer Male": "km-KH-PisethNeural",
}


async def _synthesize(text: str, voice: str, output_path: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_path))


def synthesize_speech(
    text: str,
    output_path: Path,
    *,
    voice_label: str = "Khmer Female",
) -> Path:
    text = (text or "").strip()
    if not text:
        raise ValueError("No text available for TTS.")

    voice = VOICE_CHOICES.get(voice_label, VOICE_CHOICES["Khmer Female"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_synthesize(text, voice, output_path))
    return output_path


async def _synthesize_segmented(
    segments: list[tuple[float, float, str]],
    voice: str,
    work_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for index, (_start, _end, text) in enumerate(segments):
        clip = work_dir / f"seg_{index:04d}.mp3"
        if text.strip():
            communicate = edge_tts.Communicate(text=text.strip(), voice=voice)
            await communicate.save(str(clip))
        paths.append(clip)
    return paths


def synthesize_timed_speech(
    segments: list[tuple[float, float, str]],
    output_path: Path,
    *,
    voice_label: str = "Khmer Female",
    work_dir: Path,
) -> Path:
    """
    Build a rough timeline audio track for Mode 3 (full AI dubbing).

    Each segment is synthesized, then placed around its original start time.
    Duration is not perfectly lip-synced; this is the beginner-friendly approach.
    """
    from pydub import AudioSegment

    voice = VOICE_CHOICES.get(voice_label, VOICE_CHOICES["Khmer Female"])
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_paths = asyncio.run(_synthesize_segmented(segments, voice, work_dir))

    if not segments:
        raise ValueError("No timed segments available for TTS.")

    total_ms = int(max(end for _, end, _ in segments) * 1000) + 1000
    timeline = AudioSegment.silent(duration=max(total_ms, 1000))

    for (start, _end, text), clip_path in zip(segments, clip_paths):
        if not text.strip() or not clip_path.exists():
            continue
        clip = AudioSegment.from_file(clip_path)
        position = max(0, int(start * 1000))
        timeline = timeline.overlay(clip, position=position)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer WAV for reliable FFmpeg remux.
    if output_path.suffix.lower() != ".wav":
        output_path = output_path.with_suffix(".wav")
    timeline.export(str(output_path), format="wav")
    return output_path
