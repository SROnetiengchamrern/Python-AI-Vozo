"""Speech-to-text via faster-whisper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    language: str
    text: str
    segments: list[TranscriptSegment]


_model = None
_model_name: str | None = None


def get_model(model_size: str = "base"):
    """Lazy-load Whisper so the GUI can start quickly."""
    global _model, _model_name
    if _model is None or _model_name != model_size:
        from faster_whisper import WhisperModel

        # CPU-friendly defaults for beginners.
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _model_name = model_size
    return _model


def transcribe_audio(
    audio_path: Path,
    *,
    model_size: str = "base",
    language: str | None = None,
) -> Transcript:
    model = get_model(model_size)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    segments: list[TranscriptSegment] = []
    parts: list[str] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(start=float(seg.start), end=float(seg.end), text=text)
        )
        parts.append(text)

    return Transcript(
        language=info.language or (language or "unknown"),
        text=" ".join(parts).strip(),
        segments=segments,
    )
