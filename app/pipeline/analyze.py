"""Analyze video: language, speech turns, speakers, gender."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .ffmpeg_utils import extract_audio, require_ffmpeg
from .transcribe import Transcript, transcribe_audio

ProgressCallback = Callable[[float, str], None]

LANG_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "vi": "Vietnamese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "km": "Khmer",
    "hi": "Hindi",
    "id": "Indonesian",
    "ms": "Malay",
    "ru": "Russian",
}


@dataclass
class SpeakerInfo:
    speaker_id: str  # "00", "01", ...
    gender: str  # "Male" | "Female" | "Unknown"
    default_voice: str
    segment_count: int = 0
    duration_sec: float = 0.0
    median_pitch_hz: float = 0.0


@dataclass
class AnalyzedSegment:
    start: float
    end: float
    text: str
    speaker_id: str
    gender: str


@dataclass
class AnalysisResult:
    video_path: Path
    wav_path: Path
    language_code: str
    language_name: str
    transcript: Transcript
    speakers: list[SpeakerInfo] = field(default_factory=list)
    segments: list[AnalyzedSegment] = field(default_factory=list)


@dataclass
class AnalyzeOptions:
    video_path: Path
    whisper_model: str = "base"
    detect_speakers: bool = True
    detect_gender: bool = True
    max_speakers: int = 3
    work_dir: Path | None = None


def analyze_video(
    options: AnalyzeOptions,
    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    def report(pct: float, status: str) -> None:
        if progress:
            progress(pct, status)

    video_path = options.video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    require_ffmpeg()
    own_work = options.work_dir is None
    work = Path(options.work_dir) if options.work_dir else Path(
        tempfile.mkdtemp(prefix="vozo_analyze_")
    )
    work.mkdir(parents=True, exist_ok=True)

    try:
        report(5, "Extracting audio for analysis...")
        wav_path = work / "analyze_audio.wav"
        extract_audio(video_path, wav_path)

        report(25, "Transcribing + detecting language...")
        transcript = transcribe_audio(wav_path, model_size=options.whisper_model)
        lang_code = (transcript.language or "unknown").lower()
        lang_name = LANG_NAMES.get(lang_code, lang_code.title())

        if not transcript.segments:
            report(100, "No speech detected")
            speaker = SpeakerInfo(
                speaker_id="00",
                gender="Unknown",
                default_voice="Khmer Female 1",
            )
            return AnalysisResult(
                video_path=video_path,
                wav_path=wav_path,
                language_code=lang_code,
                language_name=lang_name,
                transcript=transcript,
                speakers=[speaker],
                segments=[],
            )

        report(55, "Estimating pitch per speech segment...")
        samples, sr = _load_wav_mono(wav_path)
        pitches = [
            _median_pitch(
                samples,
                sr,
                seg.start,
                seg.end,
            )
            for seg in transcript.segments
        ]

        report(75, "Detecting speakers...")
        if options.detect_speakers and len(transcript.segments) > 1:
            labels = _cluster_speakers(
                pitches,
                max_speakers=min(options.max_speakers, len(transcript.segments)),
            )
        else:
            labels = [0] * len(transcript.segments)

        # Remap to compact 00, 01, ...
        unique = sorted(set(labels))
        remap = {old: f"{i:02d}" for i, old in enumerate(unique)}
        speaker_ids = [remap[lab] for lab in labels]

        report(88, "Detecting male / female voices...")
        gender_by_speaker: dict[str, str] = {}
        pitch_by_speaker: dict[str, list[float]] = {sid: [] for sid in remap.values()}
        for sid, pitch in zip(speaker_ids, pitches):
            if pitch > 0:
                pitch_by_speaker[sid].append(pitch)

        for sid, values in pitch_by_speaker.items():
            if not options.detect_gender or not values:
                gender_by_speaker[sid] = "Unknown"
                continue
            med = float(np.median(values))
            # Rough adult speech threshold.
            gender_by_speaker[sid] = "Female" if med >= 165 else "Male"

        male_n = 0
        female_n = 0
        speakers: list[SpeakerInfo] = []
        for sid in sorted(set(speaker_ids)):
            gender = gender_by_speaker.get(sid, "Unknown")
            values = pitch_by_speaker.get(sid, [])
            med = float(np.median(values)) if values else 0.0
            idxs = [i for i, s in enumerate(speaker_ids) if s == sid]
            duration = sum(
                max(0.0, transcript.segments[i].end - transcript.segments[i].start)
                for i in idxs
            )
            if gender == "Male":
                male_n += 1
                voice = f"Khmer Male {min(male_n, 2)}"
            elif gender == "Female":
                female_n += 1
                voice = f"Khmer Female {min(female_n, 2)}"
            else:
                voice = "Khmer Female 1"
            speakers.append(
                SpeakerInfo(
                    speaker_id=sid,
                    gender=gender,
                    default_voice=voice,
                    segment_count=len(idxs),
                    duration_sec=round(duration, 2),
                    median_pitch_hz=round(med, 1),
                )
            )

        analyzed = [
            AnalyzedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker_id=sid,
                gender=gender_by_speaker.get(sid, "Unknown"),
            )
            for seg, sid in zip(transcript.segments, speaker_ids)
        ]

        report(100, f"Found {len(speakers)} speaker(s) · language: {lang_name}")
        return AnalysisResult(
            video_path=video_path,
            wav_path=wav_path,
            language_code=lang_code,
            language_name=lang_name,
            transcript=transcript,
            speakers=speakers,
            segments=analyzed,
        )
    finally:
        # Keep wav for optional reuse only when caller owns work_dir.
        if own_work:
            # Leave analyze cache under temp; gui holds result in memory.
            pass


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(n)

    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    data /= np.max(np.abs(data)) + 1e-9
    return data, sr


def _median_pitch(
    samples: np.ndarray,
    sr: int,
    start: float,
    end: float,
) -> float:
    i0 = max(0, int(start * sr))
    i1 = min(len(samples), int(end * sr))
    chunk = samples[i0:i1]
    if len(chunk) < sr // 20:
        return 0.0

    # Frame-wise autocorrelation peak (lightweight F0 estimate).
    frame = int(0.03 * sr)
    hop = int(0.015 * sr)
    f0s: list[float] = []
    min_lag = int(sr / 350)  # ~350 Hz
    max_lag = int(sr / 70)  # ~70 Hz
    for i in range(0, max(1, len(chunk) - frame), hop):
        win = chunk[i : i + frame]
        if len(win) < frame:
            break
        win = win - win.mean()
        if np.std(win) < 0.01:
            continue
        corr = np.correlate(win, win, mode="full")[frame - 1 :]
        if max_lag >= len(corr):
            continue
        segment = corr[min_lag:max_lag]
        if len(segment) == 0:
            continue
        lag = int(np.argmax(segment)) + min_lag
        if corr[lag] <= 0.3 * corr[0]:
            continue
        f0s.append(sr / lag)

    if not f0s:
        return 0.0
    return float(np.median(f0s))


def _cluster_speakers(pitches: list[float], *, max_speakers: int) -> list[int]:
    """Cluster segments by pitch (and index continuity for same-ish pitch)."""
    vals = np.array([p if p > 0 else np.nan for p in pitches], dtype=float)
    # Fill missing pitches with global median.
    med = float(np.nanmedian(vals)) if np.any(~np.isnan(vals)) else 150.0
    vals = np.where(np.isnan(vals), med, vals)

    k = max(1, min(max_speakers, len(vals)))
    if k == 1:
        return [0] * len(vals)

    # 1D k-means on pitch.
    centers = np.linspace(vals.min(), vals.max(), k)
    labels = np.zeros(len(vals), dtype=int)
    for _ in range(12):
        dists = np.abs(vals[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = vals[mask].mean()

    # Merge tiny clusters into nearest.
    for j in range(k):
        if int(np.sum(labels == j)) >= 1:
            continue
        # unused center
        pass

    # Smooth: if a single segment is surrounded by another speaker and pitch
    # is close, reassign (reduces flicker).
    out = labels.tolist()
    for i in range(1, len(out) - 1):
        if out[i - 1] == out[i + 1] != out[i]:
            if abs(vals[i] - vals[i - 1]) < 35:
                out[i] = out[i - 1]
    return out
