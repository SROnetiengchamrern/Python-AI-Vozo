"""Orchestrates Mode 1 / Mode 2 / Mode 3 conversion pipelines."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .ffmpeg_utils import (
    burn_subtitles,
    extract_audio,
    mix_audio_with_background,
    probe_duration_seconds,
    replace_audio,
    require_ffmpeg,
)
from .numbers_km import khmerize_numbers_in_text
from .subtitles import (
    build_timed_cues,
    speak_text_from_cue,
    subtitle_max_chars,
    wrap_two_lines,
    write_ass_cues,
    write_srt_cues,
)
from .transcribe import Transcript, TranscriptSegment, transcribe_audio
from .translate import translate_transcript
from .tts import synthesize_timed_speech

ProgressCallback = Callable[[float, str], None]


@dataclass
class ConversionOptions:
    video_path: Path
    mode: str  # "subtitle" | "voice" | "full"
    language: str = "Khmer"
    voice: str = "Khmer Female 1"
    translate_speech: bool = True
    generate_voice: bool = True
    replace_audio: bool = True
    burn_in_subtitles: bool = True
    keep_background: bool = False
    whisper_model: str = "base"
    subtitle_font_size: int = 64
    voice_style: str = "Drama"
    output_dir: Path | None = None
    speaker_voices: dict[str, str] | None = None
    analyzed_segments: list | None = None
    pre_transcript: object | None = None


@dataclass
class ConversionResult:
    output_video: Path
    srt_path: Path | None
    transcript_path: Path | None
    khmer_text_path: Path | None


def _voice_for_time(
    start: float,
    end: float,
    *,
    default_voice: str,
    speaker_voices: dict[str, str] | None,
    analyzed_segments: list | None,
) -> str:
    """Pick Khmer voice by overlapping Analyze speaker segment."""
    if not speaker_voices or not analyzed_segments:
        return default_voice
    mid = (start + end) / 2.0
    best_sid = None
    best_overlap = 0.0
    for seg in analyzed_segments:
        s = float(getattr(seg, "start", 0.0))
        e = float(getattr(seg, "end", 0.0))
        overlap = max(0.0, min(end, e) - max(start, s))
        if overlap > best_overlap:
            best_overlap = overlap
            best_sid = getattr(seg, "speaker_id", None)
        elif best_sid is None and s <= mid <= e:
            best_sid = getattr(seg, "speaker_id", None)
    if best_sid and best_sid in speaker_voices:
        return speaker_voices[best_sid]
    return default_voice


def _khmerize_transcript_numbers(transcript: Transcript) -> Transcript:
    segs = [
        TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=khmerize_numbers_in_text(seg.text or ""),
        )
        for seg in transcript.segments
    ]
    full = khmerize_numbers_in_text(transcript.text or "")
    if not full:
        full = " ".join(s.text for s in segs if s.text).strip()
    return Transcript(language=transcript.language, text=full, segments=segs)


def run_conversion(
    options: ConversionOptions,
    progress: ProgressCallback | None = None,
) -> ConversionResult:
    def report(pct: float, status: str) -> None:
        if progress:
            progress(pct, status)

    video_path = options.video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    require_ffmpeg()

    out_dir = (options.output_dir or video_path.parent / "khmer_output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    video_duration = probe_duration_seconds(video_path)

    work = Path(tempfile.mkdtemp(prefix="vozo_khmer_"))
    try:
        report(5, "Extracting audio...")
        wav_path = work / "audio.wav"
        extract_audio(video_path, wav_path)

        if options.pre_transcript is not None:
            report(20, "Using Analyze transcript...")
            transcript = options.pre_transcript  # type: ignore[assignment]
        else:
            report(20, "Transcribing speech (Whisper)...")
            transcript = transcribe_audio(wav_path, model_size=options.whisper_model)
        transcript_path = out_dir / f"{stem}_original.txt"
        transcript_path.write_text(transcript.text, encoding="utf-8")

        khmer = transcript
        khmer_text_path: Path | None = None
        if options.translate_speech:
            report(40, "Translating to Khmer...")
            src_lang = getattr(transcript, "language", "auto") or "auto"
            khmer = translate_transcript(transcript, source=src_lang, target="km")

        # Speak/show money-style numbers in Khmer units (៥មុឺន, ១លាន, …).
        khmer = _khmerize_transcript_numbers(khmer)
        khmer_text_path = out_dir / f"{stem}_khmer.txt"
        khmer_text_path.write_text(khmer.text, encoding="utf-8")

        # One cue list drives BOTH speech and subtitles (same words/timing).
        max_chars = subtitle_max_chars(options.subtitle_font_size)
        cues = build_timed_cues(khmer, max_chars=max_chars)
        srt_path = out_dir / f"{stem}_khmer.srt"
        ass_path = out_dir / f"{stem}_khmer.ass"

        mode = options.mode.lower()
        output_video: Path

        def _write_subs(final_cues: list) -> None:
            report(88, "Writing Khmer subtitles (matched to voice)...")
            write_srt_cues(final_cues, srt_path)
            write_ass_cues(
                final_cues,
                ass_path,
                font_size=options.subtitle_font_size,
            )

        def _maybe_burn(src: Path, dest: Path) -> Path:
            if options.burn_in_subtitles:
                burn_subtitles(src, ass_path, dest)
                return dest
            if src != dest:
                shutil.copy2(src, dest)
            return dest

        def _build_timed_voice(status_pct: float) -> tuple[Path, list]:
            report(status_pct, "Generating timed Khmer voice...")
            timed = []
            for start, end, text in cues:
                spoken = speak_text_from_cue(text)
                if not spoken:
                    continue
                vlabel = _voice_for_time(
                    start,
                    end,
                    default_voice=options.voice,
                    speaker_voices=options.speaker_voices,
                    analyzed_segments=options.analyzed_segments,
                )
                timed.append((start, end, spoken, vlabel))
            voice_audio = work / "khmer_timed.wav"
            audio_path, placements = synthesize_timed_speech(
                timed,
                voice_audio,
                voice_label=options.voice,
                work_dir=work / "segments",
                total_duration_sec=video_duration or None,
                style=options.voice_style,
            )
            # Subtitles show exactly what was spoken, at when it was spoken.
            matched = [
                (s, e, wrap_two_lines(t, max_chars=max_chars))
                for s, e, t in placements
            ]
            return audio_path, matched

        def _apply_khmer_audio(voice_audio: Path, dest_video: Path) -> Path:
            report(82, "Clearing original voice / applying Khmer audio...")
            speech_times = [(s.start, s.end) for s in khmer.segments]
            # Default path: hard replace (no original dialogue).
            if options.keep_background:
                mix_audio_with_background(
                    video_path,
                    voice_audio,
                    dest_video,
                    speech_segments=speech_times,
                    duck_original_speech=True,
                    background_volume=0.06,
                    voice_volume=1.45,
                )
            else:
                replace_audio(video_path, voice_audio, dest_video)
            return dest_video

        if mode == "subtitle":
            _write_subs(cues)
            report(70, "Burning Khmer subtitles into video...")
            output_video = out_dir / f"{stem}_khmer_subtitle.mp4"
            if options.burn_in_subtitles:
                _maybe_burn(video_path, output_video)
            else:
                output_video = out_dir / f"{stem}_original_copy{video_path.suffix}"
                shutil.copy2(video_path, output_video)

        elif mode in ("voice", "full"):
            if not options.generate_voice and mode == "voice":
                raise ValueError("Voice mode requires Generate Khmer speech.")
            if options.generate_voice:
                voice_audio, matched_cues = _build_timed_voice(
                    65 if mode == "voice" else 60
                )
                _write_subs(matched_cues or cues)
                staged = work / "dubbed_stage.mp4"
                _apply_khmer_audio(voice_audio, staged)
            else:
                _write_subs(cues)
                staged = video_path

            report(90, "Adding Khmer subtitles...")
            suffix = "voice" if mode == "voice" else "dubbed"
            output_video = out_dir / f"{stem}_khmer_{suffix}.mp4"
            _maybe_burn(staged, output_video)
        else:
            raise ValueError(f"Unknown mode: {options.mode}")

        report(100, "Done!")
        return ConversionResult(
            output_video=output_video,
            srt_path=srt_path,
            transcript_path=transcript_path,
            khmer_text_path=khmer_text_path,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
