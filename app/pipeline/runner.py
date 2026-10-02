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
    replace_audio,
    require_ffmpeg,
)
from .subtitles import write_ass, write_srt
from .transcribe import transcribe_audio
from .translate import translate_transcript
from .tts import synthesize_speech, synthesize_timed_speech

ProgressCallback = Callable[[float, str], None]


@dataclass
class ConversionOptions:
    video_path: Path
    mode: str  # "subtitle" | "voice" | "full"
    language: str = "Khmer"
    voice: str = "Khmer Female"
    translate_speech: bool = True
    generate_voice: bool = True
    replace_audio: bool = True
    burn_in_subtitles: bool = True
    whisper_model: str = "base"
    subtitle_font_size: int = 110
    output_dir: Path | None = None


@dataclass
class ConversionResult:
    output_video: Path
    srt_path: Path | None
    transcript_path: Path | None
    khmer_text_path: Path | None


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

    work = Path(tempfile.mkdtemp(prefix="vozo_khmer_"))
    try:
        report(5, "Extracting audio...")
        wav_path = work / "audio.wav"
        extract_audio(video_path, wav_path)

        report(20, "Transcribing speech (Whisper)...")
        transcript = transcribe_audio(wav_path, model_size=options.whisper_model)
        transcript_path = out_dir / f"{stem}_original.txt"
        transcript_path.write_text(transcript.text, encoding="utf-8")

        khmer = transcript
        khmer_text_path: Path | None = None
        if options.translate_speech:
            report(40, "Translating to Khmer...")
            khmer = translate_transcript(transcript, target="km")
            khmer_text_path = out_dir / f"{stem}_khmer.txt"
            khmer_text_path.write_text(khmer.text, encoding="utf-8")

        report(55, "Writing Khmer subtitles...")
        srt_path = out_dir / f"{stem}_khmer.srt"
        ass_path = out_dir / f"{stem}_khmer.ass"
        write_srt(khmer, srt_path)
        write_ass(khmer, ass_path, font_size=options.subtitle_font_size)

        mode = options.mode.lower()
        output_video: Path

        def _maybe_burn(src: Path, dest: Path) -> Path:
            if options.burn_in_subtitles:
                burn_subtitles(src, ass_path, dest)
                return dest
            if src != dest:
                shutil.copy2(src, dest)
            return dest

        if mode == "subtitle":
            report(70, "Burning Khmer subtitles into video...")
            output_video = out_dir / f"{stem}_khmer_subtitle.mp4"
            if options.burn_in_subtitles:
                _maybe_burn(video_path, output_video)
            else:
                output_video = out_dir / f"{stem}_original_copy{video_path.suffix}"
                shutil.copy2(video_path, output_video)

        elif mode == "voice":
            if not options.generate_voice:
                raise ValueError("Mode 2 requires Generate Khmer voice.")
            report(65, "Generating Khmer voice...")
            voice_audio = work / "khmer_voice.mp3"
            synthesize_speech(khmer.text, voice_audio, voice_label=options.voice)

            report(80, "Replacing original audio...")
            voiced = work / "voiced.mp4"
            if options.replace_audio:
                replace_audio(video_path, voice_audio, voiced)
            else:
                mix_audio_with_background(video_path, voice_audio, voiced)

            report(90, "Adding Khmer subtitles...")
            output_video = out_dir / f"{stem}_khmer_voice.mp4"
            _maybe_burn(voiced, output_video)

        elif mode == "full":
            report(60, "Generating timed Khmer speech...")
            timed = [(s.start, s.end, s.text) for s in khmer.segments]
            voice_audio = work / "khmer_timed.wav"
            synthesize_timed_speech(
                timed,
                voice_audio,
                voice_label=options.voice,
                work_dir=work / "segments",
            )

            report(80, "Mixing voice with background...")
            dubbed = work / "dubbed.mp4"
            if options.replace_audio:
                replace_audio(video_path, voice_audio, dubbed)
            else:
                mix_audio_with_background(video_path, voice_audio, dubbed)

            report(90, "Adding Khmer subtitles...")
            output_video = out_dir / f"{stem}_khmer_dubbed.mp4"
            _maybe_burn(dubbed, output_video)
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
