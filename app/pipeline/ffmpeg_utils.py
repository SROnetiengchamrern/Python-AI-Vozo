"""FFmpeg helpers for audio extraction and video export."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegError(
            "FFmpeg not found. Install it and ensure `ffmpeg` is on your PATH.\n"
            "Windows: https://www.gyan.dev/ffmpeg/builds/"
        )
    return path


def require_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        # Usually beside ffmpeg.
        ffmpeg = require_ffmpeg()
        candidate = Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix else "ffprobe")
        if candidate.exists():
            return str(candidate)
        raise FFmpegError("ffprobe not found on PATH.")
    return path


def run_ffmpeg(args: list[str], *, cwd: Path | None = None) -> None:
    ffmpeg = require_ffmpeg()
    cmd = [ffmpeg, "-y", *args]
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise FFmpegError(detail[-2000:] or "FFmpeg failed.")


def probe_duration_seconds(media_path: Path) -> float:
    """Return media duration in seconds (0 if unknown)."""
    ffprobe = require_ffprobe()
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return 0.0
    try:
        data = json.loads(result.stdout or "{}")
        return float(data.get("format", {}).get("duration") or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """Extract mono 16kHz WAV audio for speech recognition."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
    )
    return audio_path


def replace_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    voice_eq: bool = True,
) -> Path:
    """
    Replace video audio completely with a new track.

    Keeps full video length. Pads short audio with silence so original
    dialogue cannot leak back in. voice_eq boosts Khmer-only tracks;
    turn it off when the input is already a voice+BGM mix.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = probe_duration_seconds(video_path)
    if voice_eq:
        afilter = (
            "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            "highpass=f=90,equalizer=f=2800:t=q:w=1.0:g=2.5,"
            "loudnorm=I=-14:TP=-1.5:LRA=8,apad[a]"
        )
    else:
        afilter = (
            "[1:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
            "loudnorm=I=-16:TP=-1.5:LRA=11,apad[a]"
        )
    args = [
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-filter_complex",
        afilter,
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]
    if duration > 0:
        args.extend(["-t", f"{duration:.3f}"])
    else:
        args.append("-shortest")
    args.append(str(output_path))
    run_ffmpeg(args)
    return output_path


def _escape_filter_path(path: Path) -> str:
    """Escape a Windows path for FFmpeg filter arguments."""
    return str(path.resolve()).replace("\\", "/").replace(":", "\\:")


def burn_subtitles(video_path: Path, subtitle_path: Path, output_path: Path) -> Path:
    """Burn ASS/SRT subtitles onto video using Battambang from assets/fonts."""
    from .fonts import ensure_battambang_fonts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fonts_dir = ensure_battambang_fonts()
    escaped = _escape_filter_path(subtitle_path)
    fonts_escaped = _escape_filter_path(fonts_dir)

    if subtitle_path.suffix.lower() == ".ass":
        vf = f"ass='{escaped}':fontsdir='{fonts_escaped}'"
    else:
        vf = (
            f"subtitles='{escaped}':fontsdir='{fonts_escaped}':"
            "force_style='FontName=Battambang,Alignment=2,MarginV=150,"
            "Fontsize=64,PrimaryColour=&H000000FF,OutlineColour=&H00FFFFFF,"
            "BorderStyle=1,Outline=4,Shadow=1.5,Bold=1'"
        )
    # Re-encode audio as AAC copy from already-replaced track.
    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-c:a",
            "copy",
            str(output_path),
        ]
    )
    return output_path


def mix_audio_with_background(
    video_path: Path,
    voice_path: Path,
    output_path: Path,
    *,
    background_volume: float = 0.22,
    voice_volume: float = 1.35,
    speech_segments: list[tuple[float, float]] | None = None,
    duck_original_speech: bool = True,
) -> Path:
    """
    Keep original music/SFX under Khmer voice.

    Lowers center vocals, ducks (does not erase) dialogue windows so BGM
    still plays during speech.
    """
    from pydub import AudioSegment

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="vozo_mix_"))
    try:
        orig_wav = work / "original.wav"
        try:
            run_ffmpeg(
                [
                    "-i",
                    str(video_path),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "2",
                    str(orig_wav),
                ]
            )
        except FFmpegError:
            run_ffmpeg(
                [
                    "-i",
                    str(video_path),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "44100",
                    "-ac",
                    "1",
                    str(orig_wav),
                ]
            )

        bed_wav = work / "bed.wav"
        try:
            # Soften centered dialogue, keep music side energy, then set bed level.
            run_ffmpeg(
                [
                    "-i",
                    str(orig_wav),
                    "-af",
                    f"stereotools=mlev=0.25,volume={background_volume}",
                    str(bed_wav),
                ]
            )
        except FFmpegError:
            run_ffmpeg(
                [
                    "-i",
                    str(orig_wav),
                    "-af",
                    f"volume={background_volume}",
                    str(bed_wav),
                ]
            )

        bed = AudioSegment.from_file(bed_wav)
        if duck_original_speech and speech_segments:
            # Duck old voice windows, but keep music audible under Khmer.
            bed = _duck_speech_regions(bed, speech_segments, duck_gain_db=-14.0)

        voice = AudioSegment.from_file(voice_path)
        if voice_volume != 1.0:
            voice = voice + (20.0 * math.log10(max(voice_volume, 0.01)))

        duration = probe_duration_seconds(video_path)
        target_ms = int(duration * 1000) if duration > 0 else max(len(bed), len(voice))
        if len(bed) < target_ms:
            bed += AudioSegment.silent(duration=target_ms - len(bed))
        else:
            bed = bed[:target_ms]
        if len(voice) < target_ms:
            voice += AudioSegment.silent(duration=target_ms - len(voice))
        else:
            voice = voice[:target_ms]

        mixed = bed.overlay(voice)
        mixed_wav = work / "mixed.wav"
        mixed.export(str(mixed_wav), format="wav")
        # Do not apply voice-only EQ — that removes music.
        return replace_audio(video_path, mixed_wav, output_path, voice_eq=False)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _merge_speech_windows(
    segments: list[tuple[float, float]],
    *,
    pad: float = 0.12,
    merge_gap: float = 0.45,
) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for start, end in segments:
        if end <= start:
            continue
        cleaned.append((max(0.0, start - pad), end + pad))
    if not cleaned:
        return []
    cleaned.sort()
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        ps, pe = merged[-1]
        if start <= pe + merge_gap:
            merged[-1] = (ps, max(pe, end))
        else:
            merged.append((start, end))
    return merged


def _duck_speech_regions(
    audio: "AudioSegment",
    segments: list[tuple[float, float]],
    *,
    duck_gain_db: float = -14.0,
) -> "AudioSegment":
    from pydub import AudioSegment

    windows = _merge_speech_windows(segments, pad=0.08, merge_gap=0.35)
    if not windows:
        return audio

    out = AudioSegment.empty()
    cursor_ms = 0
    total_ms = len(audio)
    for start, end in windows:
        s = max(0, int(start * 1000))
        e = min(total_ms, int(end * 1000))
        if e <= s or s < cursor_ms:
            continue
        if s > cursor_ms:
            out += audio[cursor_ms:s]
        # Keep music bed under Khmer (do not wipe to silence).
        chunk = audio[s:e] + duck_gain_db
        out += chunk
        cursor_ms = e
    if cursor_ms < total_ms:
        out += audio[cursor_ms:total_ms]
    return out if len(out) > 0 else audio
