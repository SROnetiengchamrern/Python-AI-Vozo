"""FFmpeg helpers for audio extraction and video export."""

from __future__ import annotations

import shutil
import subprocess
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


def replace_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    """Mux new audio over video, dropping the original audio track."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(output_path),
        ]
    )
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

    # Point libass at project fonts so Battambang Bold/Regular resolve.
    if subtitle_path.suffix.lower() == ".ass":
        vf = f"ass='{escaped}':fontsdir='{fonts_escaped}'"
    else:
        vf = (
            f"subtitles='{escaped}':fontsdir='{fonts_escaped}':"
            "force_style='FontName=Battambang,Alignment=2,MarginV=55,"
            "Fontsize=64,PrimaryColour=&H000000FF,OutlineColour=&H00FFFFFF,"
            "BorderStyle=1,Outline=4,Shadow=1.5,Bold=1'"
        )
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
    background_volume: float = 0.18,
) -> Path:
    """Replace dialogue-heavy mix: keep quiet original track under Khmer voice."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = (
        f"[0:a]volume={background_volume}[bg];"
        f"[1:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )
    run_ffmpeg(
        [
            "-i",
            str(video_path),
            "-i",
            str(voice_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-shortest",
            str(output_path),
        ]
    )
    return output_path
