# KH AI Video → Khmer Dubbing

Python desktop app: analyze speakers, translate speech to Khmer, generate Khmer voice + subtitles, export video.

```
Video
  → Extract Audio (FFmpeg)
  → Speech-to-Text (Whisper)
  → Analyze speakers / gender
  → Translate → Khmer
  → Khmer TTS (per speaker)
  → Mix / replace audio (FFmpeg)
  → Burn Khmer subtitles
  → Export
```

## Recommended beginner stack

| Function | Tool |
|----------|------|
| **GUI** | CustomTkinter |
| **Extract audio** | FFmpeg |
| **Speech → text** | Whisper / faster-whisper |
| **English/Chinese/etc. → Khmer** | Translation API (Bing → Google → MyMemory) |
| **Khmer text → speech** | edge-tts (Khmer Male / Female) |
| **Audio/video processing** | FFmpeg |
| **Subtitles** | `.srt` + styled `.ass` (Battambang, red + white outline) |
| **Progress** | CustomTkinter progress bar |

## Recommended flow (GUI)

1. **Browse** video (`MP4 / MKV / AVI / MOV`)
2. **Analyze Video** → detect language, speakers, male/female
3. Review **Speakers** table → pick Khmer voice per speaker
4. **KH Translate & Dub** → translate + TTS + mix + subtitles + export

### Options (checkboxes)

- Detect speakers  
- Detect male / female voice  
- Keep different voices  
- Translate speech  
- Generate Khmer speech  
- Keep background music  
- Generate Khmer subtitles  

## Modes (pipeline)

| Mode | Pipeline | Output |
|------|----------|--------|
| **1 — Subtitle** | Whisper → Translate → Khmer subs | `*_khmer_subtitle.mp4` |
| **2 — Khmer voice** | STT → Translate → TTS → replace audio | `*_khmer_voice.mp4` |
| **3 — Full AI dubbing** | Timed segments → TTS → mix BG → subs | `*_khmer_dubbed.mp4` |

## Setup

1. Python 3.10+  
2. FFmpeg on PATH (`ffmpeg -version`)  
3. Install deps:

```bash
cd VOZO-AI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Output location

Next to your video, in `khmer_output/`:

- Dubbed / subtitled `.mp4`
- `*_khmer.srt` / `*_khmer.ass`
- `*_original.txt` / `*_khmer.txt`

## Notes

- Start with Whisper **`base`**; use **`small`** if accuracy is weak.
- Subtitle font: **Battambang Bold** (`assets/fonts/`), size Biggest → Small in GUI.
- Analyze speakers uses pitch clustering (beginner). For pro diarization later: pyannote.
- Translation is free/multi-provider with local cache (`~/.vozo-ai/translate_cache.json`).
