# Video → Khmer AI

Python desktop app that converts a video into Khmer subtitles and/or Khmer dubbed audio.

```
Video → Extract Audio → Whisper STT → Translate to Khmer → Khmer TTS → FFmpeg Export
```

## Modes

| Mode | What it does | Output |
|------|----------------|--------|
| **Mode 1 — Subtitle** | Transcribe → translate → burn Khmer SRT | `*_khmer_subtitle.mp4` + `.srt` |
| **Mode 2 — Khmer voice** | Transcribe → translate → TTS → replace audio | `*_khmer_voice.mp4` |
| **Mode 3 — Full AI dubbing** | Timed segments → TTS on timeline → mix + subs | `*_khmer_dubbed.mp4` |

## Stack

| Function | Tool |
|----------|------|
| GUI | CustomTkinter |
| Extract / mux audio | FFmpeg |
| Speech → text | faster-whisper |
| Translate → Khmer | Bing → Google → MyMemory (auto-fallback + cache) |
| Khmer text → speech | edge-tts (`km-KH-SreymomNeural` / `PisethNeural`) |
| Timed audio assembly | pydub |
| Subtitles | `.srt` generation |

## Requirements

1. **Python 3.10+**
2. **FFmpeg** on your PATH  
   - Windows: [gyan.dev FFmpeg builds](https://www.gyan.dev/ffmpeg/builds/)  
   - Check: `ffmpeg -version`
3. Internet (first Whisper model download + translation + Edge TTS)

## Setup

```bash
cd VOZO-AI
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

1. Click **Browse…** and pick an MP4/MKV/AVI/MOV  
2. Choose a **Mode**  
3. Pick **Khmer Female** or **Khmer Male**  
4. Click **START CONVERSION**  
5. Output lands in a `khmer_output` folder next to your video

## GUI layout

Matches the beginner mockup:

- Video browse  
- Mode / Language / Voice dropdowns  
- Checkboxes: Translate speech · Generate Khmer voice · Replace original audio · Burn subtitles  
- Progress bar + live status  

## Notes for beginners

- Start with Whisper model **`base`**. Use **`small`** if accuracy is weak.
- Mode 3 is a *beginner* dubbing path: it places speech near original timestamps but does **not** do perfect lip-sync or stem separation.
- Translation uses Google via `deep-translator` (no API key). For production, swap in DeepL / Google Cloud Translate.
- Edge TTS is free and needs no cloud key. Swap `app/pipeline/tts.py` for Google/Azure TTS later if you want.

## Project layout

```
VOZO-AI/
├── main.py                 # launch GUI
├── requirements.txt
├── app/
│   ├── gui.py              # CustomTkinter UI
│   └── pipeline/
│       ├── runner.py       # Mode 1 / 2 / 3 orchestration
│       ├── ffmpeg_utils.py
│       ├── transcribe.py
│       ├── translate.py
│       ├── tts.py
│       └── subtitles.py
└── README.md
```
