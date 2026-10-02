"""CustomTkinter GUI — Video → Khmer AI."""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from .pipeline.runner import ConversionOptions, run_conversion
from .pipeline.tts import VOICE_CHOICES

SUPPORTED_VIDEO = [
    ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"),
    ("All files", "*.*"),
]

MODES = {
    "Mode 1 — Subtitle": "subtitle",
    "Mode 2 — Khmer voice": "voice",
    "Mode 3 — Full AI dubbing": "full",
}

# Preset labels → ASS Fontsize (PlayRes 1920x1080).
SUBTITLE_SIZES = {
    "Biggest": 120,
    "Large": 96,
    "Medium": 72,
    "Small": 52,
}


class VideoKhmerApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Video → Khmer AI")
        self.geometry("560x620")
        self.minsize(520, 580)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.video_path: Path | None = None
        self._worker: threading.Thread | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = ctk.CTkFrame(self, corner_radius=12)
        root.pack(fill="both", expand=True, padx=18, pady=18)

        title = ctk.CTkLabel(
            root,
            text="🎬  Video → Khmer AI",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.pack(anchor="w", padx=16, pady=(16, 8))

        # Video browse
        file_row = ctk.CTkFrame(root, fg_color="transparent")
        file_row.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(file_row, text="Video:", width=80, anchor="w").pack(side="left")
        self.video_entry = ctk.CTkEntry(file_row, placeholder_text="Select a video file…")
        self.video_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(file_row, text="Browse…", width=100, command=self._browse).pack(
            side="left"
        )

        # Mode
        mode_row = ctk.CTkFrame(root, fg_color="transparent")
        mode_row.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(mode_row, text="Mode:", width=80, anchor="w").pack(side="left")
        self.mode_var = ctk.StringVar(value="Mode 2 — Khmer voice")
        self.mode_menu = ctk.CTkOptionMenu(
            mode_row,
            values=list(MODES.keys()),
            variable=self.mode_var,
            width=280,
            command=self._on_mode_change,
        )
        self.mode_menu.pack(side="left")

        # Language
        lang_row = ctk.CTkFrame(root, fg_color="transparent")
        lang_row.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(lang_row, text="Language:", width=80, anchor="w").pack(side="left")
        self.lang_var = ctk.StringVar(value="Khmer")
        ctk.CTkOptionMenu(
            lang_row, values=["Khmer"], variable=self.lang_var, width=180
        ).pack(side="left")

        # Voice
        voice_row = ctk.CTkFrame(root, fg_color="transparent")
        voice_row.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(voice_row, text="Voice:", width=80, anchor="w").pack(side="left")
        self.voice_var = ctk.StringVar(value="Khmer Female")
        ctk.CTkOptionMenu(
            voice_row,
            values=list(VOICE_CHOICES.keys()),
            variable=self.voice_var,
            width=180,
        ).pack(side="left")

        # Options
        opts = ctk.CTkFrame(root, fg_color="transparent")
        opts.pack(fill="x", padx=16, pady=(10, 4))
        self.translate_var = ctk.BooleanVar(value=True)
        self.voice_opt_var = ctk.BooleanVar(value=True)
        self.replace_var = ctk.BooleanVar(value=True)
        self.burn_subs_var = ctk.BooleanVar(value=True)

        ctk.CTkCheckBox(
            opts, text="Translate speech", variable=self.translate_var
        ).pack(anchor="w", pady=2)
        self.voice_check = ctk.CTkCheckBox(
            opts, text="Generate Khmer voice", variable=self.voice_opt_var
        )
        self.voice_check.pack(anchor="w", pady=2)
        self.replace_check = ctk.CTkCheckBox(
            opts, text="Replace original audio", variable=self.replace_var
        )
        self.replace_check.pack(anchor="w", pady=2)
        self.subs_check = ctk.CTkCheckBox(
            opts, text="Burn Khmer subtitles into video", variable=self.burn_subs_var
        )
        self.subs_check.pack(anchor="w", pady=2)

        # Subtitle size (default Biggest; change to Small/Medium/Large)
        size_row = ctk.CTkFrame(root, fg_color="transparent")
        size_row.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(size_row, text="Sub size:", width=80, anchor="w").pack(side="left")
        self.sub_size_var = ctk.StringVar(value="Biggest")
        ctk.CTkOptionMenu(
            size_row,
            values=list(SUBTITLE_SIZES.keys()),
            variable=self.sub_size_var,
            width=140,
        ).pack(side="left")
        ctk.CTkLabel(
            size_row,
            text="Biggest → Small",
            text_color="gray70",
            font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=8)

        # Whisper model size (small = faster, large = more accurate)
        model_row = ctk.CTkFrame(root, fg_color="transparent")
        model_row.pack(fill="x", padx=16, pady=6)
        ctk.CTkLabel(model_row, text="Whisper:", width=80, anchor="w").pack(side="left")
        self.model_var = ctk.StringVar(value="base")
        ctk.CTkOptionMenu(
            model_row,
            values=["tiny", "base", "small", "medium"],
            variable=self.model_var,
            width=120,
        ).pack(side="left")
        ctk.CTkLabel(
            model_row,
            text="(base is a good start)",
            text_color="gray70",
            font=ctk.CTkFont(size=12),
        ).pack(side="left", padx=8)

        # Start
        self.start_btn = ctk.CTkButton(
            root,
            text="▶  START CONVERSION",
            height=42,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._start,
        )
        self.start_btn.pack(fill="x", padx=16, pady=(16, 10))

        # Progress
        self.progress = ctk.CTkProgressBar(root)
        self.progress.pack(fill="x", padx=16, pady=(4, 4))
        self.progress.set(0)

        self.status_label = ctk.CTkLabel(
            root, text="Status: Idle", anchor="w", text_color="gray80"
        )
        self.status_label.pack(fill="x", padx=16, pady=(0, 8))

        self.hint = ctk.CTkLabel(
            root,
            text="Requires FFmpeg on PATH. First Whisper run downloads the model.",
            anchor="w",
            text_color="gray60",
            font=ctk.CTkFont(size=11),
        )
        self.hint.pack(fill="x", padx=16, pady=(0, 12))

        self._on_mode_change(self.mode_var.get())

    def _on_mode_change(self, value: str) -> None:
        mode = MODES.get(value, "voice")
        if mode == "subtitle":
            self.voice_opt_var.set(False)
            self.replace_var.set(False)
            self.burn_subs_var.set(True)
            self.voice_check.configure(state="disabled")
            self.replace_check.configure(state="disabled")
            self.subs_check.configure(state="normal")
        elif mode == "voice":
            self.voice_opt_var.set(True)
            self.replace_var.set(True)
            self.burn_subs_var.set(True)
            self.voice_check.configure(state="normal")
            self.replace_check.configure(state="normal")
            self.subs_check.configure(state="normal")
        else:  # full
            self.voice_opt_var.set(True)
            self.replace_var.set(False)  # mix with background by default
            self.burn_subs_var.set(True)
            self.voice_check.configure(state="normal")
            self.replace_check.configure(state="normal")
            self.subs_check.configure(state="normal")

    def _browse(self) -> None:
        path = filedialog.askopenfilename(title="Select video", filetypes=SUPPORTED_VIDEO)
        if not path:
            return
        self.video_path = Path(path)
        self.video_entry.delete(0, "end")
        self.video_entry.insert(0, str(self.video_path))

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.start_btn.configure(state=state)
        if not busy:
            self.start_btn.configure(text="▶  START CONVERSION")
        else:
            self.start_btn.configure(text="Working…")

    def _report(self, pct: float, status: str) -> None:
        self.after(0, lambda: self._update_progress(pct, status))

    def _update_progress(self, pct: float, status: str) -> None:
        self.progress.set(max(0.0, min(1.0, pct / 100.0)))
        self.status_label.configure(text=f"Status: {status}")

    def _start(self) -> None:
        raw = self.video_entry.get().strip()
        video = Path(raw) if raw else self.video_path
        if not video or not video.exists():
            messagebox.showerror("Missing video", "Please browse and select a video file.")
            return
        if self._worker and self._worker.is_alive():
            return

        options = ConversionOptions(
            video_path=video,
            mode=MODES[self.mode_var.get()],
            language=self.lang_var.get(),
            voice=self.voice_var.get(),
            translate_speech=self.translate_var.get(),
            generate_voice=self.voice_opt_var.get(),
            replace_audio=self.replace_var.get(),
            burn_in_subtitles=self.burn_subs_var.get(),
            whisper_model=self.model_var.get(),
            subtitle_font_size=SUBTITLE_SIZES.get(self.sub_size_var.get(), 120),
        )

        self._set_busy(True)
        self._update_progress(0, "Starting…")

        def worker() -> None:
            try:
                result = run_conversion(options, progress=self._report)
                self.after(0, lambda: self._on_success(result.output_video))
            except Exception as exc:  # noqa: BLE001 — show any pipeline error in GUI
                err = f"{exc}\n\n{traceback.format_exc()}"
                self.after(0, lambda: self._on_error(err))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _on_success(self, output: Path) -> None:
        self._set_busy(False)
        self._update_progress(100, "Done!")
        messagebox.showinfo(
            "Conversion complete",
            f"Saved:\n{output}\n\nCompanion files are in:\n{output.parent}",
        )

    def _on_error(self, err: str) -> None:
        self._set_busy(False)
        self._update_progress(0, "Error")
        messagebox.showerror("Conversion failed", err[:2500])


def run_app() -> None:
    app = VideoKhmerApp()
    app.mainloop()
