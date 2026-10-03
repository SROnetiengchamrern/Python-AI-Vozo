"""KH AI Studio — Video Dubbing GUI (CustomTkinter)."""

from __future__ import annotations

import threading
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from .pipeline.analyze import AnalysisResult, AnalyzeOptions, analyze_video
from .pipeline.runner import ConversionOptions, run_conversion
from .pipeline.tts import KHMER_VOICE_OPTIONS

# --- Theme (matches KH AI Studio mockup) ---
BG = "#070B16"
PANEL = "#10182B"
PANEL_2 = "#151F36"
BORDER = "#24314F"
TEXT = "#F4F7FF"
MUTED = "#9AA6C1"
ACCENT = "#2F6BFF"
ACCENT_HOVER = "#2557D6"
PINK = "#E85D9A"
OK = "#3DDC97"

SUPPORTED_VIDEO = [
    ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v"),
    ("All files", "*.*"),
]

SUBTITLE_SIZES = {
    "Biggest": 72,
    "Large": 64,
    "Medium": 56,
    "Small": 46,
}

SOURCE_LANGS = ["Auto Detect", "English", "Chinese", "Thai", "Vietnamese", "Japanese"]

PROCESS_STEPS = [
    "Extracting audio",
    "Transcribing speech",
    "Detecting speakers",
    "Translating to Khmer",
    "Generating Khmer voice",
    "Clearing original voice",
    "Burning subtitles",
    "Exporting video",
]


def _step_from_status(status: str) -> int:
    s = (status or "").lower()
    rules = [
        ("extract", 0),
        ("transcrib", 1),
        ("analyz", 2),
        ("detect", 2),
        ("translat", 3),
        ("generat", 4),
        ("voice", 4),
        ("clear", 5),
        ("replac", 5),
        ("appl", 5),
        ("subtit", 6),
        ("burn", 6),
        ("export", 7),
        ("done", 7),
        ("ready", 7),
    ]
    for key, idx in rules:
        if key in s:
            return idx
    return 0


class VideoKhmerApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("KH AI Studio — Video & Song AI Tools")
        self.geometry("1280x860")
        self.minsize(1100, 760)
        self.configure(fg_color=BG)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.video_path: Path | None = None
        self.output_dir: Path | None = None
        self._worker: threading.Thread | None = None
        self._busy = False
        self.analysis: AnalysisResult | None = None
        self._speaker_voice_vars: dict[str, ctk.StringVar] = {}
        self._speaker_rows: list[ctk.CTkFrame] = []
        self._speaker_cards: list[ctk.CTkFrame] = []
        self._step_labels: list[ctk.CTkLabel] = []
        self._step_icons: list[ctk.CTkLabel] = []
        self._lockable: list = []
        self._job_started: datetime | None = None

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _card(self, parent, **kwargs) -> ctk.CTkFrame:
        opts = {
            "fg_color": PANEL,
            "corner_radius": 14,
            "border_width": 1,
            "border_color": BORDER,
        }
        opts.update(kwargs)
        return ctk.CTkFrame(parent, **opts)

    def _build_ui(self) -> None:
        self._build_header()

        body = ctk.CTkFrame(self, fg_color=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        self.main_col = ctk.CTkScrollableFrame(
            body, fg_color=BG, scrollbar_button_color=BORDER
        )
        self.main_col.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.side_col = ctk.CTkScrollableFrame(
            body, fg_color=BG, scrollbar_button_color=BORDER, width=360
        )
        self.side_col.grid(row=0, column=1, sticky="nsew")

        self._build_main()
        self._build_sidebar()
        self._collect_lockable()
        self._set_step(None)

    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color=PANEL, height=64, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", padx=18, pady=10)
        ctk.CTkLabel(
            left,
            text="▶  KH AI Studio",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w")
        ctk.CTkLabel(
            left,
            text="Video & Song AI Tools",
            font=ctk.CTkFont(size=11),
            text_color=MUTED,
        ).pack(anchor="w")

        tabs = ctk.CTkFrame(header, fg_color="transparent")
        tabs.pack(side="left", padx=30)
        self.tab_dub = ctk.CTkButton(
            tabs,
            text="Video Dubbing",
            width=130,
            height=32,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            corner_radius=8,
            command=lambda: None,
        )
        self.tab_dub.pack(side="left", padx=4)
        ctk.CTkButton(
            tabs,
            text="AI Song",
            width=100,
            height=32,
            fg_color=PANEL_2,
            hover_color=BORDER,
            text_color=MUTED,
            corner_radius=8,
            command=lambda: messagebox.showinfo("AI Song", "Coming soon."),
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            tabs,
            text="Settings",
            width=100,
            height=32,
            fg_color=PANEL_2,
            hover_color=BORDER,
            text_color=MUTED,
            corner_radius=8,
            command=lambda: messagebox.showinfo("Settings", "Coming soon."),
        ).pack(side="left", padx=4)

        ctk.CTkLabel(
            header,
            text="🇰🇭  ភាសាខ្មែរ",
            font=ctk.CTkFont(size=13),
            text_color=TEXT,
        ).pack(side="right", padx=18)

    def _build_main(self) -> None:
        root = self.main_col

        ctk.CTkLabel(
            root,
            text="KH AI VIDEO → Khmer Dubbing",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", pady=(4, 2))
        ctk.CTkLabel(
            root,
            text="Analyze speakers, translate dialogue, generate Khmer voice & burn subtitles.",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
        ).pack(anchor="w", pady=(0, 12))

        # Video file
        card = self._card(root)
        card.pack(fill="x", pady=6)
        ctk.CTkLabel(
            card, text="Video File", font=ctk.CTkFont(size=12, weight="bold"), text_color=MUTED
        ).pack(anchor="w", padx=14, pady=(12, 4))
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 14))
        self.video_entry = ctk.CTkEntry(
            row,
            placeholder_text="Select a video file…",
            height=36,
            fg_color=PANEL_2,
            border_color=BORDER,
        )
        self.video_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.browse_btn = ctk.CTkButton(
            row,
            text="Browse",
            width=100,
            height=36,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._browse,
        )
        self.browse_btn.pack(side="left")

        # Languages
        lang_card = self._card(root)
        lang_card.pack(fill="x", pady=6)
        inner = ctk.CTkFrame(lang_card, fg_color="transparent")
        inner.pack(fill="x", padx=14, pady=14)
        inner.grid_columnconfigure((0, 1), weight=1)

        left = ctk.CTkFrame(inner, fg_color="transparent")
        left.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkLabel(left, text="Source Language", text_color=MUTED, font=ctk.CTkFont(size=12)).pack(
            anchor="w"
        )
        self.source_var = ctk.StringVar(value="Auto Detect")
        self.source_menu = ctk.CTkOptionMenu(
            left,
            values=SOURCE_LANGS,
            variable=self.source_var,
            height=34,
            fg_color=PANEL_2,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
        )
        self.source_menu.pack(fill="x", pady=(4, 0))

        right = ctk.CTkFrame(inner, fg_color="transparent")
        right.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ctk.CTkLabel(right, text="Target Language", text_color=MUTED, font=ctk.CTkFont(size=12)).pack(
            anchor="w"
        )
        self.target_var = ctk.StringVar(value="Khmer")
        self.target_menu = ctk.CTkOptionMenu(
            right,
            values=["Khmer"],
            variable=self.target_var,
            height=34,
            fg_color=PANEL_2,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
        )
        self.target_menu.pack(fill="x", pady=(4, 0))

        # Options 2-column
        opt_card = self._card(root)
        opt_card.pack(fill="x", pady=6)
        ctk.CTkLabel(
            opt_card,
            text="Options",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=14, pady=(12, 6))

        self.detect_speakers_var = ctk.BooleanVar(value=True)
        self.detect_gender_var = ctk.BooleanVar(value=True)
        self.keep_voices_var = ctk.BooleanVar(value=True)
        self.translate_var = ctk.BooleanVar(value=True)
        self.gen_speech_var = ctk.BooleanVar(value=True)
        self.keep_bg_var = ctk.BooleanVar(value=False)
        self.gen_subs_var = ctk.BooleanVar(value=True)
        self.auto_detect_var = ctk.BooleanVar(value=True)
        self.manual_voice_var = ctk.BooleanVar(value=True)

        grid = ctk.CTkFrame(opt_card, fg_color="transparent")
        grid.pack(fill="x", padx=10, pady=(0, 8))
        grid.grid_columnconfigure((0, 1), weight=1)

        left_opts = [
            ("Detect speakers (diarization)", self.detect_speakers_var),
            ("Detect male / female voice", self.detect_gender_var),
            ("Keep different voices", self.keep_voices_var),
            ("Translate speech", self.translate_var),
        ]
        right_opts = [
            ("Generate Khmer speech", self.gen_speech_var),
            ("Keep BGM only (hard-mute original voice)", self.keep_bg_var),
            ("Generate Khmer subtitles", self.gen_subs_var),
        ]
        self.option_checks: list[ctk.CTkCheckBox] = []
        for i, (text, var) in enumerate(left_opts):
            cb = ctk.CTkCheckBox(
                grid,
                text=text,
                variable=var,
                fg_color=ACCENT,
                hover_color=ACCENT_HOVER,
                text_color=TEXT,
            )
            cb.grid(row=i, column=0, sticky="w", padx=8, pady=3)
            self.option_checks.append(cb)
        for i, (text, var) in enumerate(right_opts):
            cb = ctk.CTkCheckBox(
                grid,
                text=text,
                variable=var,
                fg_color=ACCENT,
                hover_color=ACCENT_HOVER,
                text_color=TEXT,
            )
            cb.grid(row=i, column=1, sticky="w", padx=8, pady=3)
            self.option_checks.append(cb)

        ctk.CTkLabel(
            opt_card,
            text="Settings lock while converting — edit again when status is Ready.",
            text_color=MUTED,
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=14, pady=(0, 12))

        # Processing dropdowns
        proc = self._card(root)
        proc.pack(fill="x", pady=6)
        prow = ctk.CTkFrame(proc, fg_color="transparent")
        prow.pack(fill="x", padx=14, pady=14)
        for i in range(3):
            prow.grid_columnconfigure(i, weight=1)

        ctk.CTkLabel(prow, text="Voice Style", text_color=MUTED, font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        ctk.CTkLabel(prow, text="Sub Size", text_color=MUTED, font=ctk.CTkFont(size=12)).grid(
            row=0, column=1, sticky="w", padx=8
        )
        ctk.CTkLabel(prow, text="Whisper / Quality", text_color=MUTED, font=ctk.CTkFont(size=12)).grid(
            row=0, column=2, sticky="w", padx=(8, 0)
        )

        self.voice_style_var = ctk.StringVar(value="Drama")
        self.style_menu = ctk.CTkOptionMenu(
            prow,
            values=["Drama", "Natural", "Soft", "Energetic"],
            variable=self.voice_style_var,
            fg_color=PANEL_2,
            button_color=ACCENT,
        )
        self.style_menu.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(4, 0))

        self.sub_size_var = ctk.StringVar(value="Large")
        self.sub_size_menu = ctk.CTkOptionMenu(
            prow,
            values=list(SUBTITLE_SIZES.keys()),
            variable=self.sub_size_var,
            fg_color=PANEL_2,
            button_color=ACCENT,
        )
        self.sub_size_menu.grid(row=1, column=1, sticky="ew", padx=8, pady=(4, 0))

        self.model_var = ctk.StringVar(value="base")
        self.model_menu = ctk.CTkOptionMenu(
            prow,
            values=["tiny", "base", "small", "medium"],
            variable=self.model_var,
            fg_color=PANEL_2,
            button_color=ACCENT,
        )
        self.model_menu.grid(row=1, column=2, sticky="ew", padx=(8, 0), pady=(4, 0))

        # Speakers
        sp_card = self._card(root)
        sp_card.pack(fill="x", pady=6)
        head = ctk.CTkFrame(sp_card, fg_color="transparent")
        head.pack(fill="x", padx=14, pady=(12, 6))
        ctk.CTkLabel(
            head,
            text="Speaker Settings",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).pack(side="left")
        self.lang_info = ctk.CTkLabel(head, text="No analysis yet", text_color=MUTED)
        self.lang_info.pack(side="right")

        extras = ctk.CTkFrame(sp_card, fg_color="transparent")
        extras.pack(fill="x", padx=14, pady=(0, 6))
        self.auto_cb = ctk.CTkCheckBox(
            extras,
            text="Auto Detection",
            variable=self.auto_detect_var,
            fg_color=ACCENT,
            text_color=TEXT,
        )
        self.auto_cb.pack(side="left", padx=(0, 12))
        self.manual_cb = ctk.CTkCheckBox(
            extras,
            text="Allow manual voice selection",
            variable=self.manual_voice_var,
            fg_color=ACCENT,
            text_color=TEXT,
        )
        self.manual_cb.pack(side="left")

        header = ctk.CTkFrame(sp_card, fg_color=PANEL_2, corner_radius=8)
        header.pack(fill="x", padx=12, pady=(4, 2))
        for text, w in (("ID", 50), ("Gender", 90), ("Detected", 110), ("Khmer Voice", 160)):
            ctk.CTkLabel(header, text=text, width=w, anchor="w", text_color=MUTED).pack(
                side="left", padx=6, pady=6
            )

        self.speakers_body = ctk.CTkFrame(sp_card, fg_color="transparent")
        self.speakers_body.pack(fill="x", padx=8, pady=(0, 12))
        self._speakers_empty = ctk.CTkLabel(
            self.speakers_body,
            text="Click Analyze Video to detect speakers…",
            text_color=MUTED,
        )
        self._speakers_empty.pack(anchor="w", padx=10, pady=10)

        # Actions + progress
        action_card = self._card(root)
        action_card.pack(fill="x", pady=(6, 16))
        self.progress = ctk.CTkProgressBar(
            action_card, progress_color=ACCENT, fg_color=PANEL_2, height=12
        )
        self.progress.pack(fill="x", padx=14, pady=(14, 6))
        self.progress.set(0)
        self.status_label = ctk.CTkLabel(
            action_card,
            text="Status: Ready — analyze first, then dub",
            anchor="w",
            text_color=MUTED,
        )
        self.status_label.pack(fill="x", padx=14, pady=(0, 10))

        btns = ctk.CTkFrame(action_card, fg_color="transparent")
        btns.pack(fill="x", padx=14, pady=(0, 14))
        self.analyze_btn = ctk.CTkButton(
            btns,
            text="Analyze Video",
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="transparent",
            border_width=2,
            border_color=ACCENT,
            text_color=TEXT,
            hover_color=PANEL_2,
            command=self._analyze,
        )
        self.analyze_btn.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.dub_btn = ctk.CTkButton(
            btns,
            text="🎙  KH Translate & Dub",
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            command=self._dub,
        )
        self.dub_btn.pack(side="left", fill="x", expand=True)

    def _build_sidebar(self) -> None:
        side = self.side_col

        # Processing steps
        steps_card = self._card(side)
        steps_card.pack(fill="x", pady=(4, 8))
        ctk.CTkLabel(
            steps_card,
            text="Processing Steps",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=14, pady=(12, 8))
        self.steps_body = ctk.CTkFrame(steps_card, fg_color="transparent")
        self.steps_body.pack(fill="x", padx=12, pady=(0, 12))
        self._step_icons.clear()
        self._step_labels.clear()
        for name in PROCESS_STEPS:
            row = ctk.CTkFrame(self.steps_body, fg_color="transparent")
            row.pack(fill="x", pady=3)
            icon = ctk.CTkLabel(row, text="○", width=22, text_color=MUTED)
            icon.pack(side="left")
            lab = ctk.CTkLabel(row, text=name, anchor="w", text_color=MUTED)
            lab.pack(side="left", fill="x", expand=True)
            self._step_icons.append(icon)
            self._step_labels.append(lab)

        # Speaker preview
        prev_card = self._card(side)
        prev_card.pack(fill="x", pady=8)
        ctk.CTkLabel(
            prev_card,
            text="Speaker Preview",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=14, pady=(12, 8))
        self.preview_body = ctk.CTkFrame(prev_card, fg_color="transparent")
        self.preview_body.pack(fill="x", padx=10, pady=(0, 12))
        self._preview_empty = ctk.CTkLabel(
            self.preview_body, text="No speakers yet", text_color=MUTED
        )
        self._preview_empty.pack(anchor="w", padx=8, pady=6)

        # Output
        out_card = self._card(side)
        out_card.pack(fill="x", pady=8)
        ctk.CTkLabel(
            out_card,
            text="Output",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=14, pady=(12, 6))

        ctk.CTkLabel(out_card, text="Save to", text_color=MUTED, font=ctk.CTkFont(size=11)).pack(
            anchor="w", padx=14
        )
        save_row = ctk.CTkFrame(out_card, fg_color="transparent")
        save_row.pack(fill="x", padx=14, pady=(2, 8))
        self.save_entry = ctk.CTkEntry(
            save_row, placeholder_text="khmer_output (next to video)", fg_color=PANEL_2, border_color=BORDER
        )
        self.save_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.save_btn = ctk.CTkButton(
            save_row,
            text="📁",
            width=40,
            fg_color=PANEL_2,
            hover_color=BORDER,
            command=self._browse_output,
        )
        self.save_btn.pack(side="left")

        ctk.CTkLabel(out_card, text="File name", text_color=MUTED, font=ctk.CTkFont(size=11)).pack(
            anchor="w", padx=14
        )
        self.name_entry = ctk.CTkEntry(
            out_card, placeholder_text="auto from video name", fg_color=PANEL_2, border_color=BORDER
        )
        self.name_entry.pack(fill="x", padx=14, pady=(2, 8))

        self.eta_label = ctk.CTkLabel(
            out_card,
            text="⏱  Estimated time: —",
            text_color=MUTED,
            font=ctk.CTkFont(size=12),
        )
        self.eta_label.pack(anchor="w", padx=14, pady=(0, 14))

    def _collect_lockable(self) -> None:
        self._lockable = [
            self.video_entry,
            self.browse_btn,
            self.source_menu,
            self.target_menu,
            self.style_menu,
            self.sub_size_menu,
            self.model_menu,
            self.analyze_btn,
            self.dub_btn,
            self.save_entry,
            self.save_btn,
            self.name_entry,
            self.auto_cb,
            self.manual_cb,
            *self.option_checks,
        ]

    # -------------------------------------------------------------- helpers
    def _set_step(self, active_idx: int | None) -> None:
        for i, (icon, lab) in enumerate(zip(self._step_icons, self._step_labels)):
            if active_idx is None:
                icon.configure(text="○", text_color=MUTED)
                lab.configure(text_color=MUTED)
            elif i < active_idx:
                icon.configure(text="✓", text_color=OK)
                lab.configure(text_color=TEXT)
            elif i == active_idx:
                icon.configure(text="●", text_color=ACCENT)
                lab.configure(text_color=TEXT)
            else:
                icon.configure(text="○", text_color=MUTED)
                lab.configure(text_color=MUTED)

    def _browse(self) -> None:
        if self._busy:
            return
        path = filedialog.askopenfilename(title="Select video", filetypes=SUPPORTED_VIDEO)
        if not path:
            return
        self.video_path = Path(path)
        self.video_entry.delete(0, "end")
        self.video_entry.insert(0, str(self.video_path))
        self.analysis = None
        self._clear_speakers()
        self.lang_info.configure(text="No analysis yet")
        if not self.save_entry.get().strip():
            self.save_entry.insert(0, str(self.video_path.parent / "khmer_output"))
        if not self.name_entry.get().strip():
            self.name_entry.insert(0, f"{self.video_path.stem}_khmer_dubbed")

    def _browse_output(self) -> None:
        if self._busy:
            return
        path = filedialog.askdirectory(title="Select output folder")
        if not path:
            return
        self.output_dir = Path(path)
        self.save_entry.delete(0, "end")
        self.save_entry.insert(0, str(self.output_dir))

    def _video(self) -> Path | None:
        raw = self.video_entry.get().strip()
        video = Path(raw) if raw else self.video_path
        if not video or not video.exists():
            messagebox.showerror("Missing video", "Please browse and select a video file.")
            return None
        return video

    def _set_busy(self, busy: bool, *, job: str = "Working") -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for widget in self._lockable:
            try:
                widget.configure(state=state)
            except Exception:  # noqa: BLE001
                pass
        for row in self._speaker_rows:
            for child in row.winfo_children():
                if isinstance(child, ctk.CTkOptionMenu):
                    try:
                        child.configure(state=state)
                    except Exception:  # noqa: BLE001
                        pass
        if busy:
            self.analyze_btn.configure(text="Please wait…")
            self.dub_btn.configure(text="Converting…")
            self._job_started = datetime.now()
        else:
            self.analyze_btn.configure(text="Analyze Video")
            self.dub_btn.configure(text="🎙  KH Translate & Dub")
            self._job_started = None

    def _report(self, pct: float, status: str) -> None:
        self.after(0, lambda: self._update_progress(pct, status))

    def _update_progress(self, pct: float, status: str) -> None:
        self.progress.set(max(0.0, min(1.0, pct / 100.0)))
        self.status_label.configure(text=f"Status: {status}")
        self._set_step(_step_from_status(status))
        if self._job_started and pct < 100:
            self.eta_label.configure(text="⏱  Estimated time: 15 – 30 minutes")
        elif pct >= 100:
            self.eta_label.configure(text="⏱  Estimated time: done")

    def _clear_speakers(self) -> None:
        for row in self._speaker_rows:
            row.destroy()
        self._speaker_rows.clear()
        self._speaker_voice_vars.clear()
        for card in self._speaker_cards:
            card.destroy()
        self._speaker_cards.clear()
        if self._speakers_empty.winfo_exists():
            self._speakers_empty.pack(anchor="w", padx=10, pady=10)
        if self._preview_empty.winfo_exists():
            self._preview_empty.pack(anchor="w", padx=8, pady=6)

    def _render_speakers(self, analysis: AnalysisResult) -> None:
        self._clear_speakers()
        if self._speakers_empty.winfo_exists():
            self._speakers_empty.pack_forget()
        if self._preview_empty.winfo_exists():
            self._preview_empty.pack_forget()

        for sp in analysis.speakers:
            row = ctk.CTkFrame(self.speakers_body, fg_color="transparent")
            row.pack(fill="x", pady=3, padx=4)
            ctk.CTkLabel(row, text=sp.speaker_id, width=50, anchor="w", text_color=TEXT).pack(
                side="left", padx=6
            )
            gender = sp.gender
            gcolor = ACCENT if gender == "Male" else (PINK if gender == "Female" else MUTED)
            ctk.CTkLabel(
                row,
                text=("♂ " if gender == "Male" else "♀ " if gender == "Female" else "? ")
                + gender,
                width=90,
                anchor="w",
                text_color=gcolor,
            ).pack(side="left", padx=6)
            ctk.CTkLabel(
                row,
                text=f"Speaker {int(sp.speaker_id) + 1}",
                width=110,
                anchor="w",
                text_color=MUTED,
            ).pack(side="left", padx=6)
            var = ctk.StringVar(value=sp.default_voice)
            self._speaker_voice_vars[sp.speaker_id] = var
            menu = ctk.CTkOptionMenu(
                row,
                values=KHMER_VOICE_OPTIONS,
                variable=var,
                width=160,
                fg_color=PANEL_2,
                button_color=ACCENT,
            )
            menu.pack(side="left", padx=6)
            self._speaker_rows.append(row)

            # Preview card
            card = ctk.CTkFrame(
                self.preview_body, fg_color=PANEL_2, corner_radius=12, border_width=1, border_color=BORDER
            )
            card.pack(fill="x", pady=5, padx=4)
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=10, pady=(10, 4))
            avatar = ctk.CTkLabel(
                top,
                text=("♂" if gender == "Male" else "♀"),
                width=36,
                height=36,
                corner_radius=18,
                fg_color=gcolor,
                text_color="white",
                font=ctk.CTkFont(size=16, weight="bold"),
            )
            avatar.pack(side="left", padx=(0, 8))
            meta = ctk.CTkFrame(top, fg_color="transparent")
            meta.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(
                meta, text=sp.default_voice, anchor="w", text_color=TEXT, font=ctk.CTkFont(weight="bold")
            ).pack(anchor="w")
            tag = ctk.CTkLabel(
                meta,
                text=gender,
                text_color=gcolor,
                font=ctk.CTkFont(size=11),
            )
            tag.pack(anchor="w")
            ctk.CTkSlider(card, from_=0, to=100, number_of_steps=20, progress_color=ACCENT).pack(
                fill="x", padx=10, pady=(2, 10)
            )
            self._speaker_cards.append(card)

        self.lang_info.configure(
            text=f"{analysis.language_name} · {len(analysis.speakers)} speakers · {len(analysis.segments)} segs"
        )

    # -------------------------------------------------------------- actions
    def _analyze(self) -> None:
        if self._busy or (self._worker and self._worker.is_alive()):
            return
        video = self._video()
        if not video:
            return

        options = AnalyzeOptions(
            video_path=video,
            whisper_model=self.model_var.get(),
            detect_speakers=self.detect_speakers_var.get(),
            detect_gender=self.detect_gender_var.get(),
            max_speakers=3 if self.keep_voices_var.get() else 1,
        )
        self._set_busy(True, job="Analyze")
        self._update_progress(0, "Analyzing… settings locked until ready")

        def worker() -> None:
            try:
                result = analyze_video(options, progress=self._report)
                self.after(0, lambda: self._on_analyze_done(result))
            except Exception as exc:  # noqa: BLE001
                err = f"{exc}\n\n{traceback.format_exc()}"
                self.after(0, lambda: self._on_error(err))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _on_analyze_done(self, result: AnalysisResult) -> None:
        self.analysis = result
        self._set_busy(False)
        self._render_speakers(result)
        self._update_progress(100, "Ready — review speakers, then Dub")
        self._set_step(2)
        messagebox.showinfo(
            "Analyze complete",
            f"Language: {result.language_name}\n"
            f"Speakers: {len(result.speakers)}\n"
            f"Segments: {len(result.segments)}\n\n"
            "Settings unlocked. Review Speakers, then click KH Translate & Dub.",
        )

    def _speaker_voice_map(self) -> dict[str, str]:
        return {sid: var.get() for sid, var in self._speaker_voice_vars.items()}

    def _dub(self) -> None:
        if self._busy or (self._worker and self._worker.is_alive()):
            return
        video = self._video()
        if not video:
            return
        if self.analysis is None:
            if not messagebox.askyesno(
                "No analysis yet",
                "Analyze Video first for better speaker voices.\n\nDub with a single voice now?",
            ):
                return

        voice_map = self._speaker_voice_map()
        default_voice = next(iter(voice_map.values()), "Khmer Female 1")
        keep_bg = self.keep_bg_var.get()
        mode = "full" if self.gen_speech_var.get() else "subtitle"
        out_raw = self.save_entry.get().strip()
        output_dir = Path(out_raw) if out_raw else None

        options = ConversionOptions(
            video_path=video,
            mode=mode,
            language=self.target_var.get(),
            voice=default_voice,
            translate_speech=self.translate_var.get(),
            generate_voice=self.gen_speech_var.get(),
            replace_audio=not keep_bg,
            keep_background=keep_bg,
            burn_in_subtitles=self.gen_subs_var.get(),
            whisper_model=self.model_var.get(),
            subtitle_font_size=SUBTITLE_SIZES.get(self.sub_size_var.get(), 64),
            voice_style=self.voice_style_var.get(),
            output_dir=output_dir,
            speaker_voices=voice_map or None,
            analyzed_segments=self.analysis.segments if self.analysis else None,
            pre_transcript=self.analysis.transcript if self.analysis else None,
        )

        self._set_busy(True, job="Convert")
        self._update_progress(0, "Converting… settings locked until ready")

        def worker() -> None:
            try:
                result = run_conversion(options, progress=self._report)
                self.after(0, lambda: self._on_success(result.output_video))
            except Exception as exc:  # noqa: BLE001
                err = f"{exc}\n\n{traceback.format_exc()}"
                self.after(0, lambda: self._on_error(err))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _on_success(self, output: Path) -> None:
        self._set_busy(False)
        self._update_progress(100, "Ready — conversion complete")
        self._set_step(7)
        messagebox.showinfo(
            "Conversion complete",
            f"Saved:\n{output}\n\nCompanion files are in:\n{output.parent}\n\n"
            "Settings unlocked — you can change options and run again.",
        )

    def _on_error(self, err: str) -> None:
        self._set_busy(False)
        self._update_progress(0, "Ready — last run failed")
        self._set_step(None)
        messagebox.showerror("Failed", err[:2500])


def run_app() -> None:
    app = VideoKhmerApp()
    app.mainloop()
