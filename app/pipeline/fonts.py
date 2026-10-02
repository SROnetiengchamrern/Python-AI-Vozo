"""Ensure Battambang Khmer fonts are available for subtitle burn-in."""

from __future__ import annotations

import urllib.request
from pathlib import Path

BATTAMBANG_FONTS = {
    "Battambang-Regular.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/battambang/Battambang-Regular.ttf"
    ),
    "Battambang-Bold.ttf": (
        "https://github.com/google/fonts/raw/main/ofl/battambang/Battambang-Bold.ttf"
    ),
}

# Family name libass / ASS Style Fontname expects.
BATTAMBANG_FAMILY = "Battambang"


def project_fonts_dir() -> Path:
    # app/pipeline/fonts.py → repo root / assets / fonts
    return Path(__file__).resolve().parents[2] / "assets" / "fonts"


def ensure_battambang_fonts() -> Path:
    """Download Battambang Regular/Bold into assets/fonts if missing."""
    fonts_dir = project_fonts_dir()
    fonts_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in BATTAMBANG_FONTS.items():
        dest = fonts_dir / filename
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        tmp = dest.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
        except Exception as err:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if not dest.exists():
                raise RuntimeError(
                    f"Failed to download Khmer font {filename}.\n"
                    f"URL: {url}\n"
                    f"Error: {err}"
                ) from err
    return fonts_dir
