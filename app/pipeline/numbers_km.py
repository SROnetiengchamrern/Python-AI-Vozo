"""Convert Western numbers into Khmer numeral + unit forms for TTS/subs."""

from __future__ import annotations

import re

_DIGIT_MAP = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")

# Prefer units the user requested: មុឺន (10k), លាន (1M).
_UNIT_LEAN = 1_000_000
_UNIT_MUEN = 10_000
_UNIT_POAN = 1_000
_UNIT_ROY = 100


def to_khmer_digits(n: int) -> str:
    return str(n).translate(_DIGIT_MAP)


def format_khmer_amount(n: int) -> str:
    """
    Format an integer the way Khmer speech often reads money/counts.

    Examples:
      10000  -> ១មុឺន
      20000  -> ២មុឺន
      50000  -> ៥មុឺន
      100000 -> ១០មុឺន
      1000000 -> ១លាន
    """
    if n < 0:
        return "-" + format_khmer_amount(-n)
    if n < 100:
        return to_khmer_digits(n)

    parts: list[str] = []

    lean = n // _UNIT_LEAN
    rem = n % _UNIT_LEAN
    if lean:
        parts.append(f"{to_khmer_digits(lean)}លាន")

    muen = rem // _UNIT_MUEN
    rem = rem % _UNIT_MUEN
    if muen:
        parts.append(f"{to_khmer_digits(muen)}មុឺន")

    poan = rem // _UNIT_POAN
    rem = rem % _UNIT_POAN
    if poan:
        parts.append(f"{to_khmer_digits(poan)}ពាន់")

    roy = rem // _UNIT_ROY
    rem = rem % _UNIT_ROY
    if roy:
        parts.append(f"{to_khmer_digits(roy)}រយ")

    if rem:
        parts.append(to_khmer_digits(rem))

    return "".join(parts) if parts else to_khmer_digits(0)


_NUM_RE = re.compile(
    r"""
    (?<![A-Za-z0-9០-៩])          # not mid-word
    (\$|USD|៛)?                   # optional currency marker
    \s*
    (
        \d{1,3}(?:,\d{3})+        # 50,000 or 1,000,000
      | \d{4,}                    # 50000 / 1000000
      | \d{1,3}                   # smaller plain numbers
    )
    (?:\.\d+)?                    # ignore decimals for unit form
    \s*
    (\$|USD|៛|dollars?|bucks?)?   # trailing currency words
    (?![A-Za-z0-9០-៩])
    """,
    re.IGNORECASE | re.VERBOSE,
)


def khmerize_numbers_in_text(text: str) -> str:
    """Replace Western numbers with Khmer digit+unit forms for speech/subs."""
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        num_token = match.group(2)
        # Keep pure years-ish 19xx/20xx as digits only (still Khmer digits).
        digits = num_token.replace(",", "")
        if not digits.isdigit():
            return raw
        value = int(digits)
        # Likely calendar year → Khmer digits only, no មុឺន.
        if 1900 <= value <= 2100 and "," not in num_token and len(digits) == 4:
            return to_khmer_digits(value)
        return format_khmer_amount(value)

    return _NUM_RE.sub(repl, text)
