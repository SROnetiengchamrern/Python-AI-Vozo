"""Translate transcript text to Khmer (quality-first, per segment)."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from .transcribe import Transcript, TranscriptSegment

_CACHE_PATH = Path.home() / ".vozo-ai" / "translate_cache_v2.json"
_KHMER_RE = re.compile(r"[\u1780-\u17FF]")
_MAX_CHARS = 4000


def translate_text(text: str, *, source: str = "auto", target: str = "km") -> str:
    text = _clean_source(text)
    if not text:
        return ""
    chunks = _chunk_text(text, max_chars=_MAX_CHARS)
    parts = [_translate_cached(chunk, source=source, target=target) for chunk in chunks]
    return " ".join(p for p in parts if p).strip()


def translate_transcript(
    transcript: Transcript,
    *,
    source: str = "auto",
    target: str = "km",
) -> Transcript:
    # Prefer Whisper-detected language over vague "auto".
    if source in ("auto", "", None) and transcript.language:
        lang = transcript.language.lower().strip()
        if lang and lang not in ("unknown", "auto"):
            source = lang

    if not transcript.segments:
        full = translate_text(transcript.text, source=source, target=target)
        return Transcript(language=target, text=full, segments=[])

    # Merge tiny fragments so the MT model sees fuller sentences.
    groups = _group_segments(transcript.segments)
    translated_segments: list[TranscriptSegment] = []

    for group in groups:
        joined = _clean_source(" ".join(g.text for g in group))
        if not joined:
            continue

        khmer = _translate_cached(joined, source=source, target=target)
        # Keep one subtitle/TTS unit per phrase (better quality than splitting Khmer).
        translated_segments.append(
            TranscriptSegment(
                start=group[0].start,
                end=max(g.end for g in group),
                text=khmer or joined,
            )
        )
        time.sleep(0.2)

    full = " ".join(s.text for s in translated_segments if s.text).strip()
    return Transcript(language=target, text=full, segments=translated_segments)


def clear_translate_cache() -> None:
    if _CACHE_PATH.exists():
        _CACHE_PATH.unlink()


def _group_segments(segments: list[TranscriptSegment]) -> list[list[TranscriptSegment]]:
    """Merge consecutive short / tightly timed fragments into better sentences."""
    groups: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            flush()
            continue
        if not current:
            current = [seg]
            continue

        prev = current[-1]
        gap = max(0.0, seg.start - prev.end)
        prev_text = (prev.text or "").strip()
        cur_len = sum(len((s.text or "").strip()) for s in current)
        # Do not merge across finished sentences.
        if prev_text.endswith((".", "?", "!", "។", "៕", "…")):
            flush()
            current = [seg]
            continue
        if gap <= 0.55 and cur_len < 70 and len(text) < 45 and not text[:1].isupper():
            current.append(seg)
        elif gap <= 0.35 and cur_len < 40 and len(text) < 40:
            # Allow merge of tiny continuation fragments even if capitalized.
            current.append(seg)
        else:
            flush()
            current = [seg]
    flush()
    return groups


def _split_khmer_by_weights(text: str, weights: list[int]) -> list[str]:
    text = (text or "").strip()
    if not text:
        return [""] * len(weights)
    if len(weights) == 1:
        return [text]

    total = sum(weights) or 1
    tokens = re.findall(r".+?(?:[។៕!?\.]+|\s+|$)", text)
    tokens = [t for t in tokens if t.strip()]
    if len(tokens) <= 1:
        out: list[str] = []
        idx = 0
        for i, w in enumerate(weights):
            if i == len(weights) - 1:
                out.append(text[idx:].strip())
                break
            take = max(1, int(round(len(text) * (w / total))))
            out.append(text[idx : idx + take].strip())
            idx += take
        return out

    budgets = [max(1, int(round(len(tokens) * (w / total)))) for w in weights]
    while sum(budgets) > len(tokens) and any(b > 1 for b in budgets):
        for i in range(len(budgets)):
            if budgets[i] > 1 and sum(budgets) > len(tokens):
                budgets[i] -= 1
    while sum(budgets) < len(tokens):
        budgets[-1] += 1

    out = []
    cursor = 0
    for b in budgets:
        chunk = "".join(tokens[cursor : cursor + b]).strip()
        out.append(chunk)
        cursor += b
    return out


def _clean_source(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text).strip()
    return text


def _looks_like_khmer(text: str) -> bool:
    if not text:
        return False
    kh = len(_KHMER_RE.findall(text))
    letters = len(re.findall(r"[A-Za-z\u1780-\u17FF]", text))
    if letters == 0:
        return kh > 0
    return (kh / letters) >= 0.45


def _translate_cached(text: str, *, source: str, target: str) -> str:
    text = _clean_source(text)
    if not text:
        return ""
    if _looks_like_khmer(text) and not re.search(r"[A-Za-z]{3,}", text):
        return text

    cached = _cache_get(text, source, target)
    if cached is not None and _looks_like_khmer(cached):
        return cached

    out = _translate_with_fallback(text, source=source, target=target)
    if out:
        _cache_put(text, source, target, out)
    return out


def _translate_with_fallback(text: str, *, source: str, target: str) -> str:
    errors: list[str] = []
    candidates: list[str] = []

    # Prefer translators Google (fast + usually natural Khmer).
    for provider in ("google", "bing"):
        try:
            out = _call_translators(provider, text, source=source, target=target)
            if not out:
                errors.append(f"{provider}: empty")
                continue
            candidates.append(out)
            if _looks_like_khmer(out):
                return out
            errors.append(f"{provider}: weak Khmer script")
        except Exception as err:  # noqa: BLE001
            errors.append(f"{provider}: {err}")
            time.sleep(0.5)

    # Optional deep-translator pass (can be slow/rate-limited).
    try:
        out = _google_deep(text, source=source, target=target)
        if out:
            candidates.append(out)
            if _looks_like_khmer(out):
                return out
    except Exception as err:  # noqa: BLE001
        errors.append(f"google-deep: {err}")

    for out in candidates:
        if out:
            return out

    raise RuntimeError(
        "Translation failed (quality/rate-limit).\n" + "\n".join(errors[-6:])
    )


def _google_deep(text: str, *, source: str, target: str) -> str:
    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import TooManyRequests

    src = "auto" if source in ("auto", "", None) else source
    translator = GoogleTranslator(source=src, target=target)
    delay = 1.5
    last: Exception | None = None
    for _ in range(2):  # keep short — often hangs when blocked
        try:
            return (translator.translate(text) or "").strip()
        except TooManyRequests as err:
            last = err
            time.sleep(delay)
            delay *= 2
        except Exception as err:  # noqa: BLE001
            last = err
            time.sleep(delay)
    raise RuntimeError(str(last) if last else "Google deep failed")


def _call_translators(provider: str, text: str, *, source: str, target: str) -> str:
    import translators as ts

    kwargs = {
        "query_text": text,
        "translator": provider,
        "to_language": target,
        "timeout": 20,
    }
    if source not in ("auto", "", None):
        kwargs["from_language"] = source

    delay = 0.8
    last: Exception | None = None
    for _ in range(3):
        try:
            return (ts.translate_text(**kwargs) or "").strip()
        except Exception as err:  # noqa: BLE001
            last = err
            time.sleep(delay)
            delay = min(delay * 2, 6.0)
    raise RuntimeError(str(last) if last else f"{provider} failed")


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for word in text.split():
        add = len(word) + (1 if current else 0)
        if size + add > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            size = len(word)
        else:
            current.append(word)
            size += add
    if current:
        chunks.append(" ".join(current))
    return chunks


def _cache_key(text: str, source: str, target: str) -> str:
    raw = f"v2|{source}|{target}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_cache() -> dict[str, str]:
    if not _CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_cache(cache: dict[str, str]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if len(cache) > 5000:
        cache = dict(list(cache.items())[-4000:])
    _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _cache_get(text: str, source: str, target: str) -> str | None:
    return _load_cache().get(_cache_key(text, source, target))


def _cache_put(text: str, source: str, target: str, value: str) -> None:
    if not value:
        return
    cache = _load_cache()
    cache[_cache_key(text, source, target)] = value
    _save_cache(cache)
