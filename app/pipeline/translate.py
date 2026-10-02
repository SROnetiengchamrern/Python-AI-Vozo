"""Translate transcript text to Khmer (multi-provider + cache)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .transcribe import Transcript, TranscriptSegment

_SEG_MARK = "⟦SEG⟧"
_MAX_CHARS = 3000
_CACHE_PATH = Path.home() / ".vozo-ai" / "translate_cache.json"

# Prefer Bing first — Google free endpoint rate-limits aggressively.
_PROVIDERS = ("bing", "google", "mymemory")


def translate_text(text: str, *, source: str = "auto", target: str = "km") -> str:
    text = (text or "").strip()
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
    if not transcript.segments:
        full = translate_text(transcript.text, source=source, target=target)
        return Transcript(language=target, text=full, segments=[])

    originals = [(seg.text or "").strip() for seg in transcript.segments]
    translated_texts = _translate_segment_batch(originals, source=source, target=target)

    translated_segments = [
        TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=(khmer or original or "").strip(),
        )
        for seg, original, khmer in zip(
            transcript.segments, originals, translated_texts
        )
    ]
    full = " ".join(s.text for s in translated_segments if s.text).strip()
    return Transcript(language=target, text=full, segments=translated_segments)


def _translate_segment_batch(
    texts: list[str],
    *,
    source: str,
    target: str,
) -> list[str]:
    results = [""] * len(texts)
    pending: list[tuple[int, str]] = [(i, t) for i, t in enumerate(texts) if t]
    if not pending:
        return results

    # Resolve cache hits first so we only hit APIs for new lines.
    still: list[tuple[int, str]] = []
    for index, text in pending:
        cached = _cache_get(text, source, target)
        if cached is not None:
            results[index] = cached
        else:
            still.append((index, text))

    for batch in _pack_batches(still, max_chars=_MAX_CHARS):
        payload = f"\n{_SEG_MARK}\n".join(text for _, text in batch)
        translated = _translate_cached(payload, source=source, target=target)
        parts = [p.strip() for p in translated.split(_SEG_MARK)]

        if len(parts) != len(batch):
            for index, text in batch:
                results[index] = _translate_cached(text, source=source, target=target)
                time.sleep(0.25)
            continue

        for (index, original), part in zip(batch, parts):
            results[index] = part
            _cache_put(original, source, target, part)
        time.sleep(0.35)

    return results


def _translate_cached(text: str, *, source: str, target: str) -> str:
    cached = _cache_get(text, source, target)
    if cached is not None:
        return cached
    out = _translate_with_fallback(text, source=source, target=target)
    _cache_put(text, source, target, out)
    return out


def _translate_with_fallback(text: str, *, source: str, target: str) -> str:
    errors: list[str] = []
    for provider in _PROVIDERS:
        try:
            out = _call_provider(provider, text, source=source, target=target)
            if out:
                return out
            errors.append(f"{provider}: empty result")
        except Exception as err:  # noqa: BLE001 — try next provider
            errors.append(f"{provider}: {err}")
            time.sleep(1.0)

    # Last resort: deep-translator Google with long backoff.
    try:
        return _google_deep_retry(text, source=source, target=target)
    except Exception as err:  # noqa: BLE001
        errors.append(f"google-retry: {err}")

    raise RuntimeError(
        "All translation providers failed.\n" + "\n".join(errors[-5:])
    )


def _call_provider(provider: str, text: str, *, source: str, target: str) -> str:
    import translators as ts

    from_lang = None if source in ("auto", "", None) else source
    kwargs = {
        "query_text": text,
        "translator": provider,
        "to_language": target,
        "timeout": 30,
    }
    if from_lang:
        kwargs["from_language"] = from_lang

    # translators can be flaky on first call; retry lightly per provider.
    delay = 1.5
    last: Exception | None = None
    for _ in range(4):
        try:
            out = ts.translate_text(**kwargs)
            return (out or "").strip()
        except Exception as err:  # noqa: BLE001
            last = err
            time.sleep(delay)
            delay = min(delay * 2, 12.0)
    raise RuntimeError(str(last) if last else f"{provider} failed")


def _google_deep_retry(text: str, *, source: str, target: str) -> str:
    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import TooManyRequests

    translator = GoogleTranslator(source=source, target=target)
    delay = 3.0
    last: Exception | None = None
    for _ in range(5):
        try:
            return (translator.translate(text) or "").strip()
        except TooManyRequests as err:
            last = err
            time.sleep(delay)
            delay = min(delay * 2, 45.0)
        except Exception as err:  # noqa: BLE001
            last = err
            time.sleep(delay)
            delay = min(delay * 1.8, 30.0)
    raise RuntimeError(str(last) if last else "Google retry failed")


def _pack_batches(
    items: list[tuple[int, str]],
    *,
    max_chars: int,
) -> list[list[tuple[int, str]]]:
    batches: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    size = 0
    mark_cost = len(_SEG_MARK) + 2

    for index, text in items:
        add = len(text) + (mark_cost if current else 0)
        if current and size + add > max_chars:
            batches.append(current)
            current = [(index, text)]
            size = len(text)
        else:
            current.append((index, text))
            size += add
    if current:
        batches.append(current)
    return batches


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
    raw = f"{source}|{target}|{text}".encode("utf-8")
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
    # Keep cache bounded.
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
