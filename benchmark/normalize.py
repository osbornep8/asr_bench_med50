"""Centralized normalization — the single most important fairness guarantee.

Audio is decoded → mono → 16kHz float32 EXACTLY ONCE per clip, here, so every
system (local model + all APIs) sees byte-identical input. No adapter may
re-resample. Text/script normalization is applied to BOTH hypothesis and
reference before any metric is computed.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np

TARGET_SR = 16000

# Devanagari combining marks / digits handled below. We keep this dependency-light:
# indic-nlp-library is used when present, else we fall back to NFC + manual rules.
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory  # type: ignore

    _INDIC_FACTORY: Optional["IndicNormalizerFactory"] = IndicNormalizerFactory()
except Exception:  # pragma: no cover - optional dep
    _INDIC_FACTORY = None


# ── Audio ─────────────────────────────────────────────────────────────────────

def load_audio_16k_mono(source: str | Path | "io.IOBase") -> np.ndarray:
    """Decode any common audio source to mono 16kHz float32 in [-1, 1].

    `source` may be a path (str/Path) or a binary file-like object (e.g. the
    BytesIO of an HTTP upload). Tries soundfile (fast, wav/flac), then librosa
    (mp3/m4a/etc. via audioread/ffmpeg). Returns a 1-D float32 array. This is the
    ONLY place resampling happens.
    """
    src = str(source) if isinstance(source, (str, Path)) else source
    audio, sr = _decode(src)
    audio = _to_mono(audio)
    if sr != TARGET_SR:
        audio = _resample(audio, sr, TARGET_SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _decode(src) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sr = sf.read(src, dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sr)
    except Exception:
        import librosa

        if hasattr(src, "seek"):
            src.seek(0)
        audio, sr = librosa.load(src, sr=None, mono=False)
        return np.asarray(audio, dtype=np.float32), int(sr)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    # soundfile gives [frames, channels]; librosa gives [channels, frames].
    axis = 1 if audio.shape[0] >= audio.shape[1] else 0
    return audio.mean(axis=axis)


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    try:
        import librosa

        return librosa.resample(audio, orig_sr=sr_in, target_sr=sr_out)
    except Exception:
        # Linear-interpolation fallback so the harness never hard-fails on resample.
        duration = audio.shape[0] / sr_in
        n_out = int(round(duration * sr_out))
        x_old = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


# ── Text / script ───────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")

# Zero-width joiners/non-joiners: delete (not space) so conjuncts aren't split.
_ZERO_WIDTH = dict.fromkeys((0x200C, 0x200D, 0xFEFF), None)

# Devanagari digits → ASCII, so "५०० mg" and "500 mg" score identically.
_DEVANAGARI_DIGITS = {ord(c): str(i) for i, c in enumerate("०१२३४५६७८९")}


def _strip_punct_keep_marks(s: str) -> str:
    """Replace punctuation/symbols with a space, but KEEP letters (L), combining
    marks (M — e.g. Kannada/Malayalam/Devanagari vowel signs & virama), and numbers (N).

    The previous regex `[^\\w\\sऀ-ॿ]` kept only Devanagari marks (`ऀ-ॿ`); for any other
    Indic script the vowel-sign marks (Unicode category M*) are NOT `\\w`, so they were
    stripped — silently corrupting Kannada/Malayalam text and its WER/term matching.
    A Unicode-category filter fixes every Indic script at once (and still drops the
    Devanagari danda ।/॥, which are punctuation, category Po).
    """
    out = []
    for ch in s:
        if ch.isspace():
            out.append(" ")
            continue
        if unicodedata.category(ch)[0] in ("L", "M", "N"):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def normalize_text(text: str, lang: str = "hi") -> str:
    """Normalize hypothesis OR reference before scoring.

    Pipeline: NFC → indic normalization (if available) → lowercase Latin →
    strip punctuation → normalize digits → collapse whitespace. Applied
    identically to hyp and ref so the comparison stays fair.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", text)
    if _INDIC_FACTORY is not None:
        s = _indic_normalize(s, lang)
    s = s.lower()                       # only affects Latin; Indic scripts are caseless
    s = s.translate(_DEVANAGARI_DIGITS)
    s = s.translate(_ZERO_WIDTH)        # drop ZWJ/ZWNJ/BOM before category filtering
    s = _strip_punct_keep_marks(s)      # keep letters+marks+digits for ALL scripts
    s = _WS.sub(" ", s).strip()
    return s


def _indic_normalize(s: str, lang: str) -> str:
    try:
        code = _ISO_TO_INDICNLP.get(lang.split("-")[0].lower(), "hi")
        normalizer = _INDIC_FACTORY.get_normalizer(code)
        return normalizer.normalize(s)
    except Exception:  # pragma: no cover - defensive
        return s


_ISO_TO_INDICNLP = {
    "hi": "hi", "mr": "mr", "bn": "bn", "ta": "ta", "te": "te",
    "kn": "kn", "ml": "ml", "gu": "gu", "pa": "pa", "or": "or", "as": "as",
}
