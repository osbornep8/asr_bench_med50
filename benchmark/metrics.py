"""Scoring. All metrics operate on ALREADY-NORMALIZED text (benchmark/normalize.py).

Primary: WER, CER (jiwer). Biasing-specific: term_recall (term partition only),
false_insertion_rate (noterm partition only — the overbias guard). Optional tcpWER
(meeteval) for DISPLACE-M comparability on diarized refs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


# ── WER / CER ─────────────────────────────────────────────────────────────────

def wer(reference: str, hypothesis: str) -> float:
    """Word error rate on normalized text. Empty ref → 0.0 if hyp empty else 1.0."""
    import jiwer

    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return float(jiwer.wer(reference, hypothesis))


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate — more informative than WER for Indic scripts."""
    import jiwer

    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return float(jiwer.cer(reference, hypothesis))


# ── Term-level biasing metrics ────────────────────────────────────────────────

def _contains(haystack: str, needle: str) -> bool:
    """Whitespace-tolerant substring match on normalized text.

    Term-recall is about whether the *term* surfaced, not exact word alignment,
    so a normalized substring check is the right granularity for multi-word terms.
    """
    if not needle:
        return False
    return f" {needle} " in f" {haystack} "


def term_recall(hypothesis: str, reference: str, terms: Sequence[str]) -> Optional[float]:
    """Of the target terms present in the REFERENCE, how many appear in the hypothesis.

    Returns None if no target term is in the reference (clip contributes no signal),
    so callers can average only over clips that actually carry a term.
    Compute ONLY on the term partition.
    """
    present = [t for t in terms if _contains(reference, t)]
    if not present:
        return None
    hit = sum(1 for t in present if _contains(hypothesis, t))
    return hit / len(present)


def false_insertion_rate(hypothesis: str, terms: Sequence[str]) -> Optional[float]:
    """Fraction of glossary terms that appear in the hypothesis on NON-term audio.

    The overbias guard: biasing that hallucinates clinical terms into clean speech
    is harmful. Compute ONLY on the noterm partition (where ref has no target term
    by construction). Returns None if the glossary is empty.
    """
    if not terms:
        return None
    inserted = sum(1 for t in terms if _contains(hypothesis, t))
    return inserted / len(terms)


# ── Fuzzy term metrics (transliteration-robust) ───────────────────────────────
# Indic drug-name transliterations have no single canonical spelling, so an exact
# substring match under-counts valid variants (पैरासिटामोल vs पेरासिटामोल). These add a
# CER-tolerant recall and a continuous per-term CER.

DEFAULT_TERM_CER_THRESHOLD = 0.25  # a term within this normalized CER counts as recalled


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _cer_str(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(reference, hypothesis) / len(reference)


def best_term_cer(hypothesis: str, term: str) -> float:
    """Min character-error-rate between `term` and the closest n±1-word window in
    `hypothesis` (0 = exact, 1 = nothing alike). Both must already be normalized."""
    h = hypothesis.split()
    t = term.split()
    if not t or not h:
        return 1.0
    n = len(t)
    best = 1.0
    for w in sorted({max(1, n - 1), n, n + 1}):
        for i in range(0, len(h) - w + 1):
            best = min(best, _cer_str(term, " ".join(h[i:i + w])))
            if best == 0.0:
                return 0.0
    return best


def term_recalled(hypothesis: str, term: str, threshold: float = DEFAULT_TERM_CER_THRESHOLD) -> bool:
    """Recalled if the term appears exactly OR some hyp span is within `threshold` CER."""
    if _contains(hypothesis, term):
        return True
    return best_term_cer(hypothesis, term) <= threshold


def term_recall_fuzzy(
    hypothesis: str,
    reference: str,
    terms: Sequence[str],
    threshold: float = DEFAULT_TERM_CER_THRESHOLD,
) -> Optional[float]:
    """term_recall but CER-near matches count as hits (transliteration-robust)."""
    present = [t for t in terms if _contains(reference, t)]
    if not present:
        return None
    hit = sum(1 for t in present if term_recalled(hypothesis, t, threshold))
    return hit / len(present)


def mean_term_cer(hypothesis: str, reference: str, terms: Sequence[str]) -> Optional[float]:
    """Mean best-CER over target terms present in the reference (lower = better).
    A continuous 'how close did the model get to each term' diagnostic that doesn't
    penalize valid spelling variants as hard as binary recall."""
    present = [t for t in terms if _contains(reference, t)]
    if not present:
        return None
    return sum(best_term_cer(hypothesis, t) for t in present) / len(present)


# ── Optional tcpWER (multi-speaker / diarized) ────────────────────────────────

def tcpwer(reference: str, hypothesis: str) -> Optional[float]:
    """tcpWER via meeteval. Only meaningful for diarized multi-speaker refs;
    for single-speaker clips it ≈ WER. Off by default (run.py --tcpwer).
    Returns None if meeteval is unavailable.
    """
    try:
        from meeteval.wer import tcpwer as _tcp  # type: ignore
        from meeteval.io.seglst import SegLST  # type: ignore
    except Exception:
        return None
    try:
        ref = SegLST([{"session_id": "s", "speaker": "A", "words": reference}])
        hyp = SegLST([{"session_id": "s", "speaker": "A", "words": hypothesis}])
        res = _tcp(ref, hyp)
        return float(res["s"].error_rate)
    except Exception:
        return None


# ── Aggregation helper ────────────────────────────────────────────────────────

def rtf(latency_ms: Optional[float], audio_sec: Optional[float]) -> Optional[float]:
    """Real-time factor = processing_time / audio_duration. <1 means faster than real-time.
    Standard ASR efficiency metric; normalizes for clip length. NOTE: fair for the LOCAL
    model (pure on-device compute); for APIs it also includes network RTT + server queue,
    so cross-system latency/RTF is descriptive, NOT apples-to-apples."""
    if latency_ms is None or not audio_sec:
        return None
    return (latency_ms / 1000.0) / audio_sec


@dataclass
class MetricRow:
    system: str
    condition: str
    wer: float
    cer: float
    term_recall: Optional[float]            # exact substring recall
    term_recall_fuzzy: Optional[float]      # CER-tolerant recall (transliteration-robust)
    term_cer: Optional[float]               # mean best-CER per term (lower = better)
    false_insertion: Optional[float]
    mean_latency_ms: Optional[float]
    mean_rtf: Optional[float]               # real-time factor (latency/audio); see rtf()
    n_clips: int


def mean_ignore_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
