"""Offline transcription contract.

Every adapter here takes one already-normalized clip and returns one transcript — a
deliberately simple shape for a file-based benchmark, not a streaming session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from speech_base.base import STTUtterance  # noqa: F401  (re-exported for adapters)


@dataclass
class OfflineTranscript:
    """One system's answer for one clip."""

    text: str                                  # primary transcript (native script)
    language_code: str
    system: str                                # "indicconformer", "sarvam", ...
    alt_text: Optional[str] = None             # IndicConformer: the OTHER head (CTC vs RNNT)
    ctc_logits: Optional[np.ndarray] = None    # [T, V]; ONLY IndicConformer, ONLY if spike succeeded
    ctc_alphabet: Optional[list[str]] = None   # token strings in vocab-index order, for pyctcdecode
    raw_payload: dict = field(default_factory=dict)
    latency_ms: Optional[float] = None         # PRIMARY head wall-clock (RNNT for IC; API call for APIs)
    # Per-STAGE wall-clock (ms), so run.py can sum only the stages a condition actually uses
    # IC-600M populates {"rnnt","ctc","logits"}; API adapters {"api"}. The Method-B
    # beam search and Method-A LLM call are timed in run.py (they live outside the adapter).
    timings: dict = field(default_factory=dict)

    @property
    def has_logits(self) -> bool:
        return self.ctc_logits is not None and self.ctc_alphabet is not None


@runtime_checkable
class OfflineSTTAdapter(Protocol):
    """Every system (local model or API) implements this."""

    name: str

    async def transcribe(self, wav_16k_mono: np.ndarray, language_code: str) -> OfflineTranscript:
        """Transcribe already-normalized 16kHz mono float32 audio.

        Adapters MUST NOT re-resample — fairness depends on every system seeing the
        identical array produced by benchmark/normalize.py.
        """
        ...
