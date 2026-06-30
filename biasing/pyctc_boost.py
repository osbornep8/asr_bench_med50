"""Method B (secondary, GATED on Milestone 0) — pyctcdecode word boosting.

ONLY available for IndicConformer, and ONLY if logit_spike.py returned GO (i.e. we
can obtain a [T, V] CTC logit array + ordered alphabet). If the spike was NO-GO,
do not import/use this — the report marks Method B "unavailable".

Beam search over CTC logits with `hotwords` (glossary surface forms) and a tunable
`hotword_weight`; optional KenLM via build_ctcdecoder(kenlm_model_path=...) with
alpha/beta. hotwords and KenLM are complementary; the default run uses hotwords only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class BoostConfig:
    hotword_weight: float = 10.0      # pyctcdecode default is ~10; sweepable via run.py
    beam_width: int = 100
    kenlm_path: Optional[str] = None  # opt-in; None = hotwords only
    alpha: float = 0.5               # KenLM LM weight (only used if kenlm_path set)
    beta: float = 1.0                # KenLM word-insertion bonus


def is_available() -> bool:
    try:
        import pyctcdecode  # noqa: F401

        return True
    except Exception:
        return False


def build_decoder(alphabet: Sequence[str], cfg: BoostConfig):
    """Construct a pyctcdecode decoder over the model's CTC alphabet.

    `alphabet` is the ordered token-string list from logit_spike, which already places
    "" at the CTC blank index (BLANK_ID) per the pyctcdecode convention — so we pass it
    through unchanged.
    """
    from pyctcdecode import build_ctcdecoder

    labels = list(alphabet)
    kwargs: dict = {}
    if cfg.kenlm_path:
        kwargs.update(kenlm_model_path=cfg.kenlm_path, alpha=cfg.alpha, beta=cfg.beta)
    return build_ctcdecoder(labels, **kwargs)


def boost(
    logits: np.ndarray,
    alphabet: Sequence[str],
    hotwords: Sequence[str],
    cfg: BoostConfig | None = None,
) -> str:
    """Decode [T, V] CTC logits with glossary hotwords. Returns the boosted transcript.

    Note: pyctcdecode operates on logits/log-probs of shape [T, V]; ensure `logits`
    is 2-D for a single clip (squeeze any batch dim before calling).
    """
    cfg = cfg or BoostConfig()
    decoder = build_decoder(alphabet, cfg)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    text = decoder.decode(
        logits,
        beam_width=cfg.beam_width,
        hotwords=[h for h in hotwords if h],
        hotword_weight=cfg.hotword_weight,
    )
    # sentencepiece word-boundary cleanup, matching the greedy decoder in logit_spike.
    return text.replace("▁", " ").strip()
