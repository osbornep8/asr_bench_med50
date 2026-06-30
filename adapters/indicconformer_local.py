"""IndicConformer-600M local adapter — ONNX (onnxruntime), both decoder heads.

IMPORTANT (learned from the model's `model_onnx.py`): this model is **ONNX**, not a
PyTorch module. It loads `encoder.onnx`, `ctc_decoder.onnx`, `rnnt_decoder.onnx`, etc.
via `onnxruntime` and runs on GPU through the **CUDAExecutionProvider** automatically
when CUDA is available. There is no torch weight tensor to cast, so `torch_dtype=float16`
/ `.half()` are no-ops — the earlier "FP16 to fit 4GB" framing does not apply here. The
600M ONNX model already fits and runs on the RTX 3050 Ti as-is (~0.7–1.7s/clip).
Lower precision, if ever needed, would be done at the ONNX level (ORT float16 conversion
or a quantized `.onnx`), not in this adapter.

Call signature (verified): `model(wav_tensor[1,S], lang, decoding)` with decoding in
{"rnnt","ctc"}; `lang` is the short code ("hi", "kn", "ml", ...).

Returns BOTH heads: text = RNNT (primary), alt_text = CTC. Populates ctc_logits/
ctc_alphabet from the real CTC log-probs (Milestone 0 — biasing/logit_spike.py).

Local storage caveat: the model's custom `from_pretrained` calls `snapshot_download`
WITHOUT cache_dir, so the ONNX assets land in the default HF hub cache. To force them
under asr_bench/models/, set HUGGINGFACE_HUB_CACHE before first load (we set it here).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from adapters.offline_base import OfflineTranscript

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class IndicConformerLocalAdapter:
    """OfflineSTTAdapter for the local IndicConformer-600M ONNX model."""

    name = "indicconformer"

    def __init__(self, models_dir: str | Path | None = None, try_logits: bool = True):
        self.models_dir = Path(models_dir or os.getenv("ASR_MODELS_DIR") or DEFAULT_MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hf_cache = self.models_dir / "hf_cache"
        self.try_logits = try_logits
        self._model = None
        self._cuda = False
        self._logits_failed = False  # stop retrying extraction once it fails

    # ── lifecycle ────────────────────────────────────────────────────────────
    def load(self) -> None:
        if self._model is not None:
            return
        # Keep the ONNX assets in-project (the model's from_pretrained ignores cache_dir).
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.hf_cache))
        import torch
        from transformers import AutoModel

        self._cuda = torch.cuda.is_available()  # the model self-selects CUDAExecutionProvider
        self._model = AutoModel.from_pretrained(
            MODEL_ID, trust_remote_code=True, cache_dir=str(self.hf_cache)
        ).eval()

    def warmup(self, language_code: str = "hi") -> None:
        """Pay the one-time CUDA/cuDNN/onnxruntime init (kernel JIT, cuDNN autotune,
        ORT graph capture) OUTSIDE the measured loop, so per-clip latency/RTF reflect
        STEADY-STATE on-device compute, not a cold first call. Runs all three
        paths (RNNT, CTC, logit extraction) on 1s of dummy audio. Best-effort: never
        raises. Local-only — API adapters have no client-side model warmup to exclude."""
        import torch

        self.load()
        lang = language_code.split("-")[0].lower()
        dummy = (np.random.RandomState(0).randn(16000).astype(np.float32) * 0.01)
        try:
            with torch.no_grad():
                t = torch.from_numpy(dummy).unsqueeze(0)
                self._decode(t, lang, "rnnt")
                self._decode(t, lang, "ctc")
            self._maybe_logits(dummy, lang)
        except Exception:
            pass

    # ── transcription ────────────────────────────────────────────────────────
    async def transcribe(self, wav_16k_mono: np.ndarray, language_code: str) -> OfflineTranscript:
        import torch

        self.load()
        lang = language_code.split("-")[0].lower()  # model wants "hi", not "hi-IN"
        wav = np.ascontiguousarray(wav_16k_mono, dtype=np.float32)

        # Time each decode path INDEPENDENTLY so run.py can attribute honest latency per
        # condition. RNNT and CTC are separate full forward passes; logit
        # extraction (encoder + ctc_decoder, the Method-B path) is timed on its own.
        # onnxruntime .run() is synchronous, so perf_counter brackets the real GPU work.
        tensor = torch.from_numpy(wav).unsqueeze(0)       # [1, S]; model moves to its device
        with torch.no_grad():
            t0 = time.perf_counter()
            rnnt_text = self._decode(tensor, lang, "rnnt")
            t_rnnt = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            ctc_text = self._decode(tensor, lang, "ctc")
            t_ctc = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ctc_logits, ctc_alphabet = self._maybe_logits(wav, lang)
        t_logits = (time.perf_counter() - t0) * 1000.0

        return OfflineTranscript(
            text=rnnt_text,
            alt_text=ctc_text,
            language_code=language_code,
            system=self.name,
            ctc_logits=ctc_logits,
            ctc_alphabet=ctc_alphabet,
            latency_ms=t_rnnt,                            # PRIMARY head = RNNT (text=rnnt_text)
            timings={"rnnt": t_rnnt, "ctc": t_ctc, "logits": t_logits},
            raw_payload={"runtime": "onnxruntime", "cuda": self._cuda},
        )

    def _decode(self, tensor, lang: str, mode: str) -> str:
        out = self._model(tensor, lang, mode)   # forward(wav, lang, decoding)
        if isinstance(out, str):
            return out.strip()
        if isinstance(out, (list, tuple)) and out:
            return str(out[0]).strip()
        return str(out).strip()

    def _maybe_logits(self, wav: np.ndarray, lang: str):
        """Extract the real CTC log-probs + alphabet for Method B (pyctcdecode).
        Returns (None, None) if extraction is disabled or fails."""
        if not self.try_logits or self._logits_failed:
            return None, None
        from biasing.logit_spike import extract_logits_and_alphabet

        logits, alphabet = extract_logits_and_alphabet(self._model, wav, lang)
        if logits is None or alphabet is None:
            self._logits_failed = True
            return None, None
        return logits, alphabet
