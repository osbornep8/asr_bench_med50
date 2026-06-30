"""Sarvam offline STT adapter — synchronous REST endpoint (file in -> text out).

Per the Sarvam docs (Research and Protocols/Sarvam AI API Docs/STT):
- Endpoint: POST https://api.sarvam.ai/speech-to-text  (multipart/form-data)
- Auth header: `api-subscription-key`.
- Form fields: file, model, mode, language_code, input_audio_codec.
    * model=saaras:v3  → supports `mode` (transcribe/translate/verbatim/translit/codemix).
      We use mode=transcribe → native-script output (matches the local IndicConformer's
      native-script output, for a fair comparison).
    * model=saarika:v2.5 is the default transcription model; `mode` is ignored for it.
- language_code is BCP-47 (e.g. hi-IN, kn-IN, en-IN) or "unknown" to auto-detect.
- Response JSON: { request_id, transcript, language_code, language_probability, ... }.

IMPORTANT: the sync REST path accepts audio **≤ 30 seconds**. Keep benchmark clips short
(longer files need the async Batch job API, which is out of scope here). The batch API
also adds diarization/timestamps we don't need.

Config via env: SARVAM_API_KEY (required), SARVAM_STT_MODEL (default saaras:v3).
"""
from __future__ import annotations

import os
import time

import numpy as np

from adapters._audio import to_wav_bytes
from adapters.offline_base import OfflineTranscript

DEFAULT_URL = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")
DEFAULT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")

# saaras:v3 supports all 23 languages; saarika:v2.5 supports the first 12.
_KNOWN_LANGS = {
    "hi", "bn", "kn", "ml", "mr", "od", "or", "pa", "ta", "te", "en", "gu",
    "as", "ur", "ne", "kok", "ks", "sd", "sa", "sat", "mni", "brx", "mai", "doi",
}


class SarvamOfflineAdapter:
    name = "sarvam"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        self.model = model
        self.url = url

    async def transcribe(self, wav_16k_mono: np.ndarray, language_code: str) -> OfflineTranscript:
        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY not set")
        import httpx

        wav_bytes = to_wav_bytes(wav_16k_mono)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": self.model,
            "mode": "transcribe",                 # only applies to saaras:v3; native script
            "language_code": _to_bcp47(language_code),
        }
        headers = {"api-subscription-key": self.api_key}

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self.url, files=files, data=data, headers=headers)
            r.raise_for_status()
            payload = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return OfflineTranscript(
            text=(payload.get("transcript") or "").strip(),
            language_code=payload.get("language_code") or language_code,
            system=self.name,
            latency_ms=latency_ms,
            timings={"api": latency_ms},     # network RTT + provider compute
            raw_payload=payload if isinstance(payload, dict) else {"raw": payload},
        )


def _to_bcp47(language_code: str) -> str:
    """Map the benchmark's lang code to Sarvam's BCP-47 form (hi -> hi-IN, kn -> kn-IN).
    Pass through 'unknown' and already-regioned codes (hi-IN stays hi-IN)."""
    lc = language_code.strip()
    if lc.lower() == "unknown":
        return "unknown"
    if "-" in lc:
        base, region = lc.split("-", 1)
        return f"{base.lower()}-{region.upper()}"
    base = lc.lower()
    if base not in _KNOWN_LANGS:
        return "unknown"
    return f"{base}-IN"
