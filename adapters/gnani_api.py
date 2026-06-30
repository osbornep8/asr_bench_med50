"""Gnani.ai (Prisma v2.5 / Vachana) offline STT adapter — REST endpoint.

Gnani launched Prisma v2.5 on 2026-06-17 (trained on ~14M hrs of Indic speech, with
native code-switching). It covers Kannada, Hindi, English and Malayalam, which makes it a
useful additional vendor in this comparison.

Per the Gnani docs (docs.gnani.ai → STT REST):
- Endpoint: POST https://api.vachana.ai/stt/v3   (multipart/form-data)
- Auth header: `X-API-Key-ID: <key>`  (no "Bearer" prefix)
- Fields:
    * audio_file      (required) — WAV/MP3/OGG/FLAC/AAC/M4A
    * language_code   (required) — BCP-47: bn-IN, en-IN, gu-IN, hi-IN, kn-IN, ml-IN,
                                   mr-IN, pa-IN, ta-IN, te-IN
    * format          — "transcribe" (ITN / proper formatting) or "verbatim" (default).
                        We default to "transcribe" to match Sarvam's mode=transcribe
                        intent, for a fair native-script comparison.
    * itn_native_numerals — native-script digits when true; scoring normalizes digits
                        anyway, so left at the API default.
- Audio: ≤ 60s (ideal ≤ 30s); auto-converted to 16kHz mono (we already send 16k mono).
- Response JSON: { success, request_id, timestamp, transcript }.

Config via env: GNANI_API_KEY (required), GNANI_STT_URL / GNANI_STT_FORMAT (overrides).

There is also a legacy gRPC API (asr.gnani.ai:443, token+accesskey+certificate) and a
realtime WebSocket (wss://api.vachana.ai/). The offline benchmark uses the REST path;
the WebSocket would be the prod-app integration later (out of scope for this folder).
"""
from __future__ import annotations

import os
import time

import numpy as np

from adapters._audio import to_wav_bytes
from adapters.offline_base import OfflineTranscript

DEFAULT_URL = os.getenv("GNANI_STT_URL", "https://api.vachana.ai/stt/v3")
DEFAULT_FORMAT = os.getenv("GNANI_STT_FORMAT", "transcribe")  # "transcribe" | "verbatim"

# Prisma v2.5 REST language set (BCP-47). Includes Malayalam (ml-IN).
SUPPORTED_BASES = {"bn", "en", "gu", "hi", "kn", "ml", "mr", "pa", "ta", "te"}


class GnaniOfflineAdapter:
    name = "gnani"

    def __init__(self, api_key: str | None = None, url: str = DEFAULT_URL, fmt: str = DEFAULT_FORMAT):
        self.api_key = api_key or os.getenv("GNANI_API_KEY")
        self.url = url
        self.fmt = fmt

    async def transcribe(self, wav_16k_mono: np.ndarray, language_code: str) -> OfflineTranscript:
        if not self.api_key:
            raise RuntimeError("GNANI_API_KEY not set")
        import httpx

        wav_bytes = to_wav_bytes(wav_16k_mono)
        files = {"audio_file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"language_code": _to_bcp47(language_code), "format": self.fmt}
        headers = {"X-API-Key-ID": self.api_key}

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self.url, files=files, data=data, headers=headers)
            r.raise_for_status()
            payload = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return OfflineTranscript(
            text=(payload.get("transcript") or "").strip(),
            language_code=language_code,
            system=self.name,
            latency_ms=latency_ms,
            timings={"api": latency_ms},     # network RTT + provider compute
            raw_payload=payload if isinstance(payload, dict) else {"raw": payload},
        )


def _to_bcp47(language_code: str) -> str:
    """Map the benchmark's lang code to Gnani's BCP-47 form (hi -> hi-IN, ml -> ml-IN)."""
    lc = language_code.strip()
    if "-" in lc:
        base, region = lc.split("-", 1)
        base = base.lower()
        bcp = f"{base}-{region.upper()}"
    else:
        base = lc.lower()
        bcp = f"{base}-IN"
    if base not in SUPPORTED_BASES:
        raise RuntimeError(
            f"Gnani Prisma does not support language {base!r}. Supported: "
            f"{sorted(SUPPORTED_BASES)}. Exclude 'gnani' for this clip."
        )
    return bcp
