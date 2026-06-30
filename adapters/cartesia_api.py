"""Cartesia offline STT adapter (file in -> text out).

Config via env:
    CARTESIA_API_KEY        (required)
    CARTESIA_STT_URL        (override endpoint if needed)
    CARTESIA_STT_MODEL      (model id; Cartesia uses "ink-whisper" family)
    CARTESIA_VERSION        (Cartesia-Version date header; default 2025-04-16)

Mapping is defensive (transcript/text across common shapes); adjust when wiring
against the live account.
"""
from __future__ import annotations

import os
import time

import numpy as np

from adapters._audio import to_wav_bytes
from adapters.offline_base import OfflineTranscript

DEFAULT_URL = os.getenv("CARTESIA_STT_URL", "https://api.cartesia.ai/stt")
DEFAULT_MODEL = os.getenv("CARTESIA_STT_MODEL", "ink-whisper")
DEFAULT_VERSION = os.getenv("CARTESIA_VERSION", "2025-04-16")
_RESPONSE_TEXT_KEYS = ("transcript", "text", "output")


class CartesiaOfflineAdapter:
    name = "cartesia"

    def __init__(self, api_key: str | None = None, url: str = DEFAULT_URL):
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        self.url = url
        self.model = DEFAULT_MODEL
        self.version = DEFAULT_VERSION

    async def transcribe(self, wav_16k_mono: np.ndarray, language_code: str) -> OfflineTranscript:
        if not self.api_key:
            raise RuntimeError("CARTESIA_API_KEY not set")
        import httpx

        wav_bytes = to_wav_bytes(wav_16k_mono)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model, "language": language_code.split("-")[0]}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Cartesia-Version": self.version,
        }

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self.url, files=files, data=data, headers=headers)
            r.raise_for_status()
            payload = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return OfflineTranscript(
            text=_extract_text(payload).strip(),
            language_code=language_code,
            system=self.name,
            latency_ms=latency_ms,
            raw_payload=payload if isinstance(payload, dict) else {"raw": payload},
        )


def _extract_text(payload) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for k in _RESPONSE_TEXT_KEYS:
            v = payload.get(k)
            if isinstance(v, str):
                return v
    return ""
