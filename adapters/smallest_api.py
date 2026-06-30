"""Smallest.ai (Waves Pulse) offline STT adapter — pre-recorded HTTP endpoint.

Per the Smallest docs (Research and Protocols/Smallest AI Docs/STT):
- Endpoint: POST https://api.smallest.ai/waves/v1/stt/
- Body is RAW audio bytes with Content-Type: application/octet-stream (NOT multipart).
- Knobs are QUERY params: `model` (required) and `language` (required).
    * model=pulse      → multilingual (use this for Hindi/English); raw bytes or URL.
    * model=pulse-pro  → English-only, leaderboard accuracy.
  We default to `pulse` so non-English clinical audio works.
- Auth: Authorization: Bearer <key>.
- Response JSON field is `transcription` (NOT `transcript`), plus `words`, `utterances`,
  `language`, `metadata`.

Known limitations (documented, not bugs):
- Kannada (kn) is NOT in the pre-recorded language set (26 codes: en, hi, de, es, ...).
  A Kannada clip raises a clear error and that cell is skipped.
- **Keyword boosting / VAD are Real-Time WebSocket features only** — the pre-recorded
  HTTP endpoint has no native contextual biasing. In this offline benchmark Smallest
  therefore relies on Method A (LLM correction) like the other APIs.

Config via env: SMALLEST_API_KEY (required), SMALLEST_STT_MODEL (default "pulse"),
SMALLEST_STT_URL (override endpoint).
"""
from __future__ import annotations

import os
import time

import numpy as np

from adapters._audio import to_wav_bytes
from adapters.offline_base import OfflineTranscript

DEFAULT_URL = os.getenv("SMALLEST_STT_URL", "https://api.smallest.ai/waves/v1/stt/")
DEFAULT_MODEL = os.getenv("SMALLEST_STT_MODEL", "pulse")

# Pre-recorded Pulse single-language codes (from the docs). No Kannada (kn).
SUPPORTED_LANGS = {
    "en", "hi", "de", "es", "ru", "it", "fr", "nl", "pt", "uk", "pl", "cs", "sk",
    "lv", "et", "ro", "fi", "sv", "bg", "hu", "da", "lt", "mt", "zh", "ja", "ko",
}


class SmallestOfflineAdapter:
    name = "smallest"

    def __init__(self, api_key: str | None = None, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("SMALLEST_API_KEY")
        self.url = url
        self.model = model

    async def transcribe(self, wav_16k_mono: np.ndarray, language_code: str) -> OfflineTranscript:
        if not self.api_key:
            raise RuntimeError("SMALLEST_API_KEY not set")
        lang = self._lang(language_code)
        import httpx

        wav_bytes = to_wav_bytes(wav_16k_mono)
        params = {"model": self.model, "language": lang}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/octet-stream",
        }

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(self.url, params=params, content=wav_bytes, headers=headers)
            r.raise_for_status()
            payload = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return OfflineTranscript(
            text=(payload.get("transcription") or "").strip(),
            language_code=language_code,
            system=self.name,
            latency_ms=latency_ms,
            timings={"api": latency_ms},     # network RTT + provider compute
            raw_payload=payload if isinstance(payload, dict) else {"raw": payload},
        )

    def _lang(self, language_code: str) -> str:
        base = language_code.split("-")[0].lower()
        if base not in SUPPORTED_LANGS:
            raise RuntimeError(
                f"Smallest pre-recorded STT does not support language {base!r} "
                f"(e.g. Kannada is unsupported). Supported includes: en, hi, …. "
                f"Exclude 'smallest' for this clip."
            )
        return base
