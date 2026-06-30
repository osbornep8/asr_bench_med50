"""Shared transcript value type used across the offline adapters."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class STTUtterance(BaseModel):
    """One transcribed utterance: text + language code, with optional confidence/timing."""

    text: str
    language_code: str
    confidence: Optional[float] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
