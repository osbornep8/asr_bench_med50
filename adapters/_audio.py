"""Shared adapter helper: turn the normalized float32 array into upload bytes.

API adapters need a file-like payload, but they MUST NOT re-resample. We just wrap
the already-16kHz-mono array into a WAV container (16-bit PCM) in memory.
"""
from __future__ import annotations

import io
import wave

import numpy as np

SR = 16000


def to_wav_bytes(wav_16k_mono: np.ndarray, sample_rate: int = SR) -> bytes:
    """Encode float32 [-1,1] mono audio as 16-bit PCM WAV bytes (no resampling)."""
    audio = np.asarray(wav_16k_mono, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()
