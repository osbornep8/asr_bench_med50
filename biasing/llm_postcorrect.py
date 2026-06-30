"""Method A (primary) — glossary-conditioned LLM correction.

Operates on TEXT, so it applies identically to every system (local model + APIs).
The corrector model MUST be the same for every system, or the comparison is unfair.

Provider-agnostic: default reads ASR_CORRECT_MODEL and routes by id. We default to
Anthropic Claude (claude-haiku-4-5 — cheap, fast, strong at instruction-following
and Indic scripts). OpenAI is supported as an alternative.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from biasing.glossary import Glossary

DEFAULT_MODEL = os.getenv("ASR_CORRECT_MODEL", "claude-haiku-4-5-20251001")

_SYSTEM_PROMPT = """\
You correct automatic speech-recognition (ASR) transcripts of Indian clinical \
conversations. You are given a transcript and a glossary of clinical terms (each \
canonical term is followed by its other known surface forms — its English name and/or \
its spelling in the other Indian script — to help you recognise a garbled token).

Rules — follow EXACTLY:
1. Correct ONLY tokens that are clearly a misrecognition of a glossary term, \
replacing them with the canonical glossary spelling.
2. Preserve everything else VERBATIM — same words, same order, same script. Do not \
paraphrase, translate, reorder, add, or remove anything else.
3. NEVER introduce a term that is not in the glossary. If unsure, leave the text \
unchanged.
4. Keep the original language/script of the transcript.
Return ONLY the corrected transcript text — no quotes, no explanation, no preamble."""


@dataclass
class CorrectionResult:
    text: str
    model: str
    changed: bool


def build_user_prompt(transcript: str, glossary: Glossary) -> str:
    return (
        f"GLOSSARY (canonical spelling  <-  English name / other-script form):\n"
        f"{glossary.prompt_block()}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"Corrected transcript:"
    )


async def correct(
    transcript: str,
    glossary: Glossary,
    model: str | None = None,
    temperature: float = 0.0,
) -> CorrectionResult:
    """Apply glossary-conditioned correction. Falls back to the raw transcript on any
    error (the benchmark must keep running)."""
    model = model or DEFAULT_MODEL
    if not transcript.strip() or not glossary.terms:
        return CorrectionResult(text=transcript, model=model, changed=False)

    user = build_user_prompt(transcript, glossary)
    try:
        if _is_anthropic(model):
            out = await _correct_anthropic(user, model, temperature)
        else:
            out = await _correct_openai(user, model, temperature)
    except Exception as e:  # keep the matrix alive; record the miss upstream if needed
        import logging

        logging.getLogger("llm_postcorrect").warning("correction failed (%s): %s", model, e)
        return CorrectionResult(text=transcript, model=model, changed=False)

    out = (out or "").strip()
    if not out:
        return CorrectionResult(text=transcript, model=model, changed=False)
    return CorrectionResult(text=out, model=model, changed=(out != transcript.strip()))


def _is_anthropic(model: str) -> bool:
    m = model.lower()
    return m.startswith("claude") or m.startswith("anthropic")


async def _correct_anthropic(user: str, model: str, temperature: float) -> str:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = await client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


async def _correct_openai(user: str, model: str, temperature: float) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""
