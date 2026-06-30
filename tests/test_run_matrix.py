"""Regression tests for benchmark.run.run_matrix — condition generation + per-stage
latency attribution — with FAKE adapters (no model, no API, no network, no GPU).

These lock in the timing/baseline fixes so a revert is caught by CI:
  - each (system, condition) cell is produced exactly once,
  - raw uses the PRIMARY-head time and raw-ctc uses the CTC-head time (not one shared
    latency reused across conditions — the original leak),
  - +pyctc latency is the boost path (logits + beam), NOT the RNNT time,
  - +llmcorrect / +pyctc+llmcorrect ADD the (timed) Haiku stage on top of their base,
  - APIs get raw + +llmcorrect only (no raw-ctc, no +pyctc).
"""
import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from adapters.offline_base import OfflineTranscript
from benchmark import run as R
from benchmark.dataset import Clip
from biasing import pyctc_boost
from biasing.glossary import Glossary, Term

# Fixed fake stage times so assertions are deterministic (no real wall-clock for these).
IC_RNNT, IC_CTC, IC_LOGITS, API_MS = 100.0, 40.0, 25.0, 200.0
LLM_SLEEP_S = 0.03   # the ONE measured stage we exercise with a real (tiny) delay


class _FakeIC:
    """Stands in for IndicConformerLocalAdapter: RNNT text + CTC alt_text + fake CTC
    logits/alphabet (so has_logits is True) + per-stage timings."""
    name = "indicconformer"

    def __init__(self):
        self._alpha = ["", "a", "b"]                       # blank at index 0
        self._logits = np.zeros((3, len(self._alpha)), dtype=np.float32)

    async def transcribe(self, wav, language_code):
        return OfflineTranscript(
            text="rnnt बुखार", alt_text="ctc बुखार",
            language_code=language_code, system=self.name,
            ctc_logits=self._logits, ctc_alphabet=self._alpha,
            latency_ms=IC_RNNT,
            timings={"rnnt": IC_RNNT, "ctc": IC_CTC, "logits": IC_LOGITS},
        )


class _FakeAPI:
    name = "sarvam"

    async def transcribe(self, wav, language_code):
        return OfflineTranscript(
            text="api बुखार", language_code=language_code, system=self.name,
            latency_ms=API_MS, timings={"api": API_MS},
        )


@dataclass
class _FakeCorrection:
    text: str
    model: str = "fake"
    changed: bool = True


async def _fake_correct(transcript, glossary, model=None, temperature=0.0):
    await asyncio.sleep(LLM_SLEEP_S)          # so the summed Haiku stage is measurable
    return _FakeCorrection(text=transcript + " [corrected]")


def _clips():
    return [
        Clip(audio_path="clips/hi_term.wav", ref_text="मुझे बुखार है", lang="hi",
             partition="term", contains_terms=["बुखार"]),
        Clip(audio_path="clips/hi_noterm.wav", ref_text="आज मौसम अच्छा है", lang="hi",
             partition="noterm", contains_terms=[]),
    ]


@pytest.fixture
def patched(monkeypatch):
    gloss = {"hi": Glossary(lang="hi", terms=[Term(canonical="बुखार", lang="hi")])}
    monkeypatch.setattr(R, "load_manifest", lambda _p: _clips())
    monkeypatch.setattr(R, "load_glossaries", lambda _p=None: gloss)
    monkeypatch.setattr(R, "glossary_for", lambda g, lang: g["hi"])
    monkeypatch.setattr(R, "load_audio_16k_mono", lambda _p: np.zeros(16000, np.float32))
    monkeypatch.setattr(R, "build_adapters",
                        lambda names: {"indicconformer": _FakeIC(), "sarvam": _FakeAPI()})
    # Method B: avoid needing pyctcdecode; deterministic boosted text, ~0 beam time.
    monkeypatch.setattr(pyctc_boost, "is_available", lambda: True)
    monkeypatch.setattr(pyctc_boost, "boost", lambda logits, alpha, hot, cfg: "boosted बुखार")
    # Method A: avoid the Anthropic call. _llm_correct imports `correct` at call time,
    # so patching the module attribute is picked up.
    import biasing.llm_postcorrect as LP
    monkeypatch.setattr(LP, "correct", _fake_correct)


def _run():
    return asyncio.run(R.run_matrix(
        manifest_path="ignored", systems=["indicconformer", "sarvam"],
        glossary_path=None, boost_cfg=pyctc_boost.BoostConfig(), do_llm=True))


def _cell(results, system, condition, clip="hi_term.wav"):
    return next(r for r in results
               if r.system == system and r.condition == condition
               and r.clip.audio_path.endswith(clip))


def test_conditions_generated_per_system(patched):
    results = _run()
    ic = {r.condition for r in results if r.system == "indicconformer"}
    api = {r.condition for r in results if r.system == "sarvam"}
    assert ic == {"raw", "raw-ctc", "+pyctc", "+llmcorrect", "+pyctc+llmcorrect"}
    assert api == {"raw", "+llmcorrect"}          # no raw-ctc / +pyctc for a black-box API


def test_raw_and_rawctc_use_their_own_head_times(patched):
    results = _run()
    assert _cell(results, "indicconformer", "raw").latency_ms == IC_RNNT
    assert _cell(results, "indicconformer", "raw-ctc").latency_ms == IC_CTC
    assert _cell(results, "sarvam", "raw").latency_ms == API_MS


def test_pyctc_latency_is_boost_path_not_rnnt(patched):
    """The original leak: +pyctc reused the RNNT latency. It must now be logits + beam."""
    results = _run()
    pyctc = _cell(results, "indicconformer", "+pyctc").latency_ms
    assert pyctc < IC_RNNT                      # would FAIL under the old reuse-raw bug
    assert pyctc == pytest.approx(IC_LOGITS, abs=40)   # logits + tiny beam


def test_llmcorrect_adds_the_haiku_stage(patched):
    results = _run()
    sleep_ms = LLM_SLEEP_S * 1000.0
    ic_llm = _cell(results, "indicconformer", "+llmcorrect").latency_ms
    api_llm = _cell(results, "sarvam", "+llmcorrect").latency_ms
    # IC Method A sits on the CTC head (same head Method B boosts): base = CTC, NOT RNNT
    assert ic_llm >= IC_CTC + sleep_ms * 0.7
    assert ic_llm < IC_RNNT                          # would FAIL if it reverted to the RNNT base
    assert api_llm >= API_MS + sleep_ms * 0.7        # API base + measured Haiku
    # combined sits on top of the boost path, not the RNNT base
    pyctc = _cell(results, "indicconformer", "+pyctc").latency_ms
    combo = _cell(results, "indicconformer", "+pyctc+llmcorrect").latency_ms
    assert combo >= pyctc + sleep_ms * 0.7


def test_ic_methoda_corrects_the_ctc_head(patched):
    """Method A and Method B share the CTC head, so IC's +llmcorrect corrects the CTC
    text (raw-ctc), not the RNNT text — the same-head comparison."""
    results = _run()
    assert _cell(results, "indicconformer", "+llmcorrect").hyp.startswith("ctc बुखार")
    assert _cell(results, "sarvam", "+llmcorrect").hyp.startswith("api बुखार")


def test_rawctc_text_is_the_ctc_head(patched):
    results = _run()
    assert _cell(results, "indicconformer", "raw-ctc").hyp == "ctc बुखार"
    assert _cell(results, "indicconformer", "raw").hyp == "rnnt बुखार"
