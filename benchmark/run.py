"""Experiment-matrix orchestrator.

For each clip × each system, produce conditions:
    raw                  always
    +llmcorrect          Method A (all systems)
    +pyctc               Method B (IndicConformer only, if Milestone 0 == GO)
    +pyctc+llmcorrect    Method B then Method A (IndicConformer only, if GO)

Aggregate per (system, condition): WER, CER, term-recall (term partition),
false-insertion (noterm partition), mean latency. For IndicConformer also log the
CTC-vs-RNNT raw WER contrast (the two heads differ — itself a finding).

Normalization happens ONCE per clip here, then the identical array is handed to every
system (the core fairness guarantee).

Usage:
    python -m benchmark.run --manifest data/manifest.jsonl \
        --systems indicconformer,sarvam --hotword-weight 10 [--tcpwer]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from adapters.offline_base import OfflineTranscript
from adapters.registry import ALL_SYSTEMS, build_adapters
from benchmark import metrics as M
from benchmark.dataset import Clip, load_manifest
from benchmark.normalize import load_audio_16k_mono, normalize_text
from biasing import pyctc_boost
from biasing.glossary import Glossary, glossary_for, load_glossaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run")

RAW = "raw"                 # IndicConformer RNNT head; APIs' sole output
RAW_CTC = "raw-ctc"         # IndicConformer CTC head — the SAME-head baseline for Method B
LLM = "+llmcorrect"
PYCTC = "+pyctc"
PYCTC_LLM = "+pyctc+llmcorrect"


@dataclass
class CellResult:
    system: str
    condition: str
    clip: Clip
    hyp: str
    latency_ms: Optional[float]
    # IndicConformer extra: raw CTC head text (RNNT is the primary `hyp`)
    ctc_hyp: Optional[str] = None
    audio_sec: Optional[float] = None   # clip duration, for RTF (= latency / audio)


async def run_matrix(
    manifest_path: str,
    systems: list[str],
    glossary_path: str | None,
    boost_cfg: pyctc_boost.BoostConfig,
    do_llm: bool = True,
    corrector_model: str | None = None,
) -> list[CellResult]:
    clips = load_manifest(manifest_path)
    glossaries = load_glossaries(glossary_path)
    adapters = build_adapters(systems)
    results: list[CellResult] = []

    # Warm up local adapters so one-time GPU/runtime init (CUDA/cuDNN/ORT) is paid OUTSIDE
    # the timed loop → steady-state latency/RTF. Done once per language present.
    # API adapters expose no warmup() → skipped (no client-side model init to exclude).
    warm_langs = sorted({clip.lang for clip in clips})
    for sys_name, adapter in adapters.items():
        warm = getattr(adapter, "warmup", None)
        if callable(warm):
            for lang in warm_langs:
                try:
                    warm(lang)
                except Exception as e:
                    log.warning("warmup %s/%s failed: %s", sys_name, lang, e)

    for clip in clips:
        try:
            wav = load_audio_16k_mono(clip.audio_path)   # ← normalize ONCE
        except Exception as e:
            log.warning("skip clip %s — cannot load audio: %s", clip.audio_path, e)
            continue
        audio_sec = len(wav) / 16000.0   # normalized to 16kHz, for RTF
        gloss = glossary_for(glossaries, clip.lang)

        for sys_name, adapter in adapters.items():
            try:
                out = await adapter.transcribe(wav, clip.lang)
            except Exception as e:
                log.warning("%s failed on %s: %s", sys_name, clip.audio_path, e)
                continue

            # raw = PRIMARY head (RNNT for IC, the API's output for APIs); latency = that head.
            results.append(CellResult(sys_name, RAW, clip, out.text, out.latency_ms,
                                       ctc_hyp=out.alt_text, audio_sec=audio_sec))

            # raw-ctc = IndicConformer CTC head, scored with FULL metrics so Method B is judged
            # against the SAME head it decodes from, not the better RNNT head.
            if sys_name == "indicconformer" and out.alt_text is not None:
                results.append(CellResult(sys_name, RAW_CTC, clip, out.alt_text,
                                          out.timings.get("ctc"), audio_sec=audio_sec))

            # +llmcorrect = Method A on the system's BIASING baseline + the (timed) Haiku call.
            # For IndicConformer that baseline is the CTC head (the same head Method B boosts),
            # so Method A vs Method B is a same-head comparison from the shared raw-ctc baseline.
            # raw (RNNT) stays as the model's best unbiased reference. APIs expose a
            # single output, so Method A sits on that.
            if do_llm:
                if sys_name == "indicconformer" and out.alt_text is not None:
                    a_text, a_base = out.alt_text, (out.timings.get("ctc") or 0.0)
                else:
                    a_text, a_base = out.text, (out.latency_ms or 0.0)
                corrected, t_llm = await _llm_correct(a_text, gloss, corrector_model)
                results.append(CellResult(sys_name, LLM, clip, corrected, a_base + t_llm,
                                          audio_sec=audio_sec))

            # Method B + B→A, IndicConformer only, only if logits are present (GO).
            # Honest Method-B latency = logit extraction (encode + ctc_decoder) + beam search;
            # it does NOT reuse the RNNT-head time.
            if sys_name == "indicconformer" and out.has_logits:
                boosted, t_beam = _pyctc(out, gloss, boost_cfg)
                t_boost_path = (out.timings.get("logits") or 0.0) + t_beam
                results.append(CellResult(sys_name, PYCTC, clip, boosted, t_boost_path,
                                          audio_sec=audio_sec))
                if do_llm:
                    boosted_corrected, t_llm2 = await _llm_correct(boosted, gloss, corrector_model)
                    results.append(
                        CellResult(sys_name, PYCTC_LLM, clip, boosted_corrected,
                                   t_boost_path + t_llm2, audio_sec=audio_sec)
                    )
    return results


async def _llm_correct(text: str, gloss: Glossary, model: str | None) -> tuple[str, float]:
    """Return (corrected_text, elapsed_ms). The elapsed time is the Method-A stage cost,
    summed into the condition's latency by the caller."""
    from biasing.llm_postcorrect import correct

    t0 = time.perf_counter()
    res = await correct(text, gloss, model=model)
    return res.text, (time.perf_counter() - t0) * 1000.0


def _pyctc(out: OfflineTranscript, gloss: Glossary,
           cfg: pyctc_boost.BoostConfig) -> tuple[str, float]:
    """Return (boosted_text, beam_elapsed_ms). Times ONLY the pyctcdecode beam search;
    the logit-extraction cost is read from out.timings['logits'] by the caller."""
    if not pyctc_boost.is_available():
        return out.text, 0.0
    t0 = time.perf_counter()
    text = pyctc_boost.boost(out.ctc_logits, out.ctc_alphabet, gloss.hotwords(), cfg)
    return text, (time.perf_counter() - t0) * 1000.0


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(
    results: list[CellResult],
    glossaries: dict[str, Glossary],
    term_cer_threshold: float = M.DEFAULT_TERM_CER_THRESHOLD,
    use_tcpwer: bool = False,
) -> list[M.MetricRow]:
    grouped: dict[tuple[str, str], list[CellResult]] = defaultdict(list)
    for r in results:
        grouped[(r.system, r.condition)].append(r)

    rows: list[M.MetricRow] = []
    for (system, condition), cells in sorted(grouped.items()):
        wers, cers, recalls, recalls_fuzzy, term_cers, insertions, latencies, rtfs = \
            [], [], [], [], [], [], [], []
        for c in cells:
            ref = normalize_text(c.clip.ref_text, c.clip.lang)
            hyp = normalize_text(c.hyp, c.clip.lang)
            wers.append(M.wer(ref, hyp))
            cers.append(M.cer(ref, hyp))
            if c.clip.partition == "term":
                # recall: only the terms KNOWN to be in this clip's reference
                terms = [normalize_text(t, c.clip.lang) for t in c.clip.contains_terms]
                recalls.append(M.term_recall(hyp, ref, terms))
                recalls_fuzzy.append(M.term_recall_fuzzy(hyp, ref, terms, term_cer_threshold))
                term_cers.append(M.mean_term_cer(hyp, ref, terms))
            else:  # noterm partition → overbias guard: did ANY glossary term leak in?
                gloss_terms = [
                    normalize_text(t, c.clip.lang)
                    for t in glossary_for(glossaries, c.clip.lang).canonical_terms()
                ]
                insertions.append(M.false_insertion_rate(hyp, gloss_terms))
            if c.latency_ms is not None:
                latencies.append(c.latency_ms)
            rtfs.append(M.rtf(c.latency_ms, c.audio_sec))
        rows.append(
            M.MetricRow(
                system=system,
                condition=condition,
                wer=_mean(wers),
                cer=_mean(cers),
                term_recall=M.mean_ignore_none(recalls),
                term_recall_fuzzy=M.mean_ignore_none(recalls_fuzzy),
                term_cer=M.mean_ignore_none(term_cers),
                false_insertion=M.mean_ignore_none(insertions),
                mean_latency_ms=M.mean_ignore_none(latencies),
                mean_rtf=M.mean_ignore_none(rtfs),
                n_clips=len(cells),
            )
        )
    return rows


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_systems(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(ALL_SYSTEMS)
    names = [s.strip().lower() for s in raw.split(",") if s.strip()]
    bad = [n for n in names if n not in ALL_SYSTEMS]
    if bad:
        raise SystemExit(f"unknown system(s): {bad}; choose from {list(ALL_SYSTEMS)} or 'all'")
    return names


def main() -> None:
    from envload import load_env

    load_env()
    ap = argparse.ArgumentParser(description="ASR contextual-biasing benchmark runner.")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--systems", default="all", help="comma list or 'all'")
    ap.add_argument("--glossary", default=None, help="override glossary CSV path")
    ap.add_argument("--no-llm", action="store_true", help="skip Method A (LLM correction)")
    ap.add_argument("--corrector-model", default=None, help="ASR_CORRECT_MODEL override")
    ap.add_argument("--hotword-weight", type=float, default=10.0)
    ap.add_argument("--beam-width", type=int, default=100)
    ap.add_argument("--kenlm", default=None, help="optional KenLM .arpa/.bin path (Method B)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--tcpwer", action="store_true", help="also compute tcpWER (diarized refs)")
    ap.add_argument("--term-cer-threshold", type=float, default=M.DEFAULT_TERM_CER_THRESHOLD,
                    help="max per-term CER to count a term as recalled (fuzzy recall)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    boost_cfg = pyctc_boost.BoostConfig(
        hotword_weight=args.hotword_weight,
        beam_width=args.beam_width,
        kenlm_path=args.kenlm,
        alpha=args.alpha,
        beta=args.beta,
    )
    systems = _parse_systems(args.systems)

    results = asyncio.run(
        run_matrix(
            manifest_path=args.manifest,
            systems=systems,
            glossary_path=args.glossary,
            boost_cfg=boost_cfg,
            do_llm=not args.no_llm,
            corrector_model=args.corrector_model,
        )
    )
    glossaries = load_glossaries(args.glossary)
    rows = aggregate(results, glossaries,
                     term_cer_threshold=args.term_cer_threshold, use_tcpwer=args.tcpwer)

    from benchmark.report import write_reports, write_transcripts, write_cells

    out_dir = Path(args.out_dir)
    write_reports(rows, out_dir)
    write_transcripts(results, out_dir)
    write_cells(results, glossaries, out_dir, term_cer_threshold=args.term_cer_threshold)
    log.info("wrote %s/{results.csv, results.md, transcripts.md, cells.csv}", out_dir)


if __name__ == "__main__":
    main()
