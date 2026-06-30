"""Emit results.csv + results.md from aggregated MetricRows, plus an auto-summary.

The summary names the best system/condition by WER and flags, per biasing condition,
whether term-recall improved WITHOUT pushing false-insertion past a threshold (the
overbias guard).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence

from benchmark.metrics import MetricRow

OVERBIAS_THRESHOLD = 0.05  # max tolerated false-insertion rate on the noterm partition

_COLUMNS = [
    "system", "condition", "wer", "cer",
    "term_recall", "term_recall_fuzzy", "term_cer",
    "false_insertion", "mean_latency_ms", "mean_rtf", "n_clips",
]


def write_reports(rows: Sequence[MetricRow], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out / "results.csv")
    _write_md(rows, out / "results.md")


def write_transcripts(results, out_dir: str | Path) -> None:
    """Dump every system's raw transcript per clip, so you can eyeball whether all
    systems emit the SAME script as the reference. If one romanizes (Latin) while the
    ref is Devanagari/Kannada, its WER is meaningless until transliterated to a common
    script. `results` is the list of run.CellResult."""
    from collections import defaultdict

    by_clip: dict[str, list] = defaultdict(list)
    order: list[str] = []
    for r in results:
        if r.clip.audio_path not in by_clip:
            order.append(r.clip.audio_path)
        by_clip[r.clip.audio_path].append(r)

    lines = ["# Transcripts — eyeball script consistency across systems", ""]
    for path in order:
        cells = by_clip[path]
        c0 = cells[0].clip
        lines.append(f"## `{Path(path).name}`  — {c0.lang}, {c0.partition}")
        lines.append(f"- **ref:** {c0.ref_text}")
        lines.append("")
        lines.append("| system | condition | transcript |")
        lines.append("|---|---|---|")
        for r in cells:
            lines.append(f"| {r.system} | {r.condition} | {_md_cell(r.hyp)} |")
        lines.append("")
    (Path(out_dir) / "transcripts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md_cell(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def write_cells(results, glossaries, out_dir: str | Path,
                term_cer_threshold: float = 0.25) -> None:
    """Per-(clip × system × condition) long-format CSV — one row per cell, with every
    metric + the raw hyp. This is the tidy data that powers rich visualizations
    (per-language / per-category boxplots, scatter, distributions)."""
    from benchmark import metrics as M
    from benchmark.normalize import normalize_text
    from biasing.glossary import glossary_for

    cols = ["clip", "lang", "partition", "system", "condition",
            "wer", "cer", "term_recall", "term_recall_fuzzy", "term_cer",
            "false_insertion", "latency_ms", "audio_sec", "rtf",
            "n_terms", "contains_terms", "ref_text", "hyp"]
    rows = []
    for r in results:
        ref = normalize_text(r.clip.ref_text, r.clip.lang)
        hyp = normalize_text(r.hyp, r.clip.lang)
        tr = trf = tc = fi = None
        if r.clip.partition == "term":
            terms = [normalize_text(t, r.clip.lang) for t in r.clip.contains_terms]
            tr = M.term_recall(hyp, ref, terms)
            trf = M.term_recall_fuzzy(hyp, ref, terms, term_cer_threshold)
            tc = M.mean_term_cer(hyp, ref, terms)
        else:
            gloss_terms = [normalize_text(t, r.clip.lang)
                           for t in glossary_for(glossaries, r.clip.lang).canonical_terms()]
            fi = M.false_insertion_rate(hyp, gloss_terms)
        rows.append([
            Path(r.clip.audio_path).name, r.clip.lang, r.clip.partition, r.system, r.condition,
            _f(M.wer(ref, hyp)), _f(M.cer(ref, hyp)),
            _f(tr), _f(trf), _f(tc), _f(fi),
            _f(r.latency_ms, 1), _f(r.audio_sec, 2), _f(M.rtf(r.latency_ms, r.audio_sec), 3),
            len(r.clip.contains_terms),
            "|".join(r.clip.contains_terms), r.clip.ref_text, r.hyp,
        ])
    out = Path(out_dir) / "cells.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def _write_csv(rows: Sequence[MetricRow], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_COLUMNS)
        for r in rows:
            w.writerow([
                r.system, r.condition,
                _f(r.wer), _f(r.cer),
                _f(r.term_recall), _f(r.term_recall_fuzzy), _f(r.term_cer),
                _f(r.false_insertion),
                _f(r.mean_latency_ms, 1), _f(r.mean_rtf, 3), r.n_clips,
            ])


def _write_md(rows: Sequence[MetricRow], path: Path) -> None:
    lines = ["# ASR Contextual-Biasing Benchmark — Results", ""]
    lines.append("Term-Recall = exact; Term-Rec(fz) = CER-tolerant (transliteration-robust); "
                 "Term-CER = mean best per-term CER (lower=better). RTF = latency/audio "
                 "(local = on-device compute; **API latency/RTF also includes network + their "
                 "servers — not apples-to-apples**).")
    lines.append("")
    lines.append("| System | Condition | WER | CER | Term-Recall | Term-Rec(fz) | Term-CER "
                 "| False-Insert | Latency(ms) | RTF | N |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r.system} | {r.condition} | {_f(r.wer)} | {_f(r.cer)} | "
            f"{_pct(r.term_recall)} | {_pct(r.term_recall_fuzzy)} | {_f(r.term_cer)} | "
            f"{_pct(r.false_insertion)} | {_f(r.mean_latency_ms, 1)} | {_f(r.mean_rtf, 3)} | "
            f"{r.n_clips} |"
        )
    lines.append("")
    lines.append("## Auto-summary")
    lines.append("")
    lines.extend(_summary(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(rows: Sequence[MetricRow]) -> list[str]:
    scored = [r for r in rows if r.condition not in ("raw-head",)]
    if not scored:
        return ["_No scored rows._"]

    best = min(scored, key=lambda r: r.wer)
    out = [
        f"- **Best by WER:** `{best.system}` `{best.condition}` "
        f"— WER {_f(best.wer)}, CER {_f(best.cer)}."
    ]

    # Per system, compare each biasing condition against the RIGHT baseline:
    #   +pyctc / +pyctc+llmcorrect → raw-ctc (same head Method B decodes from)
    #   everything else            → raw (the system's primary output)
    # Use FUZZY recall (transliteration-robust) as the headline, fall back to exact.
    raw_by_system = {r.system: r for r in scored if r.condition == "raw"}
    rawctc_by_system = {r.system: r for r in scored if r.condition == "raw-ctc"}
    for r in scored:
        if r.condition in ("raw", "raw-ctc"):
            continue
        base = (rawctc_by_system.get(r.system) if r.condition.startswith("+pyctc")
                else raw_by_system.get(r.system))
        rec = r.term_recall_fuzzy if r.term_recall_fuzzy is not None else r.term_recall
        base_rec = None if base is None else (
            base.term_recall_fuzzy if base.term_recall_fuzzy is not None else base.term_recall)
        if base is None or rec is None or base_rec is None:
            continue
        recall_gain = rec - base_rec
        fi = r.false_insertion if r.false_insertion is not None else 0.0
        verdict = (
            "improved fuzzy term-recall within the overbias budget"
            if recall_gain > 0 and fi <= OVERBIAS_THRESHOLD
            else "did NOT clear the overbias guard"
            if fi > OVERBIAS_THRESHOLD
            else "no fuzzy term-recall gain"
        )
        cer_note = ""
        if r.term_cer is not None and base.term_cer is not None:
            cer_note = f", term-CER {_f(base.term_cer)}→{_f(r.term_cer)}"
        out.append(
            f"- `{r.system}` `{r.condition}`: fuzzy term-recall {_pct(base_rec)} → "
            f"{_pct(rec)} ({recall_gain:+.1%}){cer_note}, false-insertion {_pct(fi)} "
            f"(threshold {OVERBIAS_THRESHOLD:.0%}) — {verdict}."
        )
    return out


def _f(x: Optional[float], ndigits: int = 3) -> str:
    return "—" if x is None else f"{x:.{ndigits}f}"


def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.1%}"
