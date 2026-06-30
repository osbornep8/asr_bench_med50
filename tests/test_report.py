"""End-to-end smoke test of the aggregate → report pipeline, with NO audio/API/model.

Builds fake CellResults (incl. the CTC-raw baseline condition), runs aggregate() +
write_reports() + write_cells(), and asserts the outputs exist and carry the expected
columns (incl. the RTF / per-cell fields). Guards the whole reporting path so adding
metrics can't silently break the benchmark.
"""
import csv

import pytest

from benchmark.dataset import Clip
from benchmark.report import write_cells, write_reports
from benchmark.run import CellResult, aggregate
from biasing.glossary import Glossary, Term


def _clip(name, lang, partition, ref, terms):
    return Clip(audio_path=f"clips/{name}.wav", ref_text=ref, lang=lang,
                partition=partition, contains_terms=terms)


def _glossaries():
    return {"hi": Glossary(lang="hi", terms=[
        Term(canonical="बुखार", lang="hi"), Term(canonical="खांसी", lang="hi")])}


def test_aggregate_report_cells_smoke(tmp_path):
    pytest.importorskip("jiwer")  # aggregate computes WER/CER
    gloss = _glossaries()
    term = _clip("hi_term", "hi", "term", "मुझे बुखार और खांसी है", ["बुखार", "खांसी"])
    noterm = _clip("hi_noterm", "hi", "noterm", "आज मौसम बहुत अच्छा है", [])
    results = [
        CellResult("indicconformer", "raw", term, "मुझे बुखार और खांसी है",
                   750.0, ctc_hyp="मुझे बुखार है", audio_sec=5.0),
        # raw-ctc = the CTC head, scored as its own condition (the Method-B baseline)
        CellResult("indicconformer", "raw-ctc", term, "मुझे बुखार है",
                   600.0, audio_sec=5.0),
        CellResult("indicconformer", "raw", noterm, "आज मौसम बहुत अच्छा है",
                   700.0, audio_sec=4.0),
        CellResult("sarvam", "raw", term, "मुझे बुखार है", 800.0, audio_sec=5.0),
    ]

    rows = aggregate(results, gloss)
    write_reports(rows, tmp_path)
    write_cells(results, gloss, tmp_path)

    for fn in ("results.csv", "results.md", "cells.csv"):
        assert (tmp_path / fn).exists(), f"{fn} not written"

    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "RTF" in md and "Term-Rec(fz)" in md

    # the CTC-raw condition is aggregated as a first-class scored row
    rctc = next(r for r in rows if r.system == "indicconformer" and r.condition == "raw-ctc")
    assert rctc.term_recall == 0.5  # caught बुखार, missed खांसी

    cells = list(csv.DictReader((tmp_path / "cells.csv").open(encoding="utf-8")))
    assert {"rtf", "audio_sec", "latency_ms", "term_recall_fuzzy"} <= set(cells[0])

    # indicconformer term raw: 750ms / 5s = 0.150 RTF, and it caught both terms
    row = next(c for c in cells if c["system"] == "indicconformer" and c["partition"] == "term")
    assert abs(float(row["rtf"]) - 0.150) < 1e-6
    assert row["term_recall"] == "1.000"

    # noterm cell carries a false_insertion number (0.0 — no glossary leak), not "—"
    nt = next(c for c in cells if c["partition"] == "noterm")
    assert nt["false_insertion"] == "0.000"
