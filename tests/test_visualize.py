"""Smoke test for benchmark/visualize.py — renders all figures from synthetic data."""
import csv

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from benchmark import visualize as V  # noqa: E402

_RESULTS = [
    ["system", "condition", "wer", "cer", "term_recall", "term_recall_fuzzy",
     "term_cer", "false_insertion", "mean_latency_ms", "mean_rtf", "n_clips"],
    ["indicconformer", "raw",            "0.113", "0.031", "0.695", "0.901", "0.081", "0.000", "1059", "0.131", "50"],
    ["indicconformer", "+pyctc",         "0.133", "0.038", "0.885", "0.966", "0.033", "0.000", "1059", "0.131", "50"],
    ["indicconformer", "+llmcorrect",    "0.084", "0.025", "0.927", "0.943", "0.038", "0.000", "1059", "0.131", "50"],
    ["indicconformer", "+pyctc+llmcorrect", "0.120", "0.042", "0.930", "0.945", "0.042", "0.000", "1059", "0.131", "50"],
    ["sarvam",         "raw",            "0.105", "0.027", "0.701", "0.927", "0.064", "0.000",  "755", "0.095", "50"],
    ["sarvam",         "+llmcorrect",    "0.077", "0.027", "0.911", "0.927", "0.049", "0.000",  "755", "0.095", "50"],
    ["smallest",       "raw",            "0.115", "0.033", "0.656", "0.870", "0.081", "0.000",  "454", "0.059", "25"],
    ["smallest",       "+llmcorrect",    "0.072", "0.019", "1.000", "1.000", "0.000", "0.000",  "454", "0.059", "25"],
    ["gnani",          "raw",            "0.123", "0.040", "0.779", "0.883", "0.066", "0.000", "3226", "0.403", "50"],
    ["gnani",          "+llmcorrect",    "0.111", "0.043", "0.919", "0.919", "0.051", "0.000", "3226", "0.403", "50"],
]

_HDR = ["clip", "lang", "partition", "system", "condition", "wer", "cer",
        "term_recall", "term_recall_fuzzy", "term_cer", "false_insertion",
        "latency_ms", "audio_sec", "rtf", "n_terms", "contains_terms", "ref_text", "hyp"]

def _row(clip, lang, sys, cond, tr_fuzzy, cer="0.03", wer="0.10"):
    return [clip, lang, "term", sys, cond, wer, cer, "0.7", tr_fuzzy, "0.05",
            "—", "1000", "5.0", "0.2", "2", "a|b", "ref", "hyp"]

_CELLS = [_HDR] + [
    # indicconformer hi
    _row("hi_t", "hi", "indicconformer", "raw",         "0.90"),
    _row("hi_t", "hi", "indicconformer", "+pyctc",      "0.97"),
    _row("hi_t", "hi", "indicconformer", "+llmcorrect", "0.94"),
    # indicconformer kn
    _row("kn_t", "kn", "indicconformer", "raw",         "0.86"),
    _row("kn_t", "kn", "indicconformer", "+pyctc",      "0.95"),
    _row("kn_t", "kn", "indicconformer", "+llmcorrect", "0.93"),
    # sarvam hi
    _row("hi_t", "hi", "sarvam", "raw",         "0.93"),
    _row("hi_t", "hi", "sarvam", "+llmcorrect", "0.93"),
    # sarvam kn
    _row("kn_t", "kn", "sarvam", "raw",         "0.90"),
    _row("kn_t", "kn", "sarvam", "+llmcorrect", "0.91"),
    # smallest hi only
    _row("hi_t", "hi", "smallest", "raw",         "0.87"),
    _row("hi_t", "hi", "smallest", "+llmcorrect", "1.00"),
    # gnani hi
    _row("hi_t", "hi", "gnani", "raw",         "0.88"),
    _row("hi_t", "hi", "gnani", "+llmcorrect", "0.92"),
    # gnani kn
    _row("kn_t", "kn", "gnani", "raw",         "0.85"),
    _row("kn_t", "kn", "gnani", "+llmcorrect", "0.89"),
]


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def test_visualize_smoke(tmp_path):
    rd = tmp_path / "results"
    rd.mkdir()
    _write(rd / "results.csv", _RESULTS)
    _write(rd / "cells.csv", _CELLS)
    out = rd / "figures"

    df = V._read_results(rd / "results.csv")
    cells = V._read_cells(rd / "cells.csv")

    V.plot_term_recall(cells, out)
    V.plot_tradeoff(df, out)
    V.plot_by_language(cells, out)
    V.plot_results_table(df, out)

    assert (out / "fig1_term_recall.png").exists()
    assert (out / "fig3_recall_vs_wer.png").exists()
    assert (out / "fig4_by_language.png").exists()
    assert (out / "fig5_results_table.png").exists()
