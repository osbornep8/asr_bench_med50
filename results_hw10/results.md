# ASR Contextual-Biasing Benchmark — Results

Term-Recall = exact; Term-Rec(fz) = CER-tolerant (transliteration-robust); Term-CER = mean best per-term CER (lower=better). RTF = latency/audio (local = on-device compute; **API latency/RTF also includes network + their servers — not apples-to-apples**, ADR-0024).

| System | Condition | WER | CER | Term-Recall | Term-Rec(fz) | Term-CER | False-Insert | Latency(ms) | RTF | N |
|---|---|---|---|---|---|---|---|---|---|---|
| indicconformer | +pyctc | 0.133 | 0.038 | 88.5% | 96.6% | 0.033 | 0.0% | 432.4 | 0.051 | 50 |
| indicconformer | raw | 0.113 | 0.031 | 69.5% | 90.1% | 0.081 | 0.0% | 282.6 | 0.035 | 50 |
| indicconformer | raw-ctc | 0.137 | 0.039 | 67.2% | 84.4% | 0.105 | 0.0% | 63.1 | 0.008 | 50 |

## Auto-summary

- **Best by WER:** `indicconformer` `raw` — WER 0.113, CER 0.031.
- `indicconformer` `+pyctc`: fuzzy term-recall 84.4% → 96.6% (+12.2%), term-CER 0.105→0.033, false-insertion 0.0% (threshold 5%) — improved fuzzy term-recall within the overbias budget.
