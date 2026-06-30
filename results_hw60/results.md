# ASR Contextual-Biasing Benchmark — Results

Term-Recall = exact; Term-Rec(fz) = CER-tolerant (transliteration-robust); Term-CER = mean best per-term CER (lower=better). RTF = latency/audio (local = on-device compute; **API latency/RTF also includes network + their servers — not apples-to-apples**, ADR-0024).

| System | Condition | WER | CER | Term-Recall | Term-Rec(fz) | Term-CER | False-Insert | Latency(ms) | RTF | N |
|---|---|---|---|---|---|---|---|---|---|---|
| indicconformer | +pyctc | 0.242 | 0.072 | 80.7% | 87.2% | 0.074 | 0.0% | 257.6 | 0.030 | 50 |
| indicconformer | raw | 0.113 | 0.031 | 69.5% | 90.1% | 0.081 | 0.0% | 300.2 | 0.037 | 50 |
| indicconformer | raw-ctc | 0.137 | 0.039 | 67.2% | 84.4% | 0.105 | 0.0% | 64.9 | 0.008 | 50 |

## Auto-summary

- **Best by WER:** `indicconformer` `raw` — WER 0.113, CER 0.031.
- `indicconformer` `+pyctc`: fuzzy term-recall 84.4% → 87.2% (+2.9%), term-CER 0.105→0.074, false-insertion 0.0% (threshold 5%) — improved fuzzy term-recall within the overbias budget.
