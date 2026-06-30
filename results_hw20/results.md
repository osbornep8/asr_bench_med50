# ASR Contextual-Biasing Benchmark — Results

Term-Recall = exact; Term-Rec(fz) = CER-tolerant (transliteration-robust); Term-CER = mean best per-term CER (lower=better). RTF = latency/audio (local = on-device compute; **API latency/RTF also includes network + their servers — not apples-to-apples**, ADR-0024).

| System | Condition | WER | CER | Term-Recall | Term-Rec(fz) | Term-CER | False-Insert | Latency(ms) | RTF | N |
|---|---|---|---|---|---|---|---|---|---|---|
| indicconformer | +pyctc | 0.138 | 0.038 | 87.0% | 95.1% | 0.039 | 0.0% | 297.4 | 0.036 | 50 |
| indicconformer | raw | 0.113 | 0.031 | 69.5% | 90.1% | 0.081 | 0.0% | 274.1 | 0.034 | 50 |
| indicconformer | raw-ctc | 0.137 | 0.039 | 67.2% | 84.4% | 0.105 | 0.0% | 62.3 | 0.008 | 50 |

## Auto-summary

- **Best by WER:** `indicconformer` `raw` — WER 0.113, CER 0.031.
- `indicconformer` `+pyctc`: fuzzy term-recall 84.4% → 95.1% (+10.7%), term-CER 0.105→0.039, false-insertion 0.0% (threshold 5%) — improved fuzzy term-recall within the overbias budget.
