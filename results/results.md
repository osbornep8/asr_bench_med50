# ASR Contextual-Biasing Benchmark — Results

Term-Recall = exact; Term-Rec(fz) = CER-tolerant (transliteration-robust); Term-CER = mean best per-term CER (lower=better). RTF = latency/audio (local = on-device compute; **API latency/RTF also includes network + their servers — not apples-to-apples**, ADR-0024).

| System | Condition | WER | CER | Term-Recall | Term-Rec(fz) | Term-CER | False-Insert | Latency(ms) | RTF | N |
|---|---|---|---|---|---|---|---|---|---|---|
| gnani | +llmcorrect | 0.104 | 0.037 | 95.8% | 95.8% | 0.029 | 0.0% | 2448.2 | 0.305 | 50 |
| gnani | raw | 0.125 | 0.041 | 77.9% | 88.3% | 0.066 | 0.0% | 670.0 | 0.083 | 50 |
| indicconformer | +llmcorrect | 0.099 | 0.029 | 91.1% | 92.7% | 0.054 | 0.0% | 2324.9 | 0.287 | 50 |
| indicconformer | +pyctc | 0.133 | 0.038 | 88.5% | 96.6% | 0.033 | 0.0% | 335.2 | 0.039 | 50 |
| indicconformer | +pyctc+llmcorrect | 0.112 | 0.035 | 94.5% | 96.1% | 0.032 | 0.0% | 2159.2 | 0.269 | 50 |
| indicconformer | raw | 0.113 | 0.031 | 69.5% | 90.1% | 0.081 | 0.0% | 544.5 | 0.068 | 50 |
| indicconformer | raw-ctc | 0.137 | 0.039 | 67.2% | 84.4% | 0.105 | 0.0% | 365.7 | 0.046 | 50 |
| sarvam | +llmcorrect | 0.071 | 0.021 | 94.3% | 95.8% | 0.033 | 0.0% | 2757.0 | 0.345 | 50 |
| sarvam | raw | 0.105 | 0.027 | 70.1% | 92.7% | 0.064 | 0.0% | 1066.9 | 0.134 | 50 |
| smallest | +llmcorrect | 0.072 | 0.019 | 100.0% | 100.0% | 0.000 | 0.0% | 2134.6 | 0.279 | 25 |
| smallest | raw | 0.115 | 0.033 | 65.6% | 87.0% | 0.081 | 0.0% | 416.9 | 0.054 | 25 |

## Auto-summary

- **Best by WER:** `sarvam` `+llmcorrect` — WER 0.071, CER 0.021.
- `gnani` `+llmcorrect`: fuzzy term-recall 88.3% → 95.8% (+7.6%), term-CER 0.066→0.029, false-insertion 0.0% (threshold 5%) — improved fuzzy term-recall within the overbias budget.
- `indicconformer` `+llmcorrect`: fuzzy term-recall 90.1% → 92.7% (+2.6%), term-CER 0.081→0.054, false-insertion 0.0% (threshold 5%) — improved fuzzy term-recall within the overbias budget.
- `indicconformer` `+pyctc`: fuzzy term-recall 84.4% → 96.6% (+12.2%), term-CER 0.105→0.033, false-insertion 0.0% (threshold 5%) — improved fuzzy term-recall within the overbias budget.
- `indicconformer` `+pyctc+llmcorrect`: fuzzy term-recall 84.4% → 96.1% (+11.7%), term-CER 0.105→0.032, false-insertion 0.0% (threshold 5%) — improved fuzzy term-recall within the overbias budget.
- `sarvam` `+llmcorrect`: fuzzy term-recall 92.7% → 95.8% (+3.1%), term-CER 0.064→0.033, false-insertion 0.0% (threshold 5%) — improved fuzzy term-recall within the overbias budget.
- `smallest` `+llmcorrect`: fuzzy term-recall 87.0% → 100.0% (+13.0%), term-CER 0.081→0.000, false-insertion 0.0% (threshold 5%) — improved fuzzy term-recall within the overbias budget.
