# asr_bench — Contextual-Biasing Benchmark for Domain-Specific ASR

`asr_bench` measures whether **contextual biasing** improves automatic speech recognition
of domain-specific terms — and at what cost to general transcription accuracy. It runs a
local model and several cloud STT APIs over identical audio, applies two training-free
biasing methods, and reports a metrics table partitioned by term-bearing versus control
audio.

The repository is a **template**. It ships the harness, an example clinical Hindi/Kannada
glossary, and the reporting tooling; you supply your own audio and reference transcripts
(a short workflow is provided) and run the same pipeline to produce your own results.

## Overview

**Question.** Does contextual biasing measurably improve recognition of target terms,
across a local model and commercial APIs, without degrading general accuracy?

**Systems.** [IndicConformer-600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
(AI4Bharat; local, ONNX) and the Sarvam, Smallest.ai, Cartesia, and Gnani Prisma cloud
APIs. Each is a thin adapter behind a common `OfflineSTTAdapter` interface, so adding a
provider is one adapter file plus one registry entry.

**Dataset design.** Clips are split into *term-bearing* clips (containing glossary terms,
for recall) and *no-term* control clips (containing none, to detect over-biasing). Every
system receives byte-identical, normalized audio — the central fairness guarantee.

## Methods

Two biasing strategies, neither requiring model training:

- **Method A — LLM post-correction** (all systems). The raw transcript is corrected against
  the glossary by an LLM instructed to fix only misrecognized domain terms and otherwise
  preserve the text verbatim. The same corrector model is used for every system.
- **Method B — hotword boosting** (local model only). A `pyctcdecode` beam search over the
  model's CTC log-probabilities adds a log-domain bonus to candidates that spell a glossary
  term. It needs the decoder's raw logits, so it applies only to the local model.

Metrics, all computed on normalized text:

| Metric | What it captures |
| --- | --- |
| **WER / CER** | General accuracy; CER is primary for Indic / transliterated content. |
| **Term-recall (exact, fuzzy)** | Primary biasing metric. Fuzzy counts a term recalled if a hypothesis span is within a CER threshold (default 0.25), tolerating transliteration variants. |
| **Term-CER** | Mean character distance to each target term (lower is better). |
| **False-insertion** | Over-biasing guard, measured on the no-term partition. |
| **Latency / RTF** | Per condition. Local figures are on-device compute; API figures include network round-trip, so the two are not directly comparable. |

For the local model a `raw-ctc` baseline scores the CTC head directly, so Method B (which
operates on that head) is compared against the same head rather than the stronger RNN-T
output.

## Installation

Python 3.11, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# GPU stack for the local model — install torch from the CUDA channel matching your driver
# (e.g. CUDA 13.1 -> cu130), not from PyPI:
pip install torch==2.12.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-gpu.txt        # transformers, onnxruntime-gpu

copy .env.example .env                      # add the API keys you have
```

The IndicConformer ONNX weights are fetched automatically on first use. A cloud adapter
runs only if its key is present in `.env`; others are skipped.

## Usage

### 1. Define the domain glossary

`bias_terms/medical_bias_terms.csv` holds the target vocabulary, one concept per row
(`category,english,hindi_variant,kannada_variant`). It ships with a clinical Hindi/Kannada
glossary as an example — replace it with your own domain and languages.

### 2. Build and verify a dataset

```powershell
python data/make_dataset.py             # checks candidate sentences; --write emits data/manifest.jsonl
python data/record_clip.py --name hi_term_01 --lang hi --seconds 10
```

`make_dataset.py` ships placeholder sentences showing the format; replace them with your
own. Its verifier enforces the partition contract — *term* clips must contain at least one
glossary term, *no-term* clips none — using the same matcher as the benchmark. Record each
clip as mono 16 kHz audio into `data/clips/`.

### 3. Run the benchmark

```powershell
# gate: confirm the CTC log-probs can be extracted (enables Method B)
python -m biasing.logit_spike --audio data\clips\hi_term_01.wav --lang hi

# run the full system x condition matrix
python -m benchmark.run --manifest data\manifest.jsonl --systems all
# or a subset: --systems indicconformer,sarvam,smallest
```

For each clip and system the matrix produces `raw` and `+llmcorrect`; for the local model
it adds `raw-ctc` and, if the gate passed, `+pyctc` and `+pyctc+llmcorrect`.

### 4. Outputs

`results/` holds the aggregate table (`results.csv`, `results.md`) and the per-clip
transcripts (`transcripts.md`, `cells.csv`) — the latter are useful for confirming every
system emits the expected script, and are excluded from version control by the bundled
`.gitignore`.

## Repository layout

```
asr_bench/
  bias_terms/         medical_bias_terms.csv — the domain glossary (single source of truth)
  speech_base/        shared transcript value type
  adapters/           OfflineSTTAdapter contract + local model + cloud API adapters + registry
  biasing/            glossary, llm_postcorrect (Method A), logit_spike (gate), pyctc_boost (Method B)
  benchmark/          normalize, metrics, dataset, run (the matrix), report
  models/             local ONNX weights (auto-fetched on first run)
  data/               make_dataset.py, record_clip.py, manifest, and your clips/
  app/                optional demo UI (stub)
  tests/
```

## Tests

```powershell
pytest
```

Tests cover normalization, the metrics (including fuzzy recall), glossary parsing, the run
matrix, and reporting. They run without the GPU, model, or API dependencies.

