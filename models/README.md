# models/ — local model storage

IndicConformer-600M weights are cached here so nothing is pulled from a global cache
at benchmark time. The adapter sets `HUGGINGFACE_HUB_CACHE` to `models/hf_cache` on
first load, so the ONNX assets land in-project automatically — no manual download step.

```
models/
  hf_cache/   # IndicConformer snapshot: config.json, model_onnx.py, *.onnx, vocab.json, ...
```

## It's an ONNX model, not a torch FP16 model

IndicConformer-600M ships as **ONNX** components (`encoder.onnx`, `ctc_decoder.onnx`,
`rnnt_decoder.onnx`, …) loaded via `onnxruntime`, which runs on the GPU through the
**CUDAExecutionProvider** automatically. There is no torch weight tensor to cast, so
`torch_dtype=float16` / `.half()` are **no-ops** — the earlier "export an FP16 copy"
idea doesn't apply, and `export_fp16.py` was removed. The 600M ONNX model already fits
and runs on the RTX 3050 Ti as-is (~0.7–1.7 s/clip). Lower precision, if ever needed,
would be an ONNX-level conversion (ORT float16) or a quantized `.onnx`, not this adapter.


