"""Milestone 0 (GATE) — extract CTC log-probs + alphabet from IndicConformer-600M.

IMPLEMENTED against the real model (ai4bharat/indic-conformer-600m-multilingual,
`model_onnx.py`). The model is **ONNX** (onnxruntime sessions), not a torch module:

    encoder_outputs, _ = model.encode(wav)                       # encoder.onnx
    logprobs_full = model.models['ctc_decoder'].run(            # ctc_decoder.onnx
        ['logprobs'], {'encoder_output': encoder_outputs})[0]    # [1, T, V_full]
    logprobs = logprobs_full[:, :, model.language_masks[lang]]    # slice to language
    # greedy: argmax → unique_consecutive → drop BLANK_ID(256) → map via model.vocab[lang]

So Method B's `[T, V]` matrix is `logprobs[0]` and the alphabet is `model.vocab[lang]`
(token strings in index order; index BLANK_ID is the CTC blank). No 5632/256 offset
arithmetic is needed — the model ships explicit per-language masks + vocab.

GO/NO-GO: GO if a clean [T, V] array + ordered alphabet is obtained AND a manual greedy
argmax reproduces the model's own CTC text. Else NO-GO → Method B unavailable.

Run:  python -m biasing.logit_spike --audio data/clips/sample_hi_term_1.wav --lang hi
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

BLANK_ID = 256  # config.BLANK_ID for IndicConformer-600M (blank index within a language)


@dataclass
class SpikeResult:
    ok: bool
    detail: str
    logits: Optional[np.ndarray] = None       # [T, V_lang]  (log-probs)
    alphabet: Optional[list[str]] = None       # token strings, index order; blank → ""
    greedy_text: Optional[str] = None
    model_ctc_text: Optional[str] = None


def greedy_ctc_decode(logits: np.ndarray, alphabet: list[str], blank: int = BLANK_ID) -> str:
    """Greedy CTC decode mirroring the model's own _ctc_decode: argmax per frame,
    collapse consecutive repeats, drop blank, map via alphabet, '▁' → space."""
    ids = logits.argmax(axis=-1)
    out: list[str] = []
    prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != blank:
            out.append(alphabet[i] if 0 <= i < len(alphabet) else "")
        prev = i
    return "".join(out).replace("▁", " ").strip()


def extract_logits_and_alphabet(model, wav: np.ndarray, lang: str):
    """Return (logits[T, V_lang] log-probs, alphabet:list[str]) for one clip, or (None, None).

    `model` is a loaded IndicASRModel; `wav` is float32 16kHz mono; `lang` is short ("hi").
    The blank position in the returned alphabet is set to "" so pyctcdecode treats it as blank.
    """
    try:
        import torch

        wav = np.ascontiguousarray(wav, dtype=np.float32)
        wav_t = torch.from_numpy(wav).unsqueeze(0)               # [1, S]
        encoder_outputs, _ = model.encode(wav_t)                 # encoder.onnx → numpy
        logprobs_full = model.models["ctc_decoder"].run(
            ["logprobs"], {"encoder_output": encoder_outputs}
        )[0]                                                     # [1, T, V_full]
        mask = model.language_masks[lang]                        # column indices for this language
        logprobs = torch.from_numpy(logprobs_full[:, :, mask]).log_softmax(dim=-1)
        logits = logprobs[0].cpu().numpy().astype(np.float32)    # [T, V_lang]

        alphabet = list(model.vocab[lang])                       # token strings, index order
        if len(alphabet) != logits.shape[-1]:
            return None, None                                    # vocab/mask mismatch → NO-GO
        if 0 <= BLANK_ID < len(alphabet):
            alphabet = list(alphabet)
            alphabet[BLANK_ID] = ""                              # pyctcdecode blank convention
        return logits, alphabet
    except Exception:
        return None, None


def run_spike(audio_path: str, lang: str = "hi", models_dir: str | None = None) -> SpikeResult:
    """Load the model, extract CTC log-probs, and validate greedy argmax vs the model's CTC text."""
    try:
        from transformers import AutoModel
    except Exception as e:
        return SpikeResult(ok=False, detail=f"transformers unavailable: {e}")

    from benchmark.normalize import load_audio_16k_mono

    try:
        wav = load_audio_16k_mono(audio_path)
        load_kwargs: dict = {"trust_remote_code": True}
        if models_dir:
            load_kwargs["cache_dir"] = models_dir
        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual", **load_kwargs
        ).eval()

        model_ctc_text = _call_ctc_text(model, wav, lang)
        logits, alphabet = extract_logits_and_alphabet(model, wav, lang)
        if logits is None or alphabet is None:
            return SpikeResult(
                ok=False,
                detail="could not extract [T,V] CTC log-probs + alphabet — NO-GO (skip Method B).",
                model_ctc_text=model_ctc_text,
            )

        greedy = greedy_ctc_decode(logits, alphabet)
        ok = _texts_match(greedy, model_ctc_text)
        return SpikeResult(
            ok=ok,
            detail=(
                "GO: greedy argmax over extracted log-probs reproduces the model CTC text."
                if ok else
                "NO-GO: extracted log-probs do not reproduce the model CTC text."
            ),
            logits=logits,
            alphabet=alphabet,
            greedy_text=greedy,
            model_ctc_text=model_ctc_text,
        )
    except Exception as e:
        return SpikeResult(ok=False, detail=f"spike crashed: {type(e).__name__}: {e}")


def _call_ctc_text(model, wav: np.ndarray, lang: str) -> Optional[str]:
    try:
        import torch

        wav_t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        return model(wav_t, lang, "ctc")   # forward(wav, lang, decoding="ctc")
    except Exception:
        return None


def _texts_match(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    from benchmark.normalize import normalize_text

    return normalize_text(a) == normalize_text(b)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Milestone 0 CTC-logit spike (go/no-go gate).")
    ap.add_argument("--audio", required=True, help="path to one clip")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--models-dir", default=str(Path(__file__).resolve().parents[1] / "models" / "hf_cache"))
    args = ap.parse_args()

    res = run_spike(args.audio, args.lang, args.models_dir)
    print(f"\n{'GO' if res.ok else 'NO-GO'} — {res.detail}")
    if res.model_ctc_text is not None:
        print(f"  model CTC : {res.model_ctc_text!r}")
    if res.greedy_text is not None:
        print(f"  greedy    : {res.greedy_text!r}")
    if res.logits is not None:
        print(f"  logits    : shape={res.logits.shape}  (T frames × V tokens)")
    raise SystemExit(0 if res.ok else 1)


if __name__ == "__main__":
    _main()
