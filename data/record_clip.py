"""Record a short test clip from your mic and (optionally) append a manifest line.

Records mono 16kHz WAV straight into data/clips/, so it's already in the benchmark's
canonical format. Then prompts for the reference transcript + which glossary terms the
clip contains + the partition, and appends one JSON line to data/manifest.jsonl.

Keep clips SHORT — the Sarvam REST endpoint accepts ≤ 30s, so aim for 8–20s.

Requires sounddevice (`pip install sounddevice`) + soundfile (already in requirements).

Examples
--------
# record 12s of Hindi, then answer the prompts to add it to the manifest
python data/record_clip.py --seconds 12 --lang hi --name sample_hi_term

# just record, don't touch the manifest
python data/record_clip.py --seconds 10 --lang hi --name scratch --no-manifest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SR = 16000
DATA_DIR = Path(__file__).resolve().parent
CLIPS_DIR = DATA_DIR / "clips"
MANIFEST = DATA_DIR / "manifest.jsonl"


def record(seconds: float) -> "np.ndarray":  # noqa: F821
    try:
        import sounddevice as sd
    except Exception as e:
        raise SystemExit("sounddevice not installed. Run: pip install sounddevice") from e
    import numpy as np

    print(f"● recording {seconds:.0f}s at {SR}Hz mono — speak now…")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    print("■ done.")
    return np.squeeze(audio)


def save_wav(audio, path: Path) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SR, subtype="PCM_16")
    print(f"saved → {path}")


def append_manifest(rel_audio: str, lang: str) -> None:
    print("\n— manifest entry — (Enter to skip a field)")
    ref_text = input("reference transcript (what was actually said): ").strip()
    terms_raw = input("glossary terms present, comma-separated (blank if none): ").strip()
    contains_terms = [t.strip() for t in terms_raw.split(",") if t.strip()]
    partition = "term" if contains_terms else "noterm"
    override = input(f"partition [{partition}]: ").strip()
    if override in ("term", "noterm"):
        partition = override

    entry = {
        "audio_path": rel_audio,
        "ref_text": ref_text,
        "lang": lang,
        "contains_terms": contains_terms,
        "partition": partition,
    }
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"appended to {MANIFEST.name}: {json.dumps(entry, ensure_ascii=False)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Record a test clip + add it to the manifest.")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--lang", default="hi", help="hi / kn / en")
    ap.add_argument("--name", required=True, help="clip filename stem, e.g. sample_hi_term")
    ap.add_argument("--no-manifest", action="store_true", help="record only, skip manifest prompt")
    args = ap.parse_args()

    out = CLIPS_DIR / f"{args.name}.wav"
    audio = record(args.seconds)
    save_wav(audio, out)
    if not args.no_manifest:
        append_manifest(f"clips/{out.name}", args.lang)


if __name__ == "__main__":
    main()
