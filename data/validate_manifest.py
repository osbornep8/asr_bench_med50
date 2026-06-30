"""Read-only pre-flight check of data/manifest.jsonl against the glossary + clips/.

Does NOT modify anything and does NOT run the benchmark. Verifies, per clip:
  - the audio file exists,
  - term clips: every `contains_terms` entry actually appears in `ref_text` (else term-
    recall silently misses it), and flags terms that are NOT glossary hotwords (recall-only,
    not boostable by Method A/B),
  - noterm clips: no glossary term leaked in (else false-insertion/overbias is inflated).

Run from the asr_bench/ root:  python data/validate_manifest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.dataset import load_manifest
from benchmark.metrics import _contains
from benchmark.normalize import normalize_text
from biasing.glossary import glossary_for, load_glossaries


def main() -> int:
    manifest_path = ROOT / "data" / "manifest.jsonl"
    clips = load_manifest(manifest_path)
    glossaries = load_glossaries()

    errors: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []
    n_term = n_noterm = 0
    langs: dict[str, int] = {}

    for c in clips:
        name = Path(c.audio_path).name
        langs[c.lang] = langs.get(c.lang, 0) + 1
        if not Path(c.audio_path).exists():
            errors.append(f"{name}: audio file MISSING ({c.audio_path})")

        ref = normalize_text(c.ref_text, c.lang)
        gloss_terms = {
            normalize_text(t, c.lang)
            for t in glossary_for(glossaries, c.lang).canonical_terms()
        }
        present = [t for t in gloss_terms if _contains(ref, t)]

        if c.partition == "term":
            n_term += 1
            if not c.contains_terms:
                warnings.append(f"{name}: term clip with empty contains_terms")
            for t in c.contains_terms:
                tn = normalize_text(t, c.lang)
                if not _contains(ref, tn):
                    errors.append(
                        f"{name}: contains_term {t!r} NOT found in ref_text "
                        f"→ term-recall will always miss it")
                elif tn not in gloss_terms:
                    infos.append(
                        f"{name}: {t!r} is recall-only (not a {c.lang} glossary "
                        f"hotword → measured but NOT boostable by Method A/B)")
        else:
            n_noterm += 1
            if present:
                errors.append(
                    f"{name} (noterm): glossary term(s) present → overbias inflated: {present}")

    print(f"\nmanifest: {len(clips)} clips  ({n_term} term / {n_noterm} noterm)  "
          f"langs={langs}")
    if infos:
        print(f"\nINFO — recall-only terms ({len(infos)}):")
        for m in infos:
            print(f"  · {m}")
    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for m in warnings:
            print(f"  ! {m}")
    if errors:
        print(f"\nERRORS ({len(errors)}) — fix before running:")
        for m in errors:
            print(f"  ✗ {m}")
        print("\nNOT READY.")
        return 1
    print("\n✓ READY — every clip exists, term clips contain their terms, noterm clips are clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
