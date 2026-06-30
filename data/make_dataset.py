"""Generate + VERIFY manifest entries for self-recorded benchmark clips.

For each candidate sentence it uses the SAME normalized, word-bounded matching the
benchmark uses (benchmark.metrics._contains over glossary_for(lang).canonical_terms())
to guarantee:
  - noterm clips contain ZERO glossary terms (else overbias/false-insertion is wrong),
  - term clips contain at least one glossary term, and
  - every listed term actually appears in ref_text (else term-recall silently misses it).

Edit CANDIDATES, then:  python data/make_dataset.py        # prints manifest + warnings
                        python data/make_dataset.py --write # overwrites data/manifest.jsonl
`extra` = surface forms (e.g. drug names) to also measure recall on that are NOT in the
language glossary; they are verified to appear in ref but flagged as "not boostable".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.metrics import _contains
from benchmark.normalize import normalize_text
from biasing.glossary import glossary_for, load_glossaries

# name, lang, ref_text (native script), partition, extra (non-glossary surface forms e.g. drugs)
CANDIDATES = [
    # ── CANDIDATES: replace or extend with your own sentences.
    # Each entry: (name, lang, ref_text_in_native_script, partition, extra_terms_not_in_glossary)
    #
    # Recommended workflow:
    #   1. Use an AI to generate clinically realistic Hindi/Kannada sentences that naturally
    #      contain terms from your domain glossary (symptoms, conditions, drugs, ayurveda, etc.).
    #      Aim for ~16 term-bearing and ~9 no-term control clips per language.
    #   2. Add entries here, then run:  python data/make_dataset.py
    #      The script verifies each sentence against the glossary — noterm clips must contain
    #      zero glossary terms; term clips must contain at least one.
    #      Note: the glossary path is relative to this project's folder structure; adjust
    #      GLOSSARY_PATH in biasing/glossary.py if your layout differs.
    #   3. Record audio to data/clips/<name>.wav (use record_clip.py or any 16kHz mono recorder).
    #   4. Run:  python data/validate_manifest.py  to verify all clips before benchmarking.
    #
    # ── Illustrative placeholders only (NOT the author's recordings). Replace with
    #    your own sentences; the verifier below confirms term/noterm partitioning. ──
    # ───────────────────────── HINDI — term ─────────────────────────
    ("hi_term_01", "hi", "डॉक्टर साहब मुझे दो दिन से बुखार और खांसी हो रही है", "term", []),
    # ───────────────────────── HINDI — noterm ───────────────────────
    ("hi_noterm_01", "hi", "कल हमारे मोहल्ले में एक बड़ा संगीत कार्यक्रम होने वाला है", "noterm", []),
    # ───────────────────────── KANNADA — term ───────────────────────
    ("kn_term_01", "kn", "ನನಗೆ ನಿನ್ನೆಯಿಂದ ಜ್ವರ ಮತ್ತು ತಲೆನೋವು ಇದೆ", "term", []),
    # ───────────────────────── KANNADA — noterm ─────────────────────
    ("kn_noterm_01", "kn", "ನಾಳೆ ನಾವು ಮಾರುಕಟ್ಟೆಗೆ ಹೋಗಿ ಹಣ್ಣುಗಳನ್ನು ಖರೀದಿಸುತ್ತೇವೆ", "noterm", []),
]


def detected_terms(ref: str, lang: str, glossaries) -> list[str]:
    ref_n = normalize_text(ref, lang)
    found = [
        t for t in glossary_for(glossaries, lang).canonical_terms()
        if _contains(ref_n, normalize_text(t, lang))
    ]
    # drop a term if it is fully contained in a longer detected term (सिर ⊂ सिर दर्द)
    out = []
    for t in found:
        tn = normalize_text(t, lang)
        if not any(t2 != t and f" {tn} " in f" {normalize_text(t2, lang)} " for t2 in found):
            out.append(t)
    # de-dup, keep order
    seen: dict[str, None] = {}
    for t in out:
        seen.setdefault(t, None)
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="overwrite data/manifest.jsonl")
    args = ap.parse_args()

    glossaries = load_glossaries()
    lines: list[str] = []
    problems: list[str] = []
    not_boostable: list[str] = []

    for name, lang, ref, partition, extra in CANDIDATES:
        ref_n = normalize_text(ref, lang)
        glossary_hits = detected_terms(ref, lang, glossaries)

        # verify extras (e.g. drug names) actually appear in ref
        good_extra = []
        for e in extra:
            if _contains(ref_n, normalize_text(e, lang)):
                good_extra.append(e)
                not_boostable.append(f"{name}: {e!r} (not in {lang} glossary → recall only, no boosting)")
            else:
                problems.append(f"{name}: extra term {e!r} NOT found in ref_text")

        if partition == "noterm" and glossary_hits:
            problems.append(f"{name} (noterm) CONTAINS glossary terms: {glossary_hits}  ← rephrase")
        if partition == "term" and not glossary_hits:
            problems.append(f"{name} (term) has NO glossary terms (only extras={good_extra})")

        contains_terms = glossary_hits + good_extra
        entry = {
            "audio_path": f"clips/{name}.wav",
            "ref_text": ref,
            "lang": lang,
            "contains_terms": contains_terms,
            "partition": partition,
        }
        lines.append(json.dumps(entry, ensure_ascii=False))

    print("\n".join(lines))
    print(f"\n# {len(lines)} clips "
          f"({sum(1 for c in CANDIDATES if c[3]=='term')} term / "
          f"{sum(1 for c in CANDIDATES if c[3]=='noterm')} noterm)", file=sys.stderr)
    if not_boostable:
        print("\n# NOT BOOSTABLE (drug names absent from native glossary columns):", file=sys.stderr)
        for n in not_boostable:
            print(f"#   {n}", file=sys.stderr)
    if problems:
        print("\n# ⚠ PROBLEMS — fix before recording:", file=sys.stderr)
        for p in problems:
            print(f"#   {p}", file=sys.stderr)
    else:
        print("# ✓ all sentences verified (noterm clean, term non-empty, extras present)", file=sys.stderr)

    if args.write and not problems:
        out = Path(__file__).resolve().parent / "manifest.jsonl"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n# wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
