"""Glossary loading for the biasing methods.

Single source of truth is the in-repo glossary:
    bias_terms/medical_bias_terms.csv

Real schema (header row):
    category,english,hindi_variant,kannada_variant
e.g.
    drugs,Paracetamol,,
    symptoms,fever,बुखार,ಜ್ವರ

Quirk handled here: in the source file every row is wrapped in an extra pair of
double quotes (`"symptoms,fever,बुखार,ಜ್ವರ"`), which makes a naive csv/pandas read
collapse the whole row into one column. `load_glossaries` strips that outer wrapper
if present, and still parses a normal (unwrapped) CSV correctly.

We expose ONE concept per row across languages, and build per-language Glossary
objects whose surface forms are the script the ASR actually emits for that language:
    en  -> english
    hi  -> hindi_variant   (english kept as a cross-lingual hint for the corrector)
    kn  -> kannada_variant (english kept as a cross-lingual hint)

These feed both the LLM corrector (Method A) and pyctcdecode hotwords (Method B).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

# In-repo glossary (self-contained so the benchmark runs as a standalone checkout).
DEFAULT_GLOSSARY = (
    Path(__file__).resolve().parents[1] / "bias_terms" / "medical_bias_terms.csv"
)

# Which CSV column supplies the surface form for each language code.
_LANG_COLUMN = {"en": "english", "hi": "hindi_variant", "kn": "kannada_variant"}


@dataclass
class Term:
    canonical: str              # surface form in this glossary's language (what ASR should emit)
    lang: str
    category: str = ""
    variants: list[str] = field(default_factory=list)  # cross-lingual / spelling hints

    @property
    def all_surfaces(self) -> list[str]:
        """Canonical + variants, de-duplicated, order-preserving."""
        seen: dict[str, None] = {}
        for s in [self.canonical, *self.variants]:
            s = (s or "").strip()
            if s and s not in seen:
                seen[s] = None
        return list(seen.keys())


@dataclass
class Glossary:
    lang: str
    terms: list[Term] = field(default_factory=list)

    def canonical_terms(self) -> list[str]:
        return [t.canonical for t in self.terms]

    def by_category(self, category: str) -> list[Term]:
        return [t for t in self.terms if t.category == category]

    def hotwords(self) -> list[str]:
        """Flat list of surface forms for pyctcdecode hotwords (Method B).

        For a single-language ASR pass we bias toward the canonical surface in that
        language only (not the cross-lingual english hint), to avoid injecting Latin
        tokens into a Devanagari hypothesis.
        """
        return [t.canonical for t in self.terms if t.canonical]

    def prompt_block(self) -> str:
        """'canonical  <-  hint, hint' lines for the LLM corrector (Method A)."""
        lines = []
        for t in self.terms:
            if t.variants:
                lines.append(f"{t.canonical}  <-  {', '.join(t.variants)}")
            else:
                lines.append(t.canonical)
        return "\n".join(lines)


def load_glossaries(path: str | Path | None = None) -> dict[str, Glossary]:
    """Load medical_bias_terms.csv into {lang: Glossary} for langs en/hi/kn.

    Rows contribute to a language's glossary only when that language's column is
    non-empty (e.g. 'drugs,Paracetamol,,' contributes to en only).
    """
    path = Path(path) if path is not None else DEFAULT_GLOSSARY
    rows = _read_rows(path)
    if not rows:
        return {}

    header, *data_rows = rows
    idx = _column_index(header, path)

    glossaries: dict[str, Glossary] = {lang: Glossary(lang=lang) for lang in _LANG_COLUMN}
    for row in data_rows:
        category = _cell(row, idx["category"])
        english = _cell(row, idx["english"])
        hindi = _cell(row, idx["hindi_variant"])
        kannada = _cell(row, idx["kannada_variant"])
        if not english and not hindi and not kannada:
            continue
        # Cross-reference each concept's OTHER known surface forms as variants:
        # the English name AND the spelling in the other Indic script. These feed the
        # Method-A corrector prompt so it can recognise a garbled token as a known concept
        # (the earlier code passed English ONLY — a translation, not the in-script anchor).
        # Method B is unaffected: hotwords() still uses canonical surface forms only.
        if english:
            glossaries["en"].terms.append(
                Term(canonical=english, lang="en", category=category,
                     variants=[v for v in (hindi, kannada) if v])
            )
        if hindi:
            glossaries["hi"].terms.append(
                Term(canonical=hindi, lang="hi", category=category,
                     variants=[v for v in (english, kannada) if v])
            )
        if kannada:
            glossaries["kn"].terms.append(
                Term(canonical=kannada, lang="kn", category=category,
                     variants=[v for v in (english, hindi) if v])
            )
    return {lang: g for lang, g in glossaries.items() if g.terms}


def glossary_for(by_lang: dict[str, Glossary], lang: str) -> Glossary:
    """Get the glossary for a BCP-47-ish code (e.g. 'hi-IN' -> 'hi'), falling back to
    a merged glossary across all languages if the language is unknown."""
    base = lang.split("-")[0].lower()
    if lang in by_lang:
        return by_lang[lang]
    if base in by_lang:
        return by_lang[base]
    merged = Glossary(lang=lang)
    for g in by_lang.values():
        merged.terms.extend(g.terms)
    return merged


# ── CSV reading (handles the wrapped-quote quirk) ─────────────────────────────

def _read_rows(path: Path) -> list[list[str]]:
    """Read the CSV into rows, stripping a wrapping pair of double quotes per line
    if present. Works for both the quirky source file and a normal CSV.

    utf-8-sig strips a leading BOM (the source file has one), so the wrapped-quote
    stripping below sees the real first character.
    """
    raw_lines = path.read_text(encoding="utf-8-sig").splitlines()
    cleaned: list[str] = []
    for line in raw_lines:
        if not line.strip():
            continue
        cleaned.append(_dequote_line(line))
    return list(csv.reader(cleaned))


def _dequote_line(line: str) -> str:
    s = line.rstrip("\r")
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        inner = s[1:-1]
        # Only strip if the inner text isn't itself a multi-field quoted CSV record
        # (the source file never escapes inner quotes, so this is safe).
        if '"' not in inner:
            return inner
    return s


def _column_index(header: list[str], path: Path) -> dict[str, int]:
    norm = [h.strip().lower() for h in header]
    required = ["category", "english", "hindi_variant", "kannada_variant"]
    missing = [c for c in required if c not in norm]
    if missing:
        raise ValueError(f"{path} missing required column(s): {missing}; got header {header}")
    return {c: norm.index(c) for c in required}


def _cell(row: list[str], i: int) -> str:
    return row[i].strip() if i < len(row) else ""
