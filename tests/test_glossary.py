from pathlib import Path

from biasing.glossary import glossary_for, load_glossaries

# The real file uses the quirky wrapped-quote format; replicate it here.
QUIRKY = (
    '"category,english,hindi_variant,kannada_variant"\n'
    '"drugs,Paracetamol,,"\n'
    '"symptoms,fever,बुखार,ಜ್ವರ"\n'
    '"anatomy,liver,जिगर,"\n'
)

# A normal (unwrapped) CSV must parse identically.
NORMAL = (
    "category,english,hindi_variant,kannada_variant\n"
    "drugs,Paracetamol,,\n"
    "symptoms,fever,बुखार,ಜ್ವರ\n"
    "anatomy,liver,जिगर,\n"
)


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "terms.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_parses_wrapped_quote_format(tmp_path):
    g = load_glossaries(_write(tmp_path, QUIRKY))
    assert set(g) == {"en", "hi", "kn"}
    assert "Paracetamol" in g["en"].canonical_terms()
    assert "बुखार" in g["hi"].canonical_terms()
    assert "ಜ್ವರ" in g["kn"].canonical_terms()


def test_parses_normal_csv_identically(tmp_path):
    g = load_glossaries(_write(tmp_path, NORMAL))
    assert "बुखार" in g["hi"].canonical_terms()
    assert "जिगर" in g["hi"].canonical_terms()  # liver, kannada empty


def test_drug_only_row_contributes_to_english_only(tmp_path):
    g = load_glossaries(_write(tmp_path, QUIRKY))
    assert "Paracetamol" in g["en"].canonical_terms()
    assert "Paracetamol" not in g["hi"].canonical_terms()


def test_hindi_term_keeps_english_as_cross_lingual_hint(tmp_path):
    g = load_glossaries(_write(tmp_path, QUIRKY))
    fever = next(t for t in g["hi"].terms if t.canonical == "बुखार")
    assert "fever" in fever.variants
    assert "fever" in fever.all_surfaces


def test_variants_include_cross_script_transliteration(tmp_path):
    """Method-A enrichment: a Hindi term carries BOTH its English name and
    its Kannada spelling as variants; a Kannada term carries English + Hindi. Locks in
    the fix so reverting to English-only is caught."""
    g = load_glossaries(_write(tmp_path, QUIRKY))
    hi_fever = next(t for t in g["hi"].terms if t.canonical == "बुखार")
    kn_fever = next(t for t in g["kn"].terms if t.canonical == "ಜ್ವರ")
    assert "ಜ್ವರ" in hi_fever.variants and "fever" in hi_fever.variants
    assert "बुखार" in kn_fever.variants and "fever" in kn_fever.variants
    # but the boost hotwords stay language-pure (canonical only) — Method B unaffected
    assert "ಜ್ವರ" not in g["hi"].hotwords()


def test_hotwords_are_language_pure(tmp_path):
    g = load_glossaries(_write(tmp_path, QUIRKY))
    # hi hotwords should be Devanagari surfaces, not the english hint
    assert g["hi"].hotwords() == ["बुखार", "जिगर"]


def test_glossary_for_falls_back_on_base_language(tmp_path):
    g = load_glossaries(_write(tmp_path, QUIRKY))
    assert glossary_for(g, "hi-IN").lang == "hi"


def test_glossary_for_unknown_lang_merges_all(tmp_path):
    g = load_glossaries(_write(tmp_path, QUIRKY))
    merged = glossary_for(g, "ta")  # not present
    assert len(merged.terms) >= 4  # en(2) + hi(2) + kn(1) ...


def test_loads_real_project_glossary():
    """Sanity-check the actual medical_bias_terms.csv ships and parses."""
    from biasing.glossary import DEFAULT_GLOSSARY

    if not DEFAULT_GLOSSARY.exists():
        import pytest

        pytest.skip(f"{DEFAULT_GLOSSARY} not found")
    g = load_glossaries()
    assert "Paracetamol" in g["en"].canonical_terms()
    assert any("बुखार" == t.canonical for t in g["hi"].terms)
