import pytest

from benchmark import metrics as M


def test_term_recall_counts_only_terms_in_reference():
    ref = "patient has fever and cough"
    hyp = "patient has fever"
    # both terms in ref; only "fever" recalled -> 0.5
    assert M.term_recall(hyp, ref, ["fever", "cough"]) == 0.5


def test_term_recall_none_when_no_term_in_reference():
    assert M.term_recall("anything", "no targets here", ["fever"]) is None


def test_term_recall_full():
    assert M.term_recall("fever cough", "fever cough", ["fever", "cough"]) == 1.0


def test_false_insertion_rate_flags_hallucinated_terms():
    # noterm clip: hyp wrongly contains "fever" -> 1 of 2 glossary terms inserted
    assert M.false_insertion_rate("i feel fever today", ["fever", "cough"]) == 0.5


def test_false_insertion_rate_zero_when_clean():
    assert M.false_insertion_rate("all good no symptoms", ["fever", "cough"]) == 0.0


def test_false_insertion_rate_none_for_empty_glossary():
    assert M.false_insertion_rate("text", []) is None


def test_mean_ignore_none():
    assert M.mean_ignore_none([1.0, None, 3.0]) == 2.0
    assert M.mean_ignore_none([None, None]) is None


def test_substring_match_is_word_bounded():
    # "ever" must NOT match inside "fever"
    assert M.term_recall("i have ever", "i have fever", ["fever"]) == 0.0


def test_wer_cer_smoke():
    jiwer = pytest.importorskip("jiwer")  # noqa: F841
    assert M.wer("fever and cough", "fever and cough") == 0.0
    assert M.wer("fever and cough", "fever cough") > 0.0
    assert M.cer("fever", "fevor") > 0.0


# ── fuzzy term metrics ────────────────────────────────────────────────────────

def test_best_term_cer_exact_is_zero():
    assert M.best_term_cer("रोगी को पैरासिटामोल दी", "पैरासिटामोल") == 0.0


def test_best_term_cer_small_for_spelling_variant():
    # पेरासिटामोल vs पैरासिटामोल — a couple of chars differ → small CER, not huge
    cer = M.best_term_cer("रोगी को पेरासिटामोल दी", "पैरासिटामोल")
    assert 0.0 < cer < 0.3


def test_term_recall_fuzzy_counts_variant_as_hit():
    ref = "रोगी को पैरासिटामोल दी"
    hyp = "रोगी को पेरासिटामोल दी"  # variant spelling
    assert M.term_recall(hyp, ref, ["पैरासिटामोल"]) == 0.0          # exact misses
    assert M.term_recall_fuzzy(hyp, ref, ["पैरासिटामोल"]) == 1.0    # fuzzy catches it


def test_term_recall_fuzzy_rejects_unrelated_words():
    # "azithral" decoded as "as it rolls" must NOT count as recalled
    assert M.term_recall_fuzzy("as it rolls today", "patient took azithral", ["azithral"]) == 0.0


def test_mean_term_cer_none_when_no_term_in_ref():
    assert M.mean_term_cer("anything", "no targets", ["fever"]) is None


def test_threshold_controls_fuzzy_recall():
    ref, hyp = "took azithral", "took azithrol"
    assert M.term_recall_fuzzy(hyp, ref, ["azithral"], threshold=0.2) == 1.0
    assert M.term_recall_fuzzy(hyp, ref, ["azithral"], threshold=0.0) == 0.0


# ── performance metric ────────────────────────────────────────────────────────

def test_rtf_basic():
    # 750ms to process 5s of audio → RTF 0.15 (faster than real-time)
    assert M.rtf(750.0, 5.0) == 0.15


def test_rtf_none_when_missing_inputs():
    assert M.rtf(None, 5.0) is None
    assert M.rtf(750.0, None) is None
    assert M.rtf(750.0, 0.0) is None      # avoid divide-by-zero
