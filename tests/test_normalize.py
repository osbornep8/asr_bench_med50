import numpy as np

from benchmark.normalize import _resample, _to_mono, normalize_text


def test_lowercases_latin_and_strips_punctuation():
    assert normalize_text("Paracetamol, 500mg!") == "paracetamol 500mg"


def test_collapses_whitespace():
    assert normalize_text("  fever   and    cough ") == "fever and cough"


def test_preserves_devanagari_and_converts_digits():
    # Devanagari digits ५०० → 500; script preserved.
    out = normalize_text("बुखार ५०० मिग्रा", lang="hi")
    assert "बुखार" in out
    assert "500" in out


def test_keeps_devanagari_characters_through_punct_strip():
    out = normalize_text("गले में दर्द।", lang="hi")  # danda (।) stripped as punctuation
    assert out == "गले में दर्द"


def test_none_is_empty():
    assert normalize_text(None) == ""


def test_to_mono_averages_stereo():
    stereo = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # [frames, channels]
    mono = _to_mono(stereo)
    assert mono.shape == (2,)
    assert np.allclose(mono, [0.5, 0.5])


def test_resample_fallback_changes_length_proportionally():
    sig = np.sin(np.linspace(0, 6.28, 16000)).astype(np.float32)  # 1s @ 16k
    out = _resample(sig, 16000, 8000)
    assert abs(len(out) - 8000) <= 1
