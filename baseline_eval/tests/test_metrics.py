import pytest

from adamoss.metrics import mcnemar_exact, normalize_answer, token_f1, wilson_interval


def test_normalize_removes_case_punctuation_and_articles():
    assert normalize_answer("The  Coffee-place!") == "coffeeplace"
    assert normalize_answer("A hernia.") == "hernia"


def test_token_f1_uses_best_gold_alternative():
    assert token_f1("lifting weights", "He was lifting weigts / Lifting weights") == 1.0


def test_token_f1_handles_partial_and_empty():
    assert token_f1("New York City", "New York") == pytest.approx(0.8)
    assert token_f1("", "Monday") == 0.0


def test_wilson_interval_matches_known_value():
    low, high = wilson_interval(10, 20)
    assert low == pytest.approx(0.2993, abs=1e-3)
    assert high == pytest.approx(0.7007, abs=1e-3)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert mcnemar_exact(3, 3) == 1.0
