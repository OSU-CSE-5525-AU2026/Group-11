import pandas as pd
import pytest

from adamoss.score import apply_overrides


def _predictions():
    return pd.DataFrame(
        {"qid": ["Q1", "Q1", "Q2"], "setting": ["audio_only", "transcript_only", "audio_only"], "correct": [True, False, True]}
    )


def test_overrides_change_only_listed_verdicts(tmp_path):
    path = tmp_path / "overrides.csv"
    path.write_text("qid,setting,correct,reason\nQ1,audio_only,False,refusal\n")
    df = apply_overrides(_predictions(), path)
    assert df["correct"].tolist() == [False, False, True]
    assert df["judge_correct"].tolist() == [True, False, True]


def test_missing_override_file_keeps_verdicts(tmp_path):
    df = apply_overrides(_predictions(), tmp_path / "absent.csv")
    assert df["correct"].tolist() == [True, False, True]


def test_unknown_override_is_rejected(tmp_path):
    path = tmp_path / "overrides.csv"
    path.write_text("qid,setting,correct,reason\nNOPE,audio_only,False,typo\n")
    with pytest.raises(ValueError, match="NOPE"):
        apply_overrides(_predictions(), path)
