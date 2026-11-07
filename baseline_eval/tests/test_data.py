from pathlib import Path

from adamoss.data import load_examples
from adamoss.prompts import AUDIO_AND_TRANSCRIPT, AUDIO_ONLY, TRANSCRIPT_ONLY, build_conversation

QA_CSV = Path(__file__).parent.parent / "data" / "meld_qa.csv"


def test_loader_cleans_the_export():
    examples = {example.qid: example for example in load_examples(QA_CSV)}
    assert len(examples) == 81  # The blank separator row is dropped.
    assert examples["test-dia210_utt4-q1"].clip == "dia210_utt4"  # Had no extension.
    assert examples["A008"].clip == "dia20_utt15"  # Had a trailing dot and the wrong clip.
    assert examples["A004"].clip == "dia9_utt1"
    assert examples["AT024"].question_type == "TEXT"  # Uses the reviewer-corrected label.


def test_audio_path_keeps_splits_apart():
    example = load_examples(QA_CSV)[0]
    assert example.audio_path(Path("audio")) == Path("audio/train/dia0_utt0.wav")


def _user_content(setting):
    return build_conversation(setting, "Q?", "Secret words", "clip.wav")[1]["content"]


def test_audio_can_follow_the_text():
    content = build_conversation(AUDIO_ONLY, "Q?", "Secret words", "clip.wav", audio_first=False)[1]["content"]
    assert [part["type"] for part in content] == ["text", "audio"]


def test_each_setting_exposes_only_its_modalities():
    audio_only = _user_content(AUDIO_ONLY)
    assert audio_only[0]["type"] == "audio"
    assert "Secret words" not in audio_only[-1]["text"]

    transcript_only = _user_content(TRANSCRIPT_ONLY)
    assert [part["type"] for part in transcript_only] == ["text"]
    assert "Secret words" in transcript_only[0]["text"]

    both = _user_content(AUDIO_AND_TRANSCRIPT)
    assert both[0]["type"] == "audio"
    assert "Secret words" in both[-1]["text"]
