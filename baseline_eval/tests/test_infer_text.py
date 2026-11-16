from adamoss.data import QAExample
from adamoss.infer_text import best_span, infer_kind, seq2seq_prompt
from adamoss.prompts import build_text_messages


def _example():
    return QAExample("Q1", "train", "dia0_utt0", "Secret words", "What?", "x", "TEXT", "Approved")


def test_text_messages_are_plain_strings_with_only_the_transcript():
    system, user = build_text_messages("What?", "Secret words")
    assert system["role"] == "system" and user["role"] == "user"
    assert isinstance(user["content"], str)
    assert 'Transcript: "Secret words"' in user["content"]
    assert "Question: What?" in user["content"]


def test_model_kind_is_guessed_from_the_id():
    assert infer_kind("deepset/roberta-base-squad2") == "extractive"
    assert infer_kind("google/flan-t5-large") == "seq2seq"
    assert infer_kind("allenai/unifiedqa-v2-t5-large-1363200") == "seq2seq"
    assert infer_kind("Qwen/Qwen2.5-7B-Instruct") == "chat"


def test_unifiedqa_uses_its_own_input_format():
    assert seq2seq_prompt(_example(), "allenai/unifiedqa-v2-t5-large-1363200") == "What? \\n Secret words"
    assert "Secret words" in seq2seq_prompt(_example(), "google/flan-t5-large")


def test_best_span_stays_inside_the_context():
    # Token 0 is the null token, 1-2 are the question, 3-5 are the context.
    in_context = [False, False, False, True, True, True]
    start = [0.0, 9.0, 0.0, 1.0, 5.0, 0.0]
    end = [0.0, 0.0, 9.0, 1.0, 2.0, 4.0]
    assert best_span(start, end, in_context) == (4, 5)  # Question tokens score higher but are ignored.


def test_best_span_returns_none_when_the_null_answer_wins():
    in_context = [False, False, True, True]
    assert best_span([5.0, 0.0, 1.0, 1.0], [5.0, 0.0, 1.0, 1.0], in_context) is None


def test_best_span_respects_the_length_limit():
    in_context = [False] + [True] * 5
    start = [0.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    end = [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    assert best_span(start, end, in_context, max_len=3) != (1, 5)
