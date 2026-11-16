from adamoss.model_family import model_family
from adamoss.prompts import (
    ANSWER_SYSTEM_PROMPT,
    AUDIO_AND_TRANSCRIPT,
    AUDIO_ONLY,
    TRANSCRIPT_ONLY,
    build_voxtral_conversation,
)


def test_model_family_is_chosen_from_the_id():
    assert model_family("Qwen/Qwen2.5-Omni-7B") == "omni"
    assert model_family("Qwen/Qwen2.5-Omni-3B") == "omni"
    assert model_family("Qwen/Qwen2-Audio-7B-Instruct") == "qwen2audio"
    assert model_family("mistralai/Voxtral-Mini-3B-2507") == "voxtral"
    assert model_family("google/gemma-4-E2B-it") == "gemma"
    assert model_family("google/gemma-3n-E4B-it") == "gemma"


def test_voxtral_conversation_is_one_user_turn_with_path_audio():
    conversation = build_voxtral_conversation(AUDIO_AND_TRANSCRIPT, "Q?", "Secret words", "clip.wav")
    assert [message["role"] for message in conversation] == ["user"]
    audio, text = conversation[0]["content"]
    assert audio == {"type": "audio", "path": "clip.wav"}
    assert text["text"].startswith(ANSWER_SYSTEM_PROMPT)
    assert "Secret words" in text["text"]


def test_voxtral_conversation_keeps_modalities_separate():
    audio_only = build_voxtral_conversation(AUDIO_ONLY, "Q?", "Secret words", "clip.wav")[0]["content"]
    assert "Secret words" not in audio_only[-1]["text"]

    transcript_only = build_voxtral_conversation(TRANSCRIPT_ONLY, "Q?", "Secret words", "clip.wav")[0]["content"]
    assert [part["type"] for part in transcript_only] == ["text"]
