"""The three input settings and the prompts used for answering and judging."""

from __future__ import annotations

AUDIO_ONLY = "audio_only"
TRANSCRIPT_ONLY = "transcript_only"
AUDIO_AND_TRANSCRIPT = "audio_and_transcript"
SETTINGS = (AUDIO_ONLY, TRANSCRIPT_ONLY, AUDIO_AND_TRANSCRIPT)

ANSWER_SYSTEM_PROMPT = (
    "Reply with a short, direct answer. If the question asks about more than one thing, "
    "answer every part."
)

INSTRUCTIONS = {
    AUDIO_ONLY: "Given the conversation in this audio, answer the following question.",
    TRANSCRIPT_ONLY: "This is a transcript of a conversation. Answer the following question.",
    AUDIO_AND_TRANSCRIPT: (
        "Given the conversation in this audio and its transcript, answer the following question."
    ),
}

JUDGE_SYSTEM_PROMPT = "You are a strict, consistent grader of short answers."

JUDGE_TEMPLATE = """\
Decide whether the candidate answer is correct, using the reference answer as ground truth.

Rules:
- Paraphrases and close synonyms are correct (for example "annoyed" and "irritated").
- Yes/no polarity must match the reference.
- If the reference has several parts, such as a fact and a tone, every part must be correct.
- Extra detail is fine unless it contradicts the reference.
- Hedging or refusing to answer is incorrect.

Question: {question}
Reference answer: {gold}
Candidate answer: {prediction}

Reply with exactly one word: CORRECT or INCORRECT."""


def build_voxtral_conversation(setting: str, question: str, transcript: str, audio_path: str) -> list[dict]:
    """The same prompt as ``build_conversation``, in the shape Voxtral's processor expects.

    Audio is given as ``{"type": "audio", "path": ...}`` and there is a single user turn,
    so the system prompt is placed at the start of the text instead of in its own message.
    """
    _, user = build_conversation(setting, question, transcript, audio_path)
    content = []
    for part in user["content"]:
        if part["type"] == "audio":
            content.append({"type": "audio", "path": part["audio"]})
        else:
            content.append({"type": "text", "text": f"{ANSWER_SYSTEM_PROMPT}\n\n{part['text']}"})
    return [{"role": "user", "content": content}]


def build_text_messages(question: str, transcript: str) -> list[dict]:
    """The transcript-only prompt as plain-string chat messages, for text-only models.

    Same wording as ``build_conversation(TRANSCRIPT_ONLY, ...)``, but with string content
    because text-only chat templates do not accept the multimodal list format.
    """
    conversation = build_conversation(TRANSCRIPT_ONLY, question, transcript, audio_path="")
    return [
        {"role": message["role"], "content": "\n\n".join(part["text"] for part in message["content"])}
        for message in conversation
    ]


def build_conversation(
    setting: str, question: str, transcript: str, audio_path: str, audio_first: bool = True
) -> list[dict]:
    """Build a chat conversation in which only the supplied modality varies.

    The speaker name is left out of the transcript so that the transcript and audio
    settings differ only in modality. ``audio_first`` sets where the audio goes relative
    to the text, since models document different preferred orders.
    """
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting: {setting!r}")

    parts = [INSTRUCTIONS[setting]]
    if setting != AUDIO_ONLY:
        parts.append(f'Transcript: "{transcript}"')
    parts.append(f"Question: {question}")
    text = "\n\n".join(parts)

    content: list[dict] = [{"type": "text", "text": text}]
    if setting != TRANSCRIPT_ONLY:
        content.insert(0 if audio_first else 1, {"type": "audio", "audio": audio_path})

    return [
        {"role": "system", "content": [{"type": "text", "text": ANSWER_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
