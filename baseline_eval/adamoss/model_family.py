"""Decide how to load and prompt an audio LLM from its Hugging Face id."""

from __future__ import annotations

FAMILIES = ("omni", "qwen2audio", "voxtral", "gemma")


def model_family(model_id: str) -> str:
    """Return the loading recipe for a model id; Qwen2.5-Omni is the default."""
    name = model_id.lower()
    if "voxtral" in name:
        return "voxtral"
    if "qwen2-audio" in name:
        return "qwen2audio"
    if "gemma" in name:
        return "gemma"
    return "omni"
