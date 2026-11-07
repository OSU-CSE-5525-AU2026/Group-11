"""Answer every QA example under each input setting with an audio LLM.

Supports Qwen2.5-Omni (the default) and Gemma 4 models with audio, such as
google/gemma-4-E4B-it.

Predictions are appended to ``<run-dir>/predictions.jsonl`` as they are produced, and
finished (question, setting) pairs are skipped on restart, so an interrupted job resumes.

    python -m adamoss.infer --run-dir results/qwen2.5-omni-7b
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import librosa
import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

from adamoss.data import QAExample, load_examples
from adamoss.prepare_audio import SAMPLE_RATE
from adamoss.prompts import SETTINGS, TRANSCRIPT_ONLY, build_conversation


class Answerer:
    """Shared generation loop; subclasses load a model and prepare its inputs."""

    processor: Any
    model: Any
    generator: Any
    max_new_tokens: int

    def prepare_inputs(self, example: QAExample, setting: str, audio_path: Path) -> Any:
        raise NotImplementedError

    @torch.inference_mode()
    def answer(self, example: QAExample, setting: str, audio_dir: Path) -> dict:
        inputs = self.prepare_inputs(example, setting, example.audio_path(audio_dir))
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        input_tokens = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        output = self.generator.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - start

        prediction = self.processor.decode(output[0, input_tokens:], skip_special_tokens=True).strip()
        return {"prediction": prediction, "input_tokens": input_tokens, "latency_s": round(latency, 3)}


class OmniAnswerer(Answerer):
    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )
        self.model.disable_talker()  # Text answers only; skips loading speech generation.
        self.model.eval()
        # The thinker produces the text. The top-level generate() needs the talker, which
        # is disabled here, and would override max_new_tokens with its own default.
        self.generator = self.model.thinker
        self.max_new_tokens = max_new_tokens

    def prepare_inputs(self, example: QAExample, setting: str, audio_path: Path) -> Any:
        conversation = build_conversation(setting, example.question, example.transcript, str(audio_path))
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio = None
        if setting != TRANSCRIPT_ONLY:
            audio = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)[0]
        return self.processor(text=prompt, audio=audio, return_tensors="pt", padding=True)


class GemmaAnswerer(Answerer):
    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
        self.model.eval()
        self.generator = self.model
        self.max_new_tokens = max_new_tokens

    def prepare_inputs(self, example: QAExample, setting: str, audio_path: Path) -> Any:
        # The Gemma 4 model card recommends placing audio after the text.
        conversation = build_conversation(
            setting, example.question, example.transcript, str(audio_path), audio_first=False
        )
        return self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )


def load_answerer(model_id: str, max_new_tokens: int) -> Answerer:
    answerer_class = GemmaAnswerer if "gemma" in model_id.lower() else OmniAnswerer
    return answerer_class(model_id, max_new_tokens)


def load_finished(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as handle:
        return {(record["qid"], record["setting"]) for record in map(json.loads, handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--qa-csv", type=Path, default=Path("data/meld_qa.csv"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Only use the first N questions (for smoke tests)")
    args = parser.parse_args()

    examples = load_examples(args.qa_csv)[: args.limit]
    missing = [str(e.audio_path(args.audio_dir)) for e in examples if not e.audio_path(args.audio_dir).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} audio files missing; run adamoss.prepare_audio first: {missing[:3]}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.run_dir / "predictions.jsonl"
    finished = load_finished(predictions_path)
    pending = [(e, s) for e in examples for s in args.settings if (e.qid, s) not in finished]
    print(f"{len(pending)} predictions to generate ({len(finished)} already done)", flush=True)
    if not pending:
        return

    answerer = load_answerer(args.model_id, args.max_new_tokens)
    with open(predictions_path, "a", encoding="utf-8") as handle:
        for index, (example, setting) in enumerate(pending, start=1):
            result = answerer.answer(example, setting, args.audio_dir)
            record = {**asdict(example), "setting": setting, "model_id": args.model_id, **result}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(pending)}] {example.qid} {setting}: {result['prediction']!r}", flush=True)


if __name__ == "__main__":
    main()
