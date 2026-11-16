"""Answer every QA example from the transcript alone with a text-only model.

These are the pure text baselines: no audio model is involved. Three kinds of model are
supported, chosen from the model id (or with ``--kind``):

- ``chat``: instruction-tuned LLMs, e.g. Qwen/Qwen2.5-7B-Instruct or
  meta-llama/Llama-3.1-8B-Instruct. Uses the same prompt as the audio LLMs' transcript-only
  setting.
- ``seq2seq``: T5-style models, e.g. google/flan-t5-large or allenai/unifiedqa-v2-t5-large-1363200.
- ``extractive``: span-extraction QA models fine-tuned on SQuAD 2.0, e.g.
  deepset/roberta-base-squad2. They can only copy words from the transcript.

Predictions are written in the same format as ``adamoss.infer`` (setting
``transcript_only``), so ``adamoss.score`` works on the run directory unchanged. Finished
questions are skipped on restart, so an interrupted job resumes.

    python -m adamoss.infer_text --run-dir results/qwen2.5-7b-instruct --model-id Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from adamoss.data import QAExample, load_examples
from adamoss.prompts import TRANSCRIPT_ONLY, build_text_messages

KINDS = ("chat", "seq2seq", "extractive")
NO_ANSWER = "No answer"  # What an extractive model says when it finds no span.
MAX_ANSWER_TOKENS = 30  # Longest span an extractive model may return.


def infer_kind(model_id: str) -> str:
    """Guess how to run a model from its id; override with ``--kind`` if this is wrong."""
    name = model_id.lower()
    if "squad" in name:
        return "extractive"
    if "t5" in name or "unifiedqa" in name:
        return "seq2seq"
    return "chat"


def seq2seq_prompt(example: QAExample, model_id: str) -> str:
    """UnifiedQA expects ``question \\n context``; other T5-style models take the chat text."""
    if "unifiedqa" in model_id.lower():
        return f"{example.question} \\n {example.transcript}"
    return "\n\n".join(message["content"] for message in build_text_messages(example.question, example.transcript))


def best_span(
    start_logits: list[float], end_logits: list[float], in_context: list[bool], max_len: int = MAX_ANSWER_TOKENS
) -> tuple[int, int] | None:
    """Pick the highest-scoring answer span inside the context, or None for "no answer".

    Follows the SQuAD 2.0 rule: if the null span (the first token, ``[CLS]``/``<s>``)
    scores at least as high as the best real span, the model is saying there is no answer.
    """
    null_score = start_logits[0] + end_logits[0]
    best, best_score = None, float("-inf")
    for start, start_ok in enumerate(in_context):
        if not start_ok:
            continue
        for end in range(start, min(start + max_len, len(in_context))):
            if not in_context[end]:
                break
            score = start_logits[start] + end_logits[end]
            if score > best_score:
                best, best_score = (start, end), score
    return best if best is not None and best_score > null_score else None


class TextAnswerer:
    """Shared timing and bookkeeping; subclasses implement ``generate_answer``."""

    def generate_answer(self, example: QAExample) -> tuple[str, int]:
        """Return (prediction, number of input tokens)."""
        raise NotImplementedError

    def answer(self, example: QAExample) -> dict:
        import torch

        start = time.perf_counter()
        with torch.inference_mode():
            prediction, input_tokens = self.generate_answer(example)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - start
        return {"prediction": prediction.strip(), "input_tokens": input_tokens, "latency_s": round(latency, 3)}


class ChatAnswerer(TextAnswerer):
    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate_answer(self, example: QAExample) -> tuple[str, int]:
        messages = build_text_messages(example.question, example.transcript)
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        input_tokens = inputs["input_ids"].shape[1]
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(output[0, input_tokens:], skip_special_tokens=True), input_tokens


class Seq2SeqAnswerer(TextAnswerer):
    def __init__(self, model_id: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate_answer(self, example: QAExample) -> tuple[str, int]:
        inputs = self.tokenizer(seq2seq_prompt(example, self.model_id), return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(output[0], skip_special_tokens=True), inputs["input_ids"].shape[1]


class ExtractiveAnswerer(TextAnswerer):
    def __init__(self, model_id: str, max_new_tokens: int) -> None:  # max_new_tokens is unused here.
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_id, dtype="auto", device_map="auto")
        self.model.eval()

    def generate_answer(self, example: QAExample) -> tuple[str, int]:
        # Utterances are short, so a single window always holds the whole transcript.
        encoding = self.tokenizer(
            example.question, example.transcript, return_tensors="pt", return_offsets_mapping=True, truncation="only_second"
        )
        offsets = encoding.pop("offset_mapping")[0].tolist()
        in_context = [sequence_id == 1 for sequence_id in encoding.sequence_ids(0)]
        outputs = self.model(**encoding.to(self.model.device))
        span = best_span(outputs.start_logits[0].float().tolist(), outputs.end_logits[0].float().tolist(), in_context)
        input_tokens = encoding["input_ids"].shape[1]
        if span is None:
            return NO_ANSWER, input_tokens
        return example.transcript[offsets[span[0]][0] : offsets[span[1]][1]], input_tokens


ANSWERERS: dict[str, Any] = {"chat": ChatAnswerer, "seq2seq": Seq2SeqAnswerer, "extractive": ExtractiveAnswerer}


def load_finished(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as handle:
        return {json.loads(line)["qid"] for line in handle}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True, help="Hugging Face id, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--kind", choices=KINDS, help="How to run the model; guessed from the id if omitted")
    parser.add_argument("--qa-csv", type=Path, default=Path("data/meld_qa.csv"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Only use the first N questions (for smoke tests)")
    args = parser.parse_args()

    kind = args.kind or infer_kind(args.model_id)
    examples = load_examples(args.qa_csv)[: args.limit]

    args.run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.run_dir / "predictions.jsonl"
    finished = load_finished(predictions_path)
    pending = [example for example in examples if example.qid not in finished]
    print(f"{args.model_id} ({kind}): {len(pending)} predictions to generate ({len(finished)} already done)", flush=True)
    if not pending:
        return

    answerer = ANSWERERS[kind](args.model_id, args.max_new_tokens)
    with open(predictions_path, "a", encoding="utf-8") as handle:
        for index, example in enumerate(pending, start=1):
            result = answerer.answer(example)
            record = {**asdict(example), "setting": TRANSCRIPT_ONLY, "model_id": args.model_id, **result}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(pending)}] {example.qid}: {result['prediction']!r}", flush=True)


if __name__ == "__main__":
    main()
