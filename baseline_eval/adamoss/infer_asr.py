"""Audio-only cascade: transcribe each clip with Whisper, then answer from the ASR text.

The answering model never sees the gold transcript or the audio, only what the speech
recognizer heard, so this is an audio-only baseline built from a text-only model. It cannot
use tone or other paralinguistic cues, which the audio LLMs may. Predictions use the
``audio_only`` setting, so ``adamoss.score`` works on the run directory unchanged.

    python -m adamoss.infer_asr --run-dir results/whisper-qwen2.5-7b --llm-id Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from adamoss.data import QAExample, load_examples
from adamoss.infer_text import ANSWERERS, KINDS, infer_kind
from adamoss.prepare_audio import SAMPLE_RATE
from adamoss.prompts import AUDIO_ONLY


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def transcribe(examples: list[QAExample], audio_dir: Path, asr_id: str, cache_path: Path) -> dict[str, dict]:
    """Return ASR output per question id, transcribing only clips missing from the cache."""
    cache = {record["qid"]: record for record in read_jsonl(cache_path)}
    pending = [example for example in examples if example.qid not in cache]
    if pending:
        import librosa
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(asr_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(asr_id, dtype=torch.bfloat16, device_map="auto")
        model.eval()
        with open(cache_path, "a", encoding="utf-8") as handle:
            for index, example in enumerate(pending, start=1):
                audio = librosa.load(example.audio_path(audio_dir), sr=SAMPLE_RATE, mono=True)[0]
                features = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
                start = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(
                        features.to(model.device, model.dtype), language="en", task="transcribe", max_new_tokens=128
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency = time.perf_counter() - start
                text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
                cache[example.qid] = {"qid": example.qid, "asr_transcript": text, "asr_latency_s": round(latency, 3)}
                handle.write(json.dumps(cache[example.qid], ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[asr {index}/{len(pending)}] {example.qid}: {text!r}", flush=True)
        del model
        torch.cuda.empty_cache()
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--llm-id", required=True, help="Text model that answers from the ASR text")
    parser.add_argument("--asr-id", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--kind", choices=KINDS, help="How to run the answering model; guessed from the id if omitted")
    parser.add_argument("--qa-csv", type=Path, default=Path("data/meld_qa.csv"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Only use the first N questions (for smoke tests)")
    args = parser.parse_args()

    examples = load_examples(args.qa_csv)[: args.limit]
    missing = [str(e.audio_path(args.audio_dir)) for e in examples if not e.audio_path(args.audio_dir).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} audio files missing; run adamoss.prepare_audio first: {missing[:3]}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.run_dir / "predictions.jsonl"
    finished = {record["qid"] for record in read_jsonl(predictions_path)}
    pending = [example for example in examples if example.qid not in finished]
    print(f"{len(pending)} predictions to generate ({len(finished)} already done)", flush=True)
    if not pending:
        return

    asr = transcribe(pending, args.audio_dir, args.asr_id, args.run_dir / "asr_transcripts.jsonl")
    answerer = ANSWERERS[args.kind or infer_kind(args.llm_id)](args.llm_id, args.max_new_tokens)
    with open(predictions_path, "a", encoding="utf-8") as handle:
        for index, example in enumerate(pending, start=1):
            heard = asr[example.qid]
            # The answerer reads ``transcript``, so give it the ASR text in place of the gold one.
            result = answerer.answer(replace(example, transcript=heard["asr_transcript"]))
            record = {
                **asdict(example),
                "setting": AUDIO_ONLY,
                "model_id": f"{args.asr_id} + {args.llm_id}",
                "asr_transcript": heard["asr_transcript"],
                "asr_latency_s": heard["asr_latency_s"],
                **result,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(pending)}] {example.qid}: {result['prediction']!r}", flush=True)


if __name__ == "__main__":
    main()
