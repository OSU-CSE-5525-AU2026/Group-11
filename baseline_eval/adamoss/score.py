"""Score a run's predictions and write the result tables.

The primary metric is accuracy according to an LLM judge. Token F1 and BERTScore are
reported as judge-free secondary metrics. Verdicts are cached in ``judgments.jsonl``, and
an optional ``judge_overrides.csv`` in the run directory holds reviewed corrections.

    python -m adamoss.score --run-dir results/qwen2.5-omni-7b
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
import torch
from bert_score import score as bert_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from adamoss.metrics import mcnemar_exact, token_f1, wilson_interval
from adamoss.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_TEMPLATE, SETTINGS

SUBSETS = {"all": lambda df: df, "approved": lambda df: df[df["review_status"] == "Approved"]}
QUESTION_TYPES = ("ALL", "TEXT", "AUDIO", "BOTH")
KEY = ["qid", "setting"]


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, dtype={"qid": str, "split": str, "gold_answer": str, "prediction": str})


class Judge:
    def __init__(self, model_id: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, question: str, gold: str, prediction: str) -> tuple[bool, str]:
        """Return (is_correct, raw_reply); anything but a clear CORRECT counts as wrong."""
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_TEMPLATE.format(question=question, gold=gold, prediction=prediction)},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        reply = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        verdict = reply.upper().split()[0].strip(".,:") if reply else ""
        if verdict not in {"CORRECT", "INCORRECT"}:
            print(f"Unparseable judge reply, counted as incorrect: {reply!r}", flush=True)
        return verdict == "CORRECT", reply


def run_judge(predictions: pd.DataFrame, run_dir: Path, model_id: str) -> pd.DataFrame:
    """Judge predictions that have no cached verdict, then return all verdicts."""
    path = run_dir / "judgments.jsonl"
    done = set()
    if path.exists():
        done = set(read_jsonl(path)[KEY].itertuples(index=False, name=None))

    pending = [row for row in predictions.itertuples() if (row.qid, row.setting) not in done]
    if pending:
        judge = Judge(model_id)
        with open(path, "a", encoding="utf-8") as handle:
            for row in pending:
                correct, reply = judge(row.question, row.gold_answer, row.prediction)
                record = {"qid": row.qid, "setting": row.setting, "correct": correct, "reply": reply, "judge": model_id}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        del judge
        torch.cuda.empty_cache()
    return read_jsonl(path)[KEY + ["correct"]]


def apply_overrides(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Replace judge verdicts with reviewed ones, keeping the original in ``judge_correct``."""
    df = df.assign(judge_correct=df["correct"])
    if not path.exists():
        return df
    overrides = pd.read_csv(path).set_index(KEY)["correct"]
    keys = pd.MultiIndex.from_frame(df[KEY])
    unknown = overrides.index.difference(keys)
    if len(unknown):
        raise ValueError(f"Overrides do not match any prediction: {list(unknown)}")
    df["correct"] = overrides.reindex(keys).where(lambda s: s.notna(), df["correct"].values).astype(bool).values
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Compute one row of metrics per subset, question type and setting."""
    rows = []
    for subset, select in SUBSETS.items():
        for question_type, setting in itertools.product(QUESTION_TYPES, SETTINGS):
            group = select(df)
            group = group[group["setting"] == setting]
            if question_type != "ALL":
                group = group[group["question_type"] == question_type]
            if group.empty:
                continue
            low, high = wilson_interval(int(group["correct"].sum()), len(group))
            rows.append(
                {
                    "subset": subset,
                    "question_type": question_type,
                    "setting": setting,
                    "n": len(group),
                    "judge_accuracy": group["correct"].mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "token_f1": group["token_f1"].mean(),
                    "bertscore_f1": group["bertscore_f1"].mean(),
                    "mean_input_tokens": group["input_tokens"].mean(),
                    "mean_latency_s": group["latency_s"].mean(),
                }
            )
    return pd.DataFrame(rows)


def pairwise_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Compare settings on the same questions with an exact McNemar test."""
    rows = []
    for subset, select in SUBSETS.items():
        for question_type in QUESTION_TYPES:
            group = select(df)
            if question_type != "ALL":
                group = group[group["question_type"] == question_type]
            table = group.pivot(index="qid", columns="setting", values="correct").dropna()
            for a, b in itertools.combinations(SETTINGS, 2):
                if a not in table or b not in table or table.empty:
                    continue
                a_only = int((table[a].astype(bool) & ~table[b].astype(bool)).sum())
                b_only = int((table[b].astype(bool) & ~table[a].astype(bool)).sum())
                rows.append(
                    {
                        "subset": subset,
                        "question_type": question_type,
                        "setting_a": a,
                        "setting_b": b,
                        "n": len(table),
                        "only_a_correct": a_only,
                        "only_b_correct": b_only,
                        "mcnemar_p": mcnemar_exact(a_only, b_only),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--judge-model", default="Qwen/Qwen2-7B-Instruct")
    args = parser.parse_args()

    predictions = read_jsonl(args.run_dir / "predictions.jsonl").fillna({"prediction": ""})
    df = predictions.merge(run_judge(predictions, args.run_dir, args.judge_model), on=KEY, validate="one_to_one")
    df = apply_overrides(df, args.run_dir / "judge_overrides.csv")

    df["token_f1"] = [token_f1(p, g) for p, g in zip(df["prediction"], df["gold_answer"])]
    df["bertscore_f1"] = bert_score(df["prediction"].tolist(), df["gold_answer"].tolist(), lang="en")[2].numpy()

    columns = ["qid", "question_type", "review_status", "setting", "question", "gold_answer", "prediction",
               "correct", "judge_correct", "token_f1", "bertscore_f1"]  # fmt: skip
    df[columns].sort_values(KEY).to_csv(args.run_dir / "per_example.csv", index=False)

    summary = summarize(df)
    pairwise = pairwise_tests(df)
    summary.to_csv(args.run_dir / "summary.csv", index=False, float_format="%.4f")
    pairwise.to_csv(args.run_dir / "pairwise_tests.csv", index=False, float_format="%.4f")
    print(summary.round(3).to_string(index=False))
    print(pairwise.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
