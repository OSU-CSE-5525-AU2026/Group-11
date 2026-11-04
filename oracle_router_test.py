# Oracle Router — offline replay for Adaptive Modality Routing
#
# For each question, read its reviewed route label (TEXT, AUDIO, or BOTH)
# and retrieve the cached prediction for that question and route. The
# script does not run a model or use answer correctness to choose a route.
#
# Inputs (UTF-8 CSV):
#   baseline_v1.csv: example_id, question_type
#   inference_outputs.csv: example_id, mode, prediction
# Use the same unique example_id in both CSVs, even when clips are reused.
# This matches the MELD QA Construction sheet. If Group 2 exports
# a different identifier column, rename it to example_id in its CSV.
#
# Run with Python 3 (no extra packages):
#   python oracle_router.py baseline_v1.csv inference_outputs.csv oracle_predictions.csv
#
# The output CSV keeps the raw prediction for downstream scoring.
# The console reports the share of questions routed to AUDIO or BOTH.

import argparse
import csv


QUESTION_ID = "example_id"
GOLD_ROUTE = "question_type"
MODE = "mode"
PREDICTION = "prediction"
VALID_ROUTES = {"TEXT", "AUDIO", "BOTH"}
AUDIO_ROUTES = {"AUDIO", "BOTH"}


def read_csv(path, required_columns):
    with open(path, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def normalize_route(value, example_id):
    route = (value or "").strip().upper()
    if route not in VALID_ROUTES:
        raise ValueError(f"{example_id}: invalid route {value!r}")
    return route


def select_oracle_predictions(dataset, inference_outputs):
    predictions_by_key = {}
    for row in inference_outputs:
        example_id = (row[QUESTION_ID] or "").strip()
        if not example_id:
            raise ValueError("Inference output contains a blank example_id.")

        mode = normalize_route(row[MODE], example_id)
        key = (example_id, mode)
        if key in predictions_by_key:
            raise ValueError(f"Duplicate inference output for {example_id} / {mode}")
        predictions_by_key[key] = row[PREDICTION]

    results = []
    seen_example_ids = set()
    for row in dataset:
        example_id = (row[QUESTION_ID] or "").strip()
        if not example_id or example_id in seen_example_ids:
            raise ValueError(f"Blank or duplicate dataset example_id: {example_id!r}")
        seen_example_ids.add(example_id)

        route = normalize_route(row[GOLD_ROUTE], example_id)
        key = (example_id, route)
        if key not in predictions_by_key:
            raise ValueError(f"Missing inference output for {example_id} / {route}")

        results.append({
            "example_id": example_id,
            "oracle_route": route,
            "prediction": predictions_by_key[key],
            "uses_audio": int(route in AUDIO_ROUTES),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Replay the annotated Oracle Router.")
    parser.add_argument("dataset_csv")
    parser.add_argument("inference_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()

    try:
        dataset = read_csv(args.dataset_csv, {QUESTION_ID, GOLD_ROUTE})
        inference_outputs = read_csv(
            args.inference_csv, {QUESTION_ID, MODE, PREDICTION}
        )
        if not dataset:
            raise ValueError("The evaluation dataset is empty.")
        results = select_oracle_predictions(dataset, inference_outputs)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    with open(args.output_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["example_id", "oracle_route", "prediction", "uses_audio"],
        )
        writer.writeheader()
        writer.writerows(results)

    audio_queries = sum(row["uses_audio"] for row in results)
    total = len(results)
    print(f"Oracle audio usage: {audio_queries}/{total} ({audio_queries / total:.1%})")


if __name__ == "__main__":
    main()
