# AdaMOSS baseline evaluation on MELD QA

Baseline for [AdaMOSS](#background), which routes each spoken-language question to the
cheapest modality that can answer it. This folder tests whether audio LLMs can still
answer questions when a modality is missing. We expect them to fail more often on questions
that strongly require the missing modality.

Every question, whatever its type, is asked in all three settings:

| Setting | Model receives | Instruction |
| --- | --- | --- |
| `transcript_only` | Transcript + question | "This is a transcript of a conversation. Answer the following question." |
| `audio_only` | Audio + question | "Given the conversation in this audio, answer the following question." |
| `audio_and_transcript` | Audio + transcript + question | "Given the conversation in this audio and its transcript, answer the following question." |

## Results

Correct answers out of the total for all 81 questions, with 95% Wilson intervals. Verdicts
come from an LLM judge plus the [reviewed corrections](#judge-corrections).

**Qwen2.5-Omni-7B**

| Question type | Transcript only | Audio only | Audio + transcript |
| --- | --- | --- | --- |
| TEXT | 25/26 (0.81–0.99) | 22/26 (0.66–0.94) | 25/26 (0.81–0.99) |
| AUDIO | 14/28 (0.33–0.67) | 15/28 (0.36–0.70) | 16/28 (0.39–0.73) |
| BOTH | 14/27 (0.34–0.69) | 12/27 (0.28–0.63) | 12/27 (0.28–0.63) |
| All | 53/81 (0.55–0.75) | 49/81 (0.50–0.70) | 53/81 (0.55–0.75) |

**Gemma 4 E4B** (`google/gemma-4-E4B-it`)

| Question type | Transcript only | Audio only | Audio + transcript |
| --- | --- | --- | --- |
| TEXT | 25/26 (0.81–0.99) | 19/26 (0.54–0.86) | 25/26 (0.81–0.99) |
| AUDIO | 8/28 (0.15–0.47) | 12/28 (0.27–0.61) | 12/28 (0.27–0.61) |
| BOTH | 12/27 (0.28–0.63) | 12/27 (0.28–0.63) | 15/27 (0.37–0.72) |
| All | 45/81 (0.45–0.66) | 43/81 (0.42–0.64) | 52/81 (0.53–0.74) |

### What this shows

- **Without audio, both models do much worse on AUDIO questions than on TEXT questions.**
  Transcript-only falls from 25/26 on TEXT to 14/28 (Qwen) and 8/28 (Gemma) on AUDIO, and
  the intervals do not overlap. This fits the hypothesis.
- **Supplying audio does not clearly fix this.** Audio-only scores 15/28 (Qwen) and 12/28
  (Gemma) on AUDIO questions, so AUDIO questions are hard in general, and the drop above
  cannot be attributed only to the missing modality.
- **Without a transcript, TEXT accuracy also drops,** to 22/26 (Qwen) and 19/26 (Gemma). For
  Gemma this is significant (p = 0.03), and many errors are mishearings such as "Joanie" for
  "Joey".
- **The models fail differently.** Without audio, Gemma often says it cannot tell (about 13
  of 28 AUDIO questions), while Qwen almost always guesses.
- **Having both modalities helps Gemma only.** Gemma overall is better with both than with
  transcript only (p = 0.04) or audio only (p = 0.05). Qwen shows no significant differences.

The p-values come from exact McNemar tests, uncorrected for the many comparisons made, so
treat them as suggestive. Each run's folder under [`results/`](results/) has `summary.csv`
(including the 46 reviewer-approved questions), `pairwise_tests.csv` and `per_example.csv`.

## Metrics

**Primary: LLM-judged accuracy.** Many gold answers depend on polarity or have two parts (a
fact and a tone), and answers vary in length. Overlap metrics handle this poorly: "The first
speaker is male, and the second speaker is female" gets token F1 = 1.0 against a gold answer
with the genders reversed. The judge is Qwen2-7B-Instruct.

**Secondary: token F1 and BERTScore,** which need no judge. Token F1 separates right from
wrong answers better (about 0.70 vs 0.27) than BERTScore (about 0.95 vs 0.90).

Accuracy intervals are Wilson intervals. Settings are compared with an exact McNemar test
because they answer the same questions. Input tokens and `generate()` latency are recorded
per question; latency excludes audio loading and preprocessing.

### Judge corrections

The judge makes mistakes, including accepting refusals such as "the transcript does not
provide enough context", which its rules count as wrong. All 486 verdicts from the two runs
were therefore read and clear errors corrected: 8 for Qwen and 16 for Gemma, with a reason
for each, in each run's `judge_overrides.csv`.

Borderline verdicts were left alone. The corrections still need checking by the team.
`per_example.csv` keeps both `judge_correct` (the raw
verdict) and `correct` (the verdict used in the tables). Delete the overrides file and rerun
`adamoss.score` to get the raw results.

## Reproducing

Requires a GPU with about 24 GB of memory and the original MELD archives (`train.tar.gz`,
`test.tar.gz`) from the [MELD site](https://affective-meld.github.io). MELD media is not
included in this repository. Run every command from this `baseline_eval/` folder.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m adamoss.prepare_audio --meld-root /path/to/MELD.Raw   # 16 kHz WAV per clip
python -m adamoss.infer --run-dir results/my-run                # resumes if interrupted
python -m adamoss.score --run-dir results/my-run                # writes the result tables
pytest
```

Add `--model-id google/gemma-4-E4B-it` to `adamoss.infer` for Gemma. Audio goes before the
text for Qwen and after it for Gemma, as Gemma's model card recommends.

On OSC Ascend (partition `nextgen`), each run took about 7 minutes on one A100:

```bash
mkdir -p logs   # Slurm writes job output here
sbatch --account=<your-account> slurm/run_eval.sbatch qwen2.5-omni-7b-instructed
sbatch --account=<your-account> slurm/run_eval.sbatch gemma-4-e4b-it google/gemma-4-E4B-it
```

## Dataset notes

[`data/meld_qa.csv`](data/meld_qa.csv) is the unmodified spreadsheet export, a newer
version than the CSV at the top of this repository (which has 61 questions and no
`question_type_corrected` column). It has 81
questions: 26 TEXT, 28 AUDIO and 27 BOTH, of which 46 are Approved, 15 marked Revise and
20 Needs Review. [`adamoss/data.py`](adamoss/data.py) handles these issues when loading:

- **Clip names repeat across MELD splits** with different content, so audio is always
  looked up by split as well as name.
- **Two rows pointed at the wrong clip**, found by comparing against the official MELD
  CSVs: A004 now uses `dia9_utt1` and A008 uses `dia20_utt15`.
- **Inconsistent file names and IDs:** a blank row is dropped, file names are normalised,
  and the 11 test rows without IDs get generated ones such as `test-dia210_utt4-q1`.
- `question_type_corrected` is used as the question type.

## Limitations

- **Small sample:** about 27 questions per type, which is too few to detect modest effects.
- **Weak test of audio use:** roughly 70% of AUDIO questions are yes/no or either/or, so
  guessing can score near 50%. Questions can also leak information through quoted words and
  names, which may explain why transcript-only Qwen does as well as it does.
- **Unreviewed data:** 35 of the 81 questions are not yet approved, and some gold answers
  are vague or subjective.
- **Single utterance only:** the model never sees the surrounding dialogue.

## Project layout

```text
adamoss/
  data.py            Load and clean the QA spreadsheet
  prepare_audio.py   Extract audio from the MELD archives
  prompts.py         Input settings and answer/judge prompts
  infer.py           Generate answers with Qwen2.5-Omni or Gemma 4
  metrics.py         Token F1, Wilson intervals, McNemar test
  score.py           Judge answers and write result tables
data/meld_qa.csv     QA dataset
results/             Predictions, verdicts, corrections and summary tables
slurm/               Batch script
tests/               Unit tests
```

## Background

AdaMOSS (Adaptive Modality Routing in Spoken-language Systems) is a project for CSE 5525:
Speech & Language Processing, by Group 11: Bohan Cui, Haojun Li, Mingyi Huang, Abraham
Owodunni, Cristian Bautista and Fan Yang. These results are the fixed-modality baseline
that a router will be compared against.
