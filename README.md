# MELD QA Construction Dataset

## Project Overview

This project studies **adaptive modality routing for spoken-language question answering**.  
The main idea is that different questions may require different input modalities:

- **TEXT**: the transcript alone is sufficient.
- **AUDIO**: acoustic or paralinguistic information is needed.
- **BOTH**: the answer requires complementary information from both the transcript and the audio.

Our goal is to build a small controlled QA dataset that can be used to test whether a router can correctly choose between **TEXT, AUDIO, and BOTH** for each query. The dataset will later support modality-ablation experiments and adaptive routing experiments.

## Source Dataset

The questions are constructed from **MELD (Multimodal EmotionLines Dataset)**. Each MELD sample provides information such as the transcript, speaker, emotion/sentiment labels, dialogue ID, utterance ID, and the corresponding audio/video clip.

Each constructed QA example is linked back to its original MELD sample using:

- `split`
- `dialogue_id`
- `utterance_id`
- `audio_video_file`

This allows every question to be traced back to the original data.

## Dataset Construction

Each team member first selected one modality category and created a set of questions from MELD samples. Questions were designed to have a clear gold answer and an intended modality requirement.

The construction table contains **17 columns**:

1. `example_id` – unique ID for the constructed QA example  
2. `split` – MELD split (`train`, `dev`, or `test`)  
3. `dialogue_id` – original MELD dialogue ID  
4. `utterance_id` – original MELD utterance ID  
5. `speaker` – speaker name  
6. `transcript` – original MELD transcript  
7. `emotion` – MELD emotion label  
8. `sentiment` – MELD sentiment label  
9. `audio_video_file` – corresponding media filename  
10. `question_type` – intended route: `TEXT`, `AUDIO`, or `BOTH`  
11. `question` – constructed QA question  
12. `gold_answer` – expected answer  
13. `route_reason` – explanation of why the assigned modality is needed  
14. `created_by` – team member who created the question  
15. `review_status` – review state (for example, `Needs Review` or `Approved`)  
16. `reviewed_by` – teammate who reviewed the question  
17. `reviewer_notes` – comments or suggested revisions from the reviewer  

## Question-Type Guidelines

### TEXT
The transcript should be sufficient to answer the question. Raw audio should not provide information necessary for the answer.

### AUDIO
The question should require acoustic or paralinguistic information that cannot be reliably recovered from the transcript alone, such as tone, pauses, speaker characteristics, emphasis, or other audio cues.

### BOTH
The question should require complementary information from both modalities. For example, audio may be needed to identify a tone or event, while the transcript is needed to recover the corresponding lexical content or speaker identity.

Gold answers do not have to come directly from an existing MELD label. Human-derived answers are acceptable when they are clear, reproducible, and can be confirmed by another reviewer.

## Review Process

After question construction, each set was reviewed by another team member. Reviewers checked the following criteria:

1. **Gold Answer Reliability**  
   The answer should be clear, specific, and reasonably reproducible by another reviewer.

2. **Route Validity**  
   The assigned route should match the actual information required:
   - TEXT: transcript is sufficient
   - AUDIO: acoustic/paralinguistic information is necessary
   - BOTH: both text and audio provide necessary information

3. **Question Diversity**  
   Questions should cover different information needs instead of repeatedly using the same template.

4. **Avoiding Router Shortcuts**  
   Questions should not make the correct route trivially identifiable from a repeated keyword or phrasing pattern. The router should ideally infer what information is required by the query.

5. **Data Consistency**  
   The split, dialogue ID, utterance ID, speaker, transcript, and media filename should match the original MELD sample.

Reviewers recorded feedback in `reviewer_notes` and updated `review_status` when a question was approved or required revision.

## Next Step

The reviewed QA dataset will be used for a controlled modality-ablation experiment under three inference conditions:

- Transcript Only
- Audio Only
- Audio + Transcript

These results will be used to test whether different question types show different modality requirements and, later, whether an adaptive router can achieve a better accuracy-efficiency tradeoff than fixed modality strategies.

## Modality-Ablation Baseline Results

We ran this experiment on the 81-question version of the dataset with two audio LLMs, **Qwen2.5-Omni-7B** and **Gemma 4 E4B** (`google/gemma-4-E4B-it`). Every question was asked in all three conditions, and answers were graded against the gold answer by an LLM judge (Qwen2-7B-Instruct), with clear judge errors corrected by hand. The table shows correct answers out of the total, split by the question's `question_type_corrected` label.

| Question type | Model | Transcript Only | Audio Only | Audio + Transcript |
| --- | --- | --- | --- | --- |
| TEXT (26) | Qwen | 25 | 22 | 25 |
| | Gemma | 25 | 19 | 25 |
| AUDIO (28) | Qwen | 14 | 15 | 16 |
| | Gemma | 8 | 12 | 12 |
| BOTH (27) | Qwen | 14 | 12 | 12 |
| | Gemma | 12 | 12 | 15 |
| All (81) | Qwen | 53 | 49 | 53 |
| | Gemma | 45 | 43 | 52 |

Main findings:

- **Without audio, both models do much worse on AUDIO questions than on TEXT questions** (transcript only: 25/26 on TEXT vs 14/28 and 8/28 on AUDIO), which supports the idea that these questions need audio.
- **Giving the audio does not fully close the gap:** AUDIO questions reach only 15/28 and 12/28 with audio alone, so they are also hard in general.
- **Without the transcript, TEXT accuracy drops,** mostly because the models mishear names and words.
- **Having both modalities helps Gemma** (52/81 vs 45/81 transcript only) **but not Qwen.**

With about 27 questions per type, most differences are not statistically significant. Code, per-question outputs, confidence intervals, significance tests and instructions for rerunning are in [`baseline_eval/`](baseline_eval/README.md).
