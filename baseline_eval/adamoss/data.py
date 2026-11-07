"""Load and clean the MELD QA spreadsheet export."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_CLIP_PATTERN = re.compile(r"(dia\d+_utt\d+)")

# Rows whose file name points at a different utterance from their transcript, found by
# comparing the transcripts against the official MELD CSVs.
CLIP_CORRECTIONS = {"A004": "dia9_utt1", "A008": "dia20_utt15"}


@dataclass(frozen=True)
class QAExample:
    qid: str
    split: str
    clip: str
    transcript: str
    question: str
    gold_answer: str
    question_type: str
    review_status: str

    def audio_path(self, audio_dir: Path) -> Path:
        """Clip names repeat across MELD splits, so the path must include the split."""
        return audio_dir / self.split / f"{self.clip}.wav"


def load_examples(csv_path: Path) -> list[QAExample]:
    """Read the QA CSV, handling the inconsistencies present in the export.

    - Rows without a question (spreadsheet separators) are dropped.
    - Clip names are normalised, since some lack ``.mp4`` or have a trailing dot, and
      known wrong clips are replaced using ``CLIP_CORRECTIONS``.
    - ``question_type_corrected`` is used, as it holds the reviewers' final label.
    - Rows without an ID get one derived from their split and clip.
    """
    examples = []
    seen_per_clip: Counter[tuple[str, str]] = Counter()

    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question = row["question"].strip()
            if not question:
                continue

            match = _CLIP_PATTERN.search(row["audio_video_file"])
            if match is None:
                raise ValueError(f"Cannot parse clip name: {row['audio_video_file']!r}")
            split = row["split"].strip()
            clip = CLIP_CORRECTIONS.get(row[""].strip(), match.group(1))

            seen_per_clip[(split, clip)] += 1
            qid = row[""].strip() or f"{split}-{clip}-q{seen_per_clip[(split, clip)]}"

            examples.append(
                QAExample(
                    qid=qid,
                    split=split,
                    clip=clip,
                    transcript=row["transcript"].strip(),
                    question=question,
                    gold_answer=row["gold_answer"].strip(),
                    question_type=row["question_type_corrected"].strip(),
                    review_status=row["review_status"].strip(),
                )
            )

    duplicates = [qid for qid, count in Counter(e.qid for e in examples).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate question IDs: {duplicates}")
    return examples
