"""Extract the audio track of every clip used by the QA set into 16 kHz mono WAV files.

Clips are streamed out of the original MELD archives, which avoids unpacking all ~10 GB.
Archives are selected by split because clip names repeat across splits with different
content: dia0_utt0.mp4 in the dev archive and in the test archive are unrelated clips.

    python -m adamoss.prepare_audio --meld-root /path/to/MELD.Raw
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path, PurePosixPath

import av
import numpy as np
import soundfile as sf

from adamoss.data import load_examples

SAMPLE_RATE = 16_000
ARCHIVES = {"train": "train.tar.gz", "dev": "dev.tar.gz", "test": "test.tar.gz"}


def decode_audio(video_file) -> np.ndarray:
    """Decode the audio stream of an open video file to mono float32 at 16 kHz."""
    with av.open(video_file) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(audio=0):
            chunks.extend(out.to_ndarray().ravel() for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray().ravel() for out in resampler.resample(None))
    return np.concatenate(chunks)


def extract_split(archive: Path, clips: set[str], output_dir: Path) -> set[str]:
    """Convert the requested clips from one archive and return those still missing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining = {clip for clip in clips if not (output_dir / f"{clip}.wav").exists()}
    if not remaining:
        return set()

    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if member.isfile() and path.suffix == ".mp4" and path.stem in remaining:
                # Streamed tar members are not seekable, which PyAV requires.
                audio = decode_audio(io.BytesIO(tar.extractfile(member).read()))
                sf.write(output_dir / f"{path.stem}.wav", audio, SAMPLE_RATE)
                remaining.remove(path.stem)
                if not remaining:
                    break
    return remaining


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meld-root", type=Path, required=True, help="Folder holding the *.tar.gz files")
    parser.add_argument("--qa-csv", type=Path, default=Path("data/meld_qa.csv"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    args = parser.parse_args()

    clips_by_split: dict[str, set[str]] = {}
    for example in load_examples(args.qa_csv):
        clips_by_split.setdefault(example.split, set()).add(example.clip)

    missing = []
    for split, clips in sorted(clips_by_split.items()):
        unresolved = extract_split(args.meld_root / ARCHIVES[split], clips, args.audio_dir / split)
        print(f"{split}: {len(clips) - len(unresolved)}/{len(clips)} clips ready")
        missing.extend(f"{split}/{clip}" for clip in sorted(unresolved))

    if missing:
        raise SystemExit(f"Clips not found in the archives: {', '.join(missing)}")


if __name__ == "__main__":
    main()
