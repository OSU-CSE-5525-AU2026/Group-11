#!/bin/bash
# Run an audio-only evaluation and score it on one GPU on OSC Ascend. Submit from baseline_eval/
# (create logs/ first). The audio must already be prepared with adamoss.prepare_audio.
#
# Audio LLM, audio-only setting (no transcript):
#   sbatch --account=<acct> slurm/run_audio_only_eval.sbatch llm <run-name> <audio-llm-id>
# Whisper -> text LLM cascade (audio only, through ASR text):
#   sbatch --account=<acct> slurm/run_audio_only_eval.sbatch asr <run-name> <text-llm-id> [asr-id]
#SBATCH --job-name=adamoss-audio
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

MODE="${1:?mode required: llm or asr}"
RUN_DIR="results/${2:?run name required}"
MODEL_ID="${3:?model id required}"
source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'No usable GPU'; print(torch.cuda.get_device_name(0))"

case "$MODE" in
  llm) python -m adamoss.infer --run-dir "$RUN_DIR" --model-id "$MODEL_ID" --settings audio_only ;;
  asr) python -m adamoss.infer_asr --run-dir "$RUN_DIR" --llm-id "$MODEL_ID" ${4:+--asr-id "$4"} ;;
  *) echo "mode must be llm or asr" >&2; exit 1 ;;
esac
python -m adamoss.score --run-dir "$RUN_DIR"
