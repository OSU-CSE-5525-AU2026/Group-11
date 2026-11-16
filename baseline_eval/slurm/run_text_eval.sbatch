#!/bin/bash
# Run a text-only baseline and score it on one GPU on OSC Ascend. Submit from baseline_eval/
# (create logs/ first):
#   sbatch --account=<your-account> slurm/run_text_eval.sbatch <run-name> <model-id> [kind]
# For example:
#   sbatch --account=<your-account> slurm/run_text_eval.sbatch qwen2.5-7b-instruct Qwen/Qwen2.5-7B-Instruct
#SBATCH --job-name=adamoss-text
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

RUN_DIR="results/${1:?run name required}"
MODEL_ID="${2:?model id required}"
source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'No usable GPU'; print(torch.cuda.get_device_name(0))"

python -m adamoss.infer_text --run-dir "$RUN_DIR" --model-id "$MODEL_ID" ${3:+--kind "$3"}
python -m adamoss.score --run-dir "$RUN_DIR"
