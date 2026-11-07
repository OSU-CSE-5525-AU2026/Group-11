#!/bin/bash
# Run inference and scoring on one GPU on OSC Ascend. Submit from baseline_eval/ (create logs/ first):
#   sbatch --account=<your-account> slurm/run_eval.sbatch [run-name] [model-id]
#SBATCH --job-name=adamoss-eval
#SBATCH --clusters=ascend
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

RUN_DIR="results/${1:-qwen2.5-omni-7b}"
source .venv/bin/activate

# Without a usable GPU the model would silently fall back to the CPU.
python -c "import torch; assert torch.cuda.is_available(), 'No usable GPU'; print(torch.cuda.get_device_name(0))"

python -m adamoss.infer --run-dir "$RUN_DIR" --model-id "${2:-Qwen/Qwen2.5-Omni-7B}"
python -m adamoss.score --run-dir "$RUN_DIR"
