#!/bin/bash
#SBATCH --job-name=celebA_trainning
#SBATCH --output=runs/%x_%j.out      # stdout goes to runs/<job>_<id>.out
#SBATCH --error=runs/%x_%j.err       # stderr goes to runs/<job>_<id>.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Make sure base runs/ dir exists BEFORE job starts
mkdir -p runs

# ---- Hyperparameters ----
LR="1e-4"
EPOCHS=20
BATCH_SIZE=64
EXP_NAME="debugging"
EVAL_EVERY=100
NUM_WORKERS=4
GRAD_CLIP=0  # 0 pour désactiver

# Sigma hyperparameters
SIGMA_MIN="1e-1"
SIGMA_MAX="0.5"
N_SIGMAS=10
SIGMA_SCHEDULE="lin"

# Model hyperparameters
BASE_CH=64
CHANNEL_MULTS="1,2,4,8"

# ---- Env ----
source ~/.venvs/testpip/bin/activate

# unbuffered prints (pour voir les logs en live)
export PYTHONUNBUFFERED=1

# ---- Run ----
srun python -u train.py \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --exp-name "$EXP_NAME" \
  --eval-every "$EVAL_EVERY" \
  --num-workers "$NUM_WORKERS" \
  --grad-clip "$GRAD_CLIP" \
  --sigma-min "$SIGMA_MIN" \
  --sigma-max "$SIGMA_MAX" \
  --n-sigmas "$N_SIGMAS" \
  --sigma-schedule "$SIGMA_SCHEDULE" \
  --base-ch "$BASE_CH" \
  --channel-mults "$CHANNEL_MULTS"

echo "Job finished at: $(date)"
