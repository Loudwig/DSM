#!/bin/bash
#SBATCH --job-name=celebA_ve_sde
#SBATCH --output=runs/%x_%j.out      # stdout goes to runs/<job>_<id>.out
#SBATCH --error=runs/%x_%j.err       # stderr goes to runs/<job>_<id>.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Make sure base runs/ dir exists BEFORE job starts
mkdir -p runs

# ---- Hyperparameters ----
LR="1e-4"
EPOCHS=40
BATCH_SIZE=32
EXP_NAME="comparaison"
EVAL_EVERY=100
NUM_WORKERS=4
GRAD_CLIP=1  # 0 pour désactiver

# Sigma hyperparameters (VE SDE: sigma(t) in [SIGMA_MIN, SIGMA_MAX])
SIGMA_MIN="1e-2"
SIGMA_MAX="1.5"

SIGMA_EMB_DIM=16
IMG_SIZE=64

# Model hyperparameters
BASE_CH=128
CHANNEL_MULTS="1,2,2,4,4"

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
  --base-ch "$BASE_CH" \
  --channel-mults "$CHANNEL_MULTS" \
  --sigma-emb-dim "$SIGMA_EMB_DIM" \
  --img-size "$IMG_SIZE"

echo "Job finished at: $(date)"
