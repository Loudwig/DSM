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
EPOCHS=70
BATCH_SIZE=64
EXP_NAME="comparaison"
EVAL_EVERY=1000
NUM_WORKERS=4
GRAD_CLIP=1  # 0 pour désactiver

SIGMA_MIN="1e-2"
SIGMA_MAX="40"

SIGMA_EMB_DIM=16
IMG_SIZE=64

# Model hyperparameters
BASE_CH=128
CHANNEL_MULTS="1,1,2,2,4,4"

# EMA + epoch checkpointing
EMA_DECAY=0.999                 # VE-SDE typical
SAVE_EPOCH_INTERVAL=5           # save every 5 epochs
SAVE_EPOCH_START_FRAC=0.7       # start saving after half of training

# ---- Env ----
source ~/.venvs/testpip/bin/activate

# unbuffered prints (pour voir les logs en live)
export PYTHONUNBUFFERED=1

# ---- Run ----
srun python -u train_ema.py \
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
  --img-size "$IMG_SIZE" \
  --ema-decay "$EMA_DECAY" \
  --save-epoch-interval "$SAVE_EPOCH_INTERVAL" \
  --save-epoch-start-frac "$SAVE_EPOCH_START_FRAC"

echo "Job finished at: $(date)"
