#!/usr/bin/env bash

# Usage:
#   ./make_ald_video.sh <frames_dir> <SNR> <T>
#
# Exemple:
#   ./make_ald_video.sh runs/baseline/002/videos/out 0.01 50

FRAMES_DIR="$1"
SNR="$2"
T="$3"

# Dossier où sauver la vidéo = parent du dossier de frames
OUT_DIR="$(dirname "$FRAMES_DIR")"

FPS=10

ffmpeg -y -framerate "$FPS" \
  -i "${FRAMES_DIR}/frame_%05d.png" \
  -c:v libx264 -pix_fmt yuv420p \
  "${OUT_DIR}/ald_${SNR}_${T}_video.mp4"
