#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${BREEZE_MODEL_DIR:-$repo_dir/../breeze-tts-2}"
gpu="${BREEZE_CUDA_DEVICE:-0}"
host="${BREEZE_HOST:-127.0.0.1}"
port="${BREEZE_PORT:-7860}"
depth_mode="${BREEZE_DEPTH_MODE:-compiled}"
stream_chunk_frames="${BREEZE_STREAM_CHUNK_FRAMES:-2}"

if [[ ! -f "$model_dir/model.safetensors.index.json" ]]; then
  echo "Breeze model is incomplete: $model_dir" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$gpu"
exec "$repo_dir/.venv/bin/python" -m breeze_infer.api "$model_dir" \
  --host "$host" \
  --port "$port" \
  --depth-mode "$depth_mode" \
  --stream-chunk-frames "$stream_chunk_frames"
