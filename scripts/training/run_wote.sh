#!/usr/bin/env bash
set -euo pipefail

# Activate the local environment first; do not use the upstream author's paths.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export NAVSIM_DEVKIT_ROOT="$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM dataset directory}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the maps directory}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to the experiment directory}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"

# Official training: 8 devices x 16 scenes, 30 epochs. The NPU adapter
# currently supports only 32-true precision.
accelerator="${WOTE_ACCELERATOR:-gpu}"
precision=16-mixed
if [[ "$accelerator" == npu ]]; then
  precision=32-true
fi

python ./navsim/planning/script/run_training.py \
  agent=WoTE_agent \
  experiment_name=WoTE/default \
  scene_filter=navtrain split=trainval \
  dataloader.params.batch_size=16 \
  trainer.params.max_epochs=30 \
  trainer.params.accelerator="$accelerator" \
  trainer.params.strategy=ddp \
  trainer.params.precision="$precision" \
  +trainer.params.devices=8 \
  "$@"
