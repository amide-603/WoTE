#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export NAVSIM_DEVKIT_ROOT="$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM dataset directory}"
: "${NUPLAN_MAPS_ROOT:?Set NUPLAN_MAPS_ROOT to the maps directory}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to the experiment directory}"
: "${WOTE_CHECKPOINT_PATH:?Set WOTE_CHECKPOINT_PATH to the checkpoint to evaluate}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"

cache_override=()
if [[ -n "${WOTE_METRIC_CACHE_ROOT:-}" ]]; then
  cache_override+=("metric_cache_path=$WOTE_METRIC_CACHE_ROOT/navtest")
fi

python ./navsim/planning/script/run_pdm_score.py \
  agent=WoTE_agent \
  "agent.checkpoint_path='$WOTE_CHECKPOINT_PATH'" \
  experiment_name=eval/WoTE/default \
  split=test scene_filter=navtest \
  "${cache_override[@]}" \
  "$@"
