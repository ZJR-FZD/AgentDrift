#!/usr/bin/env bash
# =============================================================================
# 01_concurrent — Concurrent data generation across all categories
#
# Uses ProcessPoolExecutor to process multiple skills in parallel.
# Each worker runs Stage1 (playbook) + Stage2 (codegen) + optional QC.
#
# Before running:
#   1. Set API credentials below or via environment
#   2. Adjust SAMPLES_PER_CATEGORY and MAX_WORKERS as needed
#   3. Optionally set ADB_OUTPUT_BATCH to tag output directories
#
# Usage:
#   bash scripts/01_data_gen_concurrent.sh                    # all categories
#   CATEGORY=Security bash scripts/01_data_gen_concurrent.sh  # one category
#   DRY_RUN=1 bash scripts/01_data_gen_concurrent.sh          # preview only
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT/1_data_generation"

# ── API credentials ──────────────────────────────────────────────────────────
export AIGCBEST_URL="${AIGCBEST_URL:-https://api.openai.com/v1}"
export AIGCBEST_API_KEY="${AIGCBEST_API_KEY:-}"  # required: export AIGCBEST_API_KEY="sk-..."
if [ -z "${AIGCBEST_API_KEY}" ]; then
  echo "[01_concurrent] warning: AIGCBEST_API_KEY is empty; Stage1/2 and critic calls will fail." >&2
fi

# Remote API retry settings
export ADB_REMOTE_API_MAX_RETRIES="${ADB_REMOTE_API_MAX_RETRIES:-8}"
export ADB_REMOTE_API_RETRY_DELAY_SEC="${ADB_REMOTE_API_RETRY_DELAY_SEC:-2}"

# ── Generation parameters ────────────────────────────────────────────────────
S1_MODEL="${S1_MODEL:-claude-sonnet-4-6}"
S2_MODEL="${S2_MODEL:-claude-sonnet-4-6}"
SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-10}"
MAX_WORKERS="${MAX_WORKERS:-8}"
AUTH_TYPE="${AUTH_TYPE:-implicit}"
SEED="${SEED:-42}"

# ── Trajectory QC ────────────────────────────────────────────────────────────
ADB_TRAJECTORY_QC="${ADB_TRAJECTORY_QC:-1}"
ADB_QC_MAX_ROUNDS="${ADB_QC_MAX_ROUNDS:-3}"
ADB_QC_SAMPLES_PER_MODEL="${ADB_QC_SAMPLES_PER_MODEL:-4}"
ADB_QC_NUM_RUNS="${ADB_QC_NUM_RUNS:-3}"
ADB_QC_AGENT_MODEL="${ADB_QC_AGENT_MODEL:-adb-qc-qwen36-35b-a3b}"
export ADB_QC_CRITIC_MODEL="${ADB_QC_CRITIC_MODEL:-claude-sonnet-4-6}"
export ADB_QC_AGENT_MODELS="${ADB_QC_AGENT_MODELS-adb-qc-qwen36-35b-a3b,adb-qc-llama4-maverick-17b}"
export ADB_TRAJECTORY_QC ADB_QC_NUM_RUNS ADB_QC_SAMPLES_PER_MODEL ADB_QC_MAX_ROUNDS ADB_QC_AGENT_MODEL

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
  python generate_concurrent.py
  --s1_model "$S1_MODEL"
  --s1_base_url "${AIGCBEST_URL}"
  --s1_api_key "${AIGCBEST_API_KEY}"
  --s2_model "$S2_MODEL"
  --s2_base_url "${AIGCBEST_URL}"
  --s2_api_key "${AIGCBEST_API_KEY}"
  --samples_per_category "$SAMPLES_PER_CATEGORY"
  --max_workers "$MAX_WORKERS"
  --auth_type "$AUTH_TYPE"
  --seed "$SEED"
  --qc_agent_model "$ADB_QC_AGENT_MODEL"
  --qc_num_runs "$ADB_QC_NUM_RUNS"
  --qc_samples_per_model "$ADB_QC_SAMPLES_PER_MODEL"
  --qc_max_rounds "$ADB_QC_MAX_ROUNDS"
)

if [ "${ADB_TRAJECTORY_QC}" = "1" ]; then
  CMD+=(--trajectory_qc)
fi

if [ -n "${ADB_QC_AGENT_MODELS:-}" ]; then
  CMD+=(--qc_agent_models "$ADB_QC_AGENT_MODELS")
fi

if [ -n "${CATEGORY:-}" ]; then
  CMD+=(--category "$CATEGORY")
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  CMD+=(--dry_run)
fi

echo "[01_concurrent] Running: ${CMD[*]}"
"${CMD[@]}"
