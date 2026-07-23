#!/usr/bin/env bash
# =============================================================================
# 01 — Data generation (Stage 1 → Stage 2) + optional post-codegen QC
#
# Before running, set API credentials (or export them below):
#   export AIGCBEST_URL="https://..."
#   export AIGCBEST_API_KEY="sk-..."
# QC critic default: ADB_QC_CRITIC_MODEL (set in this script). Gateway keys still
# use AIGCBEST_* / ADB_QC_CRITIC_* (see trajectory_qc.py).
#
# Working directory can be anywhere; invoke via absolute path or repo-root relative:
#   bash /path/to/AgentDrift/scripts/01_data_gen.sh
#   cd /path/to/AgentDrift && bash scripts/01_data_gen.sh
# Do not run `bash scripts/01_data_gen.sh` from inside scripts/ (paths will break).
#
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT/1_data_generation"

# Set credentials via environment (do not commit secrets):
#   export AIGCBEST_URL="https://..."
#   export AIGCBEST_API_KEY="sk-..."
export AIGCBEST_URL="${AIGCBEST_URL:-https://api.openai.com/v1}"
export AIGCBEST_API_KEY="${AIGCBEST_API_KEY:-}"
if [ -z "${AIGCBEST_API_KEY}" ]; then
  echo "[01] warning: AIGCBEST_API_KEY is empty; Stage1/2 and critic calls will fail." >&2
fi

# Remote OpenAI-compatible API (incl. trajectory critic): retry on network / 429 / 5xx
# (llm_api.RemoteAPI)
export ADB_REMOTE_API_MAX_RETRIES="${ADB_REMOTE_API_MAX_RETRIES:-8}"
export ADB_REMOTE_API_RETRY_DELAY_SEC="${ADB_REMOTE_API_RETRY_DELAY_SEC:-2}"

# ── Post-codegen QC (default: 2 models × 4 samples = 8 trajectories per outer round) ──
#
# Outer loop (ADB_QC_MAX_ROUNDS, default 3):
#   Each outer round = (1) run all local trajectories → (2) gate check →
#   (3) if failed, one critic call (may auto-fix env files).
#   Up to 3 "edit / resample" chances — not "8×3 trajectories in one critic prompt".
#
# Trajectories per outer round / per critic prompt:
#   · Default (this script): ADB_QC_SAMPLES_PER_MODEL=4 and exactly 2 names in
#       ADB_QC_AGENT_MODELS → model1×4 then model2×4 → 8 trajectories in one critic.
#       ADB_QC_NUM_RUNS is ignored (total fixed at 2×4=8).
#   · If ADB_QC_SAMPLES_PER_MODEL=1: legacy mode → ADB_QC_NUM_RUNS trajectories
#       (default 3); short model lists are cycled.
#
ADB_TRAJECTORY_QC="${ADB_TRAJECTORY_QC:-1}"
ADB_QC_MAX_ROUNDS="${ADB_QC_MAX_ROUNDS:-3}"
ADB_QC_SAMPLES_PER_MODEL="${ADB_QC_SAMPLES_PER_MODEL:-4}"
ADB_QC_NUM_RUNS="${ADB_QC_NUM_RUNS:-3}"
ADB_QC_AGENT_MODEL="${ADB_QC_AGENT_MODEL:-adb-qc-qwen3-next-35b-a3b}"
# Remote critic used by trajectory QC (trajectory_qc / generate.py read ADB_QC_CRITIC_MODEL)
export ADB_QC_CRITIC_MODEL="${ADB_QC_CRITIC_MODEL:-claude-sonnet-4-6}"
# Exactly 2 eval registry names (paired with default SAMPLES_PER_MODEL=4)
export ADB_QC_AGENT_MODELS="${ADB_QC_AGENT_MODELS-adb-qc-qwen3-next-35b-a3b,adb-qc-llama-3-3-70b-instruct}"
export ADB_TRAJECTORY_QC ADB_QC_NUM_RUNS ADB_QC_SAMPLES_PER_MODEL ADB_QC_MAX_ROUNDS ADB_QC_AGENT_MODEL
export ADB_QC_AGENT_MODELS ADB_QC_CRITIC_MODEL
export ADB_REMOTE_API_MAX_RETRIES ADB_REMOTE_API_RETRY_DELAY_SEC

# --- Single-model / legacy 3-traj mode (no 2×4 grid): set before running ---
#   export ADB_QC_SAMPLES_PER_MODEL=1
#   export ADB_QC_NUM_RUNS=3
#   export ADB_QC_AGENT_MODELS=
#   export ADB_QC_AGENT_MODEL=adb-qc-qwen3-next-35b-a3b

# ---------------------------------------------------------------------------
# Mode A: existing samples — QC only (default: 8 traj/outer round = 2×4; max 3 rounds)
#
# Single sample:
#   RUN_QC_ONLY=1 SAMPLE_ID=security_1claw_implicit_001 bash scripts/01_data_gen.sh
#   (run from AgentDrift repo root, or use this script's absolute path)
#
# Batch (all sample ids in data/processed_data/<QC_CATEGORY>/data.json):
#   RUN_QC_ONLY=1 QC_CATEGORY=Security bash scripts/01_data_gen.sh
#   or: RUN_QC_ONLY=1 SAMPLE_ID=all QC_CATEGORY=Security bash scripts/01_data_gen.sh
# ---------------------------------------------------------------------------
if [ "${RUN_QC_ONLY:-0}" = "1" ]; then
  QC_CATEGORY="${QC_CATEGORY:-Security}"
  QC_ONLY_EXTRA=()
  if [ -n "$ADB_QC_AGENT_MODELS" ]; then
    QC_ONLY_EXTRA+=(--agent_models "$ADB_QC_AGENT_MODELS")
  fi

  _run_one_qc() {
    local sid="$1"
    echo "[01] RUN_QC_ONLY: trajectory_qc.py --source ${QC_CATEGORY} --sample_id ${sid}"
    python trajectory_qc.py --source "$QC_CATEGORY" --sample_id "$sid" \
      --agent_model "$ADB_QC_AGENT_MODEL" \
      --qc_num_runs "$ADB_QC_NUM_RUNS" \
      --qc_samples_per_model "$ADB_QC_SAMPLES_PER_MODEL" \
      "${QC_ONLY_EXTRA[@]}" \
      --max_qc_rounds "$ADB_QC_MAX_ROUNDS"
  }

  DATA_JSON="$PROJECT_ROOT/data/processed_data/$QC_CATEGORY/data.json"
  if [ ! -f "$DATA_JSON" ]; then
    echo "[01] error: missing $DATA_JSON" >&2
    exit 1
  fi

  if [ -n "${SAMPLE_ID:-}" ] && [ "$SAMPLE_ID" != "all" ]; then
    _run_one_qc "$SAMPLE_ID"
  else
    echo "[01] RUN_QC_ONLY: batch QC (all samples in $QC_CATEGORY/data.json), default 8 traj/outer round, max ${ADB_QC_MAX_ROUNDS} outer rounds"
    IDS="$(
      python3 -c "
import json, sys
path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    data = json.load(f)
for row in data:
    if isinstance(row, dict) and row.get('id'):
        print(row['id'])
" "$DATA_JSON"
    )"
    if [ -z "${IDS// /}" ]; then
      echo "[01] error: no sample ids parsed from data.json" >&2
      exit 1
    fi
    for sid in $IDS; do
      _run_one_qc "$sid" || exit $?
    done
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# Parameters (edit as needed)
# ---------------------------------------------------------------------------
# SOURCE: category dir name (e.g. Communication), or Category/skill_stem for one raw skill
SOURCE="${SOURCE:-Security}"
MAX_TOTAL="${MAX_TOTAL:-5}"

# If a sample for the same skill already exists, generation is skipped. To force
# regenerate, set both:
#   REGEN_ID=communication_aa_implicit_001
#   SOURCE=Communication/aa          (must include / as required by generate.py)
REGEN_ARGS=()
if [ -n "${REGEN_ID:-}" ]; then
  REGEN_ARGS=(--regen "$REGEN_ID")
fi

S1_MODEL="${S1_MODEL:-claude-sonnet-4-6}"
S2_MODEL="${S2_MODEL:-claude-sonnet-4-6}"
S1_BASE_URL="${S1_BASE_URL:-${AIGCBEST_URL:-}}"
S2_BASE_URL="${S2_BASE_URL:-${AIGCBEST_URL:-}}"
S1_API_KEY="${S1_API_KEY:-${AIGCBEST_API_KEY:-}}"
S2_API_KEY="${S2_API_KEY:-${AIGCBEST_API_KEY:-}}"

# Whether to run QC after codegen (writes qc_runs/): default 8 traj/outer + max 3 rounds

# STAGE: unset = 0 (S1+S2); STAGE=1 playbook only; STAGE=2 codegen only
# ---------------------------------------------------------------------------

QC_ARGS=()
if [ "${ADB_TRAJECTORY_QC}" = "1" ]; then
  QC_ARGS=(--trajectory_qc --qc_num_runs "${ADB_QC_NUM_RUNS}" --qc_samples_per_model "${ADB_QC_SAMPLES_PER_MODEL}" --qc_max_rounds "${ADB_QC_MAX_ROUNDS}" --qc_agent_model "${ADB_QC_AGENT_MODEL}")
  if [ -n "$ADB_QC_AGENT_MODELS" ]; then
    QC_ARGS+=(--qc_agent_models "$ADB_QC_AGENT_MODELS")
  fi
fi

STAGE_ARG=()
if [ -n "${STAGE:-}" ]; then
  STAGE_ARG=(--stage "$STAGE")
fi

python generate.py --source "$SOURCE" "${STAGE_ARG[@]}" \
  --s1_model "$S1_MODEL" --s1_base_url "$S1_BASE_URL" --s1_api_key "$S1_API_KEY" \
  --s2_model "$S2_MODEL" --s2_base_url "$S2_BASE_URL" --s2_api_key "$S2_API_KEY" \
  --max_total "$MAX_TOTAL" \
  "${REGEN_ARGS[@]}" \
  "${QC_ARGS[@]}"

# =============================================================================
# Examples (copy into a terminal; comments only)
# =============================================================================
# 1a) QC one existing sample (from repo root):
#   cd ~/AgentDrift && RUN_QC_ONLY=1 SAMPLE_ID=security_example_implicit_001 bash scripts/01_data_gen.sh
# 1b) QC all samples in that category's data.json (default 8 traj/round, max 3 rounds):
#   cd ~/AgentDrift && RUN_QC_ONLY=1 QC_CATEGORY=Security bash scripts/01_data_gen.sh
#
# 2) Force regenerate one sample (playbook/code); need Category/skill + regen id:
#   REGEN_ID=security_example_implicit_001 SOURCE=Security/example bash scripts/01_data_gen.sh
#
# 3) New sample + auto QC after codegen (default 2 models × 4 / round):
#   ADB_TRAJECTORY_QC=1 SOURCE=Security/example MAX_TOTAL=1 bash scripts/01_data_gen.sh
# =============================================================================
