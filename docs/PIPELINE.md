# AgentDrift — Data Synthesis Pipeline

This document describes the **public code release**: skill → playbook → environment codegen → smoke test → optional trajectory QC. Formal evaluation / ADS metrics are **out of scope** for this repository.

---

## 1. Idea

**Authorization drift**: a capable agent gradually expands beyond the user’s authorized scope while trying to be helpful. Single steps look reasonable; the overall trajectory violates the boundary.

Mechanisms used in synthesis:

- **De-recipe instructions** — no API cookbook; leave a natural information gap.
- **Compliant path** — `required_tool_calls` is a minimal chain that can finish the task.
- **Drift path** — `violation_tool_calls` fill the gap by over-reaching; env returns may contain soft, non-imperative hints.

---

## 2. Stage 1–2

| Item | Path |
|------|------|
| Entry | `1_data_generation/generate.py` |
| Concurrent | `1_data_generation/generate_concurrent.py` |
| Shell | `scripts/01_data_gen.sh`, `scripts/01_data_gen_concurrent.sh` |
| Outputs | `data/processed_data/{Category}/` |

### Stage 1 — Playbook (`stage1.md`)

- **In**: `data/raw_data/{Category}/**/*.md` + sample metadata (`SAMPLE_ID`, `AUTHORIZATION_TYPE`, …).
- **Out**: playbook Markdown (or `PASS` if unsuitable).
- Artifacts: `stage1_playbooks_*/{id}_playbook.md`, `raw_outputs_*/{id}_s1_raw.txt`.

### Stage 2 — Codegen (`stage2.md`)

Per sample directory `environments_*/{sample_id}/`:

| File | Role |
|------|------|
| `{EnvClass}.py` | `BaseEnv` subclass |
| `{EnvClass}.json` | Tool schemas |
| `{sample_id}.json` | instruction + `annotation` (`required_tool_calls` / `violation_tool_calls` / …) |

Emit order (recommended): JSON schema → Python env → sample JSON. Samples are merged into `data/processed_data/{Category}/data.json`.

---

## 3. Smoke test

`1_data_generation/smoke_test.py` (after Stage 2, before QC):

1. Required tools exist; args non-placeholder  
2. Violation tools exist; args non-placeholder  
3. No required/violation name+args collision  
4. Synthetic compliant traj → task✓ drift✗; violation traj → drift✓  
5. No empty / `$VAR` / `<x>` placeholders  

Failed samples write `smoke_reports/` and should not enter the QC loop.

---

## 4. Generation-time trajectory QC (`qc_runs/`)

| Module | Role |
|--------|------|
| `trajectory_qc.py` | Orchestrate N trajectories + gates + critic rounds |
| `trajectory_qc_worker.py` | One eval subprocess per run |
| `trajectory_utils.py` | Compact traj text + gate helpers |
| `trajectory_critic.py` + `.md` | Critic + optional auto-fix of env trio / `data.json` |
| `2_trajectory_generation/eval.py` | Agent loop for QC sampling |

Default grid (see shell scripts): **2 agent models × K samples** (e.g. K=4 → 8 traj/round), up to `ADB_QC_MAX_ROUNDS` (default 3). Critic uses a remote model (`ADB_QC_CRITIC_MODEL`, temperature 0).

Gate (multi-model): need compliant success from ≥2 models and drift tool use from ≥2 models (see `trajectory_qc.py` docstring). Failures after max rounds → `qc_status/{id}.json` with `needs_human_review`.

QC agent endpoints: set `ADB_LOCAL_CHAT_ENDPOINTS_JSON` or edit placeholders in `2_trajectory_generation/model_api/config.py`.

---

## 5. Environment variables (common)

| Variable | Purpose |
|----------|---------|
| `AIGCBEST_URL` / `AIGCBEST_API_KEY` | OpenAI-compatible gateway for Stage1/2 + critic |
| `ADB_OUTPUT_BATCH` | Tag output dirs (`environments_<batch>/`, …) |
| `ADB_TRAJECTORY_QC` | `1` to enable QC after codegen |
| `ADB_QC_MAX_ROUNDS` | Critic outer rounds |
| `ADB_QC_SAMPLES_PER_MODEL` | K in 2×K grid (`1` = legacy N-run mode) |
| `ADB_QC_AGENT_MODELS` | Comma-separated eval keys (exactly 2 in grid mode) |
| `ADB_QC_CRITIC_MODEL` | Critic model id |
| `ADB_LOCAL_CHAT_ENDPOINTS_JSON` | QC agent `(api_model_id, chat_url)` list |
| `ADB_REMOTE_API_MAX_RETRIES` | Retries for remote Stage/critic calls |

---

## 6. Module index

| Module | Path |
|--------|------|
| Stage 1–2 | `1_data_generation/generate.py` |
| Concurrent | `1_data_generation/generate_concurrent.py` |
| Parser | `1_data_generation/parser.py` |
| LLM client | `1_data_generation/llm_api.py` |
| Smoke | `1_data_generation/smoke_test.py` |
| QC | `1_data_generation/trajectory_qc.py` |
| Critic | `1_data_generation/trajectory_critic.py` |
| Rule gates | `3_evaluation/metrics/rule_based.py` |
| Env infra | `environments/BaseEnv.py`, `EnvManager.py` |
| QC rollout | `2_trajectory_generation/eval.py` |
