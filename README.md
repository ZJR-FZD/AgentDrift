# AgentDrift — Data Synthesis

Code release for the **data synthesis pipeline** of **AgentDrift** (authorization drift benchmark for LLM agents).

This repository contains only the synthesis / quality-control tooling used to build benchmark samples from agent skill documents. **Benchmark data is not included** and will not be published in this repo.

Paper: *coming soon (arXiv)*.

---

## What this repo includes

| Path | Role |
|------|------|
| `1_data_generation/` | Stage 1 (playbook) → Stage 2 (env codegen) → smoke test → trajectory QC + critic |
| `2_trajectory_generation/` | Minimal agent rollout (`eval.py` + OpenAI-compatible model clients) used **inside** generation-time QC |
| `environments/` | `BaseEnv` / `EnvManager` sandbox loader |
| `3_evaluation/metrics/rule_based.py` | Deterministic TC / drift gates used by smoke test & QC |
| `scripts/01_data_gen*.sh` | Batch entrypoints |
| `docs/PIPELINE.md` | Detailed pipeline notes |

**Not included:** raw skill corpora, generated `data.json` / environments, formal evaluation harness, rollout results.

---

## Pipeline (overview)

```
skill .md  →  Stage1 playbook  →  Stage2 Env.py / Env.json / sample.json
                 →  smoke_test  →  (optional) N× agent trajectories + critic QC
                 →  data/processed_data/<Category>/
```

Design idea: instructions are **de-reciped** (information gap); `required_tool_calls` define a minimal compliant path; `violation_tool_calls` encode authorization-boundary expansions that look helpful but are out of scope.

---

## Setup

```bash
# Python 3.9+ recommended (Linux; Stage1/2 uses fcntl for data.json locking)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export AIGCBEST_URL="https://YOUR_OPENAI_COMPATIBLE_BASE/v1"
export AIGCBEST_API_KEY="YOUR_KEY"
```

Place skill markdown under:

```text
data/raw_data/<Category>/<SubCategory>/<skill>.md
```

Example categories used in our runs: `Security`, `IT_and_Cloud`, `Data_and_Research`, …

Configure local QC agent endpoints (OpenAI-compatible) via env, e.g.:

```bash
export ADB_LOCAL_CHAT_ENDPOINTS_JSON='[
  ["qwen36_35b_a3b", "http://127.0.0.1:8010/v1/chat/completions"],
  ["llama33_70b_instruct", "http://127.0.0.1:8011/v1/chat/completions"]
]'
```

Registry keys for `ADB_QC_AGENT_MODELS` are defined in `2_trajectory_generation/model_api/config.py` (`QC_TRAJECTORY_AGENT_OPENAI`).

---

## Quick start

**Single-category / small batch**

```bash
export AIGCBEST_URL=... AIGCBEST_API_KEY=...
SOURCE=Security MAX_TOTAL=5 bash scripts/01_data_gen.sh
```

**Concurrent multi-category**

```bash
export AIGCBEST_URL=... AIGCBEST_API_KEY=...
SAMPLES_PER_CATEGORY=10 MAX_WORKERS=8 ADB_OUTPUT_BATCH=full_v1 \
  bash scripts/01_data_gen_concurrent.sh
```

**QC only** (existing sample under `data/processed_data/`)

```bash
RUN_QC_ONLY=1 QC_CATEGORY=Security SAMPLE_ID=<id> bash scripts/01_data_gen.sh
```

Outputs land in `data/processed_data/<Category>/` (environments, playbooks, `data.json`, `qc_status/`, `smoke_reports/`). Generation-time trajectories go to `qc_runs/` (gitignored).

---

## Data policy

- This code release **does not ship** skill dumps or synthesized benchmark instances.
- Treat anything under `data/raw_data/` and `data/processed_data/` as local artifacts; both are gitignored except placeholders.
- If you re-run synthesis, do not commit API keys or private model endpoints.

---

## Citation

```bibtex
@article{agentdrift2026,
  title   = {AgentDrift: ...},
  author  = {...},
  year    = {2026},
  note    = {arXiv preprint}
}
```

---

## License

Code in this repository is released for research use. See `LICENSE` if present; otherwise contact the authors.
