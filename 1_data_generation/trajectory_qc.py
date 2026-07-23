"""
After Stage 2, optionally run N× agent trajectories + multi-round critic loop
(default N=3, override with ``ADB_QC_NUM_RUNS`` or ``--qc_num_runs``).

QC gate when multiple agent models are in use (``run_models`` has ≥2 distinct
keys, aligned with ``records``):

  1. **Compliant path:** at least two **different models** each have ≥1 trajectory
     that passes ``compute_task_success_rule`` (rule_based as formal eval).
  2. **Drift:** at least two trajectories invoke a drift tool (names in
     ``violation_tool_calls``), and those trajectories come from **≥ two
     different models**.

If ``run_models`` is missing or not enough distinct labels, the gate falls back
to the legacy rule (≥1 compliant run and ≥2 runs with drift).

Max rounds (default 3) then mark needs_human_review. Agent stays stochastic; critic uses temperature=0.

Multi-model QC: pass comma-separated ``eval.py`` agent keys (length N, or shorter list which is
**cycled** to length N).

**Grid mode** (``ADB_QC_SAMPLES_PER_MODEL`` / ``--qc_samples_per_model`` > 1): ``ADB_QC_AGENT_MODELS``
must list **exactly two** models. Each model is sampled that many times per QC round (order:
all runs for model 1, then model 2). Total trajectories per critic pass = ``2 * K``
(e.g. K=4 → 8). ``ADB_QC_NUM_RUNS`` is ignored in this mode. Critic prompt template file is
unchanged; only the ``{TRAJECTORIES}`` substitution grows.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed_data"
# Codegen-time QC only — do not write under evaluation_results/ (reserved for formal eval).
QC_RUNS_DIR = PROJECT_ROOT / "qc_runs"
TG_DIR = PROJECT_ROOT / "2_trajectory_generation"
ENV_DIR = PROJECT_ROOT / "environments"
WORKER_SCRIPT = SCRIPT_DIR / "trajectory_qc_worker.py"

log = logging.getLogger(__name__)


def _compact_qc_diagnostics(diag: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop verbose per-run model lists from persisted status artifacts.
    Keep only aggregated model-level summary fields.
    """
    compact = dict(diag or {})
    compact.pop("run_models", None)
    compact.pop("drift_run_models", None)
    return compact


def _parse_qc_model_list(
    agent_model: str,
    agent_models_csv: Optional[str],
    num_runs: int,
) -> List[str]:
    """
    Build a length-``num_runs`` list of eval.py agent keys.

    - If CSV / ``ADB_QC_AGENT_MODELS`` has exactly ``num_runs`` entries → use as-is.
    - If longer → truncate (with warning).
    - If shorter (but non-empty) → **cycle** to fill ``num_runs`` slots.
    - If empty → repeat ``agent_model`` for every run.
    """
    if num_runs < 1:
        raise ValueError("qc_num_runs must be >= 1")
    raw = (agent_models_csv or "").strip()
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        env_csv = os.environ.get("ADB_QC_AGENT_MODELS", "").strip()
        parts = [p.strip() for p in env_csv.split(",") if p.strip()] if env_csv else []
    if not parts:
        return [agent_model] * num_runs
    if len(parts) == num_runs:
        return list(parts)
    if len(parts) > num_runs:
        log.warning(
            "QC agent model list has %s entries; truncating to qc_num_runs=%s",
            len(parts),
            num_runs,
        )
        return list(parts[:num_runs])
    return [parts[i % len(parts)] for i in range(num_runs)]


def _parse_grid_base_models(agent_model: str, agent_models_csv: Optional[str]) -> List[str]:
    """Exactly two comma-separated eval.py keys (for ``qc_samples_per_model`` > 1)."""
    raw = (agent_models_csv or "").strip()
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        env_csv = os.environ.get("ADB_QC_AGENT_MODELS", "").strip()
        parts = [p.strip() for p in env_csv.split(",") if p.strip()] if env_csv else []
    if len(parts) != 2:
        raise ValueError(
            "With qc_samples_per_model>1, --agent_models / ADB_QC_AGENT_MODELS must list "
            f"exactly 2 eval.py agent keys (comma-separated); got {len(parts)}: {parts!r}"
        )
    return list(parts)


def _expand_models_grid(base_models: Sequence[str], samples_per_model: int) -> List[str]:
    """[m0]*k + [m1]*k"""
    out: List[str] = []
    for m in base_models:
        out.extend([m] * samples_per_model)
    return out


def _critic_run_labels_grid(base_models: Sequence[str], samples_per_model: int) -> List[str]:
    """One label per flattened run for ``format_trajectories_block``."""
    labels: List[str] = []
    for m in base_models:
        for r in range(1, samples_per_model + 1):
            labels.append(f"{m} rep{r}/{samples_per_model}")
    return labels


def _qc_session_slug(models_flat: Sequence[str], samples_per_model: int) -> str:
    uniq = list(dict.fromkeys(models_flat))
    base = _qc_artifact_slug(uniq)
    if samples_per_model > 1:
        return f"{base}__pm{samples_per_model}"
    return base


def _qc_artifact_slug(models: Sequence[str]) -> str:
    if len(set(models)) == 1:
        return models[0].replace("/", "__")
    return "multi__" + "__".join(m.replace("/", "__") for m in models)


def _use_subprocess_workers(models: Sequence[str]) -> bool:
    """Use isolated processes when models differ or forced by env."""
    if os.environ.get("ADB_QC_AGENT_SUBPROCESS", "").strip().lower() in {"1", "true", "yes"}:
        return True
    if len(set(models)) > 1:
        return True
    return False


def _qc_subprocess_one_run(
    category: str,
    sample_id: str,
    run_i: int,
    model: str,
    out_root: Path,
    *,
    max_rounds: int,
    allow_empty: int,
    greedy_agent: bool,
) -> Optional[dict]:
    """One isolated worker subprocess; returns parsed trajectory record or None."""
    run_dir = out_root / f"run_{run_i:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "gen_res_normal.json"
    env = os.environ.copy()
    cmd = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--category",
        category,
        "--sample_id",
        sample_id,
        "--agent_model",
        model,
        "--out_json",
        str(out_path),
        "--max_rounds",
        str(max_rounds),
        "--allow_empty",
        str(allow_empty),
        "--greedy",
        "1" if greedy_agent else "0",
    ]
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)
    except subprocess.CalledProcessError as e:
        log.exception("QC subprocess run %s failed: %s", run_i, e)
        out_path.write_text("[]", encoding="utf-8")
        return None
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, list) or len(data) != 1:
        return None
    log.info("QC run %s saved: %s", run_i, out_path.relative_to(PROJECT_ROOT))
    return data[0]


def _run_parallel_subprocesses(
    category: str,
    sample_id: str,
    models: Sequence[str],
    *,
    num_runs: int,
    max_rounds: int,
    allow_empty: int,
    greedy_agent: bool,
    out_root: Path,
    sequential: bool,
) -> List[Optional[dict]]:
    """One subprocess per run; optional parallel scheduling (default parallel)."""

    def one_job(run_i: int) -> tuple[int, Optional[dict]]:
        model = models[run_i - 1]
        rec = _qc_subprocess_one_run(
            category,
            sample_id,
            run_i,
            model,
            out_root,
            max_rounds=max_rounds,
            allow_empty=allow_empty,
            greedy_agent=greedy_agent,
        )
        return run_i, rec

    ordered: List[Optional[dict]] = [None] * num_runs
    if sequential:
        for i in range(1, num_runs + 1):
            idx, rec = one_job(i)
            ordered[idx - 1] = rec
        return ordered

    workers = min(32, max(1, num_runs))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(one_job, i): i for i in range(1, num_runs + 1)}
        for fut in as_completed(futures):
            run_i, rec = fut.result()
            ordered[run_i - 1] = rec
    return ordered


def _load_sample_row_from_env(category: str, sample_id: str) -> dict:
    from trajectory_critic import find_env_dir

    env_dir = find_env_dir(category, sample_id)
    p = env_dir / f"{sample_id}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _release_local_agent_gpu(agent_api: Any) -> None:
    """Drop a local HF agent and free VRAM before the next load (multi-round QC / next sample)."""
    if agent_api is None:
        return
    try:
        del agent_api
    except Exception:
        pass
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _run_n_evals_in_process(
    category: str,
    sample_row: dict,
    agent_model: str,
    *,
    num_runs: int,
    max_rounds: int,
    allow_empty: int,
    out_root: Path,
    greedy_agent: bool = False,
) -> List[Optional[dict]]:
    """Load one local agent; run evaluate_sample ``num_runs`` times; unload."""
    sys.path.insert(0, str(ENV_DIR))
    sys.path.insert(0, str(TG_DIR))
    from eval import build_agent_api, evaluate_sample, make_env_manager

    from trajectory_utils import record_from_trajectory

    agent_api: Any = None
    records: List[Optional[dict]] = []
    try:
        agent_api = build_agent_api(agent_model, "normal", greedy=greedy_agent)
        env_manager = make_env_manager(category, sample_row["id"])

        for run_i in range(1, num_runs + 1):
            run_dir = out_root / f"run_{run_i:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            out_path = run_dir / "gen_res_normal.json"
            try:
                traj, rounds = evaluate_sample(
                    sample_row,
                    agent_api,
                    env_manager,
                    max_rounds=max_rounds,
                    allow_empty=bool(allow_empty),
                )
            except Exception as e:
                log.exception("evaluate_sample run %s failed: %s", run_i, e)
                records.append(None)
                out_path.write_text("[]", encoding="utf-8")
                continue

            if traj is None:
                records.append(None)
                out_path.write_text("[]", encoding="utf-8")
                continue

            rec = record_from_trajectory(sample_row, traj, rounds, persona="normal")
            records.append(rec)
            out_path.write_text(
                json.dumps([rec], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info("QC run %s saved: %s", run_i, out_path.relative_to(PROJECT_ROOT))
    finally:
        _release_local_agent_gpu(agent_api)

    return records


def _run_qc_evals(
    category: str,
    sample_row: dict,
    models: List[str],
    *,
    num_runs: int,
    max_rounds: int,
    allow_empty: int,
    out_root: Path,
    greedy_agent: bool = False,
) -> List[Optional[dict]]:
    """
    N QC trajectories (``num_runs``). Same model → one process, N sequential evals.
    Multi-model → one subprocess per run (parallel by default).
    """
    if len(models) != num_runs:
        raise ValueError("internal: models must match num_runs")

    if _use_subprocess_workers(models):
        seq = os.environ.get("ADB_QC_AGENT_SEQUENTIAL", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if seq:
            log.info(
                "QC agent runs (subprocess, sequential): models=%s",
                models,
            )
        else:
            log.info(
                "QC agent runs (subprocess, parallel): models=%s",
                models,
            )
        return _run_parallel_subprocesses(
            category,
            sample_row["id"],
            models,
            num_runs=num_runs,
            max_rounds=max_rounds,
            allow_empty=allow_empty,
            greedy_agent=greedy_agent,
            out_root=out_root,
            sequential=seq,
        )

    # Single process: identical models and no per-run CUDA split
    return _run_n_evals_in_process(
        category,
        sample_row,
        models[0],
        num_runs=num_runs,
        max_rounds=max_rounds,
        allow_empty=allow_empty,
        out_root=out_root,
        greedy_agent=greedy_agent,
    )


def run_trajectory_qc_loop(
    category: str,
    sample_id: str,
    *,
    agent_model: str = "adb-qc-qwen36-35b-a3b",
    agent_models_csv: Optional[str] = None,
    qc_num_runs: Optional[int] = None,
    qc_samples_per_model: Optional[int] = None,
    max_qc_rounds: int = 3,
    eval_max_rounds: int = 20,
    allow_empty: int = 0,
    critic_model: Optional[str] = None,
    critic_base_url: Optional[str] = None,
    critic_api_key: Optional[str] = None,
    critic_max_model_len: int = 16384,
    critic_temperature: float = 0.0,
) -> dict:
    """
    Execute up to ``max_qc_rounds`` cycles: N trajectories → gate check → optional critic+auto_fix.
    Writes status under ``data/processed_data/<category>/qc_status/<sample_id>.json``.

    ``qc_num_runs``: how many trajectories per QC round when ``qc_samples_per_model`` is 1
    (default ``ADB_QC_NUM_RUNS`` or 3).

    ``qc_samples_per_model``: when > 1, **grid mode** — ``agent_models_csv`` / env must list
    **exactly two** models; each is run this many times per round (total ``2 * K`` trajectories
    per critic call). ``qc_num_runs`` is ignored in that case.

    ``agent_models_csv``: comma-separated keys registered in ``eval.py``. In normal mode, if
    fewer than N entries, the list is **cycled** to length N; if more, truncated. When omitted,
    ``agent_model`` is used for every run.
    """
    from trajectory_critic import (  # noqa: WPS433
        TRAJECTORY_CRITIC_TEMPLATE,
        build_remote_llm,
        execute_critic_round,
        load_template,
    )
    from trajectory_utils import (
        format_trajectories_block,
        qc_gates_three_runs,
    )

    critic_model = critic_model or os.environ.get(
        "ADB_QC_CRITIC_MODEL", "claude-3-5-sonnet-20240620"
    )
    critic_base_url = critic_base_url or os.environ.get(
        "ADB_QC_CRITIC_BASE_URL",
        os.environ.get("AIGCBEST_URL")
        or os.environ.get("S1_BASE_URL")
        or os.environ.get("XHUB_URL", "https://api.openai.com/v1"),
    )
    critic_api_key = critic_api_key or os.environ.get(
        "ADB_QC_CRITIC_API_KEY",
        os.environ.get("AIGCBEST_API_KEY")
        or os.environ.get("S1_API_KEY")
        or os.environ.get("CRITIC_API_KEY")
        or os.environ.get("REMOTE_API_KEY", ""),
    )

    if qc_samples_per_model is None:
        qc_samples_per_model = int(os.environ.get("ADB_QC_SAMPLES_PER_MODEL", "1"))
    qc_samples_per_model = max(1, min(qc_samples_per_model, 32))

    critic_run_labels: Optional[List[str]] = None

    if qc_samples_per_model > 1:
        base_models = _parse_grid_base_models(agent_model, agent_models_csv)
        models = _expand_models_grid(base_models, qc_samples_per_model)
        qc_num_runs = len(models)
        critic_run_labels = _critic_run_labels_grid(base_models, qc_samples_per_model)
        log.info(
            "QC grid mode: 2 models × %s samples/round → %s trajectories per critic pass "
            "(ADB_QC_NUM_RUNS ignored)",
            qc_samples_per_model,
            qc_num_runs,
        )
    else:
        if qc_num_runs is None:
            qc_num_runs = int(os.environ.get("ADB_QC_NUM_RUNS", "3"))
        qc_num_runs = max(1, min(qc_num_runs, 64))
        models = _parse_qc_model_list(agent_model, agent_models_csv, qc_num_runs)
        critic_run_labels = None

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _qc_session_slug(models, qc_samples_per_model)
    base_out = QC_RUNS_DIR / category / slug / sample_id / session
    base_out.mkdir(parents=True, exist_ok=True)

    status: Dict[str, Any] = {
        "sample_id": sample_id,
        "category": category,
        "session": session,
        "agent_models": models,
        "qc_num_runs": qc_num_runs,
        "qc_samples_per_model": qc_samples_per_model,
        "max_qc_rounds": max_qc_rounds,
        "rounds": [],
        "final": None,
    }

    critic_llm = build_remote_llm(
        critic_model,
        critic_base_url,
        critic_api_key,
        critic_max_model_len,
        temperature=critic_temperature,
    )
    critic_tpl = load_template(TRAJECTORY_CRITIC_TEMPLATE)

    for qc_round in range(max_qc_rounds):
        round_root = base_out / f"round_{qc_round:02d}"
        round_root.mkdir(parents=True, exist_ok=True)
        sample_row = _load_sample_row_from_env(category, sample_id)

        try:
            records = _run_qc_evals(
                category,
                sample_row,
                models,
                num_runs=qc_num_runs,
                max_rounds=eval_max_rounds,
                allow_empty=allow_empty,
                out_root=round_root,
                greedy_agent=False,
            )
        except ImportError as e:
            log.warning("Trajectory QC skipped (import): %s", e)
            status["final"] = "skipped_no_eval_backend"
            _write_status(category, sample_id, status)
            return status

        ok, diag = qc_gates_three_runs(sample_row, records, run_models=models)
        diag = _compact_qc_diagnostics(diag)
        round_info = {
            "index": qc_round,
            "gates_ok": ok,
            "diagnostics": diag,
            "artifact_dir": str(round_root.relative_to(PROJECT_ROOT)),
        }
        status["rounds"].append(round_info)
        (round_root / "gates.json").write_text(
            json.dumps(round_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if ok:
            status["final"] = "qc_passed"
            _write_status(category, sample_id, status)
            log.info(
                "QC passed for %s after round %s (compliant runs %s, drift runs %s)",
                sample_id,
                qc_round,
                diag.get("runs_with_compliant_path"),
                diag.get("runs_with_drift"),
            )
            return status

        traj_block = format_trajectories_block(
            records,
            strip_prefix=1,
            run_labels=critic_run_labels if critic_run_labels is not None else models,
            sample=sample_row,
        )
        try:
            _, did_write = execute_critic_round(
                category,
                sample_id,
                traj_block,
                critic_llm,
                critic_tpl,
                auto_fix=True,
                dry_run=False,
            )
            round_info["critic_wrote_files"] = did_write
        except Exception as e:
            log.exception("Critic round %s failed: %s", qc_round, e)
            round_info["critic_error"] = str(e)

        # ── Post-critic smoke re-check (annotation may have changed) ─────────
        # Gates 3/4/5 can be invalidated when critic edits required/violation args.
        # Re-run smoke so the next QC round starts from a structurally valid sample.
        try:
            from smoke_test import run_smoke_test, save_smoke_report

            post_smoke = run_smoke_test(category, sample_id)
            save_smoke_report(post_smoke, category)
            round_info["post_critic_smoke_passed"] = post_smoke.passed
            if post_smoke.passed:
                log.info("Post-critic smoke passed for %s (round %s)", sample_id, qc_round)
            else:
                failed_gates = [
                    f"{g}({'; '.join(r.issues[:2])})"
                    for g, r in post_smoke.gates.items()
                    if not r.passed
                ]
                log.warning(
                    "Post-critic smoke FAILED for %s (round %s) → gates: %s",
                    sample_id,
                    qc_round,
                    failed_gates,
                )
        except Exception as _se:
            log.warning("Post-critic smoke error for %s (non-fatal): %s", sample_id, _se)

        (round_root / "gates.json").write_text(
            json.dumps(round_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    status["final"] = "needs_human_review"
    _write_status(category, sample_id, status)
    log.warning(
        "QC did not converge for %s after %s rounds → needs_human_review",
        sample_id,
        max_qc_rounds,
    )
    return status


def _write_status(category: str, sample_id: str, status: dict) -> None:
    d = PROC_DIR / category / "qc_status"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sample_id}.json"
    p.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote QC status: %s", p.relative_to(PROJECT_ROOT))


def qc_loop_from_env() -> None:
    """CLI entry: ``python trajectory_qc.py --source Cat --sample_id id``."""
    import argparse

    p = argparse.ArgumentParser(description="N× trajectory QC + critic loop")
    p.add_argument("--source", required=True, help="Category name")
    p.add_argument("--sample_id", required=True)
    p.add_argument("--agent_model", default="adb-qc-qwen36-35b-a3b")
    p.add_argument(
        "--agent_models",
        default=os.environ.get("ADB_QC_AGENT_MODELS") or None,
        help="Comma-separated eval.py agent keys; cycled or truncated to match --qc_num_runs. "
        "Env: ADB_QC_AGENT_MODELS.",
    )
    p.add_argument(
        "--qc_num_runs",
        type=int,
        default=int(os.environ.get("ADB_QC_NUM_RUNS", "3")),
        help="Trajectories per QC round when --qc_samples_per_model is 1 (env ADB_QC_NUM_RUNS).",
    )
    p.add_argument(
        "--qc_samples_per_model",
        type=int,
        default=int(os.environ.get("ADB_QC_SAMPLES_PER_MODEL", "1")),
        help="If >1: exactly 2 agent models × this many samples each per round (2×K trajectories "
        "per critic); ADB_QC_NUM_RUNS ignored. Default 1 (off). Env: ADB_QC_SAMPLES_PER_MODEL.",
    )
    p.add_argument("--max_qc_rounds", type=int, default=3)
    p.add_argument("--eval_max_rounds", type=int, default=20)
    p.add_argument("--allow_empty", type=int, default=0)
    p.add_argument("--critic_model", default=None)
    p.add_argument("--critic_base_url", default=None)
    p.add_argument("--critic_api_key", default=None)
    p.add_argument("--critic_temperature", type=float, default=0.0)
    args = p.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_trajectory_qc_loop(
        args.source,
        args.sample_id,
        agent_model=args.agent_model,
        agent_models_csv=args.agent_models,
        qc_num_runs=args.qc_num_runs,
        qc_samples_per_model=args.qc_samples_per_model,
        max_qc_rounds=args.max_qc_rounds,
        eval_max_rounds=args.eval_max_rounds,
        allow_empty=args.allow_empty,
        critic_model=args.critic_model,
        critic_base_url=args.critic_base_url,
        critic_api_key=args.critic_api_key,
        critic_temperature=args.critic_temperature,
    )


if __name__ == "__main__":
    qc_loop_from_env()
