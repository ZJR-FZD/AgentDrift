"""
rerun_qc_repaired.py

Re-run full trajectory QC (new agent runs + critic loop) on samples that were
previously repaired by repair_critic_blocked.py (final = repaired_pending_rerun).

Usage:
    cd 1_data_generation
    python rerun_qc_repaired.py \\
        [--category IT_and_Cloud]   \\
        [--sample_id <id>]          \\
        [--max_workers 4]           \\
        [--qc_num_runs 3]           \\
        [--qc_samples_per_model 4]  \\
        [--max_qc_rounds 3]         \\
        [--agent_models adb-qc-qwen36-35b-a3b,adb-qc-llama4-maverick-17b] \\
        [--dry_run]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed_data"

sys.path.insert(0, str(SCRIPT_DIR))

log = logging.getLogger(__name__)


def find_repaired_samples(
    category_filter: Optional[str] = None,
    sample_id_filter: Optional[str] = None,
) -> List[Dict]:
    pattern = (
        f"{category_filter}/qc_status/*.json"
        if category_filter
        else "*/qc_status/*.json"
    )
    samples = []
    for qc_file in sorted(PROC_DIR.glob(pattern)):
        try:
            status = json.loads(qc_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status.get("final") != "repaired_pending_rerun":
            continue
        sid = status.get("sample_id") or qc_file.stem
        if sample_id_filter and sid != sample_id_filter:
            continue
        cat = status.get("category") or qc_file.parent.parent.name
        # Reuse original QC settings (agent_models, qc_samples_per_model)
        samples.append({
            "category": cat,
            "sample_id": sid,
            "orig_agent_models": status.get("agent_models") or [],
            "orig_qc_samples_per_model": status.get("qc_samples_per_model", 1),
        })
    return samples


def rerun_one(
    entry: Dict,
    *,
    agent_models_csv: Optional[str],
    qc_num_runs: int,
    qc_samples_per_model: int,
    max_qc_rounds: int,
    critic_model: Optional[str],
    critic_base_url: Optional[str],
    critic_api_key: Optional[str],
    dry_run: bool,
) -> Dict:
    from trajectory_qc import run_trajectory_qc_loop

    cat = entry["category"]
    sid = entry["sample_id"]

    # Use overrides if provided, else fall back to original settings
    _qc_spm = qc_samples_per_model if qc_samples_per_model > 0 else entry["orig_qc_samples_per_model"]

    if dry_run:
        log.info("[dry_run] Would rerun QC for %s/%s", cat, sid)
        return {"category": cat, "sample_id": sid, "outcome": "dry_run"}

    log.info("Rerunning QC for %s/%s …", cat, sid)
    try:
        status = run_trajectory_qc_loop(
            cat,
            sid,
            agent_models_csv=agent_models_csv,
            qc_num_runs=qc_num_runs,
            qc_samples_per_model=_qc_spm,
            max_qc_rounds=max_qc_rounds,
            critic_model=critic_model,
            critic_base_url=critic_base_url,
            critic_api_key=critic_api_key,
        )
        final = status.get("final")
        log.info("[%s] QC result: %s", sid, final)
        return {"category": cat, "sample_id": sid, "outcome": final}
    except Exception as e:
        log.exception("[%s] QC failed: %s", sid, e)
        return {"category": cat, "sample_id": sid, "outcome": "exception", "error": str(e)}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Re-run full QC on repaired_pending_rerun samples"
    )
    p.add_argument("--category", default=None)
    p.add_argument("--sample_id", default=None)
    p.add_argument(
        "--agent_models",
        default=os.environ.get("ADB_QC_AGENT_MODELS") or None,
        help="Comma-separated eval.py agent keys (default: reuse original settings). "
             "Env: ADB_QC_AGENT_MODELS.",
    )
    p.add_argument(
        "--qc_num_runs",
        type=int,
        default=int(os.environ.get("ADB_QC_NUM_RUNS", "3")),
    )
    p.add_argument(
        "--qc_samples_per_model",
        type=int,
        default=int(os.environ.get("ADB_QC_SAMPLES_PER_MODEL", "4")),
        help="0 = reuse original value from qc_status",
    )
    p.add_argument("--max_qc_rounds", type=int, default=3)
    p.add_argument(
        "--critic_model",
        default=os.environ.get("ADB_QC_CRITIC_MODEL", "claude-sonnet-4-6"),
    )
    p.add_argument(
        "--critic_base_url",
        default=(
            os.environ.get("ADB_OPENAI_COMPAT_BASE_URL")
            or os.environ.get("AIGCBEST_URL")
            or os.environ.get("S1_BASE_URL")
            or "https://api.openai.com/v1"
        ),
    )
    p.add_argument(
        "--critic_api_key",
        default=(
            os.environ.get("ADB_OPENAI_COMPAT_API_KEY")
            or os.environ.get("AIGCBEST_API_KEY")
            or os.environ.get("S1_API_KEY")
            or os.environ.get("REMOTE_API_KEY", "")
        ),
    )
    p.add_argument("--max_workers", type=int, default=4,
                   help="Parallel samples (each sample itself uses subprocess workers for agents)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    samples = find_repaired_samples(
        category_filter=args.category,
        sample_id_filter=args.sample_id,
    )
    log.info("Found %d repaired_pending_rerun samples to re-QC", len(samples))
    if not samples:
        log.info("Nothing to do.")
        return

    results = []
    if args.max_workers <= 1:
        for entry in samples:
            results.append(rerun_one(
                entry,
                agent_models_csv=args.agent_models,
                qc_num_runs=args.qc_num_runs,
                qc_samples_per_model=args.qc_samples_per_model,
                max_qc_rounds=args.max_qc_rounds,
                critic_model=args.critic_model,
                critic_base_url=args.critic_base_url,
                critic_api_key=args.critic_api_key,
                dry_run=args.dry_run,
            ))
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {
                ex.submit(
                    rerun_one, entry,
                    agent_models_csv=args.agent_models,
                    qc_num_runs=args.qc_num_runs,
                    qc_samples_per_model=args.qc_samples_per_model,
                    max_qc_rounds=args.max_qc_rounds,
                    critic_model=args.critic_model,
                    critic_base_url=args.critic_base_url,
                    critic_api_key=args.critic_api_key,
                    dry_run=args.dry_run,
                ): entry
                for entry in samples
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    entry = futures[fut]
                    log.error("[%s] Unhandled: %s", entry["sample_id"], e)
                    results.append({"sample_id": entry["sample_id"], "outcome": "unhandled"})

    # Summary
    outcomes: Dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    log.info("=== Re-QC summary (%d samples) ===", len(results))
    for outcome, count in sorted(outcomes.items()):
        log.info("  %-35s %d", outcome, count)


if __name__ == "__main__":
    main()
