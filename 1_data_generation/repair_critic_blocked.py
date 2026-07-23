"""
repair_critic_blocked.py

Re-run the Critic (with the fixed prompt + retry logic) on samples that were
blocked by the parse-error bug in the previous full_v1 batch.

It does NOT regenerate trajectories — it reuses the existing QC run dirs
(gen_res_normal.json files) from the last recorded round of each blocked sample.

Usage:
    cd 1_data_generation
    python repair_critic_blocked.py \\
        --critic_model claude-sonnet-4-6 \\
        --critic_base_url <url> \\
        --critic_api_key  <key> \\
        [--category Security]  \\
        [--sample_id  <id>]    \\
        [--dry_run]            \\
        [--max_workers 4]

The script:
1. Scans qc_status/ for needs_human_review samples with all-[Parser] critic errors.
2. Loads the existing gen_res_normal.json files from the last round's artifact_dir.
3. Calls execute_critic_round(auto_fix=True) with the now-fixed Critic (prompt + retry).
4. Runs post-critic smoke test and patches qc_status with the new outcome.
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


# ---------------------------------------------------------------------------
# Identification helpers
# ---------------------------------------------------------------------------

def _is_parse_blocked(status: dict) -> bool:
    """True iff sample is needs_human_review and every critic call was a [Parser] error."""
    if status.get("final") != "needs_human_review":
        return False
    rounds = status.get("rounds") or []
    if not rounds:
        return False
    critic_rounds = [r for r in rounds if "critic_error" in r or "critic_wrote_files" in r]
    if not critic_rounds:
        return False
    return all(
        r.get("critic_wrote_files") is None
        and "[Parser]" in str(r.get("critic_error", ""))
        for r in critic_rounds
    )


def find_blocked_samples(
    category_filter: Optional[str] = None,
    sample_id_filter: Optional[str] = None,
) -> List[Dict]:
    blocked = []
    pattern = (
        f"{category_filter}/qc_status/*.json"
        if category_filter
        else "*/qc_status/*.json"
    )
    for qc_file in sorted(PROC_DIR.glob(pattern)):
        try:
            status = json.loads(qc_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _is_parse_blocked(status):
            continue
        sid = status.get("sample_id") or qc_file.stem
        if sample_id_filter and sid != sample_id_filter:
            continue
        cat = status.get("category") or qc_file.parent.parent.name
        # The last round's artifact_dir has the trajectories we want to reuse.
        last_round = (status.get("rounds") or [])[-1]
        blocked.append(
            {
                "category": cat,
                "sample_id": sid,
                "qc_file": qc_file,
                "status": status,
                "last_round": last_round,
            }
        )
    return blocked


# ---------------------------------------------------------------------------
# Trajectory reconstruction helpers
# ---------------------------------------------------------------------------

def load_records_from_artifact_dir(
    artifact_dir: str,
    agent_models: List[str],
    sample_id: str,
) -> List[Optional[dict]]:
    """
    Reconstruct the list of trajectory records (same length as agent_models) from
    the run_01..run_N subdirectories of *artifact_dir*.
    """
    root = PROJECT_ROOT / artifact_dir
    n = len(agent_models)
    records: List[Optional[dict]] = []
    for i in range(1, n + 1):
        run_path = root / f"run_{i:02d}" / "gen_res_normal.json"
        if not run_path.exists():
            records.append(None)
            continue
        try:
            data = json.loads(run_path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Could not read %s: %s", run_path, e)
            records.append(None)
            continue
        if isinstance(data, list) and len(data) >= 1:
            # gen_res_normal.json is a single-element list
            rec = data[0] if data[0].get("id") == sample_id else next(
                (x for x in data if x.get("id") == sample_id), None
            )
            records.append(rec)
        else:
            records.append(None)
    return records


# ---------------------------------------------------------------------------
# Repair one sample
# ---------------------------------------------------------------------------

def repair_one(
    entry: dict,
    critic_llm,
    critic_tpl: str,
    *,
    dry_run: bool,
) -> dict:
    from trajectory_critic import execute_critic_round
    from trajectory_utils import format_trajectories_block

    cat = entry["category"]
    sid = entry["sample_id"]
    status = entry["status"]
    last_round = entry["last_round"]
    qc_file: Path = entry["qc_file"]

    artifact_dir = last_round.get("artifact_dir", "")
    agent_models: List[str] = status.get("agent_models") or []
    qc_samples_per_model: int = status.get("qc_samples_per_model", 1)

    if not artifact_dir:
        log.warning("[%s] No artifact_dir in last round — skipping", sid)
        return {"sample_id": sid, "outcome": "no_artifact_dir"}

    records = load_records_from_artifact_dir(artifact_dir, agent_models, sid)
    if all(r is None for r in records):
        log.warning("[%s] All trajectory records are None — skipping", sid)
        return {"sample_id": sid, "outcome": "no_trajectories"}

    # Build run labels matching what QC originally used
    if qc_samples_per_model > 1:
        from trajectory_qc import _critic_run_labels_grid, _parse_grid_base_models
        base_models = list(dict.fromkeys(agent_models))
        run_labels = _critic_run_labels_grid(base_models, qc_samples_per_model)
    else:
        run_labels = agent_models

    # Load sample row for format_trajectories_block
    try:
        from trajectory_critic import find_env_dir
        env_dir = find_env_dir(cat, sid)
        sample_row = json.loads((env_dir / f"{sid}.json").read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("[%s] Could not load sample row: %s — using minimal stub", sid, e)
        sample_row = {"id": sid}

    traj_block = format_trajectories_block(
        records,
        strip_prefix=1,
        run_labels=run_labels,
        sample=sample_row,
    )

    log.info("[%s] Running Critic (dry_run=%s) …", sid, dry_run)
    try:
        report, did_write = execute_critic_round(
            cat,
            sid,
            traj_block,
            critic_llm,
            critic_tpl,
            auto_fix=True,
            dry_run=dry_run,
        )
    except Exception as e:
        log.error("[%s] Critic failed: %s", sid, e)
        return {"sample_id": sid, "outcome": "critic_exception", "error": str(e)}

    # Post-critic smoke test
    smoke_passed: Optional[bool] = None
    smoke_error: Optional[str] = None
    if not dry_run:
        try:
            from smoke_test import run_smoke_test, save_smoke_report
            post_smoke = run_smoke_test(cat, sid)
            save_smoke_report(post_smoke, cat)
            smoke_passed = post_smoke.passed
            if smoke_passed:
                log.info("[%s] Post-critic smoke PASSED", sid)
            else:
                failed_gates = [
                    f"{g}({'; '.join(r.issues[:2])})"
                    for g, r in post_smoke.gates.items()
                    if not r.passed
                ]
                log.warning("[%s] Post-critic smoke FAILED: %s", sid, failed_gates)
        except Exception as e:
            smoke_error = str(e)
            log.warning("[%s] Smoke test error (non-fatal): %s", sid, e)

    # Patch qc_status: append a new pseudo-round entry recording this repair
    if not dry_run:
        repair_entry = {
            "index": "repair",
            "gates_ok": last_round.get("gates_ok", False),
            "diagnostics": last_round.get("diagnostics", {}),
            "artifact_dir": artifact_dir,
            "critic_wrote_files": did_write,
            "post_critic_smoke_passed": smoke_passed,
            "repaired_from_existing_trajectories": True,
        }
        if smoke_error:
            repair_entry["smoke_error"] = smoke_error
        status["rounds"].append(repair_entry)
        # Keep final as needs_human_review — a human should still review
        # (the Critic may have fixed annotation, but QC gates need a fresh run
        # to confirm). Mark as repaired so it's easy to re-queue.
        status["final"] = "repaired_pending_rerun"
        qc_file.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("[%s] Updated qc_status → repaired_pending_rerun", sid)

    return {
        "sample_id": sid,
        "outcome": "repaired",
        "critic_wrote_files": did_write,
        "smoke_passed": smoke_passed,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Re-run Critic on parse-error-blocked QC samples (reuses existing trajectories)"
    )
    p.add_argument("--category", default=None, help="Limit to one category")
    p.add_argument("--sample_id", default=None, help="Limit to one sample id")
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
            or os.environ.get("XHUB_URL")
            or "https://api.openai.com/v1"
        ),
    )
    p.add_argument(
        "--critic_api_key",
        default=(
            os.environ.get("ADB_OPENAI_COMPAT_API_KEY")
            or os.environ.get("AIGCBEST_API_KEY")
            or os.environ.get("S1_API_KEY")
            or os.environ.get("XHUB_API_KEY")
            or os.environ.get("CRITIC_API_KEY")
            or os.environ.get("REMOTE_API_KEY", "")
        ),
    )
    p.add_argument("--critic_max_model_len", type=int, default=16384)
    p.add_argument("--critic_temperature", type=float, default=0.0)
    p.add_argument("--max_workers", type=int, default=4, help="Parallel critic calls")
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Run Critic and save reports, but do NOT overwrite env files or update qc_status",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    from trajectory_critic import (
        TRAJECTORY_CRITIC_TEMPLATE,
        build_remote_llm,
        load_template,
    )

    critic_llm = build_remote_llm(
        args.critic_model,
        args.critic_base_url,
        args.critic_api_key,
        args.critic_max_model_len,
        temperature=args.critic_temperature,
    )
    critic_tpl = load_template(TRAJECTORY_CRITIC_TEMPLATE)

    blocked = find_blocked_samples(
        category_filter=args.category,
        sample_id_filter=args.sample_id,
    )
    log.info("Found %d parse-error-blocked samples to repair", len(blocked))
    if not blocked:
        log.info("Nothing to do.")
        return

    results = []
    if args.max_workers <= 1:
        for entry in blocked:
            results.append(
                repair_one(entry, critic_llm, critic_tpl, dry_run=args.dry_run)
            )
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {
                ex.submit(repair_one, entry, critic_llm, critic_tpl, dry_run=args.dry_run): entry
                for entry in blocked
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    entry = futures[fut]
                    log.error("[%s] Unhandled exception: %s", entry["sample_id"], e)
                    results.append(
                        {"sample_id": entry["sample_id"], "outcome": "unhandled_exception", "error": str(e)}
                    )

    # Summary
    outcomes: Dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    log.info("=== Repair summary (%d samples) ===", len(results))
    for outcome, count in sorted(outcomes.items()):
        log.info("  %-35s %d", outcome, count)
    wrote = sum(1 for r in results if r.get("critic_wrote_files"))
    smoke_ok = sum(1 for r in results if r.get("smoke_passed") is True)
    smoke_fail = sum(1 for r in results if r.get("smoke_passed") is False)
    log.info("  critic wrote files: %d / %d", wrote, len(results))
    log.info("  post-smoke passed:  %d  failed: %d", smoke_ok, smoke_fail)

    if args.dry_run:
        log.info("Dry-run: no files were modified.")


if __name__ == "__main__":
    main()
