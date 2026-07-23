"""
Concurrent data generation orchestrator.

Scans data/raw_data/ to discover skill categories and .md files, samples N
skills per category, then runs Stage1 + Stage2 + optional trajectory QC in
parallel across skills.

Concurrency model:
  - ProcessPoolExecutor across skills (outer parallelism)
  - Each worker independently calls generate.process_skill() then trajectory_qc
  - data.json writes use fcntl file locking (generate.save_data_atomic)
  - Within trajectory QC, two small models run in parallel (ThreadPoolExecutor,
    already built into trajectory_qc.py)

Usage:
  python generate_concurrent.py --samples_per_category 10 --max_workers 8
  python generate_concurrent.py --category Security --samples_per_category 5
  python generate_concurrent.py --config model_config.yaml --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_data"

sys.path.insert(0, str(SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(process)d] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _batch_suffix_for_report() -> str:
    """Same convention as generate._batch_suffix — keeps per-batch reports apart."""
    batch = os.getenv("ADB_OUTPUT_BATCH", "").strip()
    if not batch:
        return ""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", batch)
    return f"_{safe}"


def scan_raw_data_categories() -> Dict[str, List[Path]]:
    """
    Scan data/raw_data/ directory structure directly.
    Returns {category_name: [skill_path, ...]} where each skill_path is a .md file.

    Structure: data/raw_data/<Category>/<SubCategory>/<skill>.md
    e.g. data/raw_data/Security/Browser/agentaudit.md
    """
    categories: Dict[str, List[Path]] = {}
    if not RAW_DIR.is_dir():
        log.error("raw_data directory not found: %s", RAW_DIR)
        return categories

    for cat_dir in sorted(RAW_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        md_files = sorted(cat_dir.rglob("*.md"))
        if md_files:
            categories[cat_dir.name] = md_files

    return categories


def sample_skills(
    cat_name: str,
    skill_paths: List[Path],
    samples_per_category: int,
    seed: Optional[int] = None,
) -> List[Tuple[str, str, Path]]:
    """
    Returns list of (category, skill_label, skill_path) tuples.
    skill_label = "SubCategory/skill_stem" for logging.
    """
    if not skill_paths:
        return []

    valid = [(cat_name, f"{p.parent.name}/{p.stem}", p) for p in skill_paths]

    if len(valid) <= samples_per_category:
        return valid

    rng = random.Random(seed)
    return rng.sample(valid, samples_per_category)


def _worker_process_skill(
    proc_category: str,
    entry_key: str,
    skill_path_str: str,
    auth_type: str,
    s1_model: str,
    s1_base_url: str,
    s1_api_key: str,
    s2_model: str,
    s2_base_url: str,
    s2_api_key: str,
    s1_max_model_len: int,
    s2_max_model_len: int,
    enable_qc: bool,
    qc_agent_model: str,
    qc_agent_models_csv: str,
    qc_num_runs: int,
    qc_samples_per_model: int,
    qc_max_rounds: int,
    qc_eval_max_rounds: int,
) -> Dict[str, Any]:
    """
    Single worker: runs Stage1 + Stage2 for one skill, optionally followed by QC.
    Executed in a subprocess via ProcessPoolExecutor.

    proc_category: the skill_categories.json key (e.g. "Security"), used as the
    output directory under processed_data/.
    """
    skill_path = Path(skill_path_str)
    skill_name = skill_path.stem
    result = {
        "category": proc_category,
        "entry": entry_key,
        "skill": skill_name,
        "status": "unknown",
        "sample_id": None,
        "error": None,
    }

    try:
        from generate import (
            build_llm,
            find_existing_sample_id,
            load_template,
            process_skill,
            save_data_atomic,
            STAGE1_PROMPT_FILE,
            STAGE2_PROMPT_FILE,
        )

        existing_id = find_existing_sample_id(proc_category, skill_name, auth_type)
        if existing_id:
            log.info("Skipping existing: %s", existing_id)
            result["status"] = "skipped_existing"
            result["sample_id"] = existing_id
            return result

        s1_llm = build_llm("remote", s1_model, s1_base_url, s1_api_key, None, None, s1_max_model_len)
        s2_llm = build_llm("remote", s2_model, s2_base_url, s2_api_key, None, None, s2_max_model_len)
        s1_tpl = load_template(STAGE1_PROMPT_FILE)
        s2_tpl = load_template(STAGE2_PROMPT_FILE)

        data = process_skill(
            category=proc_category,
            skill_path=skill_path,
            auth_type=auth_type,
            regen_id=None,
            stage=0,
            s1_llm=s1_llm,
            s1_template=s1_tpl,
            s2_llm=s2_llm,
            s2_template=s2_tpl,
        )

        if data is None:
            result["status"] = "stage12_failed_or_pass"
            return result

        sample_id = data.get("id", "unknown")
        result["sample_id"] = sample_id
        result["status"] = "stage12_done"
        log.info("Stage1+2 completed: %s", sample_id)

        # ── Smoke test (after Stage2, BEFORE data.json save + QC) ────────────
        # Failing smoke must NOT pollute the shared benchmark file. The env
        # directory and smoke report are still written to disk for debugging.
        smoke_passed = True
        if sample_id != "unknown":
            try:
                from smoke_test import run_smoke_test, save_smoke_report

                smoke = run_smoke_test(proc_category, sample_id)
                save_smoke_report(smoke, proc_category)
                smoke_passed = smoke.passed
                result["smoke_passed"] = smoke.passed

                if smoke.passed:
                    log.info("Smoke test passed for %s", sample_id)
                else:
                    failed_gates = [
                        f"{g}({'; '.join(r.issues[:2])})"
                        for g, r in smoke.gates.items()
                        if not r.passed
                    ]
                    log.warning(
                        "Smoke test FAILED for %s → skipping data.json + QC. Gates: %s",
                        sample_id,
                        failed_gates,
                    )
                    result["status"] = "smoke_failed"
            except Exception as e:
                log.warning("Smoke test error for %s (non-fatal): %s", sample_id, e)

        skip_data_save = os.getenv("ADB_SKIP_DATA_SAVE", "").strip().lower() in {"1", "true", "yes"}
        if smoke_passed and not skip_data_save:
            save_data_atomic(proc_category, data)

        if enable_qc and sample_id != "unknown" and smoke_passed:
            try:
                from trajectory_qc import run_trajectory_qc_loop

                qc_status = run_trajectory_qc_loop(
                    proc_category,
                    sample_id,
                    agent_model=qc_agent_model,
                    agent_models_csv=qc_agent_models_csv or None,
                    qc_num_runs=qc_num_runs,
                    qc_samples_per_model=qc_samples_per_model,
                    max_qc_rounds=qc_max_rounds,
                    eval_max_rounds=qc_eval_max_rounds,
                    allow_empty=int(os.getenv("ADB_QC_ALLOW_EMPTY", "0")),
                )
                result["qc_final"] = qc_status.get("final", "unknown")
                result["status"] = "completed_with_qc"
                log.info("QC completed for %s: %s", sample_id, result["qc_final"])
            except Exception as e:
                log.warning("QC failed for %s (non-fatal): %s", sample_id, e)
                result["qc_final"] = f"error: {e}"
                result["status"] = "stage12_done_qc_failed"
        else:
            result["status"] = "completed_no_qc"

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        log.error("Worker failed for %s/%s: %s", proc_category, skill_name, traceback.format_exc())

    return result


def parse_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Parse YAML config with env var expansion, return flat dict."""
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        log.warning("Config file not found: %s", path)
        return {}
    try:
        import yaml
    except ImportError:
        log.warning("PyYAML not installed; ignoring config file")
        return {}
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded) or {}


def main():
    p = argparse.ArgumentParser(description="Concurrent data generation across skill categories")
    p.add_argument("--config", default=None, help="Path to model_config.yaml")
    p.add_argument("--category", default=None, help="Process only this category (key from skill_categories.json)")
    p.add_argument("--samples_per_category", type=int, default=10)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--auth_type", default="implicit", choices=["implicit", "explicit"])
    p.add_argument("--seed", type=int, default=42, help="Random seed for skill sampling")
    p.add_argument("--dry_run", action="store_true", help="Only list selected skills, don't generate")

    p.add_argument("--s1_model", default=None)
    p.add_argument("--s1_base_url", default=None)
    p.add_argument("--s1_api_key", default=None)
    p.add_argument("--s1_max_model_len", type=int, default=16384)
    p.add_argument("--s2_model", default=None)
    p.add_argument("--s2_base_url", default=None)
    p.add_argument("--s2_api_key", default=None)
    p.add_argument("--s2_max_model_len", type=int, default=16384)

    p.add_argument("--trajectory_qc", action="store_true")
    p.add_argument("--qc_agent_model", default=None)
    p.add_argument("--qc_agent_models", default=None, help="Comma-separated eval.py agent keys")
    p.add_argument("--qc_num_runs", type=int, default=3)
    p.add_argument("--qc_samples_per_model", type=int, default=4)
    p.add_argument("--qc_max_rounds", type=int, default=3)
    p.add_argument("--qc_eval_max_rounds", type=int, default=20)

    args = p.parse_args()

    cfg = parse_config(args.config)
    s12_cfg = cfg.get("stage12", {})
    critic_cfg = cfg.get("critic", {})
    qc_cfg = cfg.get("qc", {})
    conc_cfg = cfg.get("concurrency", {})

    s1_model = args.s1_model or s12_cfg.get("model", "claude-sonnet-4-6")
    s1_base_url = args.s1_base_url or s12_cfg.get("base_url") or os.getenv("AIGCBEST_URL", "https://api.openai.com/v1")
    s1_api_key = args.s1_api_key or s12_cfg.get("api_key") or os.getenv("AIGCBEST_API_KEY") or os.getenv("REMOTE_API_KEY", "")
    s2_model = args.s2_model or s12_cfg.get("model", s1_model)
    s2_base_url = args.s2_base_url or s12_cfg.get("base_url") or s1_base_url
    s2_api_key = args.s2_api_key or s12_cfg.get("api_key") or s1_api_key

    enable_qc = args.trajectory_qc or os.getenv("ADB_TRAJECTORY_QC", "").strip().lower() in {"1", "true", "yes"}
    qc_agent_model = args.qc_agent_model or os.getenv("ADB_QC_AGENT_MODEL", "adb-qc-qwen3-next-35b-a3b")
    qc_agent_models_csv = args.qc_agent_models or os.getenv("ADB_QC_AGENT_MODELS", "")
    qc_samples_per_model = args.qc_samples_per_model or qc_cfg.get("samples_per_model", 4)
    qc_max_rounds = args.qc_max_rounds or qc_cfg.get("max_rounds", 3)
    qc_eval_max_rounds = args.qc_eval_max_rounds or qc_cfg.get("eval_max_rounds", 20)

    max_workers = args.max_workers or conc_cfg.get("max_workers", 8)
    samples_per_category = args.samples_per_category or conc_cfg.get("samples_per_category", 10)

    categories = scan_raw_data_categories()
    log.info("Scanned %d categories from %s", len(categories), RAW_DIR)

    all_tasks: List[Tuple[str, str, Path]] = []
    for cat_name, skill_paths in categories.items():
        if args.category and cat_name != args.category:
            continue
        tasks = sample_skills(cat_name, skill_paths, samples_per_category, seed=args.seed)
        all_tasks.extend(tasks)
        log.info("Category %s: selected %d/%d skills", cat_name, len(tasks), len(skill_paths))

    log.info("Total skills to process: %d", len(all_tasks))

    if args.dry_run:
        log.info("=== DRY RUN: listing selected skills ===")
        for proc_cat, entry_key, spath in all_tasks:
            log.info("  [%s] %s → %s", proc_cat, entry_key, spath)
        log.info("DRY RUN complete. %d skills would be processed.", len(all_tasks))
        return

    results: List[Dict[str, Any]] = []
    effective_workers = min(max_workers, len(all_tasks))
    log.info("Launching %d workers for %d tasks", effective_workers, len(all_tasks))

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        future_map = {}
        for proc_cat, entry_key, skill_path in all_tasks:
            future = executor.submit(
                _worker_process_skill,
                proc_cat,
                entry_key,
                str(skill_path),
                args.auth_type,
                s1_model, s1_base_url, s1_api_key,
                s2_model, s2_base_url, s2_api_key,
                args.s1_max_model_len, args.s2_max_model_len,
                enable_qc,
                qc_agent_model, qc_agent_models_csv,
                args.qc_num_runs, qc_samples_per_model,
                qc_max_rounds, qc_eval_max_rounds,
            )
            future_map[future] = (proc_cat, entry_key, skill_path.stem)

        for future in as_completed(future_map):
            cat_info = future_map[future]
            try:
                result = future.result()
                results.append(result)
                log.info(
                    "[%s] %s → %s (id=%s)",
                    result["category"],
                    result["skill"],
                    result["status"],
                    result.get("sample_id", "N/A"),
                )
            except Exception as e:
                log.error("Unhandled error for %s: %s", cat_info, e)
                results.append({
                    "category": cat_info[0],
                    "entry": cat_info[1],
                    "skill": cat_info[2],
                    "status": "executor_error",
                    "error": str(e),
                })

    report_path = PROJECT_ROOT / "data" / "processed_data" / f"concurrent_run_report{_batch_suffix_for_report()}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Run report saved: %s", report_path)

    completed = sum(1 for r in results if "completed" in r.get("status", ""))
    skipped = sum(1 for r in results if r.get("status") == "skipped_existing")
    failed = sum(1 for r in results if "error" in r.get("status", "") or "failed" in r.get("status", ""))
    log.info(
        "=== SUMMARY: %d completed, %d skipped, %d failed / %d total ===",
        completed, skipped, failed, len(results),
    )


if __name__ == "__main__":
    main()
