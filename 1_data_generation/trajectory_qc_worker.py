"""
Single QC trajectory run in an isolated process (own CUDA_VISIBLE_DEVICES).

Invoked by trajectory_qc.py when using multiple local agent models and/or
per-run GPU assignment. Loads torch/eval only inside the child process.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKER_DIR.parent
TG_DIR = PROJECT_ROOT / "2_trajectory_generation"
ENV_INFRA = PROJECT_ROOT / "environments"


def _bootstrap_eval_paths() -> None:
    sys.path.insert(0, str(ENV_INFRA))
    sys.path.insert(0, str(TG_DIR))
    os.chdir(TG_DIR)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("trajectory_qc_worker")

    p = argparse.ArgumentParser(description="Run one trajectory QC eval in a subprocess")
    p.add_argument("--category", required=True)
    p.add_argument("--sample_id", required=True)
    p.add_argument("--agent_model", required=True)
    p.add_argument("--out_json", required=True, type=Path)
    p.add_argument("--max_rounds", type=int, default=20)
    p.add_argument("--allow_empty", type=int, default=0)
    p.add_argument("--greedy", type=int, default=0)
    args = p.parse_args()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(WORKER_DIR))
    from trajectory_critic import find_env_dir
    from trajectory_utils import record_from_trajectory

    _bootstrap_eval_paths()
    from eval import build_agent_api, evaluate_sample, make_env_manager

    env_dir = find_env_dir(args.category, args.sample_id)
    sample_path = env_dir / f"{args.sample_id}.json"
    if not sample_path.is_file():
        log.error("Missing sample json: %s", sample_path)
        args.out_json.write_text("[]", encoding="utf-8")
        return 0

    sample_row = json.loads(sample_path.read_text(encoding="utf-8"))
    greedy_agent = bool(args.greedy)

    agent_api = None
    try:
        agent_api = build_agent_api(args.agent_model, "normal", greedy_agent)
        env_manager = make_env_manager(args.category, sample_row["id"])
        traj, rounds = evaluate_sample(
            sample_row,
            agent_api,
            env_manager,
            max_rounds=args.max_rounds,
            allow_empty=bool(args.allow_empty),
        )
        if traj is None:
            args.out_json.write_text("[]", encoding="utf-8")
            return 0
        rec = record_from_trajectory(sample_row, traj, rounds, persona="normal")
        args.out_json.write_text(
            json.dumps([rec], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Wrote %s", args.out_json)
        return 0
    except Exception:
        log.exception("evaluate_sample failed")
        args.out_json.write_text("[]", encoding="utf-8")
        return 0
    finally:
        if agent_api is not None:
            try:
                del agent_api
            except Exception:
                pass
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
