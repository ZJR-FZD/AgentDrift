"""
Authorization Drift Benchmark — Two-Stage Data Generation

Stage 1: Design — security architect → playbook Markdown (stage1.md)
Stage 2: Codegen — playbook → EnvXxx.py, EnvXxx.json, {sample_id}.json (stage2.md)

Stages 1–2 use --s1_* / --s2_* (default: claude-sonnet-4-6). Optional post-codegen
trajectory QC (N× local agent + critic loop; N default 3) lives in trajectory_qc.py; standalone
critic remains in trajectory_critic.py.
"""

import os
import sys
import json
import argparse
import fcntl
import logging
import re
from pathlib import Path
from typing import Any, Optional, Tuple

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_data"
PROC_DIR = PROJECT_ROOT / "data" / "processed_data"

STAGE1_PROMPT_FILE = SCRIPT_DIR / "stage1.md"
STAGE2_PROMPT_FILE = SCRIPT_DIR / "stage2.md"

# Default model IDs (OpenAI-compatible remote). Override via flags or env ADB_REMOTE_BASE_URL.
_DEFAULT_REMOTE_BASE = "https://api.openai.com/v1"
DEFAULT_S12_MODEL = "claude-sonnet-4-6"

sys.path.insert(0, str(SCRIPT_DIR))

try:
    from llm_api import get_llm
except ImportError:
    def get_llm(*args, **kwargs):
        raise RuntimeError("llm_api.py not found. Please provide a valid LLM module.")

try:
    from parser import parse_llm_output
    from parser import ParsedOutput
except ImportError:
    def parse_llm_output(*args, **kwargs):
        raise RuntimeError("parser.py not found. Please provide a valid parser module.")

    ParsedOutput = Any  # type: ignore

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────
def load_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Template missing: {path}")
    with open(path, encoding="utf-8") as f:
        return f.read()


def collect_skill_files(source: str) -> list:
    results = []
    if source == "all":
        for cat_dir in sorted(RAW_DIR.iterdir()):
            if cat_dir.is_dir():
                for f in sorted(cat_dir.glob("*.md")):
                    results.append((cat_dir.name, f))
    elif "/" in source:
        cat, skill_name = source.split("/", 1)
        skill_path = RAW_DIR / cat / f"{skill_name}.md"
        if skill_path.exists():
            results.append((cat, skill_path))
    else:
        cat_dir = RAW_DIR / source
        if cat_dir.is_dir():
            for f in sorted(cat_dir.glob("*.md")):
                results.append((source, f))
    return results


def build_sample_id(category: str, skill_name: str, auth_type: str, index: int) -> str:
    cat_snake = category.lower().replace("-", "_").replace(" ", "_")
    skill_snake = skill_name.lower().replace("-", "_").replace(" ", "_")
    return f"{cat_snake}_{skill_snake}_{auth_type}_{index:03d}"


def build_env_class_name(skill_name: str) -> str:
    parts = re.sub(r'[^a-zA-Z0-9_]', '_', skill_name).split("_")
    name_parts = [p.capitalize() for p in parts if p]
    class_name = "".join(name_parts) + "Env"
    if class_name and class_name[0].isdigit():
        class_name = "Env" + class_name
    return class_name


def next_index(category: str, skill_name: str, auth_type: str) -> int:
    data_file = PROC_DIR / category / "data.json"
    if not data_file.exists():
        return 1
    try:
        with open(data_file, encoding="utf-8") as f:
            existing = json.load(f)
    except Exception:
        return 1

    prefix = (
        f"{category.lower().replace('-','_')}_"
        f"{skill_name.lower().replace('-','_')}_{auth_type}_"
    )
    indices = []
    for sample in existing:
        sid = sample.get("id", "")
        if sid.startswith(prefix):
            try:
                indices.append(int(sid[len(prefix):]))
            except ValueError:
                continue
    return max(indices, default=0) + 1


def load_existing_data(category: str) -> list:
    data_file = PROC_DIR / category / "data.json"
    if not data_file.exists():
        return []
    try:
        with open(data_file, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def find_existing_sample_id(category: str, skill_name: str, auth_type: str) -> Optional[str]:
    prefix = (
        f"{category.lower().replace('-', '_')}_"
        f"{skill_name.lower().replace('-', '_')}_{auth_type}_"
    )

    for sample in load_existing_data(category):
        sid = sample.get("id", "")
        if isinstance(sid, str) and sid.startswith(prefix):
            return sid

    root = env_root_dir(category)
    if root.exists():
        for p in sorted(root.iterdir()):
            if p.is_dir() and p.name.startswith(prefix):
                return p.name
    return None


def save_data(category: str, samples: list):
    data_file = PROC_DIR / category / "data.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def save_data_atomic(category: str, new_sample: dict):
    """Append one sample to data.json with file locking (safe for concurrent workers)."""
    data_file = PROC_DIR / category / "data.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = data_file.with_suffix(".lock")
    with open(lock_file, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            existing = []
            if data_file.exists():
                try:
                    with open(data_file, encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            existing = [s for s in existing if s.get("id") != new_sample.get("id")]
            existing.append(new_sample)
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def save_env_files(category: str, parsed, sample_id: str):
    env_dir = env_root_dir(category) / sample_id
    env_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(parsed, "env_py_name") and hasattr(parsed, "env_py_code"):
        (env_dir / f"{parsed.env_py_name}.py").write_text(parsed.env_py_code, encoding="utf-8")
    if hasattr(parsed, "env_json_name") and hasattr(parsed, "env_json_content"):
        (env_dir / f"{parsed.env_json_name}.json").write_text(parsed.env_json_content, encoding="utf-8")
    if hasattr(parsed, "data_json_name") and hasattr(parsed, "data_json_content"):
        (env_dir / f"{parsed.data_json_name}.json").write_text(parsed.data_json_content, encoding="utf-8")
    log.info(f"  Saved env files: {env_dir.relative_to(PROC_DIR)}")


def _batch_suffix() -> str:
    batch = os.getenv("ADB_OUTPUT_BATCH", "").strip()
    if not batch:
        return ""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", batch)
    return f"_{safe}"


def playbook_dir(category: str) -> Path:
    return PROC_DIR / category / f"stage1_playbooks{_batch_suffix()}"


def playbook_path(category: str, sample_id: str) -> Path:
    return playbook_dir(category) / f"{sample_id}_playbook.md"


def spec_dir(category: str) -> Path:
    return PROC_DIR / category / f"stage2_specs{_batch_suffix()}"


def spec_path(category: str, sample_id: str) -> Path:
    return spec_dir(category) / f"{sample_id}_spec.json"


def raw_dir(category: str) -> Path:
    return PROC_DIR / category / f"raw_outputs{_batch_suffix()}"


def env_root_dir(category: str) -> Path:
    return PROC_DIR / category / f"environments{_batch_suffix()}"


def rationale_path(category: str, sample_id: str) -> Path:
    return raw_dir(category) / f"{sample_id}_rationale.txt"


def load_rationale_from_disk(category: str, sample_id: str) -> str:
    p = rationale_path(category, sample_id)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return ""


def save_rationale_to_disk(category: str, sample_id: str, rationale: str) -> None:
    if not (rationale or "").strip():
        return
    p = rationale_path(category, sample_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(rationale.strip(), encoding="utf-8")


def is_stage1_pass(playbook: str) -> bool:
    if not playbook:
        return False
    text = playbook.strip()
    fenced = re.match(r"^```[^\n]*\n(?P<body>.*)\n```$", text, re.DOTALL)
    if fenced:
        text = fenced.group("body").strip()
    if text == "PASS":
        return True
    return bool(re.match(r"^PASS\s*\nReason\s*:", text, re.IGNORECASE))


# ── STAGE 1: playbook ──────────────────────────────────────────────────────────
def run_stage1(
    llm, template: str, category: str, skill_path: Path,
    auth_type: str, sample_id: str,
    prompt_suffix: str = "",
) -> Optional[str]:
    try:
        skill_content = skill_path.read_text(encoding="utf-8")
    except Exception as e:
        log.error(f"Failed to read skill: {e}")
        return None

    prompt = template \
        .replace("{SAMPLE_ID}", sample_id) \
        .replace("{AUTHORIZATION_TYPE}", auth_type) \
        .replace("{SKILL_CONTENT}", skill_content)
    if prompt_suffix:
        prompt = prompt + prompt_suffix

    log.info("  [Stage 1] Generating playbook...")
    try:
        playbook = llm.generate(prompt)
    except Exception as e:
        log.error(f"Stage1 LLM error: {e}")
        return None

    rdir = raw_dir(category)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"{sample_id}_s1_raw.txt").write_text(playbook, encoding="utf-8")

    pdir = playbook_dir(category)
    pdir.mkdir(parents=True, exist_ok=True)
    pp = playbook_path(category, sample_id)
    pp.write_text(playbook, encoding="utf-8")
    log.info(f"  [Stage1] Saved: {pp.relative_to(PROC_DIR)}")
    if is_stage1_pass(playbook):
        log.info("  [Stage1] Model returned PASS; later stages will be skipped.")
    return playbook


# ── STAGE 2: codegen (playbook → env files + sample JSON) ─────────────────────
def run_stage2(
    llm, template: str, category: str, sample_id: str,
    env_class_name: str, playbook: str, auth_type: str,
    rationale: str = "",
    prompt_suffix: str = "",
) -> Tuple[Optional[dict], Optional[ParsedOutput]]:
    prompt = template \
        .replace("{ENV_CLASS_NAME}", env_class_name) \
        .replace("{SAMPLE_ID}", sample_id) \
        .replace("{STAGE1_OUTPUT}", playbook)
    if (rationale or "").strip():
        prompt = prompt + f"\n\n## Design Rationale\n{rationale.strip()}"
    if prompt_suffix:
        prompt = prompt + prompt_suffix

    log.info("  [Stage 2] Generating env files...")
    try:
        raw = llm.generate(prompt)
    except Exception as e:
        log.error(f"Stage2 LLM error: {e}")
        return None, None

    rdir = raw_dir(category)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"{sample_id}_s2_raw.txt").write_text(raw, encoding="utf-8")

    try:
        parsed = parse_llm_output(raw, sample_id, env_class_name_hint=env_class_name)
    except Exception as e:
        log.error(f"Parser failed: {e}")
        return None, None

    save_env_files(category, parsed, sample_id)

    data: Optional[dict] = None
    if hasattr(parsed, "data_json_content") and parsed.data_json_content:
        try:
            data = json.loads(parsed.data_json_content)
            data["id"] = sample_id
            data.setdefault("authorization_type", auth_type)
        except Exception:
            data = None
    return data, parsed


# ── flow ───────────────────────────────────────────────────────────────────────
def process_skill(
    category: str, skill_path: Path, auth_type: str, regen_id: Optional[str],
    stage: int,
    s1_llm, s1_template,
    s2_llm, s2_template,
) -> Optional[dict]:
    skill_name = skill_path.stem
    sample_id = regen_id or build_sample_id(
        category, skill_name, auth_type, next_index(category, skill_name, auth_type)
    )
    env_class_name = build_env_class_name(skill_name)
    log.info(f"Processing: {sample_id} | EnvClass: {env_class_name}")

    playbook: Optional[str] = None

    if stage in (0, 1):
        playbook = run_stage1(s1_llm, s1_template, category, skill_path, auth_type, sample_id)
        if not playbook:
            return None
        if is_stage1_pass(playbook):
            return None
        if stage == 1:
            return None

    if stage in (0, 2):
        if not playbook:
            pp = playbook_path(category, sample_id)
            if not pp.exists():
                log.error("Playbook not found")
                return None
            playbook = pp.read_text(encoding="utf-8")
        if is_stage1_pass(playbook):
            return None
        rationale = load_rationale_from_disk(category, sample_id)
        data, parsed = run_stage2(
            s2_llm, s2_template, category, sample_id, env_class_name, playbook, auth_type,
            rationale=rationale, prompt_suffix="",
        )
        if not parsed:
            return None
        return data

    return None


# ── LLM ────────────────────────────────────────────────────────────────────────
def build_llm(backend: str, model: str, base_url: str, api_key: str,
              model_path: str, gpu_ids: str, max_model_len: int):
    try:
        if backend == "remote":
            remote_max_tokens = min(max_model_len or 8192, 16384)
            return get_llm(
                "remote",
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_tokens=remote_max_tokens,
            )
        else:
            gpu_list = [int(x) for x in gpu_ids.split(",")] if gpu_ids else None
            return get_llm(
                "local",
                model_path=model_path,
                gpu_ids=gpu_list,
                max_model_len=max_model_len,
            )
    except Exception as e:
        log.error(f"LLM init failed: {e}")
        raise


# ── args ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Two-stage authorization drift benchmark generation")
    p.add_argument("--source", required=True, help="all / Category / Category/skill")
    p.add_argument("--auth_type", default="implicit", choices=["implicit", "explicit"])
    p.add_argument("--regen", default=None)
    p.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                   help="0=run Stage1 then Stage2; 1=Stage1 only; 2=Stage2 only (codegen from saved playbook)")
    p.add_argument("--max_total", type=int, default=None)

    _b = os.environ.get("ADB_REMOTE_BASE_URL", _DEFAULT_REMOTE_BASE)

    s1 = p.add_argument_group("Stage 1 (design playbook)")
    s1.add_argument("--s1_backend", default="remote")
    s1.add_argument("--s1_model", default=DEFAULT_S12_MODEL,
                    help=f"default: {DEFAULT_S12_MODEL!r}")
    s1.add_argument("--s1_base_url", default=_b,
                    help="OpenAI-compatible base URL; env: ADB_REMOTE_BASE_URL")
    s1.add_argument("--s1_api_key")
    s1.add_argument("--s1_model_path")
    s1.add_argument("--s1_gpu_ids")
    s1.add_argument("--s1_max_model_len", type=int, default=16384)

    s2 = p.add_argument_group("Stage 2 (codegen)")
    s2.add_argument("--s2_backend", default="remote")
    s2.add_argument("--s2_model", default=DEFAULT_S12_MODEL,
                    help=f"default: {DEFAULT_S12_MODEL!r}")
    s2.add_argument("--s2_base_url", default=_b)
    s2.add_argument("--s2_api_key")
    s2.add_argument("--s2_model_path")
    s2.add_argument("--s2_gpu_ids")
    s2.add_argument("--s2_max_model_len", type=int, default=16384)

    qc = p.add_argument_group("Trajectory QC (after Stage 2, local agent + critic loop)")
    qc.add_argument(
        "--trajectory_qc",
        action="store_true",
        help="Run N× local agent trajectories + gated critic loop (also enable with env ADB_TRAJECTORY_QC=1). "
        "N defaults to 3; set --qc_num_runs or ADB_QC_NUM_RUNS. Requires PyTorch/transformers and GPU.",
    )
    qc.add_argument(
        "--qc_max_rounds",
        type=int,
        default=int(os.environ.get("ADB_QC_MAX_ROUNDS", "3")),
        help="Max critic/auto-fix cycles before needs_human_review (default 3, env ADB_QC_MAX_ROUNDS).",
    )
    qc.add_argument(
        "--qc_agent_model",
        type=str,
        default=os.environ.get("ADB_QC_AGENT_MODEL", "adb-qc-qwen3-next-35b-a3b"),
        help="Agent key when all QC runs use the same model (default adb-qc-qwen3-next-35b-a3b, env ADB_QC_AGENT_MODEL).",
    )
    qc.add_argument(
        "--qc_num_runs",
        type=int,
        default=int(os.environ.get("ADB_QC_NUM_RUNS", "3")),
        help="QC trajectories per round when --qc_samples_per_model is 1 (env ADB_QC_NUM_RUNS).",
    )
    qc.add_argument(
        "--qc_samples_per_model",
        type=int,
        default=int(os.environ.get("ADB_QC_SAMPLES_PER_MODEL", "1")),
        help="If >1: exactly 2 models in ADB_QC_AGENT_MODELS, each sampled this many times per "
        "round (2×K critic trajectories); env ADB_QC_SAMPLES_PER_MODEL.",
    )
    qc.add_argument(
        "--qc_agent_models",
        type=str,
        default=os.environ.get("ADB_QC_AGENT_MODELS") or None,
        help="Comma-separated eval.py agent keys; cycled/truncated to --qc_num_runs (env ADB_QC_AGENT_MODELS). "
        "Subprocess execution is used when models differ.",
    )

    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.regen and "/" not in args.source:
        log.error("--regen must use format: Category/skill")
        return
    skip_data_save = os.getenv("ADB_SKIP_DATA_SAVE", "").strip().lower() in {"1", "true", "yes"}

    s1_llm = s2_llm = None

    if args.stage in (0, 1):
        s1_api_key = args.s1_api_key or os.getenv("S1_API_KEY") or os.getenv("REMOTE_API_KEY")
        s1_llm = build_llm(
            args.s1_backend, args.s1_model, args.s1_base_url, s1_api_key,
            args.s1_model_path, args.s1_gpu_ids, args.s1_max_model_len
        )

    if args.stage in (0, 2):
        s2_api_key = args.s2_api_key or os.getenv("S2_API_KEY") or os.getenv("REMOTE_API_KEY")
        s2_llm = build_llm(
            args.s2_backend, args.s2_model, args.s2_base_url, s2_api_key,
            args.s2_model_path, args.s2_gpu_ids, args.s2_max_model_len
        )

    s1_tpl = load_template(STAGE1_PROMPT_FILE) if args.stage in (0, 1) else ""
    s2_tpl = load_template(STAGE2_PROMPT_FILE) if args.stage in (0, 2) else ""

    skill_files = collect_skill_files(args.source)
    log.info(f"Found {len(skill_files)} skill files")
    count = 0

    for cat, path in skill_files:
        if args.max_total and count >= args.max_total:
            break
        log.info(f"\n=== Processing: {cat} / {path.name} ===")

        skill_name = path.stem
        existing = load_existing_data(cat)
        if args.regen:
            existing = [s for s in existing if s.get("id") != args.regen]
        elif args.stage == 0:
            existing_id = find_existing_sample_id(cat, skill_name, args.auth_type)
            if existing_id:
                count += 1
                log.info(f"Skipping existing sample: {existing_id}")
                continue

        data = process_skill(
            category=cat,
            skill_path=path,
            auth_type=args.auth_type,
            regen_id=args.regen,
            stage=args.stage,
            s1_llm=s1_llm,
            s1_template=s1_tpl,
            s2_llm=s2_llm,
            s2_template=s2_tpl,
        )

        if args.stage == 1:
            count += 1
            continue

        if data:
            count += 1
            log.info(f"✓ Completed sample: {data.get('id', 'unknown')}")

            # ── Smoke test (after Stage2, BEFORE data.json save + QC) ──────
            # Failing smoke must NOT pollute the shared benchmark file. The env
            # directory and smoke report are still written for debugging.
            _sample_id = data.get("id", "")
            _smoke_passed = True
            if _sample_id:
                try:
                    from smoke_test import run_smoke_test, save_smoke_report

                    _smoke = run_smoke_test(cat, _sample_id)
                    save_smoke_report(_smoke, cat)
                    _smoke_passed = _smoke.passed
                    if _smoke.passed:
                        log.info("Smoke test passed for %s", _sample_id)
                    else:
                        _failed = [
                            f"{g}({'; '.join(r.issues[:2])})"
                            for g, r in _smoke.gates.items()
                            if not r.passed
                        ]
                        log.warning(
                            "Smoke test FAILED for %s → skipping data.json + QC. Gates: %s",
                            _sample_id, _failed,
                        )
                except Exception as _e:
                    log.warning("Smoke test error for %s (non-fatal): %s", _sample_id, _e)

            if skip_data_save:
                log.info("Skipping shared data.json write because ADB_SKIP_DATA_SAVE is enabled.")
            elif _smoke_passed:
                existing.append(data)
                save_data(cat, existing)

            enable_qc = getattr(args, "trajectory_qc", False) or (
                os.getenv("ADB_TRAJECTORY_QC", "").strip().lower()
                in {"1", "true", "yes"}
            )
            if enable_qc and args.stage != 1 and data.get("id") and _smoke_passed:
                try:
                    from trajectory_qc import run_trajectory_qc_loop

                    run_trajectory_qc_loop(
                        cat,
                        data["id"],
                        agent_model=args.qc_agent_model,
                        agent_models_csv=args.qc_agent_models,
                        qc_num_runs=args.qc_num_runs,
                        qc_samples_per_model=args.qc_samples_per_model,
                        max_qc_rounds=args.qc_max_rounds,
                        eval_max_rounds=int(
                            os.getenv("ADB_QC_EVAL_MAX_ROUNDS", "20")
                        ),
                        allow_empty=int(os.getenv("ADB_QC_ALLOW_EMPTY", "0")),
                    )
                except Exception as e:
                    log.warning(
                        "Trajectory QC loop failed (non-fatal): %s", e, exc_info=True
                    )

    log.info(f"\nAll done! Total processed: {count}")


if __name__ == "__main__":
    main()
