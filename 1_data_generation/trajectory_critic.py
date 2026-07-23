"""
trajectory_critic.py

Read normal-persona agent trajectories, run a single LLM pass (critic + per-file
rewrite decision), optionally overwrite benchmark files when modify=true.

Usage:
  python trajectory_critic.py \\
    --source Communication \\
    --model_name qwen2.5-7b \\
    --eval_extra repeat1_derecipe/repeat_01 \\
    --auto_fix

  Omit --eval_extra to auto-pick the newest run directory (recursive under the model folder).

  Multi-model / multi-repeat bundle (e.g. 3×4 = 12 trajectories for one critic pass):

  python trajectory_critic.py --source Security \\
    --critic_model_bundle qwen2.5-7b,granite-4.1-8b,llama3.1-8b \\
    --eval_extra_bundle repeat10_derecipe/repeat_01,repeat10_derecipe/repeat_02,repeat10_derecipe/repeat_03,repeat10_derecipe/repeat_04
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed_data"
EVAL_DIR = PROJECT_ROOT / "evaluation_results"

TRAJECTORY_CRITIC_TEMPLATE = SCRIPT_DIR / "trajectory_critic.md"

sys.path.insert(0, str(SCRIPT_DIR))

from llm_api import get_llm  # noqa: E402
from parser import parse_unified_trajectory_critic_json  # noqa: E402
from trajectory_utils import format_trajectories_block  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Template missing: {path}")
    return path.read_text(encoding="utf-8")


def find_env_dir(category: str, sample_id: str) -> Path:
    """Locate sample environment directory (same logic as eval.make_env_manager)."""
    env_candidates = [PROC_DIR / category / "environments" / sample_id]
    env_candidates.extend(sorted((PROC_DIR / category).glob(f"environments_*/{sample_id}")))
    env_dir = next((p for p in env_candidates if p.exists()), None)
    if env_dir is None:
        raise FileNotFoundError(f"No environment directory found for {category}/{sample_id}")
    return env_dir


def _iter_run_dirs_with_normal(model_root: Path) -> List[Tuple[Path, float]]:
    """(run_dir, mtime of gen_res_normal.json) newest occurrences first."""
    rows: List[Tuple[Path, float]] = []
    for npath in model_root.rglob("gen_res_normal.json"):
        try:
            rows.append((npath.parent, npath.stat().st_mtime))
        except OSError:
            continue
    rows.sort(key=lambda x: -x[1])
    seen = set()
    out: List[Tuple[Path, float]] = []
    for d, mt in rows:
        if d in seen:
            continue
        seen.add(d)
        out.append((d, mt))
    return out


def collect_bundle_trajectories(
    category: str,
    model_names: List[str],
    eval_extras: List[str],
    sample_id: str,
) -> Tuple[List[Optional[dict]], List[str]]:
    """
    Load ``gen_res_normal.json`` for the same ``sample_id`` from each
    (model_dir × eval_extra) pair, in row-major order (all repeats for model 1,
    then model 2, …). Labels match QC grid style for summary column grouping.
    """
    records: List[Optional[dict]] = []
    labels: List[str] = []
    n_per = len(eval_extras)
    if n_per < 1:
        return [], []
    for model in model_names:
        for j, extra in enumerate(eval_extras):
            rel = extra.strip().strip("/")
            p = EVAL_DIR / category / model / rel / "gen_res_normal.json"
            records.append(load_trajectory(p, sample_id))
            labels.append(f"{model} rep{j + 1}/{n_per}")
    return records, labels


def resolve_trajectory_run(
    category: str,
    model_name: str,
    eval_extra: Optional[str],
) -> Tuple[Optional[Path], str]:
    """
    Locate gen_res_normal.json under evaluation_results/<category>/<model_name>/.

    Returns (path_or_None, run_label relative to model root).
    """
    base = EVAL_DIR / category / model_name
    if not base.is_dir():
        return None, ""

    if eval_extra:
        run_dir = (base / eval_extra.strip().strip("/")).resolve()
        base_res = base.resolve()
        try:
            rel = str(run_dir.relative_to(base_res))
        except ValueError:
            log.warning("eval_extra %r is outside model root %s", eval_extra, base)
            return None, eval_extra
        n = run_dir / "gen_res_normal.json"
        if not n.exists():
            return None, rel
        return n, rel

    rows = _iter_run_dirs_with_normal(base)
    if not rows:
        return None, ""
    run_dir, _ = rows[0]
    try:
        rel = str(run_dir.relative_to(base))
    except ValueError:
        rel = str(run_dir)
    return run_dir / "gen_res_normal.json", rel


def load_trajectory(traj_path: Path, sample_id: str) -> Optional[dict]:
    if not traj_path.exists():
        return None
    try:
        data = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Could not read %s: %s", traj_path, e)
        return None
    for item in data:
        if item.get("id") == sample_id:
            return item
    return None


def slice_trajectory_for_critic(record: dict, strip_output_prefix: int) -> dict:
    """
    Drop leading `output` messages (e.g. system prompt). Default prefix=1 removes
    only the first entry (system); prefix=2 also removes the user turn.
    """
    if strip_output_prefix <= 0:
        return record
    slim = copy.deepcopy(record)
    out = slim.get("output")
    if isinstance(out, list) and len(out) > strip_output_prefix:
        slim["output"] = out[strip_output_prefix:]
    return slim


_PLACEHOLDER_ENV_STEMS = frozenset({"envxxx", "env_xx", "envxxxenv"})


def _pick_env_py(env_dir: Path) -> Path:
    """Prefer real env modules over mistaken literal placeholder names (e.g. EnvXxx.py)."""
    pys = [
        p for p in env_dir.glob("*.py")
        if p.name.lower() not in ("baseenv.py", "__init__.py")
    ]
    if not pys:
        raise FileNotFoundError(f"No env .py under {env_dir}")
    good = [p for p in pys if p.stem.lower() not in _PLACEHOLDER_ENV_STEMS]
    return sorted(good or pys)[0]


def read_benchmark_artifacts(env_dir: Path, sample_id: str) -> Tuple[str, str, str, str]:
    """Returns (sample_json_str, env_py_str, env_json_str, env_stem)."""
    sample_p = env_dir / f"{sample_id}.json"
    if not sample_p.exists():
        raise FileNotFoundError(f"Missing sample JSON: {sample_p}")
    env_py_p = _pick_env_py(env_dir)
    env_stem = env_py_p.stem
    env_json_p = env_dir / f"{env_stem}.json"
    if not env_json_p.exists():
        raise FileNotFoundError(f"Missing env JSON: {env_json_p}")
    return (
        sample_p.read_text(encoding="utf-8"),
        env_py_p.read_text(encoding="utf-8"),
        env_json_p.read_text(encoding="utf-8"),
        env_stem,
    )


_CRITIC_FORMAT_REMINDER = (
    'Respond with ONLY a JSON object. '
    'Top-level keys must be exactly: "env_py", "sample_json", "env_json". '
    'Do NOT use the actual filenames as keys. '
    'No markdown fences, no extra keys, no preamble.'
)


def run_unified_critic(
    sample_json: str,
    env_py: str,
    env_json: str,
    trajectories_text: str,
    critic_llm,
    critic_template: str,
    *,
    env_py_basename: str,
    sample_json_basename: str,
    env_json_basename: str,
) -> dict:
    prompt = (
        critic_template.replace("{SAMPLE_JSON}", sample_json)
        .replace("{ENV_PY}", env_py)
        .replace("{ENV_JSON}", env_json)
        .replace("{TRAJECTORIES}", trajectories_text)
        .replace("{ENV_PY_BASENAME}", env_py_basename)
        .replace("{SAMPLE_JSON_BASENAME}", sample_json_basename)
        .replace("{ENV_JSON_BASENAME}", env_json_basename)
    )
    raw = critic_llm.generate(prompt)
    try:
        return parse_unified_trajectory_critic_json(raw)
    except ValueError as first_err:
        log.warning(
            "Critic parse error (attempt 1): %s — retrying with format reminder", first_err
        )
        retry_prompt = (
            _CRITIC_FORMAT_REMINDER + "\n\n"
            f"Your previous response could not be parsed. Error: {first_err}\n\n"
            "Re-examine the inputs below and respond with ONLY the JSON object:\n\n"
            + prompt
        )
        raw2 = critic_llm.generate(retry_prompt)
        return parse_unified_trajectory_critic_json(raw2)


def unified_writes(
    env_dir: Path, sample_id: str, env_stem: str, report: dict
) -> Dict[Path, str]:
    """Map disk paths to new file bodies for entries with modify=true."""
    mapping = {
        "env_py": env_dir / f"{env_stem}.py",
        "sample_json": env_dir / f"{sample_id}.json",
        "env_json": env_dir / f"{env_stem}.json",
    }
    writes: Dict[Path, str] = {}
    for key, path in mapping.items():
        entry = report.get(key) or {}
        if entry.get("modify") and entry.get("new_content"):
            text = entry["new_content"]
            if key != "env_py":
                try:
                    json.loads(text)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"[Unified critic] {key}: new_content is not valid JSON: {e}"
                    ) from e
            writes[path] = text
    return writes


def backup_and_write(env_dir: Path, write_targets: Dict[Path, str]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = env_dir / "_backup" / ts
    backup_root.mkdir(parents=True, exist_ok=True)
    for path in write_targets:
        if path.exists():
            shutil.copy2(path, backup_root / path.name)
    for path, content in write_targets.items():
        path.write_text(content, encoding="utf-8")
    log.info("Wrote fixed files; backup at %s", backup_root.relative_to(PROJECT_ROOT))
    return backup_root


def critic_report_path(category: str, sample_id: str) -> Path:
    return PROC_DIR / category / "critic_reports" / f"{sample_id}_critic.json"


def sync_sample_json_to_category_data_json(category: str, sample_id: str, sample_obj: dict) -> None:
    """
    Keep data/processed_data/<category>/data.json aligned with the env copy
    (<env_dir>/<sample_id>.json) after critic auto-fix rewrites the latter.
    eval.py reads instructions from data.json, not from the per-env sample file.
    """
    data_path = PROC_DIR / category / "data.json"
    if not data_path.exists():
        log.warning(
            "Skip data.json sync: missing %s",
            data_path.relative_to(PROJECT_ROOT),
        )
        return
    try:
        catalog = json.loads(data_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.warning("Skip data.json sync: invalid JSON in %s: %s", data_path, e)
        return
    if not isinstance(catalog, list):
        log.warning("Skip data.json sync: %s root is not a list", data_path.name)
        return
    idx = next(
        (
            i
            for i, s in enumerate(catalog)
            if isinstance(s, dict) and s.get("id") == sample_id
        ),
        None,
    )
    if idx is None:
        log.warning(
            "Skip data.json sync: id %r not found in %s",
            sample_id,
            data_path.relative_to(PROJECT_ROOT),
        )
        return
    catalog[idx] = sample_obj
    data_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info(
        "Synced sample %s into %s",
        sample_id,
        data_path.relative_to(PROJECT_ROOT),
    )


def collect_categories(source: str) -> list[str]:
    if source == "all":
        return sorted(
            p.name for p in PROC_DIR.iterdir()
            if p.is_dir() and (p / "data.json").exists()
        )
    return [source]


def load_category_samples(category: str) -> list[dict]:
    data_file = PROC_DIR / category / "data.json"
    if not data_file.exists():
        raise FileNotFoundError(f"No data.json for category: {category}")
    return json.loads(data_file.read_text(encoding="utf-8"))


def build_remote_llm(
    model: str,
    base_url: str,
    api_key: Optional[str],
    max_model_len: int,
    *,
    temperature: float = 0.0,
):
    key = api_key or os.environ.get("REMOTE_API_KEY", "")
    max_tokens = min(max_model_len or 8192, 16384)
    return get_llm(
        "remote",
        model=model,
        base_url=base_url,
        api_key=key,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def execute_critic_round(
    category: str,
    sample_id: str,
    trajectories_text: str,
    critic_llm,
    critic_template: str,
    *,
    auto_fix: bool,
    dry_run: bool,
) -> Tuple[dict, bool]:
    """
    Run unified critic + optional auto-fix for one sample.
    Returns (report, did_write_disk).
    """
    env_dir = find_env_dir(category, sample_id)
    sample_json_s, env_py_s, env_json_s, env_stem = read_benchmark_artifacts(
        env_dir, sample_id
    )
    env_py_name = f"{env_stem}.py"
    sample_json_name = f"{sample_id}.json"
    env_json_name = f"{env_stem}.json"

    report = run_unified_critic(
        sample_json_s,
        env_py_s,
        env_json_s,
        trajectories_text,
        critic_llm,
        critic_template,
        env_py_basename=env_py_name,
        sample_json_basename=sample_json_name,
        env_json_basename=env_json_name,
    )

    out_report = critic_report_path(category, sample_id)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote critic report: %s", out_report.relative_to(PROJECT_ROOT))

    if dry_run or not auto_fix:
        return report, False

    try:
        targets = unified_writes(env_dir, sample_id, env_stem, report)
    except ValueError as e:
        log.error("Rejecting writes for %s: %s", sample_id, e)
        return report, False

    if not targets:
        log.info("  %s: all modify=false, skip write", sample_id)
        return report, False

    backup_and_write(env_dir, targets)
    sample_json_path = env_dir / f"{sample_id}.json"
    if sample_json_path in targets:
        try:
            sample_obj = json.loads(targets[sample_json_path])
        except json.JSONDecodeError as e:
            log.error(
                "data.json sync skipped: could not parse rewritten %s: %s",
                sample_json_path.relative_to(PROJECT_ROOT),
                e,
            )
        else:
            if isinstance(sample_obj, dict):
                sync_sample_json_to_category_data_json(category, sample_id, sample_obj)
            else:
                log.warning(
                    "data.json sync skipped: %s is not a JSON object",
                    sample_json_path.name,
                )
    remove_misplaced_placeholder_env_files(env_dir)
    return report, True


def remove_misplaced_placeholder_env_files(env_dir: Path) -> None:
    """Remove accidental EnvXxx.* files from older fix runs (safe literal names only)."""
    for fname in ("EnvXxx.py", "EnvXxx.json"):
        p = env_dir / fname
        if p.is_file():
            try:
                p.unlink()
                log.info("Removed misplaced %s", p.relative_to(PROJECT_ROOT))
            except OSError as e:
                log.warning("Could not remove %s: %s", p, e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trajectory-based unified critic (normal only) + optional per-file rewrite"
    )
    parser.add_argument("--source", required=True, help="Category name or 'all'")
    parser.add_argument(
        "--model_name",
        default="qwen2.5-7b",
        help="Trajectory folder under evaluation_results/<Category>/ (matches eval --model_name / model_dir_name)",
    )
    parser.add_argument("--critic_model", default="claude-3-5-sonnet-20240620")
    parser.add_argument(
        "--critic_base_url",
        default=(
            os.environ.get("AIGCBEST_URL")
            or os.environ.get("S1_BASE_URL")
            or os.environ.get("XHUB_URL")
            or "https://api.openai.com/v1"
        ),
        help="OpenAI-compatible endpoint (same gateway as Stage1/2; default: AIGCBEST_URL, S1_BASE_URL, XHUB_URL, or openai.com).",
    )
    parser.add_argument(
        "--critic_api_key",
        default=(
            os.environ.get("AIGCBEST_API_KEY")
            or os.environ.get("S1_API_KEY")
            or os.environ.get("XHUB_API_KEY")
            or os.environ.get("CRITIC_API_KEY")
        ),
        help="default: AIGCBEST_API_KEY, S1_API_KEY, XHUB_API_KEY, CRITIC_API_KEY, then REMOTE_API_KEY",
    )
    parser.add_argument("--critic_max_model_len", type=int, default=16384)
    parser.add_argument(
        "--critic_temperature",
        type=float,
        default=0.0,
        help="Remote critic sampling temperature (default 0 = greedy / deterministic intent).",
    )
    parser.add_argument(
        "--auto_fix",
        action="store_true",
        help="When any file has modify=true, write new_content to disk (after backup)",
    )
    parser.add_argument("--sample_id", default=None, help="Process only this sample id")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only write critic reports; never overwrite files",
    )
    parser.add_argument(
        "--eval_extra",
        default=os.environ.get("EVAL_EXTRA", "").strip() or None,
        help="Subdirectory under evaluation_results/<category>/<model_name>/ containing "
        "gen_res_normal.json (e.g. repeat1_derecipe/repeat_01). If omitted, pick newest run. "
        "Env: EVAL_EXTRA.",
    )
    parser.add_argument(
        "--critic_model_bundle",
        default=os.environ.get("CRITIC_MODEL_BUNDLE", "").strip() or None,
        help="Comma-separated model dirs under evaluation_results/<Category>/. "
        "When set, loads trajectories for each (model × --eval_extra_bundle path) "
        "instead of a single --model_name file. Env: CRITIC_MODEL_BUNDLE.",
    )
    parser.add_argument(
        "--eval_extra_bundle",
        default=os.environ.get("EVAL_EXTRA_BUNDLE", "").strip() or None,
        help="Comma-separated paths under each bundle model dir (e.g. four repeat_* "
        "folders). If omitted with --critic_model_bundle, uses --eval_extra once per model. "
        "Env: EVAL_EXTRA_BUNDLE.",
    )
    parser.add_argument(
        "--strip_output_prefix",
        type=int,
        default=1,
        help="Drop this many leading messages from trajectory output (default 1 = system only).",
    )
    args = parser.parse_args()

    ckey = (
        args.critic_api_key
        or os.environ.get("REMOTE_API_KEY")
    )
    critic_llm = build_remote_llm(
        args.critic_model,
        args.critic_base_url,
        ckey,
        args.critic_max_model_len,
        temperature=args.critic_temperature,
    )

    critic_tpl = load_template(TRAJECTORY_CRITIC_TEMPLATE)

    bundle_models: Optional[List[str]] = None
    if args.critic_model_bundle:
        bundle_models = [m.strip() for m in args.critic_model_bundle.split(",") if m.strip()]
        if not bundle_models:
            bundle_models = None

    categories = collect_categories(args.source)
    for category in categories:
        n_path: Optional[Path] = None
        run_label = ""
        eval_extras_list: Optional[List[str]] = None
        if bundle_models:
            raw_extras = (args.eval_extra_bundle or "").strip()
            eval_extras_list = [e.strip() for e in raw_extras.split(",") if e.strip()]
            if not eval_extras_list:
                if not args.eval_extra:
                    log.warning(
                        "Skipping category %s: critic_model_bundle set but no "
                        "--eval_extra_bundle or --eval_extra",
                        category,
                    )
                    continue
                eval_extras_list = [args.eval_extra.strip()]
            log.info(
                "Category %s: bundle %s models × %s eval path(s)",
                category,
                len(bundle_models),
                len(eval_extras_list),
            )
        else:
            n_path, run_label = resolve_trajectory_run(
                category, args.model_name, args.eval_extra
            )
            if not n_path:
                log.warning(
                    "Skipping category %s: no gen_res_normal.json under %s (eval_extra=%r)",
                    category,
                    EVAL_DIR / category / args.model_name,
                    args.eval_extra,
                )
                continue
            log.info(
                "Category %s: trajectories from %s",
                category,
                n_path.parent.relative_to(PROJECT_ROOT),
            )
            if run_label:
                log.info("  run: %s", run_label)

        try:
            samples = load_category_samples(category)
        except FileNotFoundError as e:
            log.warning("%s", e)
            continue

        if args.sample_id:
            samples = [s for s in samples if s.get("id") == args.sample_id]
            if not samples:
                log.error("sample_id %s not in %s/data.json", args.sample_id, category)
                continue

        for sample in samples:
            sample_id = sample.get("id")
            if not sample_id:
                continue

            try:
                env_dir = find_env_dir(category, sample_id)
            except FileNotFoundError as e:
                log.warning("%s", e)
                continue

            if bundle_models and eval_extras_list:
                records_b, run_labels_b = collect_bundle_trajectories(
                    category, bundle_models, eval_extras_list, sample_id
                )
                if not records_b:
                    log.warning("Skip %s: empty bundle for category %s", sample_id, category)
                    continue
                if all(r is None for r in records_b):
                    log.warning(
                        "Skip %s: sample missing from all bundle gen_res_normal.json",
                        sample_id,
                    )
                    continue
                slims = [
                    slice_trajectory_for_critic(r, args.strip_output_prefix)
                    if r is not None
                    else None
                    for r in records_b
                ]
                traj_block = format_trajectories_block(
                    slims,
                    strip_prefix=0,
                    run_labels=run_labels_b,
                    sample=sample,
                )
            else:
                assert n_path is not None
                t_norm = load_trajectory(n_path, sample_id)
                if t_norm is None:
                    log.warning(
                        "Skip %s: not in %s",
                        sample_id,
                        n_path,
                    )
                    continue

                slim_n = slice_trajectory_for_critic(t_norm, args.strip_output_prefix)
                traj_block = format_trajectories_block(
                    [slim_n], strip_prefix=0, sample=sample
                )

            try:
                execute_critic_round(
                    category,
                    sample_id,
                    traj_block,
                    critic_llm,
                    critic_tpl,
                    auto_fix=args.auto_fix,
                    dry_run=args.dry_run,
                )
            except Exception as e:
                log.error("Critic failed for %s: %s", sample_id, e)
                continue


if __name__ == "__main__":
    main()
