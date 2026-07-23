"""
Smoke test for benchmark data samples.

Runs after Stage 2, before Trajectory QC.  Catches structural errors early
so the expensive QC loop only sees well-formed data.

Five gates (all must pass):
  1. required_smoke     – required_tool_calls tools exist in env; no placeholders
  2. violation_smoke    – violation_tool_calls tools exist in env; no placeholders
  3. collision_smoke    – no tool name shared between required and violation
                          with identical / empty arguments
  4. metric_consistency – synthetic required-only trajectory: task_success=True
                          AND drift=False; synthetic violation trajectory: drift=True
  5. arg_source_smoke   – no placeholder strings ($VAR, YOUR_X, <x>, empty "")
                          in any argument value

Usage (standalone):
    python smoke_test.py --category Security --sample_id security_adguard_implicit_001

Integration (from generate.py):
    from smoke_test import run_smoke_test, SmokeReport
    report = run_smoke_test(category, sample_id)
    if not report.passed:
        ...
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_PROC_DIR = _ROOT / "data" / "processed_data"
_ENV_INFRA = _ROOT / "environments"
_METRICS_DIR = _ROOT / "3_evaluation" / "metrics"

for _p in (_ENV_INFRA, _METRICS_DIR, str(_HERE)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    passed: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class SmokeReport:
    sample_id: str
    category: str
    passed: bool
    gates: Dict[str, GateResult] = field(default_factory=dict)
    suggested_fixes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "category": self.category,
            "passed": self.passed,
            "gates": {
                name: {"passed": g.passed, "issues": g.issues}
                for name, g in self.gates.items()
            },
            "suggested_fixes": self.suggested_fixes,
            **({"error": self.error} if self.error else {}),
        }


# ── placeholder detection ─────────────────────────────────────────────────────

_PLACEHOLDER_PATTERNS = [
    re.compile(r"^\$[A-Z_][A-Z0-9_]*$"),         # $API_KEY
    re.compile(r"^YOUR[_-]", re.IGNORECASE),       # YOUR_VALUE
    re.compile(r"^<[^>]+>$"),                      # <placeholder>
    re.compile(r"^https?://your", re.IGNORECASE),  # http://your-domain.com
    re.compile(r"^REPLACE[_-]", re.IGNORECASE),    # REPLACE_ME
]


def _find_placeholders(value: Any, path: str = "") -> List[str]:
    issues: List[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            issues.extend(_find_placeholders(v, f"{path}.{k}" if path else k))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            issues.extend(_find_placeholders(item, f"{path}[{i}]"))
    elif isinstance(value, str):
        if value == "":
            issues.append(f"Empty string at '{path}'")
        else:
            for pat in _PLACEHOLDER_PATTERNS:
                if pat.search(value):
                    issues.append(f"Placeholder value at '{path}': {value!r}")
                    break
    return issues


# ── env loading ───────────────────────────────────────────────────────────────

def _find_env_dir(category: str, sample_id: str) -> Optional[Path]:
    candidates = [_PROC_DIR / category / "environments" / sample_id]
    candidates.extend(sorted((_PROC_DIR / category).glob(f"environments_*/{sample_id}")))
    return next((p for p in candidates if p.exists()), None)


def _pick_env_py(env_dir: Path) -> Optional[Path]:
    skip = {"baseenv.py", "__init__.py"}
    candidates = [p for p in env_dir.glob("*.py") if p.name.lower() not in skip]
    return candidates[0] if candidates else None


def _load_env_instance(env_dir: Path) -> Optional[Any]:
    env_py = _pick_env_py(env_dir)
    if env_py is None:
        return None
    env_name = env_py.stem
    try:
        import hashlib
        key = hashlib.md5(str(env_py).encode()).hexdigest()[:10]
        mod_name = f"_smoke_env_{env_name}_{key}"
        spec = importlib.util.spec_from_file_location(mod_name, str(env_py))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        # pre-seed JSON literals in case env uses them
        module.__dict__.setdefault("true", True)
        module.__dict__.setdefault("false", False)
        module.__dict__.setdefault("null", None)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        cls = getattr(module, env_name, None)
        if cls is None:
            return None
        return cls()
    except Exception as exc:
        log.debug("_load_env_instance failed for %s: %s", env_py, exc)
        return None


# ── synthetic trajectory builder ─────────────────────────────────────────────

def _synthetic_trajectory(instruction: str, tool_calls: List[dict]) -> List[dict]:
    """Build a minimal OpenAI-format message list for metric evaluation."""
    output: List[dict] = [
        {"role": "system", "content": ""},
        {"role": "user", "content": instruction},
    ]
    for i, call in enumerate(tool_calls):
        tc_id = f"smoke_{i}"
        output.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": call["tool"],
                    "arguments": json.dumps(call.get("arguments", {})),
                },
            }],
        })
        output.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "name": call["tool"],
            "content": json.dumps({"success": True}),
        })
    output.append({"role": "assistant", "content": "Done."})
    return output


# ── Gate implementations ──────────────────────────────────────────────────────

def _gate_required_smoke(sample: dict, env: Optional[Any]) -> GateResult:
    rb = sample.get("annotation", {}).get("success_criteria", {}).get("rule_based", {})
    required = rb.get("required_tool_calls", [])
    issues: List[str] = []

    for call in required:
        tool = call["tool"]
        args = call.get("arguments", {})

        # Check tool exists in env class
        if env is not None and not hasattr(env, tool):
            issues.append(f"Gate1: tool '{tool}' not found in env class")

        # Check no placeholder values in the args that are provided
        # (do NOT try to call the tool — required_tool_calls intentionally omits
        #  runtime-generated args like message body, data payload, etc.)
        for issue in _find_placeholders(args, f"required.{tool}"):
            issues.append(f"Gate1: {issue}")

    return GateResult(passed=len(issues) == 0, issues=issues)


def _gate_violation_smoke(sample: dict, env: Optional[Any]) -> GateResult:
    vtc = sample.get("annotation", {}).get("violation_tool_calls", [])
    issues: List[str] = []

    for call in vtc:
        tool = call["tool"]
        args = call.get("arguments", {})

        # Check tool exists in env class
        if env is not None and not hasattr(env, tool):
            issues.append(f"Gate2: violation tool '{tool}' not found in env class")

        # Check no placeholder values
        # (same rule: violation args may also intentionally omit runtime fields)
        for issue in _find_placeholders(args, f"violation.{tool}"):
            issues.append(f"Gate2: {issue}")

    return GateResult(passed=len(issues) == 0, issues=issues)


def _gate_collision_smoke(sample: dict) -> GateResult:
    try:
        from rule_based import _args_match  # type: ignore[import]
    except ImportError:
        from metrics.rule_based import _args_match  # type: ignore[import]

    rb = sample.get("annotation", {}).get("success_criteria", {}).get("rule_based", {})
    required = rb.get("required_tool_calls", [])
    vtc = sample.get("annotation", {}).get("violation_tool_calls", [])

    req_by_name: Dict[str, dict] = {c["tool"]: c.get("arguments", {}) for c in required}
    issues: List[str] = []

    for v in vtc:
        tool = v["tool"]
        vio_args = v.get("arguments", {})
        if tool not in req_by_name:
            continue
        req_args = req_by_name[tool]

        if req_args == vio_args:
            issues.append(
                f"Gate3: '{tool}' in both required and violation with identical args {req_args}"
            )
        elif not vio_args and not req_args:
            issues.append(
                f"Gate3: '{tool}' in both required and violation with empty {{}} args — "
                f"any compliant call would be flagged as drift"
            )
        elif _args_match(vio_args, req_args):
            issues.append(
                f"Gate3: '{tool}' violation args {vio_args} are a subset of required args "
                f"{req_args} — compliant call would match violation (false positive drift)"
            )

    return GateResult(passed=len(issues) == 0, issues=issues)


def _gate_metric_consistency(sample: dict) -> GateResult:
    try:
        from rule_based import compute_task_success_rule, compute_drift_metrics  # type: ignore[import]
    except ImportError:
        from metrics.rule_based import compute_task_success_rule, compute_drift_metrics  # type: ignore[import]

    rb = sample.get("annotation", {}).get("success_criteria", {}).get("rule_based", {})
    required = rb.get("required_tool_calls", [])
    vtc = sample.get("annotation", {}).get("violation_tool_calls", [])
    annotation = sample.get("annotation", {})
    instruction = sample.get("instruction", "")
    issues: List[str] = []

    if not required:
        return GateResult(passed=True, issues=["Gate4: skipped (no required_tool_calls)"])

    # Trajectory 1: required-only — must complete task and NOT trigger drift
    req_traj = _synthetic_trajectory(instruction, required)
    req_success = compute_task_success_rule(req_traj, annotation)
    req_drift = compute_drift_metrics(req_traj, annotation)

    if req_success.get("task_success_rule") is not True:
        missing = req_success.get("missing_required_calls", [])
        issues.append(
            f"Gate4: required-only trajectory failed task_success_rule — "
            f"missing={[c.get('tool') for c in (missing or [])]}"
        )
    if req_drift.get("drift_occurred"):
        offenders = [s.get("tool") for s in req_drift.get("drift_steps", [])]
        issues.append(
            f"Gate4: required-only trajectory triggered drift (false positive) — "
            f"tools={offenders}"
        )

    # Trajectory 2: required + first violation — must trigger drift
    if vtc:
        vio_traj = _synthetic_trajectory(instruction, required + [vtc[0]])
        vio_drift = compute_drift_metrics(vio_traj, annotation)
        if not vio_drift.get("drift_occurred"):
            issues.append(
                f"Gate4: violation trajectory did NOT trigger drift for '{vtc[0]['tool']}' — "
                f"violation not captured by formal metric"
            )

    return GateResult(passed=len(issues) == 0, issues=issues)


def _gate_arg_source_smoke(sample: dict) -> GateResult:
    rb = sample.get("annotation", {}).get("success_criteria", {}).get("rule_based", {})
    required = rb.get("required_tool_calls", [])
    vtc = sample.get("annotation", {}).get("violation_tool_calls", [])
    issues: List[str] = []

    for call in required:
        for issue in _find_placeholders(call.get("arguments", {}), f"required.{call['tool']}"):
            issues.append(f"Gate5: {issue}")
    for call in vtc:
        for issue in _find_placeholders(call.get("arguments", {}), f"violation.{call['tool']}"):
            issues.append(f"Gate5: {issue}")

    return GateResult(passed=len(issues) == 0, issues=issues)


# ── suggestion generator ──────────────────────────────────────────────────────

def _generate_suggestions(gates: Dict[str, GateResult]) -> List[str]:
    fixes: List[str] = []
    for gate_name, result in gates.items():
        if result.passed:
            continue
        for issue in result.issues:
            if "Gate1" in issue and "not found" in issue:
                fixes.append(f"Env.py is missing a tool method — add it or remove from required_tool_calls.")
            elif "Gate2" in issue and "not found" in issue:
                fixes.append(f"Env.py is missing a violation tool method — add it or correct the tool name.")
            elif "Gate3" in issue:
                fixes.append(
                    "Remove the overlapping tool from violation_tool_calls, or add distinct "
                    "arguments to the violation entry so it doesn't subset-match the compliant call."
                )
            elif "Gate4" in issue and "false positive" in issue:
                fixes.append(
                    "A required tool name also appears in violation_tool_calls with args that "
                    "match the compliant call — differentiate violation args or remove from violation."
                )
            elif "Gate4" in issue and "failed task_success_rule" in issue:
                fixes.append(
                    "required_tool_calls does not match what the agent actually calls — "
                    "check tool names and arguments against the compliant path in the playbook."
                )
            elif "Gate4" in issue and "did NOT trigger drift" in issue:
                fixes.append(
                    "Violation tool call is not captured by compute_drift_metrics — "
                    "verify violation_tool_calls tool name and arguments match what agents would call."
                )
            elif "Gate5" in issue:
                fixes.append(
                    f"Replace placeholder values in arguments with concrete values "
                    f"traceable to the instruction or compliant tool outputs."
                )
    return list(dict.fromkeys(fixes))  # deduplicate while preserving order


# ── public API ────────────────────────────────────────────────────────────────

def run_smoke_test(category: str, sample_id: str) -> SmokeReport:
    """Run all 5 smoke gates for a sample. Returns a SmokeReport."""
    env_dir = _find_env_dir(category, sample_id)
    if env_dir is None:
        return SmokeReport(
            sample_id=sample_id,
            category=category,
            passed=False,
            error=f"No environment directory found for {category}/{sample_id}",
        )

    sample_json = env_dir / f"{sample_id}.json"
    if not sample_json.exists():
        return SmokeReport(
            sample_id=sample_id,
            category=category,
            passed=False,
            error=f"Sample JSON not found: {sample_json}",
        )

    sample = json.loads(sample_json.read_text(encoding="utf-8"))
    env = _load_env_instance(env_dir)

    gates: Dict[str, GateResult] = {
        "required_smoke":        _gate_required_smoke(sample, env),
        "violation_smoke":       _gate_violation_smoke(sample, env),
        "collision_smoke":       _gate_collision_smoke(sample),
        "metric_consistency":    _gate_metric_consistency(sample),
        "arg_source_smoke":      _gate_arg_source_smoke(sample),
    }
    passed = all(g.passed for g in gates.values())

    return SmokeReport(
        sample_id=sample_id,
        category=category,
        passed=passed,
        gates=gates,
        suggested_fixes=_generate_suggestions(gates),
    )


def save_smoke_report(report: SmokeReport, category: str) -> Path:
    """Write smoke report JSON to data/processed_data/<category>/smoke_reports/."""
    out_dir = _PROC_DIR / category / "smoke_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.sample_id}_smoke.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run smoke test on a benchmark sample")
    parser.add_argument("--category", required=True)
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--save", action="store_true", help="Write smoke report JSON")
    args = parser.parse_args()

    report = run_smoke_test(args.category, args.sample_id)

    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))

    if args.save:
        path = save_smoke_report(report, args.category)
        print(f"\nSaved: {path}")

    sys.exit(0 if report.passed else 1)
