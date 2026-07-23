"""
Parse LLM output into three separate files:
  - {EnvClassName}.py
  - {EnvClassName}.json
  - {sample_id}.json

Supports two output formats:
  1. Primary:  === FILE: filename === blocks
  2. Fallback: bare ```python / ```json blocks in order
"""

import re
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict

log = logging.getLogger(__name__)


@dataclass
class ParsedOutput:
    env_py_name: str
    env_py_code: str
    env_json_name: str
    env_json_content: str    # clean re-serialised JSON string
    data_json_name: str
    data_json_content: str   # clean re-serialised JSON string


# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_fences(content: str) -> str:
    """Strip optional markdown fences. Closing fence must be removable even when the
    model adds trailing chat formatting (e.g. ---) after ```."""
    content = content.strip()
    content = re.sub(r"^```(?:python|json)?\s*\n", "", content)
    # Prefer end-anchored strip when the block ends cleanly with a fence.
    content = re.sub(r"(\n```[a-zA-Z0-9]*\s*)+$", "", content)
    # If the model wrote ``` then --- / blank lines, the above $ anchor fails;
    # drop from the first line that is only a closing fence through end of block.
    content = re.sub(r"(?ms)^```[a-zA-Z0-9]*\s*$(?:\n.*)?\Z", "", content)
    return content.strip()


def _extract_json(content: str, label: str) -> tuple:
    content = content.strip()
    decoder = json.JSONDecoder()
    for start, ch in enumerate(content):
        if ch in ("{", "["):
            try:
                obj, _ = decoder.raw_decode(content, start)
                return obj, json.dumps(obj, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                continue
    raise ValueError(
        f"[Parser] {label}: no valid JSON found.\nPreview:\n{content[:400]}"
    )

def _strip_preamble_blocks(text: str) -> str:
    text = re.sub(r"<drift_design>.*?</drift_design>", "", text, flags=re.DOTALL)
    text = re.sub(r"<self_check>.*?</self_check>", "", text, flags=re.DOTALL)
    return text


def _strip_json_fences(content: str) -> str:
    s = content.strip()
    s = re.sub(r"^```(?:json|JSON|text|)\s*\n", "", s)
    s = re.sub(r"(\n```[a-zA-Z0-9]*\s*)+$", "", s).strip()
    return s


# ── Critic LLM output (raw JSON) ───────────────────────────────────────────

def parse_fix_output(raw_text: str) -> Dict[str, str]:
    """
    Parse fix_prompt.md model output: === FILE: <filename> === blocks.
    Keys are filenames as emitted (e.g. sample.json, MyEnv.py, MyEnv.json).
    Does not assume a fixed env class name. Uses document order: first two
    .json blocks and the single .py block (may appear in any position).
    """
    if not raw_text or not str(raw_text).strip():
        raise ValueError("[Parser] Fix output is empty.")
    ordered = _extract_file_blocks_ordered(str(raw_text))
    if len(ordered) < 3:
        raise ValueError(
            f"[Parser] Fix output: need at least 3 FILE blocks, got {len(ordered)}"
        )
    py_entries = [(n, c) for n, c in ordered if n.endswith(".py")]
    json_entries = [(n, c) for n, c in ordered if n.endswith(".json")]
    if len(py_entries) != 1:
        raise ValueError(
            f"[Parser] Fix output: expected exactly one .py block, got {len(py_entries)}"
        )
    if len(json_entries) < 2:
        raise ValueError(
            f"[Parser] Fix output: expected at least two .json blocks, got {len(json_entries)}"
        )
    n_py, c_py = py_entries[0]
    (n_j1, c_j1), (n_j2, c_j2) = json_entries[0], json_entries[1]
    return {n_py: c_py, n_j1: c_j1, n_j2: c_j2}


def parse_critic_json(raw_text: str) -> dict:
    """
    Parse a Critic model reply that must be a single JSON object.
    Strips optional markdown fences and decodes the first top-level { ... }.
    """
    if not raw_text or not str(raw_text).strip():
        raise ValueError("[Parser] Critic output is empty.")
    text = _strip_json_fences(str(raw_text))
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(
        f"[Parser] Critic: no JSON object found.\nPreview:\n{text[:500]}"
    )


UNIFIED_TRAJECTORY_CRITIC_KEYS = ("env_py", "sample_json", "env_json")


def parse_unified_trajectory_critic_json(raw_text: str) -> dict:
    """
    Parse unified trajectory critic output: exactly env_py, sample_json, env_json;
    each value is { "modify": bool, "new_content"?: str }.
    """
    obj = parse_critic_json(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("[Parser] Unified critic: root must be a JSON object.")
    missing = [k for k in UNIFIED_TRAJECTORY_CRITIC_KEYS if k not in obj]
    if missing:
        raise ValueError(
            f"[Parser] Unified critic: missing keys {missing}; "
            f"required {list(UNIFIED_TRAJECTORY_CRITIC_KEYS)}"
        )
    for key in UNIFIED_TRAJECTORY_CRITIC_KEYS:
        entry = obj[key]
        if not isinstance(entry, dict):
            raise ValueError(f"[Parser] Unified critic: {key} must be an object.")
        if "modify" not in entry:
            raise ValueError(f"[Parser] Unified critic: {key} missing boolean 'modify'.")
        if not isinstance(entry["modify"], bool):
            raise ValueError(f"[Parser] Unified critic: {key}.modify must be boolean.")
        body = entry.get("new_content")
        if body is None and "content" in entry:
            body = entry.get("content")
        if entry["modify"]:
            if not isinstance(body, str) or not body.strip():
                raise ValueError(
                    f"[Parser] Unified critic: {key} has modify=true but "
                    "missing non-empty new_content (or content)."
                )
            obj[key] = {"modify": True, "new_content": _strip_fences(body.strip())}
        else:
            obj[key] = {"modify": False}
    return obj


# ── strategy 1: === FILE: === blocks ─────────────────────────────────────────

def _extract_file_blocks(text: str) -> dict:
    # strip drift_design block first to avoid its content interfering
    text = _strip_preamble_blocks(text)
    pattern = r"=== FILE:\s*(.+?)\s*===\s*\n(.*?)(?====\s*FILE:|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    return {name.strip(): _strip_fences(content) for name, content in matches}


def _extract_file_blocks_ordered(text: str) -> list[tuple[str, str]]:
    """Same as FILE blocks but preserves document order (list of (name, content))."""
    text = _strip_preamble_blocks(text)
    pattern = r"=== FILE:\s*(.+?)\s*===\s*\n(.*?)(?====\s*FILE:|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(name.strip(), _strip_fences(content)) for name, content in matches]


def _parse_from_file_blocks(blocks: dict, sample_id: str) -> ParsedOutput:
    py_files   = {k: v for k, v in blocks.items() if k.endswith(".py")}
    json_files = {k: v for k, v in blocks.items() if k.endswith(".json")}

    if not py_files:
        raise ValueError("[Parser] No .py block found.")
    if len(json_files) < 2:
        raise ValueError(
            f"[Parser] Need 2 .json blocks, got {len(json_files)}: {list(json_files)}"
        )

    env_py_name_full, env_py_code = next(iter(py_files.items()))
    env_class_name = env_py_name_full.replace(".py", "")

    env_json_key  = f"{env_class_name}.json"
    data_json_key = f"{sample_id}.json"

    if env_json_key not in json_files:
        candidates = [k for k in json_files if k != data_json_key]
        if not candidates:
            raise ValueError(f"[Parser] Cannot find env json block '{env_json_key}'.")
        env_json_key = candidates[0]

    if data_json_key not in json_files:
        candidates = [k for k in json_files if k != env_json_key]
        if not candidates:
            raise ValueError(f"[Parser] Cannot find data json block '{data_json_key}'.")
        data_json_key = candidates[0]

    _, env_json_clean         = _extract_json(json_files[env_json_key],  f"env json ({env_json_key})")
    data_obj, data_json_clean = _extract_json(json_files[data_json_key], f"data json ({data_json_key})")

    _check_data_schema(data_obj)

    return ParsedOutput(
        env_py_name=env_class_name,
        env_py_code=env_py_code,
        env_json_name=env_class_name,
        env_json_content=env_json_clean,
        data_json_name=sample_id,
        data_json_content=data_json_clean,
    )


# ── strategy 2: bare code-fence fallback ─────────────────────────────────────

def _extract_bare_blocks(text: str) -> list:
    text = _strip_preamble_blocks(text)
    pattern = r"```(python|json)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang.strip(), content.strip()) for lang, content in matches]


def _parse_from_bare_blocks(
    text: str, sample_id: str, env_class_name_hint: Optional[str]
) -> ParsedOutput:
    blocks = _extract_bare_blocks(text)

    py_blocks   = [c for l, c in blocks if l == "python"]
    json_blocks = [c for l, c in blocks if l == "json"]

    if not py_blocks:
        raise ValueError("[Parser][fallback] No ```python block found.")
    if len(json_blocks) < 2:
        raise ValueError(
            f"[Parser][fallback] Need 2 ```json blocks, got {len(json_blocks)}."
        )

    env_py_code    = py_blocks[0]
    env_class_name = env_class_name_hint
    if not env_class_name:
        m = re.search(r"class\s+(\w+)\s*\(", env_py_code)
        env_class_name = m.group(1) if m else "UnknownEnv"

    parsed_jsons = []
    for jcontent in json_blocks:
        try:
            obj, clean = _extract_json(jcontent, "bare json block")
            parsed_jsons.append((obj, clean))
        except ValueError:
            continue

    if len(parsed_jsons) < 2:
        raise ValueError(
            f"[Parser][fallback] Could not parse 2 valid JSON blocks, "
            f"got {len(parsed_jsons)}."
        )

    env_candidates  = [(o, c) for o, c in parsed_jsons if isinstance(o, list)]
    data_candidates = [(o, c) for o, c in parsed_jsons if isinstance(o, dict) and "id" in o]

    if env_candidates and data_candidates:
        env_obj,  env_json_clean  = env_candidates[0]
        data_obj, data_json_clean = data_candidates[0]
    else:
        env_obj,  env_json_clean  = parsed_jsons[0]
        data_obj, data_json_clean = parsed_jsons[1]

    _check_data_schema(data_obj)

    return ParsedOutput(
        env_py_name=env_class_name,
        env_py_code=env_py_code,
        env_json_name=env_class_name,
        env_json_content=env_json_clean,
        data_json_name=sample_id,
        data_json_content=data_json_clean,
    )


# ── schema check ──────────────────────────────────────────────────────────────

def _check_data_schema(data_obj: dict):
    required = ["id", "authorization_type", "instruction", "environments", "annotation"]
    missing  = [f for f in required if f not in data_obj]
    if missing:
        raise ValueError(f"[Parser] Data JSON missing fields: {missing}")


# ── public entry point ────────────────────────────────────────────────────────

def parse_llm_output(
    raw_text: str,
    sample_id: str,
    env_class_name_hint: Optional[str] = None,
) -> ParsedOutput:
    blocks = _extract_file_blocks(raw_text)
    if len(blocks) >= 3:
        log.info("[Parser] Using FILE-block strategy.")
        return _parse_from_file_blocks(blocks, sample_id)

    log.warning(
        f"[Parser] FILE blocks found: {len(blocks)} — "
        f"falling back to bare code-fence strategy."
    )

    try:
        result = _parse_from_bare_blocks(raw_text, sample_id, env_class_name_hint)
        log.info("[Parser] Fallback strategy succeeded.")
        return result
    except ValueError as e:
        raise ValueError(
            f"[Parser] Both strategies failed.\n"
            f"  FILE-block result: {list(blocks.keys())}\n"
            f"  Bare-block error:  {e}\n"
            f"  Raw output preview:\n{raw_text[:600]}"
        )