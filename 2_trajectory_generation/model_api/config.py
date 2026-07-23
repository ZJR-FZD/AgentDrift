"""
Model / endpoint configuration for data-generation trajectory QC.

All secrets and cluster hosts come from environment variables.
Do not hard-code API keys or private hostnames in this file.
"""

import json
import os

AIGCBEST_URL = os.getenv("AIGCBEST_URL", "https://api.openai.com/v1")
AIGCBEST_API_KEY = os.getenv("AIGCBEST_API_KEY", "")
XHUB_URL = os.getenv("XHUB_URL", "")
XHUB_API_KEY = os.getenv("XHUB_API_KEY", "")

# OpenAI-compatible client: one host + key for all models that gateway exposes.
URL = (
    os.getenv("ADB_OPENAI_COMPAT_BASE_URL")
    or os.getenv("AIGCBEST_URL")
    or XHUB_URL
    or "https://api.openai.com/v1"
)
API_KEY = (
    os.getenv("ADB_OPENAI_COMPAT_API_KEY")
    or os.getenv("AIGCBEST_API_KEY")
    or XHUB_API_KEY
)

# -----------------------------------------------------------------------------
# Local / cluster OpenAI-compatible chat/completions endpoints used by trajectory QC.
#
# Prefer env ``ADB_LOCAL_CHAT_ENDPOINTS_JSON``:
#   export ADB_LOCAL_CHAT_ENDPOINTS_JSON='[
#     ["qwen36_35b_a3b", "http://127.0.0.1:8010/v1/chat/completions"],
#     ["llama33_70b_instruct", "http://127.0.0.1:8011/v1/chat/completions"]
#   ]'
#
# If unset, the example endpoints below are placeholders — replace before running QC.
# -----------------------------------------------------------------------------
_DEFAULT_LOCAL_CHAT_COMPLETIONS_ENDPOINTS: list[tuple[str, str]] = [
    ("qwen36_35b_a3b", "http://127.0.0.1:8010/v1/chat/completions"),
    ("llama33_70b_instruct", "http://127.0.0.1:8011/v1/chat/completions"),
    ("llama4_maverick_17b_128e_instruct_fp8", "http://127.0.0.1:8012/v1/chat/completions"),
]


def _load_local_chat_endpoints() -> list[tuple[str, str]]:
    raw = os.getenv("ADB_LOCAL_CHAT_ENDPOINTS_JSON", "").strip()
    if not raw:
        return list(_DEFAULT_LOCAL_CHAT_COMPLETIONS_ENDPOINTS)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            "ADB_LOCAL_CHAT_ENDPOINTS_JSON must be a JSON list of "
            '[api_model_id, chat_completions_url] pairs'
        ) from e
    out: list[tuple[str, str]] = []
    for item in parsed:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError(f"Invalid endpoint entry: {item!r}")
        out.append((str(item[0]), str(item[1])))
    return out


LOCAL_CHAT_COMPLETIONS_ENDPOINTS: list[tuple[str, str]] = _load_local_chat_endpoints()


def openai_base_url_from_chat_completions_url(chat_url: str) -> str:
    """``.../v1/chat/completions`` → OpenAI client ``base_url`` (``.../v1``)."""
    u = (chat_url or "").strip().rstrip("/")
    suf = "/chat/completions"
    if u.endswith(suf):
        return u[: -len(suf)]
    return u


def _build_qc_trajectory_agent_openai() -> dict[str, tuple[str, str]]:
    """
    Trajectory-QC agent registry: eval.py key → (api_model_id, base_url).
    Keys are what ``ADB_QC_AGENT_MODELS`` / ``--qc_agent_models`` should use.
    """
    by_mid = {mid: curl for mid, curl in LOCAL_CHAT_COMPLETIONS_ENDPOINTS}
    want = {
        "qwen36_35b_a3b": [
            "adb-qc-qwen36-35b-a3b",
            "adb-qc-qwen3-next-35b-a3b",
            "adb-local-qwen3-6-35b-a3b",
        ],
        "llama33_70b_instruct": [
            "adb-qc-llama-3-3-70b-instruct",
            "adb-local-llama-3-3-70b-instruct",
        ],
        "llama4_maverick_17b_128e_instruct_fp8": [
            "adb-qc-llama4-maverick-17b",
        ],
    }
    out: dict[str, tuple[str, str]] = {}
    for mid, eval_keys in want.items():
        curl = by_mid.get(mid)
        if not curl:
            continue
        for eval_key in eval_keys:
            out[eval_key] = (mid, openai_base_url_from_chat_completions_url(curl))
    return out


QC_TRAJECTORY_AGENT_OPENAI: dict[str, tuple[str, str]] = _build_qc_trajectory_agent_openai()

# Optional local HF weight dirs (USE_LOCAL=1). Override via env or edit placeholders.
MODEL_PATHS = {
    "Qwen2.5-7B-Instruct": os.getenv("ADB_MODEL_PATH_QWEN25_7B", "***"),
    "Qwen3-32B": os.getenv("ADB_MODEL_PATH_QWEN3_32B", "***"),
    "Qwen3-8B": os.getenv("ADB_MODEL_PATH_QWEN3_8B", "***"),
    "DeepSeek-R1-Distill-Qwen-7B": "***",
    "DeepSeek-R1-Distill-Llama-8B": "***",
    "glm-4-9b-chat-hf": "***",
    "Llama-3.1-8B-Instruct": os.getenv("ADB_MODEL_PATH_LLAMA31_8B", "***"),
    "granite-4.1-8b": "***",
    "Qwen3-235B-A22B-Instruct": "***",
    "Llama-3.3-70B-Instruct": os.getenv("ADB_MODEL_PATH_LLAMA33_70B", "***"),
    "Qwen3-Next-35B-A3B": os.getenv("ADB_MODEL_PATH_QWEN3_NEXT_35B", "***"),
    "Llama-4-Maverick-17B-128E-Instruct-FP8": os.getenv(
        "ADB_MODEL_PATH_LLAMA4_MAVERICK", "***"
    ),
}
