from openai import OpenAI
import time
import json
import sys
import re
import os
from copy import deepcopy
from uuid import uuid4

sys.path.append("./model_api")
from .BaseAPI import BaseAPI
from .config import URL, API_KEY

_OPENAI_MSG_ALLOWED = frozenset({
    "role", "content", "name", "tool_calls", "tool_call_id", "function_call",
})
_CTX_ERR_RE = re.compile(
    r"maximum context length is\s+(\d+)\s+tokens.*?contains at least\s+(\d+)\s+input tokens",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
_FUNCTION_TAG_RE = re.compile(
    r"<function\s*=\s*([A-Za-z0-9_.:-]+)\s*>(.*?)</function>",
    re.IGNORECASE | re.DOTALL,
)
_PARAMETER_TAG_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z0-9_.:-]+)\s*>(.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)


def _sanitize_messages(messages: list) -> list:
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        clean = {k: deepcopy(m[k]) for k in m if k in _OPENAI_MSG_ALLOWED}
        if clean.get("content") is None and not clean.get("tool_calls"):
            clean["content"] = ""
        out.append(clean)
    return out


def _to_dict_arguments(raw):
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _parse_function_tag_arguments(raw: str):
    s = (raw or "").strip()
    if not s:
        return {}
    from_json = _to_dict_arguments(s)
    if from_json:
        return from_json
    out = {}
    for pm in _PARAMETER_TAG_RE.finditer(s):
        key = (pm.group(1) or "").strip()
        val = (pm.group(2) or "").strip()
        if key:
            out[key] = val
    return out


def _parse_xmlish_tool_calls(text: str):
    if not isinstance(text, str) or "<tool_call" not in text.lower():
        return [], text

    tool_calls = []
    for block_m in _TOOL_CALL_BLOCK_RE.finditer(text):
        inner = (block_m.group(1) or "").strip()
        if not inner:
            continue

        parsed = None
        try:
            obj = json.loads(inner)
            if isinstance(obj, dict):
                name = (obj.get("name") or obj.get("tool_name") or "").strip()
                if name:
                    parsed = {
                        "tool_call_id": f"xml_{uuid4().hex[:12]}",
                        "tool_name": name,
                        "arguments": _to_dict_arguments(obj.get("arguments")),
                    }
        except json.JSONDecodeError:
            parsed = None

        if parsed is None:
            fm = _FUNCTION_TAG_RE.search(inner)
            if fm:
                name = (fm.group(1) or "").strip()
                raw_args = (fm.group(2) or "").strip()
                if name:
                    args = _parse_function_tag_arguments(raw_args)
                    if raw_args and not args:
                        continue
                    parsed = {
                        "tool_call_id": f"xml_{uuid4().hex[:12]}",
                        "tool_name": name,
                        "arguments": args,
                    }

        if parsed is not None:
            tool_calls.append(parsed)

    cleaned = _TOOL_CALL_BLOCK_RE.sub("", text).strip()
    return tool_calls, (cleaned or None)


def _shrink_output_tokens_on_context_error(err: Exception, gen: dict):
    """On context overflow, shrink max_tokens and retry once."""
    msg = str(err or "")
    m = _CTX_ERR_RE.search(msg)
    if not m:
        return False

    try:
        max_ctx = int(m.group(1))
        input_tokens = int(m.group(2))
    except Exception:
        return False

    if "max_tokens" not in gen:
        return False
    cur = gen.get("max_tokens")
    try:
        cur = int(cur)
    except Exception:
        return False

    # Keep a larger margin for provider-side counting drift and hidden tokens.
    safety_margin = int(os.getenv("ADB_CTX_SAFETY_MARGIN", "256"))
    budget = max_ctx - input_tokens - safety_margin
    if budget < 1:
        return False

    # Drop faster than -1 to converge quickly under repeated 400s.
    new_val = max(64, min(cur - 64, budget))
    if new_val >= cur:
        return False

    gen["max_tokens"] = new_val
    print(
        f"[QwenCloudAPI] Context overflow; shrinking max_tokens from {cur} to {new_val} "
        f"(max_ctx={max_ctx}, input_tokens={input_tokens})"
    )
    return True


class QwenCloudAPI(BaseAPI):
    def __init__(self, model_name, generation_config={}, *, base_url=None, api_key=None):
        super().__init__(generation_config)
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url or URL,
            api_key=api_key or API_KEY,
        )
        # Cloud non-streaming APIs do not support enable_thinking
        self.generation_config['extra_body'] = {"enable_thinking": False}
     
    def response(self, messages, tools):
        if not tools:
            tools = None
        safe_messages = _sanitize_messages(messages)
        gen = deepcopy(self.generation_config or {})
        for _ in range(10):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    tools=tools,
                    messages=safe_messages,
                    **gen
                )
                if completion is None or completion.choices is None:
                    continue
                return completion
            except Exception as e:
                if _shrink_output_tokens_on_context_error(e, gen):
                    # Persist the lowered cap so next round does not reset to 2048.
                    self.generation_config["max_tokens"] = gen["max_tokens"]
                    continue
                print(e)
                time.sleep(1)

    def generate_response(self, messages, tools):
        completion = self.response(messages, tools)
        print('======================================')
        print(completion)
        print('======================================')
        if completion is None:
            return None

        msg = completion.choices[0].message

        reasoning = getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None)

        content = msg.content if msg.content else None

        raw_tool_calls = msg.tool_calls or []
        tool_calls = []
        for tc in raw_tool_calls:
            tool_calls.append({
                "tool_call_id": tc.id,
                "tool_name": tc.function.name,
                "arguments": _to_dict_arguments(tc.function.arguments)
            })

        # Fallback for backends that place tool calls in assistant content
        # like <tool_call>{"name":"...","arguments":{...}}</tool_call>.
        if not tool_calls and isinstance(content, str):
            xml_calls, cleaned_content = _parse_xmlish_tool_calls(content)
            if xml_calls:
                tool_calls = xml_calls
                content = cleaned_content

        return {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning": reasoning
        }