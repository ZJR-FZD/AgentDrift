from openai import OpenAI
import time
import json
import sys
import copy
import re
from uuid import uuid4
sys.path.append('./model_api')
from .BaseAPI import BaseAPI
from .config import URL, API_KEY

# Keys the OpenAI Chat Completions API accepts on message objects.
_OPENAI_MSG_ALLOWED = frozenset({
    "role", "content", "name", "tool_calls", "tool_call_id", "function_call",
})

# chat.completions.create() only accepts a subset of sampling params; eval.py also
# sets top_k / min_p for local HF models — strip those here to avoid TypeError.
_OPENAI_CREATE_ALLOWED = frozenset({
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "stop",
    "n",
    "logit_bias",
    "response_format",
    "reasoning_effort",
    "parallel_tool_calls",
    "service_tier",
})


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
_FUNCTION_TAG_RE = re.compile(
    r"<function\s*=\s*([A-Za-z0-9_.:-]+)\s*>(.*?)</function>",
    re.IGNORECASE | re.DOTALL,
)
_PARAMETER_TAG_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z0-9_.:-]+)\s*>(.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)
_CTX_ERR_RE = re.compile(
    r"maximum context length is\s+(\d+)\s+tokens.*?contains at least\s+(\d+)\s+input tokens",
    re.IGNORECASE | re.DOTALL,
)


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
    """
    Parse arguments in function-tag body.
    Supported forms:
      - JSON object string
      - XML-ish parameter tags: <parameter=path>api-keys/stripe</parameter>
    """
    s = (raw or "").strip()
    if not s:
        return {}

    # First try strict JSON object.
    from_json = _to_dict_arguments(s)
    if from_json:
        return from_json

    # Then parse <parameter=...> blocks.
    out = {}
    for pm in _PARAMETER_TAG_RE.finditer(s):
        key = (pm.group(1) or "").strip()
        val = (pm.group(2) or "").strip()
        if key:
            out[key] = val
    return out


def _parse_xmlish_tool_calls(text: str):
    """
    Parse non-standard `<tool_call>...</tool_call>` blocks into OpenAI-like tool calls.
    Supports:
      1) JSON payload: {"name":"fn","arguments":{...}}
      2) function tag: <function=fn>{...}</function> (args may be empty)
    """
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
                    # If function body is present but cannot be parsed, skip this
                    # tool call instead of issuing an empty-argument invocation.
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


def _parse_json_in_content_tool_calls(text: str) -> list:
    """
    Parse tool calls emitted as JSON in content by some vLLM backends (e.g. llama4-maverick).
    Supports two formats:
      - Single object: {"type": "function", "name": "fn", "parameters": {...}}
      - JSON array:    ["{...}", "{...}"] where each element is a JSON-encoded object
    Uses "parameters" key (not "arguments") as emitted by llama4.
    """
    text = text.strip()
    if not text or text[0] not in ('{', '['):
        return []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []

    def _extract_one(item) -> dict | None:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception:
                return None
        if not isinstance(item, dict):
            return None
        if item.get("type") != "function":
            return None
        name = (item.get("name") or "").strip()
        if not name:
            return None
        args = item.get("parameters") or item.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        return {
            "tool_call_id": f"jic_{uuid4().hex[:12]}",
            "tool_name": name,
            "arguments": args if isinstance(args, dict) else {},
        }

    if isinstance(obj, list):
        calls = [_extract_one(item) for item in obj]
        return [c for c in calls if c is not None]
    elif isinstance(obj, dict):
        call = _extract_one(obj)
        return [call] if call else []
    return []


def _shrink_output_tokens_on_context_error(err: Exception, gen: dict):
    """
    If provider reports context overflow, lower max output tokens and retry.
    Returns True when gen is updated and caller should retry.
    """
    msg = str(err or "")
    m = _CTX_ERR_RE.search(msg)
    if not m:
        return False

    try:
        max_ctx = int(m.group(1))
        input_tokens = int(m.group(2))
    except Exception:
        return False

    # Keep a tiny safety margin for provider-side accounting differences.
    budget = max_ctx - input_tokens - 16
    if budget < 1:
        return False

    out_key = "max_tokens" if "max_tokens" in gen else "max_completion_tokens"
    cur = gen.get(out_key, budget)
    try:
        cur = int(cur)
    except Exception:
        cur = budget

    new_val = max(1, min(cur - 1, budget))
    if new_val >= cur:
        return False
    gen[out_key] = new_val
    print(
        f"[OpenaiAPI] Context overflow; shrinking {out_key} from {cur} to {new_val} "
        f"(max_ctx={max_ctx}, input_tokens={input_tokens})"
    )
    return True


def _openai_sanitize_messages(messages: list) -> list:
    """Strip non-standard keys (e.g. our local `reasoning`) so upstream does not 400."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        clean = {k: copy.deepcopy(m[k]) for k in m if k in _OPENAI_MSG_ALLOWED}
        role = clean.get("role")
        # Some providers 400 on null/None `content` when not a tool+tool_calls turn.
        if clean.get("content") is None and not clean.get("tool_calls"):
            clean["content"] = ""
        if role == "tool" and clean.get("content") is None:
            clean["content"] = ""
        out.append(clean)
    return out


def _fix_array_schema(schema):
    """Recursively patch array schemas missing the items field."""
    if not isinstance(schema, dict):
        return schema
    if schema.get("type") == "array" and "items" not in schema:
        schema["items"] = {"type": "string"}
    for value in schema.values():
        if isinstance(value, dict):
            _fix_array_schema(value)
        elif isinstance(value, list):
            for item in value:
                _fix_array_schema(item)
    return schema


class OpenaiAPI(BaseAPI):
    def __init__(
        self,
        model_name,
        generation_config={},
        *,
        base_url=None,
        api_key=None,
    ):
        super().__init__(generation_config)
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url or URL,
            api_key=api_key or API_KEY,
        )

    def response(self, messages, tools):
        if not tools:
            tools = None
        else:
            tools = copy.deepcopy(tools)
            for tool in tools:
                fn = tool.get("function")
                if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict):
                    fn["parameters"] = _fix_array_schema(fn["parameters"])
        safe_messages = _openai_sanitize_messages(messages)
        gen = {
            k: v
            for k, v in (self.generation_config or {}).items()
            if k in _OPENAI_CREATE_ALLOWED
        }
        for _ in range(10):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    tools=tools,
                    messages=safe_messages,
                    **gen,
                )
                if completion is None or completion.choices is None:
                    continue
                return completion
            except Exception as e:
                if _shrink_output_tokens_on_context_error(e, gen):
                    continue
                import traceback
                traceback.print_exc()
                time.sleep(1)

    def generate_response(self, messages, tools):
        completion = self.response(messages, tools)
        if completion is None:
            return None

        msg = completion.choices[0].message

        # reasoning: OpenAI models usually omit this; getattr as fallback
        reasoning = getattr(msg, 'reasoning_content', None) or getattr(msg, 'reasoning', None)

        # content: normalize empty string to None
        content = msg.content if msg.content else None
        # gpt-5 / o-series: visible answer may live in reasoning while content is ""
        if not content and reasoning:
            content = reasoning

        # tool_calls: normalize None / empty to []
        raw_tool_calls = msg.tool_calls or []
        tool_calls = []
        for tc in raw_tool_calls:
            tool_calls.append({
                "tool_call_id": tc.id,
                "tool_name": tc.function.name,
                "arguments": _to_dict_arguments(tc.function.arguments)
            })

        # Fallback for OpenAI-compatible backends that emit XML-like tool tags
        # in content but leave message.tool_calls empty.
        if not tool_calls and isinstance(content, str):
            xml_calls, cleaned_content = _parse_xmlish_tool_calls(content)
            if xml_calls:
                tool_calls = xml_calls
                content = cleaned_content

        # Fallback for backends (e.g. llama4-maverick vLLM) that emit tool calls
        # as JSON in content: single {"type":"function","name":...,"parameters":{...}}
        # or a JSON array of such objects.
        if not tool_calls and isinstance(content, str):
            json_calls = _parse_json_in_content_tool_calls(content)
            if json_calls:
                tool_calls = json_calls
                content = None

        return {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning": reasoning
        }