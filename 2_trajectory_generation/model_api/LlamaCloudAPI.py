from openai import OpenAI
import time
import json
import sys
import re
import os
from copy import deepcopy

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
        f"[LlamaCloudAPI] Context overflow; shrinking max_tokens from {cur} to {new_val} "
        f"(max_ctx={max_ctx}, input_tokens={input_tokens})"
    )
    return True


class LlamaCloudAPI(BaseAPI):
    def __init__(self, model_name, generation_config={}, *, base_url=None, api_key=None):
        super().__init__(generation_config)
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url or URL,
            api_key=api_key or API_KEY,
        )
        if 'temperature' not in self.generation_config:
            self.generation_config['temperature'] = 1.0

    def response(self, messages, tools):
        if not tools:
            tools = None
        safe_messages = _sanitize_messages(messages)
        gen = deepcopy(self.generation_config or {})
        for _ in range(5):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    tools=tools,
                    messages=safe_messages,
                    **gen,
                )
                if completion is None or completion.choices is None:
                    continue
                # regenerate if the completion is empty and not greedy
                if not completion.choices[0].message.content and completion.choices[0].message.tool_calls is None and gen.get('temperature', 1.0) != 0.:
                    print('empty completion, regenerate')
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
        _tools = deepcopy(tools)
        for tool in _tools:
            if len(tool['function']['parameters']['properties']) == 0:
                tool['function']['parameters']['properties'] = {'_tool_caller': {'type': 'string', 'description': 'The caller of the function. Simply set it to assistant.'}}

        completion = self.response(messages, _tools)
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
                "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {}
            })

        return {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning": reasoning
        }