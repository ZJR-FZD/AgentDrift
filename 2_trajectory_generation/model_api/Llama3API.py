from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import random
import string
import sys
sys.path.append('./model_api')
from .BaseAPI import BaseAPI


def _iter_top_level_json_values(s: str):
    """
    Repeatedly ``JSONDecoder().raw_decode`` from ``s`` so multiple values separated
    by ``;`` or whitespace (e.g. Llama: ``{...}; {...}``) parse correctly.
    """
    dec = json.JSONDecoder()
    idx = 0
    n = len(s)
    while idx < n:
        while idx < n and s[idx] in " \t\n\r;":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = dec.raw_decode(s, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        yield obj
        idx = end


class Llama3API(BaseAPI):
    def __init__(self, model_path, tokenizer_path, generation_config={}):
        super().__init__(generation_config)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device('cuda')
        print('loading model...')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(self.device).eval()
        self.tokenizer.padding_side = 'left'
        print('finish loading')

    def response(self, messages, tools):
        if tools:
            inputs = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
        else:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, **self.generation_config)
        return self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

    def fix_json_brackets(self, s):
        """Drop unmatched closing brackets."""
        stack = []
        result = []
        for ch in s:
            if ch == '{':
                stack.append('{')
                result.append(ch)
            elif ch == '[':
                stack.append('[')
                result.append(ch)
            elif ch == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                    result.append(ch)
            elif ch == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                    result.append(ch)
            else:
                result.append(ch)
        return ''.join(result)

    def extract_tool_calls(self, completion):
        """Extract tool calls; accept either ``arguments`` or ``parameters`` field names."""
        tool_calls = []
        cleaned = completion.replace("```json", "").replace("```", "").strip()
        candidates: list = []

        # Path 0: multiple ``{...}; {...}`` (common for Llama) or consecutive JSON values
        for data in _iter_top_level_json_values(cleaned):
            if isinstance(data, dict) and "name" in data:
                candidates.append(data)
            elif isinstance(data, list):
                for x in data:
                    if isinstance(x, dict) and "name" in x:
                        candidates.append(x)

        # Path 1: whole string is one JSON object or array
        if not candidates and self.is_json(cleaned):
            data = self.parse_json(cleaned)
            if isinstance(data, dict):
                candidates = [data]
            elif isinstance(data, list):
                candidates = [x for x in data if isinstance(x, dict)]

        # Path 2: slice first [...] or {...} fragment (single-block rindex fallback)
        if not candidates:
            for start_char, end_char in [("[", "]"), ("{", "}")]:
                if start_char in cleaned:
                    start = cleaned.index(start_char)
                    end = cleaned.rindex(end_char) + 1
                    fragment = cleaned[start:end]
                    fragment = self.fix_json_brackets(fragment)
                    if self.is_json(fragment):
                        data = self.parse_json(fragment)
                        if isinstance(data, dict):
                            candidates = [data]
                        elif isinstance(data, list):
                            candidates = [x for x in data if isinstance(x, dict)]
                        break

        # Keep valid tool calls; map ``parameters`` → ``arguments`` when needed
        for item in candidates:
            if not isinstance(item, dict) or 'name' not in item:
                continue
            # Accept ``parameters`` as an alias for ``arguments``
            if 'parameters' in item and 'arguments' not in item:
                item['arguments'] = item['parameters']
            if 'arguments' not in item:
                continue
            arguments = item['arguments']
            if not isinstance(arguments, dict):
                if self.is_json(arguments):
                    arguments = self.parse_json(arguments)
                else:
                    continue
            tool_calls.append({
                "tool_call_id": ''.join(random.sample(string.ascii_letters + string.digits, 9)),
                "tool_name": item['name'],
                "arguments": arguments
            })

        return tool_calls

    def generate_response(self, messages, tools):
        completion = self.response(messages, tools)
        print('======================================')
        print(completion)
        print('======================================')

        content = completion.strip() or None
        tool_calls = self.extract_tool_calls(completion)
        # If the whole output is multiple JSON tools (no prose), do not put raw text in content
        if tool_calls and len(tool_calls) > 1:
            content = None

        return {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning": None  # Llama3 has no reasoning field
        }