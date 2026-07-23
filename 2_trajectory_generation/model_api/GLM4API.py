from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import random
import string
import sys
sys.path.append('./model_api')
from .BaseAPI import BaseAPI
from copy import deepcopy

class GLM4API(BaseAPI):

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
        if str(tools) not in messages[0]['content']:
            messages[0]['content'] = f"{messages[0]['content']}\n\n{tools}"
        ipt_messages = []
        for msg in messages:
            msg = deepcopy(msg)  # do not mutate the caller's messages
            if msg['role'] == 'assistant':
                msg.pop('tool_calls') if 'tool_calls' in msg else None
                msg.pop('reasoning') if 'reasoning' in msg else None
            ipt_messages.append(msg)
        inputs = self.tokenizer.apply_chat_template(ipt_messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, **self.generation_config)
        return self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

    def fix_json_brackets(self, s):
        """Drop unmatched closing brackets (e.g. ``}}}`` → ``}``)."""
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
                # Drop extra ``}``
            elif ch == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                    result.append(ch)
                # Drop extra ``]``
            else:
                result.append(ch)
        return ''.join(result)
    
    def _try_name_newline_dict(self, cleaned: str):
        """
        GLM often emits ``tool_name`` then a newline then ``{...}`` (or a single-quoted
        Python dict). Try this before brace-slicing; otherwise ``{...}`` is treated as a
        whole JSON candidate, dropped for missing name/arguments, and this branch never runs
        because candidates is already non-empty.
        """
        if "\n{" not in cleaned and "\r\n{" not in cleaned:
            return []
        text = cleaned.replace("\r\n", "\n").strip()
        try:
            name, arguments = text.split("\n", 1)
        except ValueError:
            return []
        name = name.strip()
        arguments = arguments.strip()
        if not name or not arguments.startswith("{"):
            return []
        if not self.is_json(arguments):
            return []
        args_obj = self.parse_json(arguments)
        if not isinstance(args_obj, dict):
            return []
        return [{"name": name, "arguments": args_obj}]

    def extract_tool_calls(self, completion):
        """Extract tool calls from model output; return [] if none parse cleanly."""
        tool_calls = []

        # Strip markdown code fences
        cleaned = completion.replace('```json', '').replace('```', '').strip()

        candidates = []

        # Path 1: whole string is JSON
        if self.is_json(cleaned):
            data = self.parse_json(cleaned)
            if isinstance(data, dict):
                candidates = [data]
            elif isinstance(data, list):
                candidates = data

        # Path 2: GLM ``name\\n{args}`` (must run before generic brace slicing)
        if not candidates:
            candidates = self._try_name_newline_dict(cleaned)

        # Path 3: slice [...] / {...}; require OpenAI-style name+arguments (avoid bare arg dicts)
        if not candidates:
            for start_char, end_char in [('[', ']'), ('{', '}')]:
                if start_char not in cleaned:
                    continue
                start = cleaned.index(start_char)
                end = cleaned.rindex(end_char) + 1
                fragment = cleaned[start:end]
                fragment = self.fix_json_brackets(fragment)
                if not self.is_json(fragment):
                    continue
                data = self.parse_json(fragment)
                if isinstance(data, dict) and "name" in data and "arguments" in data:
                    candidates = [data]
                    break
                if (
                    isinstance(data, list)
                    and data
                    and all(
                        isinstance(x, dict) and "name" in x and "arguments" in x
                        for x in data
                    )
                ):
                    candidates = data
                    break

        # Keep valid tool calls
        for item in candidates:
            if not isinstance(item, dict) or 'name' not in item or 'arguments' not in item:
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

        return tool_calls if tool_calls else []

    def generate_response(self, messages, tools):
        completion = self.response(messages, tools)
        print('======================================')
        print(completion)
        print('======================================')

        tool_calls = self.extract_tool_calls(completion)
        # Like Qwen: if tools were parsed, do not put raw name+args text into content
        content = None if tool_calls else (completion.strip() or None)

        print('🌟Extracted content:', content)
        print('🚌Extracted tool_calls:', tool_calls)

        return {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning": None  # GLM4 has no reasoning field
        }