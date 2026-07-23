"""
IBM Granite 4.x Instruct — local inference aligned with official chat + tool format.

Prompting: apply_chat_template(..., tokenize=False, tools=..., add_generation_prompt=True),
then tokenizer(prompt, return_tensors="pt"). Tool replies use <tool_call>...</tool_call> XML
with JSON: {"name": ..., "arguments": ...}.
"""
import random
import re
import string

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    from modelscope import AutoModelForCausalLM, AutoTokenizer

from .BaseAPI import BaseAPI

_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*([\s\S]*?)\s*</tool_call>",
    re.IGNORECASE,
)


class GraniteAPI(BaseAPI):
    def __init__(self, model_path, tokenizer_path, generation_config=None):
        super().__init__(generation_config or {})
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        print("loading Granite model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        load_kw = {"trust_remote_code": True}
        if torch.cuda.is_available():
            load_kw["torch_dtype"] = torch.float16
            load_kw["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, **load_kw
            ).eval()
            self.device = next(self.model.parameters()).device
        else:
            load_kw["torch_dtype"] = torch.float32
            self.model = (
                AutoModelForCausalLM.from_pretrained(model_path, **load_kw)
                .to(torch.device("cpu"))
                .eval()
            )
            self.device = torch.device("cpu")

        print("finish loading Granite model")

    def _build_prompt(self, messages, tools):
        """Match IBM docs: string template first, tokenize in a second step."""
        use_tools = tools if tools else None
        if use_tools:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                tools=use_tools,
                add_generation_prompt=True,
            )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def response(self, messages, tools):
        prompt_text = self._build_prompt(messages, tools)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        gen_kw = dict(self.generation_config or {})
        output = self.model.generate(
            **inputs,
            pad_token_id=pad_id,
            **gen_kw,
        )
        in_len = inputs["input_ids"].shape[1]
        new_tokens = output[0, in_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def extract_tool_calls(self, completion):
        """Parse Granite <tool_call>{"name": "...", "arguments": {...}}</tool_call> blocks."""
        tool_calls = []
        for m in _TOOL_CALL_BLOCK.finditer(completion or ""):
            raw = m.group(1).strip()
            if not raw:
                continue
            data = self.parse_json(raw)
            if not isinstance(data, dict) or "name" not in data:
                continue
            args = data.get("arguments")
            if args is None and "parameters" in data:
                args = data["parameters"]
            if not isinstance(args, dict):
                if isinstance(args, str) and self.is_json(args):
                    args = self.parse_json(args)
                else:
                    continue
            tool_calls.append({
                "tool_call_id": "".join(
                    random.sample(string.ascii_letters + string.digits, 9)
                ),
                "tool_name": data["name"],
                "arguments": args,
            })
        return tool_calls

    def generate_response(self, messages, tools):
        completion = self.response(messages, tools)
        print("======================================")
        print(completion)
        print("======================================")

        tool_calls = self.extract_tool_calls(completion)
        visible = _TOOL_CALL_BLOCK.sub("", completion or "").strip()
        content = visible or None

        return {
            "content": content,
            "tool_calls": tool_calls,
            "reasoning": None,
        }
