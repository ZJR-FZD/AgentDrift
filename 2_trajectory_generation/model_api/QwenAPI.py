from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import random
import string
import os
import re
import logging

from .BaseAPI import BaseAPI

log = logging.getLogger(__name__)

class QwenAPI(BaseAPI):
    def __init__(self, model_path, tokenizer_path, generation_config={}):
        super().__init__(generation_config)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        print('loading model...')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # "balanced" spreads shards more evenly than "auto" on multi-GPU; override via ADB_QWEN_DEVICE_MAP.
        device_map = (os.environ.get("ADB_QWEN_DEVICE_MAP", "balanced") or "balanced").strip()

        load_kw = dict(
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        if os.environ.get("ADB_QWEN_DISABLE_SDPA", "").strip().lower() not in {"1", "true", "yes"}:
            load_kw["attn_implementation"] = "sdpa"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw).eval()
        except (TypeError, ValueError) as e:
            log.warning("Qwen load without SDPA (retry): %s", e)
            load_kw.pop("attn_implementation", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kw).eval()

        # Per-model cap (optional); also see eval.py ADB_LOCAL_MAX_NEW_TOKENS for all locals.
        cap = os.environ.get("ADB_QWEN_MAX_NEW_TOKENS", "").strip()
        if cap.isdigit():
            m = max(64, int(cap))
            self.generation_config = dict(self.generation_config or {})
            cur = int(self.generation_config.get("max_new_tokens", m))
            self.generation_config["max_new_tokens"] = min(cur, m)

        # For sharded models (device_map=...), place input tensors on the first model shard.
        self.device = next(self.model.parameters()).device
        self.tokenizer.padding_side = 'left'
        print('finish loading')

    def response(self, messages, tools):
        # Qwen3 thinking models need enable_thinking=False; Qwen2.5 tokenizers lack that
        # kwarg (TypeError) — fall back once without it.
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
        except TypeError:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
        gen_kw = dict(self.generation_config or {})
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.eos_token_id,
                **gen_kw,
            )
        completion = self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return completion

    def extract_tag(self, text, tag):
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else None

    def extract_all_tags(self, text, tag):
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches]

    def remove_tag(self, text, tag):
        pattern = f"<{tag}>.*?</{tag}>"
        return re.sub(pattern, "", text, flags=re.DOTALL)

    def generate_response(self, messages, tools):
        completion = self.response(messages, tools)
        print('======================================')
        print(completion)
        print('======================================')

        # reasoning
        think_content = self.extract_tag(completion, "think") or ''

        # Strip tags; remainder is content
        cleaned = self.remove_tag(completion, "think")
        cleaned = self.remove_tag(cleaned, "tool_call")
        normal_content = cleaned.strip() or None

        # Parse all <tool_call> tags
        tool_call_contents = self.extract_all_tags(completion, "tool_call")
        tool_calls = []
        for i, tc_str in enumerate(tool_call_contents):
            # Fail if inner text is not valid JSON / missing fields / args not a dict
            if not self.is_json(tc_str):
                log.warning(
                    "QwenAPI: tool_call #%s inner text is not valid JSON (is_json=False). "
                    "Preview: %r",
                    i,
                    (tc_str[:300] + "...") if len(tc_str) > 300 else tc_str,
                )
                return None  # malformed; same as other backends returning None
            res = self.parse_json(tc_str)
            if not isinstance(res, dict) or "name" not in res or "arguments" not in res:
                log.warning(
                    "QwenAPI: tool_call #%s JSON missing 'name' or 'arguments' (or not an object). "
                    "Keys: %s",
                    i,
                    list(res.keys()) if isinstance(res, dict) else type(res).__name__,
                )
                return None

            arguments = res["arguments"]
            if not isinstance(arguments, dict):
                if self.is_json(arguments):
                    arguments = self.parse_json(arguments)
                else:
                    log.warning(
                        "QwenAPI: tool_call #%s (%r): arguments is not a dict and not a JSON string. "
                        "Preview: %r",
                        i,
                        res.get("name"),
                        (arguments[:200] + "...")
                        if isinstance(arguments, str) and len(arguments) > 200
                        else arguments,
                    )
                    return None
                    
            tool_calls.append({
                "tool_call_id": ''.join(random.sample(string.ascii_letters + string.digits, 9)),
                "tool_name": res['name'],
                "arguments": arguments
            })

        return {
            "content": normal_content,
            "tool_calls": tool_calls,
            "reasoning": think_content
        }