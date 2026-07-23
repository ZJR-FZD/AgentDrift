"""
LLM API wrappers for benchmark data generation.

Two backends:
  - RemoteAPI : any OpenAI-compatible endpoint
  - LocalVLLM : local vLLM (multi-GPU)

Qwen3 thinking is disabled by default via /no_think prefix.

RemoteAPI.generate retries transient failures (network, 429, 5xx) when
``ADB_REMOTE_API_MAX_RETRIES`` / ``REMOTE_API_MAX_RETRIES`` > 1 (see generate()).
"""

import logging
import os
import time
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class RemoteAPI(BaseLLM):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = None,
        temperature: float = 0.9,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
    ):
        from openai import OpenAI
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("REMOTE_API_KEY", ""),
        )

    def _max_remote_retries(self) -> int:
        for key in ("ADB_REMOTE_API_MAX_RETRIES", "REMOTE_API_MAX_RETRIES"):
            raw = os.environ.get(key, "").strip()
            if raw:
                try:
                    return max(1, int(raw))
                except ValueError:
                    pass
        return 5

    def _retry_delay_sec(self) -> float:
        raw = os.environ.get("ADB_REMOTE_API_RETRY_DELAY_SEC", "").strip()
        if not raw:
            raw = os.environ.get("REMOTE_API_RETRY_DELAY_SEC", "2").strip()
        try:
            return max(0.1, float(raw))
        except ValueError:
            return 2.0

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """Best-effort classification without hard dependency on OpenAI exception layout."""
        try:
            import openai

            conn_errs: tuple = tuple(
                c
                for c in (
                    getattr(openai, "APIConnectionError", None),
                    getattr(openai, "APITimeoutError", None),
                    getattr(openai, "RateLimitError", None),
                )
                if isinstance(c, type)
            )
            if conn_errs and isinstance(exc, conn_errs):
                return True
            status_error = getattr(openai, "APIStatusError", None)
            if isinstance(status_error, type) and isinstance(exc, status_error):
                code = getattr(exc, "status_code", None)
                if code is None:
                    return False
                return int(code) == 429 or int(code) >= 500
        except Exception:
            pass
        return False

    def generate(self, prompt: str) -> str:
        prefix = "/think\n" if self.enable_thinking else "/no_think\n"
        content = prefix + prompt
        attempts = self._max_remote_retries()
        base_delay = self._retry_delay_sec()
        last_exc = None
        for attempt in range(attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                out = response.choices[0].message.content
                if out is None:
                    raise RuntimeError("RemoteAPI: empty message content in API response")
                return out
            except Exception as e:
                last_exc = e
                if attempt >= attempts - 1 or not self._is_retryable(e):
                    raise
                sleep_s = base_delay * (2**attempt)
                log.warning(
                    "RemoteAPI.generate retry %s/%s after %s: %s (sleep %.1fs)",
                    attempt + 1,
                    attempts,
                    type(e).__name__,
                    e,
                    sleep_s,
                )
                time.sleep(sleep_s)
        assert last_exc is not None
        raise last_exc


class LocalVLLM(BaseLLM):
    def __init__(
        self,
        model_path: str,
        gpu_ids: list = None,
        max_model_len: int = 16384,
        temperature: float = 0.9,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
    ):
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        from vllm import LLM, SamplingParams

        self.enable_thinking = enable_thinking
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=len(gpu_ids) if gpu_ids else 1,
            max_model_len=max_model_len,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            repetition_penalty=1.05,
        )

    def generate(self, prompt: str) -> str:
        prefix = "/think\n" if self.enable_thinking else "/no_think\n"
        outputs = self.llm.generate([prefix + prompt], self.sampling_params)
        return outputs[0].outputs[0].text


def get_llm(backend: str, **kwargs) -> BaseLLM:
    """
    Factory function.

    backend: 'remote' | 'local'

    For 'remote':
        model           (str, required)
        base_url        (str, required)
        api_key         (str, optional)  falls back to REMOTE_API_KEY env var
        temperature     (float, default 0.9)
        max_tokens      (int,   default 4096)
        enable_thinking (bool,  default False)

    For 'local':
        model_path      (str, required)
        gpu_ids         (list, optional)  e.g. [0, 1, 2, 3]
        max_model_len   (int,  default 16384)
        temperature     (float, default 0.9)
        max_tokens      (int,   default 4096)
        enable_thinking (bool,  default False)

    Examples
    --------
    llm = get_llm("remote",
                  model="qwen3-235b-a22b-instruct-2507",
                  base_url="https://your-proxy.com/v1")

    llm = get_llm("local",
                  model_path="/data/models/qwen3-235b",
                  gpu_ids=[0, 1, 2, 3])
    """
    if backend == "remote":
        for key in ("model", "base_url"):
            if not kwargs.get(key):
                raise ValueError(f"'remote' backend requires '{key}' kwarg.")
        return RemoteAPI(
            model=kwargs["model"],
            base_url=kwargs["base_url"],
            api_key=kwargs.get("api_key"),
            temperature=kwargs.get("temperature", 0.9),
            max_tokens=kwargs.get("max_tokens", 4096),
            enable_thinking=kwargs.get("enable_thinking", False),
        )
    elif backend == "local":
        if not kwargs.get("model_path"):
            raise ValueError("'local' backend requires 'model_path' kwarg.")
        return LocalVLLM(
            model_path=kwargs["model_path"],
            gpu_ids=kwargs.get("gpu_ids"),
            max_model_len=kwargs.get("max_model_len", 16384),
            temperature=kwargs.get("temperature", 0.9),
            max_tokens=kwargs.get("max_tokens", 4096),
            enable_thinking=kwargs.get("enable_thinking", False),
        )
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Choose 'remote' or 'local'.")