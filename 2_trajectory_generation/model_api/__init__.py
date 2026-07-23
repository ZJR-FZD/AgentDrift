from .OpenaiAPI import OpenaiAPI
from .ClaudeAPI import ClaudeAPI
from .GeminiAPI import GeminiAPI
from .DeepseekAPI import DeepseekAPI
from .QwenCloudAPI import QwenCloudAPI
from .LlamaCloudAPI import LlamaCloudAPI
from .config import MODEL_PATHS

try:
    from .GLM4API import GLM4API
except ModuleNotFoundError:
    GLM4API = None

try:
    from .Llama3API import Llama3API
except ModuleNotFoundError:
    Llama3API = None

try:
    from .GraniteAPI import GraniteAPI
except (ModuleNotFoundError, ImportError):
    GraniteAPI = None

try:
    from .QwenAPI import QwenAPI
except (ModuleNotFoundError, ImportError):
    # Usually missing torch/transformers or broken optional local stack.
    QwenAPI = None
