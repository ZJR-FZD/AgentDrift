import importlib
import importlib.util
import hashlib
from pathlib import Path
import sys
import traceback
from copy import deepcopy


def _seed_json_literal_aliases(module_dict: dict) -> None:
    """Codegen sometimes pastes JSON (true/false/null). Pre-seed module globals so they resolve like Python."""
    module_dict.setdefault("true", True)
    module_dict.setdefault("false", False)
    module_dict.setdefault("null", None)


class EnvManager:
    def __init__(self, env_dir=None):
        self.env_dir = Path(env_dir).resolve() if env_dir else None
    
    def init_env(self, env_name, env_params):
        # print(sys.path)
        try:
            if self.env_dir is not None:
                env_file = self.env_dir / f"{env_name}.py"
                if not env_file.exists():
                    raise FileNotFoundError(f"Environment module file not found: {env_file}")
                module_key = hashlib.md5(str(env_file).encode("utf-8")).hexdigest()[:12]
                module_name = f"_adb_env_{env_name}_{module_key}"
                spec = importlib.util.spec_from_file_location(module_name, str(env_file))
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load module spec from {env_file}")
                env_module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = env_module
                _seed_json_literal_aliases(env_module.__dict__)
                spec.loader.exec_module(env_module)
            else:
                env_module = importlib.import_module(env_name)
                _seed_json_literal_aliases(env_module.__dict__)
        except Exception as e:
            print(e)
            traceback.print_exc()
            return None
        # print(env_module)
        env = getattr(env_module, env_name)
        
        if env_params is not None:
            return env(parameters=deepcopy(env_params))
        return env()