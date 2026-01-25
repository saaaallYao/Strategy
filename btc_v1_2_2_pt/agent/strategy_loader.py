import importlib.util
import os
from typing import Any


def load_strategy(path: str, config: dict) -> Any:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Strategy file not found: {path}")
    spec = importlib.util.spec_from_file_location("user_strategy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load strategy module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Strategy"):
        raise AttributeError("Strategy file must define a Strategy class")
    return module.Strategy(config)
