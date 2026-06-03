import joblib
import os
from typing import Any

def save_object(obj: Any, filename: str, output_dir: str = "models"):
    """Saves a python object using joblib."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    joblib.dump(obj, filepath)

def load_object(filepath: str) -> Any:
    """Loads a python object using joblib."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return joblib.load(filepath)
