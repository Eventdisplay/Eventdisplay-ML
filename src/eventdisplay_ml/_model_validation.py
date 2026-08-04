"""Isolated validation for serialized model artifacts."""

import sys
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None


def _set_memory_limit(memory_limit_bytes):
    """Limit virtual memory on Linux before loading third-party model objects."""
    if resource is not None and sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))


def _validate_payload(payload):
    """Check the minimum structure required by the application model loaders."""
    if not isinstance(payload, dict):
        raise TypeError("Model payload must be a dictionary.")

    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Model payload must contain a non-empty 'models' dictionary.")

    for name, config in models.items():
        if not isinstance(config, dict) or config.get("model") is None:
            raise ValueError(f"Model configuration '{name}' does not contain a trained model.")

        model = config["model"]
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            booster.num_boosted_rounds()
            booster.num_features()


def main():
    """Load and validate one joblib model file."""
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m eventdisplay_ml._model_validation FILE MEMORY_BYTES")

    model_path = Path(sys.argv[1])
    memory_limit_bytes = int(sys.argv[2])
    _set_memory_limit(memory_limit_bytes)

    from eventdisplay_ml.utils import load_joblib

    _validate_payload(load_joblib(model_path))


if __name__ == "__main__":
    main()
