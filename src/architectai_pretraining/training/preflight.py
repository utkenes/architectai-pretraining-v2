"""Colab environment inspection with intentionally conservative recommendations."""

import importlib.metadata
import json
from typing import Any


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_environment() -> dict[str, Any]:
    import platform

    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "accelerate": _version("accelerate"),
        "peft": _version("peft"),
        "datasets": _version("datasets"),
        "bitsandbytes": _version("bitsandbytes"),
        "gpu": None,
        "vram_gb": 0.0,
        "cuda_available": False,
        "bf16_supported": False,
        "recommended_dtype": "float32",
        "full_parameter_8b_feasible": False,
        "recommended_smoke_mode": "gpu_required",
        "recommended_train_batch_size": None,
        "compute_capability": None,
    }
    try:
        import torch  # type: ignore[import-not-found]

        result["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["gpu"] = props.name
            result["vram_gb"] = round(props.total_memory / 1024**3, 1)
            result["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            result["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            result["recommended_dtype"] = "bfloat16" if result["bf16_supported"] else "float16"
            result["recommended_smoke_mode"] = "qlora_or_smaller_model"
            result["recommended_train_batch_size"] = 1 if result["vram_gb"] < 24 else 2
            # Full optimizer states and activations make 16 GB class GPUs unsuitable.
            result["full_parameter_8b_feasible"] = result["vram_gb"] >= 80.0
    except ImportError:
        pass
    return result


def main() -> None:
    print(json.dumps(inspect_environment(), indent=2))


if __name__ == "__main__":
    main()
