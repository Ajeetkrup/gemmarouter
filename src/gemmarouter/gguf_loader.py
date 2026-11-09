"""Single-file GGUF checkpoint loader and dequantizer for GemmaDecisionRouterModel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch
from transformers import AutoTokenizer

from .configuration import GemmaDecisionRouterConfig
from .modeling import GemmaDecisionRouterModel
from .router import DEFAULT_REPO_ID, DecisionRouter, select_safe_device_and_dtype


def _dequantize_tensor_to_torch(t: Any, target_shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a GGUF tensor to float32 NumPy and convert to target PyTorch tensor."""
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        arr = t.data.astype(np.float32)
    else:
        arr = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
    arr = arr.reshape(target_shape)
    return torch.from_numpy(arr).to(dtype)


def _load_model_state_from_gguf(reader: gguf.GGUFReader, model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Populate model state_dict from GGUFReader tensors."""
    target_state = model.state_dict()
    tensor_map = {t.name: t for t in reader.tensors}
    loaded_state: dict[str, torch.Tensor] = {}

    for name, target_t in target_state.items():
        if name in tensor_map:
            t = tensor_map[name]
            loaded_state[name] = _dequantize_tensor_to_torch(t, tuple(target_t.shape), dtype)

    model.load_state_dict(loaded_state, strict=False)


def load_gguf(
    gguf_path: str | Path,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    threshold: float | None = None,
    margin_threshold: float | None = None,
    tokenizer: Any = None,
) -> DecisionRouter:
    """Load a single-file GGUF model checkpoint into an active DecisionRouter."""
    p = Path(gguf_path)
    if not p.exists():
        raise FileNotFoundError(f"GGUF checkpoint not found: {p}")

    reader = gguf.GGUFReader(str(p))
    resolved_dev, resolved_dt = select_safe_device_and_dtype(device=device, dtype=dtype)

    cfg_field = reader.fields.get("gemma_decision_router.config_json")
    if cfg_field is not None and len(cfg_field.parts) > 0:
        raw_json = str(bytes(cfg_field.parts[-1]).decode("utf-8", errors="replace"))
        cfg = GemmaDecisionRouterConfig.from_dict(json.loads(raw_json))
    else:
        cfg = GemmaDecisionRouterConfig()

    thresh_f = reader.fields.get("gemma_decision_router.null_threshold")
    margin_f = reader.fields.get("gemma_decision_router.margin_threshold")
    eff_thresh = float(thresh_f.parts[-1][0]) if thresh_f is not None and threshold is None else threshold
    eff_margin = float(margin_f.parts[-1][0]) if margin_f is not None and margin_threshold is None else margin_threshold

    model = GemmaDecisionRouterModel(cfg).to(resolved_dt)
    _load_model_state_from_gguf(reader, model, resolved_dt)
    if hasattr(reader, "data") and hasattr(reader.data, "_mmap"):
        reader.data._mmap.close()
    model.to(resolved_dev).eval()

    if tokenizer is None:
        tok_source = getattr(cfg, "_name_or_path", None) or DEFAULT_REPO_ID
        tok = AutoTokenizer.from_pretrained(tok_source)
    else:
        tok = tokenizer

    return DecisionRouter(
        model=model,
        tokenizer=tok,
        threshold=eff_thresh,
        margin_threshold=eff_margin,
    )


__all__ = ["load_gguf"]
