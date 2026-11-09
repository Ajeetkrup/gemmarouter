"""Pure ONNX Runtime Decision & Tool Router without PyTorch runtime dependency."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .router import DEFAULT_REPO_ID, RegisteredTool, RouteResult, _normalize_options


def _build_route_result_np(
    opt_keys: list[str],
    probs_np: np.ndarray,
    p_null: float,
    eff_thresh: float,
    eff_margin: float,
    cache_hit: bool,
    total_ms: float,
    head_ms: float,
) -> RouteResult:
    """Construct a calibrated RouteResult from NumPy arrays without PyTorch."""
    probs_list = [float(p) for p in probs_np]
    sorted_idxs = np.argsort(-probs_np)[: min(2, len(opt_keys))]
    top_idx = int(sorted_idxs[0])
    second_idx = int(sorted_idxs[1]) if len(sorted_idxs) > 1 else top_idx
    top_key, second_key = opt_keys[top_idx], opt_keys[second_idx]
    top_prob, second_prob = float(probs_np[top_idx]), float(probs_np[second_idx])
    margin = round(top_prob - second_prob, 4)

    is_null = p_null >= eff_thresh
    needs_clarification = (not is_null) and (margin < eff_margin)
    abstain = is_null or needs_clarification
    clarify_msg = (
        f"Ambiguous collision between '{top_key}' ({top_prob * 100:.1f}%) and "
        f"'{second_key}' ({second_prob * 100:.1f}%). Please clarify which option to run."
        if needs_clarification
        else None
    )
    return RouteResult(
        decision=None if abstain else top_key,
        top_option=top_key,
        confidence=round(top_prob, 4),
        null_probability=round(p_null, 6),
        abstain=abstain,
        threshold=eff_thresh,
        probabilities={k: round(float(p), 4) for k, p in zip(opt_keys, probs_list)},
        option_cache_hit=cache_hit,
        latency_ms=round(total_ms, 2),
        head_latency_ms=round(head_ms, 3),
        margin=margin,
        margin_threshold=eff_margin,
        needs_clarification=needs_clarification,
        clarify_options=(top_key, second_key) if needs_clarification else None,
        clarification_prompt=clarify_msg,
    )


class ONNXDecisionRouter:
    """Fast, low-dependency DecisionRouter powered purely by ONNX Runtime and NumPy."""

    def __init__(
        self,
        encoder_session: ort.InferenceSession,
        head_session: ort.InferenceSession,
        tokenizer: Any,
        config_dict: dict[str, Any] | None = None,
        threshold: float | None = None,
        margin_threshold: float | None = None,
    ) -> None:
        self.encoder_session = encoder_session
        self.head_session = head_session
        self.tokenizer = tokenizer
        cfg = config_dict or {}
        self.threshold = float(cfg.get("null_threshold", 0.58) if threshold is None else threshold)
        self.margin_threshold = float(cfg.get("margin_threshold", 0.40) if margin_threshold is None else margin_threshold)
        self.prompt = str(cfg.get("prompt", "task: classification | query: "))
        self.option_prompt = str(cfg.get("option_prompt", "title: none | text: "))
        self.max_seq_length = int(cfg.get("max_seq_length", 2048))

        self._tools: dict[str, RegisteredTool] = {}
        self._local_opt_cache: OrderedDict[tuple[str, ...], np.ndarray] = OrderedDict()
        self.max_cache_size: int = 128

    def clear_cache(self) -> None:
        """Clear cached option embeddings."""
        self._local_opt_cache.clear()

    def _encode_texts(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """Encode strings into L2-normalized 768-d vectors via encoder_session."""
        prompted = [f"{prefix}{t}" for t in texts]
        enc = self.tokenizer(
            prompted,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )
        feed = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        return self.encoder_session.run(None, feed)[0]

    def _get_cached_option_vectors(self, opt_texts: list[str]) -> tuple[np.ndarray, bool]:
        """Fetch option vectors from bounded LRU cache or compute via encoder."""
        key = (self.option_prompt, *opt_texts)
        cached = self._local_opt_cache.get(key)
        if cached is not None:
            self._local_opt_cache.move_to_end(key)
            return cached, True
        vecs = self._encode_texts(opt_texts, prefix=self.option_prompt)
        if len(self._local_opt_cache) >= self.max_cache_size:
            self._local_opt_cache.popitem(last=False)
        self._local_opt_cache[key] = vecs
        return vecs, False

    def prewarm_options(self, options: dict[str, str] | list[str] | None = None) -> int:
        """Pre-encode and cache candidate option vectors ahead of live queries."""
        norm = _normalize_options(options, self._tools)
        opt_texts = [f"{k}: {v}" if v and v != k else k for k, v in norm.items()]
        self._get_cached_option_vectors(opt_texts)
        return len(opt_texts)

    def route(
        self,
        query: str,
        options: dict[str, str] | list[str] | None = None,
        threshold: float | None = None,
        margin_threshold: float | None = None,
    ) -> RouteResult:
        """Route query across candidate options using ONNX Runtime."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("`query` must be a non-empty string.")
        norm_opts = _normalize_options(options, self._tools)
        opt_keys = list(norm_opts.keys())
        opt_texts = [f"{k}: {v}" if v and v != k else k for k, v in norm_opts.items()]
        eff_thresh = float(self.threshold if threshold is None else threshold)
        eff_margin = float(self.margin_threshold if margin_threshold is None else margin_threshold)

        t0 = time.perf_counter()
        o_vecs, cache_hit = self._get_cached_option_vectors(opt_texts)
        q_vec = self._encode_texts([f"State: {query.strip()}"], prefix=self.prompt)

        feed = {
            "query_emb": q_vec.astype(np.float32),
            "option_embs": o_vecs[np.newaxis, ...].astype(np.float32),
        }
        t_head0 = time.perf_counter()
        ort_out = self.head_session.run(None, feed)
        head_ms = (time.perf_counter() - t_head0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        probs_np = ort_out[2][0]
        p_null = float(ort_out[3][0])
        return _build_route_result_np(
            opt_keys, probs_np, p_null, eff_thresh, eff_margin, cache_hit, total_ms, head_ms
        )

    def tool(self, description: str, name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a Python function as a candidate tool."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = (name or func.__name__).strip()
            self._tools[tool_name] = RegisteredTool(name=tool_name, description=description.strip(), func=func)
            return func
        return decorator

    def register_mcp_tools(self, tools: list[dict[str, Any]]) -> int:
        """Register a list of MCP tool descriptors."""
        for item in tools:
            t_name = str(item.get("name", "")).strip()
            t_desc = str(item.get("description", t_name)).strip()
            if t_name:
                self._tools[t_name] = RegisteredTool(name=t_name, description=t_desc, func=item.get("func"))
        return len(self._tools)

    def dispatch(
        self,
        query: str,
        *args: Any,
        threshold: float | None = None,
        margin_threshold: float | None = None,
        **kwargs: Any,
    ) -> tuple[RouteResult, Any]:
        """Route and execute a registered tool only if abstain is False."""
        res = self.route(query, options=None, threshold=threshold, margin_threshold=margin_threshold)
        if res.abstain or res.decision is None:
            return res, None
        reg = self._tools.get(res.decision)
        if reg is None or reg.func is None:
            return res, None
        return res, reg.func(*args, **kwargs)


def _resolve_model_files(
    model_dir_or_repo: str | Path,
    quantized: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve encoder, head, and config file paths from local directory or Hugging Face Hub."""
    p = Path(model_dir_or_repo)
    enc_name = "encoder_int8.onnx" if quantized else "encoder.onnx"
    head_name = "decision_head_int8.onnx" if quantized else "decision_head.onnx"

    if p.is_dir():
        enc_p = p / enc_name
        head_p = p / head_name
        cfg_p = p / "router_onnx_config.json"
        cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
        return enc_p, head_p, cfg

    from huggingface_hub import hf_hub_download

    repo_id = str(model_dir_or_repo)
    enc_p = Path(hf_hub_download(repo_id, f"onnx/{enc_name}"))
    head_p = Path(hf_hub_download(repo_id, f"onnx/{head_name}"))
    cfg_p = Path(hf_hub_download(repo_id, "onnx/router_onnx_config.json"))
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    return enc_p, head_p, cfg


def load_onnx(
    model_dir_or_repo: str | Path = DEFAULT_REPO_ID,
    provider: str = "CPUExecutionProvider",
    quantized: bool = False,
    threshold: float | None = None,
    margin_threshold: float | None = None,
    tokenizer: Any = None,
) -> ONNXDecisionRouter:
    """Load an ONNXDecisionRouter from local directory or Hugging Face Hub."""
    enc_path, head_path, cfg = _resolve_model_files(model_dir_or_repo, quantized=quantized)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    enc_sess = ort.InferenceSession(str(enc_path), sess_options=opts, providers=[provider])
    head_sess = ort.InferenceSession(str(head_path), sess_options=opts, providers=[provider])

    if tokenizer is None:
        tok_source = str(model_dir_or_repo) if not Path(model_dir_or_repo).is_dir() else DEFAULT_REPO_ID
        tok = AutoTokenizer.from_pretrained(tok_source)
    else:
        tok = tokenizer

    return ONNXDecisionRouter(
        encoder_session=enc_sess,
        head_session=head_sess,
        tokenizer=tok,
        config_dict=cfg,
        threshold=threshold,
        margin_threshold=margin_threshold,
    )


__all__ = ["ONNXDecisionRouter", "load_onnx"]
