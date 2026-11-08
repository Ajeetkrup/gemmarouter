"""High-level DecisionRouter SDK with built-in option caching, float16 NaN guardrails, and @router.tool dispatch."""

from __future__ import annotations

from collections import OrderedDict
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .modeling import GemmaDecisionRouterModel

DEFAULT_REPO_ID = "Ajeet-001/embeddinggemma2-decision-router"


@dataclass(frozen=True)
class RouteResult:
    """Safe-by-default routing result where `decision` is None whenever `abstain` is True."""

    decision: str | None
    top_option: str
    confidence: float
    null_probability: float
    abstain: bool
    threshold: float
    probabilities: dict[str, float]
    option_cache_hit: bool
    latency_ms: float
    head_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegisteredTool:
    """Metadata and callable for a tool registered via `@router.tool` or `register_mcp_tools`."""

    name: str
    description: str
    func: Callable[..., Any] | None = None


def select_safe_device_and_dtype(
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> tuple[torch.device, torch.dtype]:
    """Select safe PyTorch device and dtype, explicitly blocking float16 and invalid dtypes."""
    if dtype in (torch.float16, "float16", "fp16", "half"):
        raise ValueError(
            "float16 is forbidden for EmbeddingGemma 2 because it produces NaN activations. "
            "Use `torch.bfloat16` (on supported GPUs) or `torch.float32`."
        )
    valid_bf16 = (torch.bfloat16, "bfloat16", "bf16")
    valid_fp32 = (torch.float32, "float32", "fp32", "float")
    if dtype is not None and dtype not in (*valid_bf16, *valid_fp32):
        raise ValueError(f"Unsupported dtype {dtype!r}. Allowed: bfloat16 or float32.")

    resolved_dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    can_use_bf16 = resolved_dev.type == "cuda" and torch.cuda.is_bf16_supported()
    if dtype in valid_fp32:
        return resolved_dev, torch.float32
    if dtype in valid_bf16:
        return resolved_dev, (torch.bfloat16 if can_use_bf16 else torch.float32)
    return resolved_dev, (torch.bfloat16 if can_use_bf16 else torch.float32)


def _normalize_options(
    options: dict[str, str] | list[str] | tuple[str, ...] | None,
    registered_tools: dict[str, RegisteredTool],
) -> dict[str, str]:
    """Convert a dict, list, or registered tool registry into a canonical {key: description} dict."""
    if options is None:
        if not registered_tools:
            raise ValueError("Provide `options` or register tools with `@router.tool(...)` first.")
        return {name: t.description for name, t in registered_tools.items()}
    if isinstance(options, dict):
        cleaned = {str(k).strip(): str(v).strip() for k, v in options.items() if str(k).strip()}
    elif isinstance(options, (list, tuple)):
        cleaned = {str(x).strip(): str(x).strip() for x in options if str(x).strip()}
    else:
        raise TypeError("`options` must be a dict[str, str], list[str], or None.")
    if len(cleaned) < 2:
        raise ValueError("Provide at least 2 candidate options for routing.")
    return cleaned


class DecisionRouter:
    """Ergonomic, cached decision and tool router wrapping GemmaDecisionRouterModel."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        threshold: float | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        cfg_thresh = getattr(getattr(model, "config", None), "null_threshold", 0.58)
        self.threshold = float(cfg_thresh if threshold is None else threshold)
        self._tools: dict[str, RegisteredTool] = {}
        self._local_opt_cache: OrderedDict[tuple[str, ...], torch.Tensor] = OrderedDict()
        self.max_cache_size: int = 128

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def clear_cache(self) -> None:
        """Clear cached option embeddings."""
        self._local_opt_cache.clear()
        if hasattr(self.model, "clear_option_cache"):
            self.model.clear_option_cache()

    def _get_cached_option_vectors(self, opt_texts: list[str]) -> tuple[torch.Tensor, bool]:
        """Fetch L2-normalized option embeddings from bounded LRU cache or encode once."""
        key = (str(self.model.config.option_prompt), *opt_texts)
        cached = self._local_opt_cache.get(key)
        if cached is not None and cached.device == self.device:
            self._local_opt_cache.move_to_end(key)
            return cached, True
        vecs = self.model.encode(opt_texts, prefix=self.model.config.option_prompt, tokenizer=self.tokenizer)
        if len(self._local_opt_cache) >= self.max_cache_size:
            self._local_opt_cache.popitem(last=False)
        self._local_opt_cache[key] = vecs
        return vecs, False

    def prewarm_options(self, options: dict[str, str] | list[str] | None = None) -> int:
        """Pre-encode and cache an option dictionary or list ahead of live requests."""
        norm = _normalize_options(options, self._tools)
        opt_texts = [f"{k}: {v}" if v and v != k else k for k, v in norm.items()]
        self._get_cached_option_vectors(opt_texts)
        return len(opt_texts)

    @torch.no_grad()
    def route(
        self,
        query: str,
        options: dict[str, str] | list[str] | None = None,
        threshold: float | None = None,
    ) -> RouteResult:
        """Route `query` across `options` (or registered tools) with safe abstention and option caching."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("`query` must be a non-empty string.")
        norm_opts = _normalize_options(options, self._tools)
        opt_keys = list(norm_opts.keys())
        opt_texts = [f"{k}: {v}" if v and v != k else k for k, v in norm_opts.items()]
        eff_thresh = float(self.threshold if threshold is None else threshold)

        t0 = time.perf_counter()
        o_vecs, cache_hit = self._get_cached_option_vectors(opt_texts)
        q_vec = self.model.encode([f"State: {query.strip()}"], prefix=self.model.config.prompt, tokenizer=self.tokenizer)

        head_dtype = next(self.model.head.parameters()).dtype
        t_head0 = time.perf_counter()
        head_logits, null_logits = self.model.head(q_vec.to(head_dtype), o_vecs.unsqueeze(0).to(head_dtype))
        head_ms = (time.perf_counter() - t_head0) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        probs_t = F.softmax(head_logits.float(), dim=-1)[0].cpu()
        probs_list = probs_t.tolist()
        top_idx = int(probs_t.argmax().item())
        top_key = opt_keys[top_idx]
        p_null = float(torch.sigmoid(null_logits.float()[0]).cpu().item())
        abstain = p_null >= eff_thresh

        return RouteResult(
            decision=None if abstain else top_key,
            top_option=top_key,
            confidence=round(float(probs_list[top_idx]), 4),
            null_probability=round(p_null, 6),
            abstain=abstain,
            threshold=eff_thresh,
            probabilities={k: round(float(p), 4) for k, p in zip(opt_keys, probs_list)},
            option_cache_hit=cache_hit,
            latency_ms=round(total_ms, 2),
            head_latency_ms=round(head_ms, 3),
        )

    def tool(self, description: str, name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a Python function as a routable candidate tool."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = (name or func.__name__).strip()
            self._tools[tool_name] = RegisteredTool(name=tool_name, description=description.strip(), func=func)
            return func
        return decorator

    def register_mcp_tools(self, tools: list[dict[str, Any]]) -> int:
        """Register a list of MCP tool descriptors (`[{'name': ..., 'description': ...}]`)."""
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
        **kwargs: Any,
    ) -> tuple[RouteResult, Any]:
        """Route `query` to a registered `@router.tool` and execute it only if `abstain` is False."""
        res = self.route(query, options=None, threshold=threshold)
        if res.abstain or res.decision is None:
            return res, None
        reg = self._tools.get(res.decision)
        if reg is None or reg.func is None:
            return res, None
        return res, reg.func(*args, **kwargs)


def load(
    model_id: str = DEFAULT_REPO_ID,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    threshold: float | None = None,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> DecisionRouter:
    """Load `GemmaDecisionRouterModel` locally with `trust_remote_code=False` by default and safe dtype."""
    resolved_dev, resolved_dt = select_safe_device_and_dtype(device=device, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    mod = (
        GemmaDecisionRouterModel.from_pretrained(
            model_id,
            dtype=resolved_dt,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        .to(resolved_dev)
        .eval()
    )
    return DecisionRouter(model=mod, tokenizer=tok, threshold=threshold)
