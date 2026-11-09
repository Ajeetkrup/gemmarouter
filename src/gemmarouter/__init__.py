"""gemmarouter: Calibrated, Set-Equivariant Decision & Tool Router on EmbeddingGemma 2."""

from .configuration import QUESTION_TYPES, GemmaDecisionRouterConfig
from .head import SetEquivariantDecisionHead
from .modeling import (
    GemmaDecisionOutput,
    GemmaDecisionRouterModel,
    GemmaDecisionRouterPreTrainedModel,
    QuestionSpec,
    TypedAnswer,
)
from .gguf_loader import load_gguf
from .onnx_router import ONNXDecisionRouter, load_onnx
from .router import (
    DecisionRouter,
    RegisteredTool,
    RouteResult,
    load,
    select_safe_device_and_dtype,
)

__version__ = "0.1.1"
__all__ = [
    "DecisionRouter",
    "GemmaDecisionOutput",
    "GemmaDecisionRouterConfig",
    "GemmaDecisionRouterModel",
    "GemmaDecisionRouterPreTrainedModel",
    "ONNXDecisionRouter",
    "QUESTION_TYPES",
    "QuestionSpec",
    "RegisteredTool",
    "RouteResult",
    "SetEquivariantDecisionHead",
    "TypedAnswer",
    "load",
    "load_gguf",
    "load_onnx",
    "select_safe_device_and_dtype",
]
