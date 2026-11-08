"""Configuration for GemmaDecisionRouterModel: Set-Equivariant Decision Head + Null Gate on EmbeddingGemma 2."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

QUESTION_TYPES = ("choice", "score", "noul")


class GemmaDecisionRouterConfig(PretrainedConfig):
    """Configuration class for GemmaDecisionRouterModel."""

    model_type = "gemma_decision_router"
    sub_configs = {"encoder_config": AutoConfig}

    def __init__(
        self,
        encoder_config: dict[str, Any] | PretrainedConfig | None = None,
        embed_dim: int = 768,
        n_heads: int = 8,
        n_layers: int = 2,
        decision_temperature: float = 0.05,
        null_threshold: float = 0.58,
        margin_threshold: float = 0.40,
        max_seq_length: int = 2048,
        prompt: str = "task: classification | query: ",
        option_prompt: str = "title: none | text: ",
        question_types: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(encoder_config, dict):
            model_type = encoder_config.get("model_type")
            if model_type is not None and model_type in CONFIG_MAPPING:
                encoder_config = CONFIG_MAPPING[model_type].from_dict(encoder_config)
        self.encoder_config = encoder_config
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.decision_temperature = decision_temperature
        self.null_threshold = null_threshold
        self.margin_threshold = margin_threshold
        self.max_seq_length = max_seq_length
        self.prompt = prompt
        self.option_prompt = option_prompt
        self.question_types = list(question_types or QUESTION_TYPES)
        super().__init__(**kwargs)


__all__ = ["GemmaDecisionRouterConfig", "QUESTION_TYPES"]
