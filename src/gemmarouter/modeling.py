"""GemmaDecisionRouter: Set-Equivariant Typed Decision & Router Model on top of EmbeddingGemma 2."""

from __future__ import annotations

from collections import OrderedDict
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration import QUESTION_TYPES, GemmaDecisionRouterConfig
from .head import SetEquivariantDecisionHead

Kind = Literal["choice", "score", "noul"]


@dataclass(frozen=True)
class QuestionSpec:
    """Normalized specification for a single routing or decision question."""

    qid: str
    kind: Kind
    instructions: str
    options: tuple[str, ...]
    descriptions: tuple[str, ...]

    @property
    def kind_id(self) -> int:
        return QUESTION_TYPES.index(self.kind)


@dataclass(frozen=True)
class TypedAnswer:
    """Structured answer returned by GemmaDecisionRouterModel.decide()."""

    kind: Kind
    label: str
    value: str | int | bool
    confidence: float
    probabilities: dict[str, float]
    expected_level: float | None = None
    null_probability: float | None = None
    abstain: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GemmaDecisionOutput(ModelOutput):
    """Output container for GemmaDecisionRouterModel.forward()."""

    loss: Tensor | None = None
    logits: Tensor | None = None
    null_logits: Tensor | None = None
    probabilities: Tensor | None = None
    text_embeds: Tensor | None = None


def _json(x: Any) -> Any:
    return json.loads(x) if isinstance(x, str) else x


def _is_int(x: Any) -> bool:
    return str(x).lstrip("-").isdigit()


def _parse_score_criteria(crit: Any, probs: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract option keys and descriptions for a `score` question."""
    if isinstance(crit, list):
        return tuple(str(i) for i in range(len(crit))), tuple(str(c) for c in crit)
    if isinstance(crit, dict) and all(_is_int(k) for k in crit):
        keys = sorted(crit, key=int)
        return tuple(str(k) for k in keys), tuple(str(crit[k]) for k in keys)
    if isinstance(crit, dict):
        return tuple(str(i) for i in range(len(crit))), tuple(f"{k}: {v}" for k, v in crit.items())
    opts = tuple(str(k) for k in sorted(probs, key=int))
    return opts, opts


def parse_question(qid: str, question: dict[str, Any], gold: dict[str, Any] | None = None) -> QuestionSpec:
    """Parse a raw question dictionary into a strongly typed QuestionSpec."""
    raw_kind = question.get("type") or (gold or {}).get("type") or "choice"
    kind: Kind = raw_kind if raw_kind in QUESTION_TYPES else "choice"
    crit = question.get("criteria")
    probs = (gold or {}).get("probabilities") or {}

    if kind == "noul":
        if isinstance(crit, dict) and "false" in crit and "true" in crit:
            opts, desc = ("false", "true"), (str(crit["false"]), str(crit["true"]))
        else:
            opts, desc = ("false", "true"), ("False / No: does not hold", "True / Yes: holds")
    elif kind == "score":
        opts, desc = _parse_score_criteria(crit, probs)
    else:
        if isinstance(crit, dict):
            opts, desc = tuple(map(str, crit)), tuple(str(v) for v in crit.values())
        elif isinstance(crit, list):
            opts = desc = tuple(map(str, crit))
        else:
            opts = desc = tuple(map(str, probs))

    if not opts:
        raise ValueError(f"question {qid!r} has no options")
    return QuestionSpec(qid, kind, str(question.get("instructions") or qid), opts, desc)


def render_state(workflow: str, state: Any) -> str:
    """Flatten a string or nested dictionary state into a canonical text string."""
    state = _json(state) if isinstance(state, str) and state.strip()[:1] in "{[" else state
    if not isinstance(state, dict):
        if state == "":
            return f"workflow: {workflow}" if workflow else ""
        return f"workflow: {workflow} | state: {state}" if workflow else str(state)
    lines: list[str] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix} · {k}" if prefix else k, v)
        else:
            lines.append(f"{prefix.replace('_', ' ')}: {value}")

    walk("", state)
    return (f"workflow: {workflow}\n" if workflow else "") + "\n".join(lines)


def _format_option_texts(sp: QuestionSpec) -> list[str]:
    """Format candidate option strings according to the question kind."""
    opt_texts: list[str] = []
    for key, desc in zip(sp.options, sp.descriptions):
        if sp.kind == "noul":
            opt_texts.append(f"Question: {sp.instructions} | Answer: {key} ({desc})")
        elif sp.kind == "score":
            opt_texts.append(f"Question: {sp.instructions} | Score level {key} of {len(sp.options)-1}: {desc}")
        else:
            opt_texts.append(f"{key}: {desc}" if desc != key else str(key))
    return opt_texts


class GemmaDecisionRouterPreTrainedModel(PreTrainedModel):
    """Base PreTrainedModel wrapper for GemmaDecisionRouter."""

    config_class = GemmaDecisionRouterConfig
    base_model_prefix = "gemma_decision_router"
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        pass


class GemmaDecisionRouterModel(GemmaDecisionRouterPreTrainedModel):
    """EmbeddingGemma 2 text encoder + Set-Equivariant Decision Head + Calibrated Null Gate."""

    def __init__(self, config: GemmaDecisionRouterConfig) -> None:
        super().__init__(config)
        self.encoder = AutoModel.from_config(config.encoder_config)
        self.head = SetEquivariantDecisionHead(
            d_model=config.embed_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )
        self._option_emb_cache: OrderedDict[tuple[str, ...], Tensor] = OrderedDict()
        self._max_option_cache_size: int = 128
        self.post_init()

    def clear_option_cache(self) -> None:
        """Clear the in-memory option embedding cache."""
        self._option_emb_cache.clear()

    def get_tokenizer(self, tokenizer: Any = None) -> Any:
        """Return or lazy-load the tokenizer associated with this model."""
        if tokenizer is not None:
            return tokenizer
        if getattr(self, "_tokenizer", None) is None:
            source = (
                getattr(self.config, "_name_or_path", None)
                or getattr(self, "name_or_path", None)
                or "Ajeet-001/embeddinggemma2-decision-router"
            )
            self._tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
        return self._tokenizer

    def embed(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Mean-pool and L2-normalize token representations from the encoder."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled[:, : self.config.embed_dim].float(), p=2, dim=-1)

    @torch.no_grad()
    def encode(
        self, texts: list[str], prefix: str | None = None, batch_size: int = 64, tokenizer: Any = None
    ) -> Tensor:
        """Encode a list of strings with the given prompt prefix into L2-normalized vectors."""
        tok = self.get_tokenizer(tokenizer)
        pref = self.config.prompt if prefix is None else prefix
        prompted = [f"{pref}{t}" for t in texts]
        all_vecs = []
        for i in range(0, len(prompted), batch_size):
            batch = prompted[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            all_vecs.append(self.embed(enc["input_ids"], enc["attention_mask"]))
        return torch.cat(all_vecs, dim=0)

    @torch.no_grad()
    def encode_options_cached(
        self, opt_texts: list[str], tokenizer: Any = None
    ) -> tuple[Tensor, bool]:
        """Encode option strings once and reuse bounded LRU-cached L2-normalized vectors."""
        cache_key = (str(self.config.option_prompt), *opt_texts)
        cached = self._option_emb_cache.get(cache_key)
        if cached is not None and cached.device == self.device:
            self._option_emb_cache.move_to_end(cache_key)
            return cached, True
        encoded = self.encode(opt_texts, prefix=self.config.option_prompt, tokenizer=tokenizer)
        if len(self._option_emb_cache) >= self._max_option_cache_size:
            self._option_emb_cache.popitem(last=False)
        self._option_emb_cache[cache_key] = encoded
        return encoded, False

    def forward(
        self,
        query_emb: Tensor,
        option_embs: Tensor,
        return_dict: bool = True,
    ) -> GemmaDecisionOutput | tuple[Tensor, Tensor, Tensor]:
        """Run the SetEquivariantDecisionHead on pre-encoded query and option embeddings."""
        head_dtype = next(self.head.parameters()).dtype
        q_in = query_emb.to(head_dtype)
        o_in = option_embs.to(head_dtype)
        sims = torch.bmm(o_in.float(), q_in.float().unsqueeze(-1)).squeeze(-1)
        head_logits, null_logits = self.head(q_in, o_in)
        logits = (sims / self.config.decision_temperature) + 0.15 * head_logits.float()
        probs = F.softmax(logits, dim=-1)
        if not return_dict:
            return (logits, null_logits, probs)
        return GemmaDecisionOutput(logits=logits, null_logits=null_logits.float(), probabilities=probs)

    def _score_single_question(
        self,
        q_vec: Tensor,
        sp: QuestionSpec,
        effective_thresh: float,
        tokenizer: Any = None,
    ) -> TypedAnswer:
        """Score a single QuestionSpec using cached option embeddings and the decision head."""
        opt_texts = _format_option_texts(sp)
        o_vecs, _ = self.encode_options_cached(opt_texts, tokenizer=tokenizer)
        out = self.forward(q_vec, o_vecs.unsqueeze(0))
        if sp.kind == "choice":
            head_dtype = next(self.head.parameters()).dtype
            head_logits, _ = self.head(q_vec.to(head_dtype), o_vecs.unsqueeze(0).to(head_dtype))
            p = F.softmax(head_logits.float(), dim=-1)[0].cpu()
        else:
            p = out.probabilities[0].cpu()
        j = int(p.argmax())
        label = sp.options[j]
        value: str | int | bool = (
            (label == "true") if sp.kind == "noul" else int(label) if sp.kind == "score" else label
        )
        expected = (
            float(sum(int(o) * float(pi) for o, pi in zip(sp.options, p))) if sp.kind == "score" else None
        )
        null_prob = float(torch.sigmoid(out.null_logits[0]).cpu()) if sp.kind == "choice" else None
        abstain = (null_prob >= effective_thresh) if null_prob is not None else False
        return TypedAnswer(
            kind=sp.kind,
            label=label,
            value=value,
            confidence=float(p[j]),
            probabilities={k: round(float(pi), 4) for k, pi in zip(sp.options, p.tolist())},
            expected_level=round(expected, 4) if expected is not None else None,
            null_probability=round(null_prob, 4) if null_prob is not None else None,
            abstain=abstain,
        )

    @torch.no_grad()
    def decide(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        workflow: str = "",
        tokenizer: Any = None,
        abstain_threshold: float | None = None,
        include_question: bool | None = None,
    ) -> dict[str, TypedAnswer]:
        """Answer every question in a schema (`choice`, `score`, `noul`) with automatic option caching."""
        was_training = self.training
        self.eval()
        state_str = render_state(workflow, state)
        q_dict = _json(questions)
        specs = [parse_question(qid, q) for qid, q in q_dict.items()]
        plain_text = isinstance(state, str) and state.strip() != "" and state.strip()[0] not in "{["

        q_texts: list[str] = []
        for sp in specs:
            drop_q = include_question is False or (include_question is None and plain_text and sp.kind == "choice")
            if drop_q and state_str:
                q_texts.append(f"State: {state.strip() if plain_text else state_str}")
            else:
                q_texts.append(f"State: {state_str}\nQuestion: {sp.instructions}" if state_str else f"Question: {sp.instructions}")

        q_vecs = self.encode(q_texts, prefix=self.config.prompt, tokenizer=tokenizer)
        effective_thresh = float(
            getattr(self.config, "null_threshold", 0.58) if abstain_threshold is None else abstain_threshold
        )
        answers = {
            sp.qid: self._score_single_question(q_vecs[i : i + 1], sp, effective_thresh, tokenizer=tokenizer)
            for i, sp in enumerate(specs)
        }
        self.train(was_training)
        return answers


__all__ = [
    "GemmaDecisionOutput",
    "GemmaDecisionRouterModel",
    "GemmaDecisionRouterPreTrainedModel",
    "QuestionSpec",
    "TypedAnswer",
]
