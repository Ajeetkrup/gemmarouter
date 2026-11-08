"""Set-equivariant decision head with cross-option competition and decoupled coverage gate."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SetEquivariantDecisionHead(nn.Module):
    """Permutation-equivariant 2-layer Transformer decision head with zero option positional encodings."""

    def __init__(self, d_model: int = 768, n_heads: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.query_role = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.option_role = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.pair_proj = nn.Linear(d_model + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.option_scorer = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        nn.init.normal_(self.option_scorer[-1].weight, std=0.01)
        nn.init.zeros_(self.option_scorer[-1].bias)

        self.log_tau = nn.Parameter(torch.tensor(-3.5))
        self.a_null = nn.Parameter(torch.tensor(-100.0))
        self.b_null = nn.Parameter(torch.tensor(74.0))
        self.null_mlp = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.normal_(self.null_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.null_mlp[-1].bias)

    def _compute_null_logits(
        self, raw_sims: Tensor, adapt_sims: Tensor, choice_logits: Tensor
    ) -> Tensor:
        """Compute 6-feature permutation-invariant set statistics and return null logits [B]."""
        k_top = min(2, raw_sims.size(1))
        top2_raw, _ = torch.topk(raw_sims, k=k_top, dim=-1)
        top2_adp, _ = torch.topk(adapt_sims, k=k_top, dim=-1)
        max_sim = top2_raw[:, 0:1]
        raw_margin = top2_raw[:, 0:1] - (top2_raw[:, 1:2] if k_top > 1 else top2_raw[:, 0:1])
        adp_max = top2_adp[:, 0:1]
        adp_margin = top2_adp[:, 0:1] - (top2_adp[:, 1:2] if k_top > 1 else top2_adp[:, 0:1])
        mean_sim = raw_sims.mean(dim=-1, keepdim=True)

        probs = F.softmax(choice_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1, keepdim=True)
        stats = torch.cat(
            [max_sim * 10.0, raw_margin * 10.0, adp_max * 10.0, adp_margin * 10.0, mean_sim * 10.0, entropy],
            dim=-1,
        )
        base_null = self.a_null * max_sim.squeeze(-1) + self.b_null
        return base_null + self.null_mlp(stats).squeeze(-1)

    def forward(self, query_emb: Tensor, option_embs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute per-option choice logits [B, K] and set-level null logits [B]."""
        raw_sims = torch.bmm(option_embs, query_emb.unsqueeze(-1)).squeeze(-1)
        q_adapt = F.normalize(query_emb + 0.2 * self.q_proj(query_emb), p=2, dim=-1)
        o_adapt = F.normalize(option_embs + 0.2 * self.o_proj(option_embs), p=2, dim=-1)
        adapt_sims = torch.bmm(o_adapt, q_adapt.unsqueeze(-1)).squeeze(-1)

        q_exp = q_adapt.unsqueeze(1).expand_as(o_adapt)
        pair_feat = torch.cat([q_exp * o_adapt, adapt_sims.unsqueeze(-1)], dim=-1)
        q_tok = q_adapt.unsqueeze(1) + self.query_role
        o_toks = o_adapt + self.option_role + 0.2 * self.pair_proj(pair_feat)

        hidden = self.transformer(torch.cat([q_tok, o_toks], dim=1))
        q_context = hidden[:, 0:1, :].expand_as(option_embs)
        o_context = hidden[:, 1:, :]

        tau = torch.exp(self.log_tau).clamp(min=1e-3, max=1.0)
        delta_scores = self.option_scorer(torch.cat([o_context, q_context * o_context], dim=-1)).squeeze(-1)
        choice_logits = (0.5 * raw_sims + 0.5 * adapt_sims) / tau + delta_scores
        null_logits = self._compute_null_logits(raw_sims, adapt_sims, choice_logits)
        return choice_logits, null_logits
