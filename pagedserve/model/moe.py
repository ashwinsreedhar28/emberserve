"""DeepSeek-V3-style Mixture-of-Experts feed-forward (DeepSeek-V2/V3, Moonshot Moonlight).

Routing follows HF `modeling_deepseek_v3` ("noaux_tc" top-k over sigmoid scores):

    scores        = sigmoid(x @ W_gate^T)                      # [N, E], fp32
    choice_scores = scores + e_score_correction_bias           # the bias steers CHOICE only
    group score   = sum of the top-2 choice_scores in each of n_group expert groups
    keep the topk_group best groups, zero the others' choice_scores
    idx           = topk(choice_scores, k)                     # [N, k]
    w             = scores.gather(idx)                         # weights come from the UNBIASED scores
    w             = w / (w.sum(-1) + 1e-20)   if norm_topk_prob
    w             = w * routed_scaling_factor
    out           = sum_k w_k * expert_{idx_k}(x) + shared_experts(x)

Experts are stored stacked: `experts_gate_up [E, 2I, H]` (rows [0, I) gate, [I, 2I) up, the
same fused layout as the dense `gate_up_proj`) and `experts_down [E, H, I]`. The forward is
a per-expert loop over the experts that received tokens (gather rows, one fused SwiGLU MLP,
`index_add_` back); a grouped GEMM is the later optimization once a profile asks for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pagedserve.model import ops


@dataclass(frozen=True)
class MoEConfig:
    hidden_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int = 0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"

    def __post_init__(self) -> None:
        if self.scoring_func != "sigmoid":
            raise ValueError(f"unsupported scoring_func {self.scoring_func!r} (sigmoid only)")
        if self.topk_method != "noaux_tc":
            raise ValueError(f"unsupported topk_method {self.topk_method!r} (noaux_tc only)")
        if self.n_routed_experts % self.n_group:
            raise ValueError("n_routed_experts must be divisible by n_group")
        if not 1 <= self.topk_group <= self.n_group:
            raise ValueError("topk_group must be in [1, n_group]")
        if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("num_experts_per_tok must be in [1, n_routed_experts]")

    @classmethod
    def from_hf(cls, cfg: dict) -> "MoEConfig":
        """From an HF `config.json` dict (DeepSeek-V2/V3 field names)."""
        return cls(
            hidden_size=cfg["hidden_size"],
            moe_intermediate_size=cfg["moe_intermediate_size"],
            n_routed_experts=cfg["n_routed_experts"],
            num_experts_per_tok=cfg["num_experts_per_tok"],
            n_shared_experts=cfg.get("n_shared_experts", 0) or 0,
            n_group=cfg.get("n_group", 1) or 1,
            topk_group=cfg.get("topk_group", 1) or 1,
            norm_topk_prob=cfg.get("norm_topk_prob", True),
            routed_scaling_factor=float(cfg.get("routed_scaling_factor", 1.0)),
            scoring_func=cfg.get("scoring_func", "sigmoid"),
            topk_method=cfg.get("topk_method", "noaux_tc"),
        )


class MoEGate(nn.Module):
    """The router. `weight: [E, H]`, `e_score_correction_bias: [E]` (HF names)."""

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.n_routed_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`x: [N, H]` -> `(topk_idx [N, k] int64, topk_weight [N, k] float32)`."""
        c = self.config
        n = x.shape[0]
        logits = F.linear(x.float(), self.weight.float())  # [N, E]
        scores = torch.sigmoid(logits)
        choice = scores + self.e_score_correction_bias.float()
        if c.n_group > 1:
            grouped = choice.view(n, c.n_group, c.n_routed_experts // c.n_group)
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)  # [N, n_group]
            keep = group_scores.topk(c.topk_group, dim=-1, sorted=False).indices  # [N, topk_group]
            group_mask = torch.zeros_like(group_scores).scatter_(1, keep, 1.0)
            choice = (grouped * group_mask[:, :, None]).reshape(n, c.n_routed_experts)
        idx = choice.topk(c.num_experts_per_tok, dim=-1, sorted=False).indices  # [N, k]
        w = scores.gather(1, idx)
        if c.norm_topk_prob:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        w = w * c.routed_scaling_factor
        return idx, w


class SharedExperts(nn.Module):
    """Dense SwiGLU MLP with intermediate `n_shared * I`; same fused layout as `Qwen2MLP`."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(ops.silu_and_mul(self.gate_up_proj(x)))


class DeepseekMoE(nn.Module):
    """Routed experts (stacked) + optional shared experts."""

    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.config = config
        e, inter, h = config.n_routed_experts, config.moe_intermediate_size, config.hidden_size
        self.gate = MoEGate(config)
        self.experts_gate_up = nn.Parameter(torch.empty(e, 2 * inter, h))
        self.experts_down = nn.Parameter(torch.empty(e, h, inter))
        self.shared_experts: SharedExperts | None = None
        if config.n_shared_experts:
            self.shared_experts = SharedExperts(h, config.n_shared_experts * inter)

    def expert(self, e: int, rows: torch.Tensor) -> torch.Tensor:
        """One expert's SwiGLU MLP on `rows: [n_e, H]`."""
        return F.linear(ops.silu_and_mul(F.linear(rows, self.experts_gate_up[e])),
                        self.experts_down[e])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, h = x.shape
        idx, w = self.gate(x)  # [N, k]
        k = idx.shape[1]
        out = torch.zeros_like(x)
        flat_idx = idx.reshape(-1)  # token t's j-th choice is at position t*k + j
        flat_w = w.reshape(-1)
        order = torch.argsort(flat_idx)  # group the (token, choice) pairs by expert
        sorted_experts = flat_idx[order]
        counts = torch.bincount(sorted_experts, minlength=self.config.n_routed_experts)
        start = 0
        for e, cnt in enumerate(counts.tolist()):
            if cnt == 0:
                continue
            pos = order[start:start + cnt]  # positions into the flat (token, choice) list
            start += cnt
            tokens = pos // k
            y = self.expert(e, x.index_select(0, tokens))
            out.index_add_(0, tokens, y * flat_w[pos].to(y.dtype)[:, None])
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


def moe_forward_reference(module: DeepseekMoE, x: torch.Tensor) -> torch.Tensor:
    """Per-token loop, for tests: the definition the batched forward must match."""
    idx, w = module.gate(x)
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        row = x[t:t + 1]
        acc = torch.zeros_like(row)
        for j in range(idx.shape[1]):
            acc = acc + w[t, j].to(row.dtype) * module.expert(int(idx[t, j]), row)
        out[t] = acc[0]
    if module.shared_experts is not None:
        out = out + module.shared_experts(x)
    return out
