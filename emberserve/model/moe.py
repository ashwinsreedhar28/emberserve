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

import torch
import torch.nn.functional as F
from torch import nn

from emberserve.config import MoEConfig
from emberserve.model import ops

__all__ = ["DeepseekMoE", "MoEConfig", "MoEGate", "SharedExperts", "moe_forward_reference"]


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
        logits = F.linear(x.float(), self._weight_f32())  # [N, E]
        if c.n_group == 1:
            from emberserve.model.moe_triton import fused_moe_enabled, topk_gate

            if fused_moe_enabled(x):
                return topk_gate(logits, self.e_score_correction_bias, c.num_experts_per_tok,
                                 c.norm_topk_prob, c.routed_scaling_factor)
        return self.select(logits)

    def _weight_f32(self) -> torch.Tensor:
        """The router runs in fp32 (HF does too); the cast is cached, weights never change."""
        if self.weight.dtype == torch.float32:
            return self.weight
        key = (self.weight.data_ptr(), self.weight._version, str(self.weight.device))
        cached = getattr(self, "_w32", None)
        if cached is None or cached[0] != key:
            cached = (key, self.weight.detach().float())
            self._w32 = cached
        return cached[1]

    def select(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Routing from the fp32 logits in torch ops (the reference for `topk_gate`, and
        the path for group-limited routing, `n_group > 1`)."""
        c = self.config
        n = logits.shape[0]
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

    def forward(self, x: torch.Tensor, add_to: torch.Tensor | None = None) -> torch.Tensor:
        """`add_to` (same shape as the output) is folded into the down GEMM's epilogue."""
        act = ops.silu_and_mul(self.gate_up_proj(x))
        if add_to is None:
            return self.down_proj(act)
        if not isinstance(self.down_proj, nn.Linear):  # quantized: no weight to hand addmm
            return self.down_proj(act) + add_to
        return torch.addmm(add_to, act, self.down_proj.weight.t())


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
        idx, w = self.gate(x)  # [N, k]
        from emberserve.model.moe_triton import fused_moe_enabled, fused_moe_forward

        if fused_moe_enabled(x):
            out = fused_moe_forward(x, idx, w, self.experts_gate_up, self.experts_down)
        else:
            out = self.forward_loop(x, idx, w)
        if self.shared_experts is not None:
            out = self.shared_experts(x, add_to=out)
        return out

    def forward_loop(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Routed experts by a per-expert Python loop (the reference; CPU path)."""
        n, h = x.shape
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
