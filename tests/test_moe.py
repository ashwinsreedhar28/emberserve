"""DeepSeek-V3-style MoE (model/moe.py): the noaux_tc router against an independent
reimplementation, the batched per-expert forward against a per-token reference, and the
stacked-expert weight mapping round-tripped through an HF-named checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from pagedserve.config import ModelConfig
from pagedserve.model.moe import DeepseekMoE, MoEConfig, MoEGate, moe_forward_reference
from pagedserve.model.weights import hf_state_dict, hf_to_local, load_hf_weights

torch.set_num_threads(2)
H, I_MOE = 32, 16


def cfg(**kw) -> MoEConfig:
    base = dict(hidden_size=H, moe_intermediate_size=I_MOE, n_routed_experts=8,
                num_experts_per_tok=2, n_shared_experts=1, n_group=1, topk_group=1,
                norm_topk_prob=True, routed_scaling_factor=2.5)
    base.update(kw)
    return MoEConfig(**base)


def seed_module(m: nn.Module, seed: int) -> nn.Module:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("e_score_correction_bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return m.eval()


def route_reference(gate: MoEGate, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight transcription of HF modeling_deepseek_v3's noaux_tc routing, per token."""
    c = gate.config
    idx_out, w_out = [], []
    for t in range(x.shape[0]):
        scores = torch.sigmoid(x[t].float() @ gate.weight.float().T)
        choice = scores + gate.e_score_correction_bias.float()
        group_size = c.n_routed_experts // c.n_group
        if c.n_group > 1:
            gs = [choice[g * group_size:(g + 1) * group_size].topk(2).values.sum()
                  for g in range(c.n_group)]
            keep = set(torch.tensor(gs).topk(c.topk_group).indices.tolist())
            mask = torch.zeros_like(choice)
            for g in keep:
                mask[g * group_size:(g + 1) * group_size] = 1.0
            choice = choice * mask
        idx = choice.topk(c.num_experts_per_tok).indices
        w = scores[idx]
        if c.norm_topk_prob:
            w = w / (w.sum() + 1e-20)
        w = w * c.routed_scaling_factor
        idx_out.append(idx)
        w_out.append(w)
    return torch.stack(idx_out), torch.stack(w_out)


def _sorted_pairs(idx: torch.Tensor, w: torch.Tensor) -> list[list[tuple[int, float]]]:
    return [sorted(zip(i.tolist(), v.tolist())) for i, v in zip(idx, w)]


@pytest.mark.parametrize("groups", [(1, 1), (2, 1), (4, 2)])
def test_router_matches_reference(groups):
    n_group, topk_group = groups
    gate = seed_module(MoEGate(cfg(n_group=n_group, topk_group=topk_group)), 1)
    x = torch.randn(50, H, generator=torch.Generator().manual_seed(2))
    idx, w = gate(x)
    ridx, rw = route_reference(gate, x)
    got, want = _sorted_pairs(idx, w), _sorted_pairs(ridx, rw)
    for a, b in zip(got, want):
        assert [e for e, _ in a] == [e for e, _ in b]
        torch.testing.assert_close(torch.tensor([v for _, v in a]), torch.tensor([v for _, v in b]))
    # weights are unbiased sigmoid scores, normalized, scaled
    torch.testing.assert_close(w.sum(-1), torch.full((50,), 2.5))
    assert w.dtype == torch.float32 and idx.dtype == torch.int64


def test_router_bias_changes_choice_but_not_weights():
    gate = seed_module(MoEGate(cfg(n_group=1)), 3)
    x = torch.randn(1, H, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        gate.e_score_correction_bias.zero_()
        idx0, _ = gate(x)
        # push a currently-unchosen expert to the top by bias alone
        unchosen = next(e for e in range(8) if e not in idx0[0].tolist())
        gate.e_score_correction_bias[unchosen] = 10.0
        idx1, w1 = gate(x)
    assert unchosen in idx1[0].tolist()
    scores = torch.sigmoid(x.float() @ gate.weight.float().T)[0]
    raw = scores[idx1[0]]
    torch.testing.assert_close(w1[0], raw / (raw.sum() + 1e-20) * 2.5)


@pytest.mark.parametrize("n", [1, 7, 64])
@pytest.mark.parametrize("shared", [0, 1, 2])
def test_forward_matches_per_token_reference(n, shared):
    moe = seed_module(DeepseekMoE(cfg(n_shared_experts=shared)), 5 + n)
    x = torch.randn(n, H, generator=torch.Generator().manual_seed(n))
    torch.testing.assert_close(moe(x), moe_forward_reference(moe, x), atol=1e-5, rtol=1e-5)


def test_some_experts_receive_no_tokens():
    moe = seed_module(DeepseekMoE(cfg(n_routed_experts=16, num_experts_per_tok=1, n_shared_experts=0)), 9)
    x = torch.randn(3, H, generator=torch.Generator().manual_seed(9))
    idx, _ = moe.gate(x)
    assert len(set(idx.reshape(-1).tolist())) < 16
    torch.testing.assert_close(moe(x), moe_forward_reference(moe, x), atol=1e-5, rtol=1e-5)


def test_config_validation():
    with pytest.raises(ValueError):
        cfg(scoring_func="softmax")
    with pytest.raises(ValueError):
        cfg(n_group=3)  # 8 experts not divisible by 3
    with pytest.raises(ValueError):
        cfg(n_group=2, topk_group=3)
    c = MoEConfig.from_hf({"hidden_size": 8, "moe_intermediate_size": 4, "n_routed_experts": 4,
                           "num_experts_per_tok": 2, "n_shared_experts": None, "n_group": None,
                           "topk_group": None, "routed_scaling_factor": 1.5})
    assert c.n_shared_experts == 0 and c.n_group == 1 and c.routed_scaling_factor == 1.5


# ---- weight mapping ------------------------------------------------------------------------

class _Layer(nn.Module):
    def __init__(self, c: MoEConfig) -> None:
        super().__init__()
        self.mlp = DeepseekMoE(c)


class _Tree(nn.Module):
    """Mimics `model.layers.0.mlp` so the loader's name mapping is exercised end to end."""

    def __init__(self, c: MoEConfig) -> None:
        super().__init__()
        self.config = ModelConfig.tiny(hidden_size=c.hidden_size, tie_word_embeddings=False)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(c)])


def test_expert_name_mapping():
    assert hf_to_local("model.layers.3.mlp.experts.5.up_proj.weight") == \
        ("model.layers.3.mlp.experts_gate_up", ("expert", 5, "up"))
    assert hf_to_local("model.layers.3.mlp.experts.12.down_proj.weight") == \
        ("model.layers.3.mlp.experts_down", ("expert", 12, "down"))
    assert hf_to_local("model.layers.3.mlp.shared_experts.gate_proj.weight") == \
        ("model.layers.3.mlp.shared_experts.gate_up_proj.weight", "gate")
    assert hf_to_local("model.layers.3.mlp.gate.weight") == ("model.layers.3.mlp.gate.weight", None)
    assert hf_to_local("model.layers.3.mlp.gate.e_score_correction_bias") == \
        ("model.layers.3.mlp.gate.e_score_correction_bias", None)


def test_checkpoint_round_trip(tmp_path: Path):
    c = cfg(n_shared_experts=2)
    ref = seed_module(_Tree(c), 11)
    sd = hf_state_dict(ref)
    names = set(sd)
    assert "model.layers.0.mlp.experts.7.gate_proj.weight" in names
    assert "model.layers.0.mlp.experts.7.down_proj.weight" in names
    assert "model.layers.0.mlp.shared_experts.up_proj.weight" in names
    assert not any("experts_gate_up" in k or "experts_down" in k or "gate_up_proj" in k for k in names)
    assert sd["model.layers.0.mlp.experts.7.gate_proj.weight"].shape == (I_MOE, H)
    assert sd["model.layers.0.mlp.shared_experts.up_proj.weight"].shape == (2 * I_MOE, H)
    (tmp_path / "config.json").write_text(json.dumps({}))
    save_file({k: v.detach().clone().contiguous() for k, v in sd.items()},
              str(tmp_path / "model.safetensors"))
    loaded = _Tree(c)
    load_hf_weights(loaded, tmp_path)
    for (n1, p1), (n2, p2) in zip(ref.state_dict().items(), loaded.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    x = torch.randn(5, H, generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(loaded.model.layers[0].mlp(x), ref.model.layers[0].mlp(x))


def test_missing_expert_shard_is_named(tmp_path: Path):
    c = cfg()
    ref = seed_module(_Tree(c), 13)
    sd = {k: v.detach().clone().contiguous() for k, v in hf_state_dict(ref).items()}
    sd.pop("model.layers.0.mlp.experts.3.up_proj.weight")
    save_file(sd, str(tmp_path / "model.safetensors"))
    with pytest.raises(KeyError, match=r"model\.layers\.0\.mlp\.experts\.3\.up_proj\.weight"):
        load_hf_weights(_Tree(c), tmp_path)
