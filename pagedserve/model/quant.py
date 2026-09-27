"""Weight-only int8 quantization: per-output-channel symmetric int8 weights, dequantized
on the fly inside a Triton GEMM.

Why: a decode step at batch 1 is the weight read (7B fp16 on an A100: 15 GB at ~1.5 TB/s,
10 ms), so halving the bytes of every projection halves the floor; the activations stay
fp16/bf16 and the arithmetic is unchanged. Per-channel scales (one fp32 per output row)
keep the rounding error small enough that greedy outputs mostly match the fp16 model; it
is a quality trade, not an exact transform, and `check_golden --quantization int8`
reports how many tokens move.

    W [N, K] fp16  ->  q = round(W / s) in int8, s[n] = max_k |W[n, k]| / 127

The GEMM (`int8_gemm`, Triton) streams int8 weight tiles, converts them to the activation
dtype, multiplies on the tensor cores and applies `s` to the fp32 accumulator, so the
result is `x @ (q * s)^T` computed as `(x @ q^T) * s` (exact rearrangement). Tile shapes
are autotuned per (M bucket, N, K) at load time (`warm_int8_kernels`): the batch-1 step
wants narrow tiles so the weight streams through many programs, a prefill chunk wants the
square tiles a dense GEMM uses; the first version used one fixed shape per regime and a
register transpose of the weight tile, and was 2x behind cuBLAS at M >= 32 on the A100.
The torch reference does the same in plain ops (CPU tests); `PAGEDSERVE_INT8_KERNEL=0`
forces it and `PAGEDSERVE_INT8_AUTOTUNE=0` pins the first config of each regime.
`quantize_model` swaps every `nn.Linear` of the decoder (and the lm_head) for
`Int8Linear`, one at a time so the fp16 copy is freed before the next one is converted.
MoE experts (stacked 3-D weights) stay fp16/bf16 for now.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

def quantize_int8_weight(w: Tensor) -> tuple[Tensor, Tensor]:
    """`w [N, K]` -> (`q [N, K]` int8, `scale [N]` fp32), symmetric per output row."""
    wf = w.detach().float()
    scale = wf.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(wf / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


def int8_gemm_torch(x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None = None) -> Tensor:
    """Reference: `(x @ q^T) * scale (+ bias)` with an fp32 accumulator, in x's dtype."""
    out = torch.matmul(x.float(), q.float().t()) * scale[None, :]
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def kernel_enabled(x: Tensor) -> bool:
    return (os.environ.get("PAGEDSERVE_INT8_KERNEL", "1") != "0"
            and (x.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1"))


_KERNEL = None  # the plain JIT kernel (fixed config; interpreter and A/B)
_REDUCE = None  # the split-K reduce kernel
_TUNED = None  # the autotuned launcher (CUDA)

# Tile configs the autotuner picks from, keyed on the M bucket (rows are decode batch
# sizes or prefill chunks; N and K are one of a handful of projection shapes per model).
# Small M: a narrow tile per program so the int8 weight streams through many programs
# (the batch-1 step is the weight read). Large M: square-ish tiles, deep pipeline.
_CONFIGS_SMALL_M = [  # M <= 16
    dict(BM=16, BN=64, BK=128, num_warps=4, num_stages=3),
    dict(BM=16, BN=128, BK=64, num_warps=4, num_stages=4),
    dict(BM=16, BN=64, BK=256, num_warps=4, num_stages=3),
    dict(BM=16, BN=32, BK=256, num_warps=2, num_stages=4),
]
_CONFIGS_LARGE_M = [
    dict(BM=64, BN=128, BK=64, num_warps=4, num_stages=4),
    dict(BM=64, BN=64, BK=64, num_warps=4, num_stages=4),
    dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=3),
    dict(BM=128, BN=256, BK=64, num_warps=8, num_stages=3),
    dict(BM=64, BN=256, BK=32, num_warps=8, num_stages=4),
    dict(BM=32, BN=128, BK=64, num_warps=4, num_stages=4),
    dict(BM=32, BN=64, BK=64, num_warps=4, num_stages=4),
]
# Split-K: a [M, N] output of a weight-read-bound GEMM has too few tiles to keep the GPU's
# memory system busy (7B down_proj at batch 1: 3584 / 64 = 56 programs on 108 SMs), so
# the K range is cut into pieces that run as separate programs writing fp32 partials, and
# a reduce kernel sums them with the scale and bias. Chosen on the host from the tile
# count (target ~2 programs per SM, at most 8 pieces), only where the workspace is small.
SPLIT_K_MAX_M = 128
_TARGET_PROGRAMS = 216


def _m_bucket(m: int) -> int:
    """Autotune key: next power of two of M (capped), so a new prefill size does not
    re-tune and the decode batch buckets tune once each."""
    import triton

    return min(triton.next_power_of_2(max(m, 1)), 8192)


def split_k(m: int, n: int, k: int, bk: int = 64) -> int:
    """Pieces to cut K into (a power of two, 1..8) for an `[m, n]` output."""
    if m > SPLIT_K_MAX_M:
        return 1
    tiles = max(1, -(-m // (16 if m <= 16 else 64))) * max(1, -(-n // 64))
    k_tiles = max(1, -(-k // bk))
    s = 1
    while s < 8 and tiles * s < _TARGET_PROGRAMS and s * 2 <= k_tiles:
        s *= 2
    return s


def _define_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _int8_gemm_kernel(
        a_ptr, w_ptr, scale_ptr, bias_ptr, out_ptr, ws_ptr, M, N, K,
        stride_am, stride_wn, stride_om,
        M_BUCKET: tl.constexpr, SPLIT_K: tl.constexpr, HAS_BIAS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        m_valid = offs_m < M
        n_valid = offs_n < N
        # this program's share of the K tiles (contiguous range; the last split may be short)
        k_tiles = tl.cdiv(K, BK)
        per = tl.cdiv(k_tiles, SPLIT_K)
        t0 = pid_k * per
        t1 = tl.minimum(t0 + per, k_tiles)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] + t0 * BK  # [BM, BK]
        # the weight is [N, K] row-major; the B operand is read as a [BK, BN] tile straight
        # from that layout (K contiguous down the tile), so no register transpose is needed
        w_ptrs = w_ptr + offs_k[:, None] + offs_n[None, :] * stride_wn + t0 * BK  # [BK, BN]
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for t in range(t0, t1):
            kk = t * BK + offs_k
            k_valid = kk < K
            a = tl.load(a_ptrs, mask=m_valid[:, None] & k_valid[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=k_valid[:, None] & n_valid[None, :], other=0)  # int8
            acc = tl.dot(a, w.to(a.dtype), acc)
            a_ptrs += BK
            w_ptrs += BK
        if SPLIT_K == 1:
            scale = tl.load(scale_ptr + offs_n, mask=n_valid, other=0.0)
            out = acc * scale[None, :]
            if HAS_BIAS:
                bias = tl.load(bias_ptr + offs_n, mask=n_valid, other=0.0).to(tl.float32)
                out = out + bias[None, :]
            tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
                     out.to(out_ptr.dtype.element_ty), mask=m_valid[:, None] & n_valid[None, :])
        else:  # fp32 partial for the reduce kernel: ws[pid_k, m, n]
            tl.store(ws_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :], acc,
                     mask=m_valid[:, None] & n_valid[None, :])

    return _int8_gemm_kernel


def _define_reduce():
    import triton
    import triton.language as tl

    @triton.jit
    def _int8_reduce_kernel(ws_ptr, scale_ptr, bias_ptr, out_ptr, M, N, stride_om,
                            SPLIT_K: tl.constexpr, HAS_BIAS: tl.constexpr,
                            BM: tl.constexpr, BN: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
        idx = offs_m[:, None] * N + offs_n[None, :]
        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for s in range(SPLIT_K):
            acc += tl.load(ws_ptr + s * M * N + idx, mask=mask, other=0.0)
        scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0)
        out = acc * scale[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            out = out + bias[None, :]
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
                 out.to(out_ptr.dtype.element_ty), mask=mask)

    return _int8_reduce_kernel


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _define_kernel()
    return _KERNEL


def _reduce():
    global _REDUCE
    if _REDUCE is None:
        _REDUCE = _define_reduce()
    return _REDUCE


def _tuned():
    """The autotuned launcher: one benchmark per (M bucket, split, N, K), then cached."""
    global _TUNED
    if _TUNED is None:
        import triton

        configs = [triton.Config({"BM": c["BM"], "BN": c["BN"], "BK": c["BK"]},
                                 num_warps=c["num_warps"], num_stages=c["num_stages"])
                   for c in _CONFIGS_SMALL_M + _CONFIGS_LARGE_M]

        def prune(configs, named_args, **kwargs):
            # Triton passes the positional args as `named_args` and the constexpr keyword
            # args (M_BUCKET among them) in `kwargs`; older releases put both in named_args.
            bucket = kwargs.get("M_BUCKET", named_args.get("M_BUCKET", 1))
            small = bucket <= 16
            keep = [c for c in configs if (c.kwargs["BM"] == 16) == small]
            return keep or configs

        kw = dict(configs=configs, key=["M_BUCKET", "SPLIT_K", "N", "K"],
                  prune_configs_by={"early_config_prune": prune})
        try:  # short benchmarks: ~70 keys are tuned at load time (warm_int8_kernels)
            _TUNED = triton.autotune(**kw, warmup=5, rep=20)(_define_kernel())
        except TypeError:  # a Triton without the warmup/rep knobs
            _TUNED = triton.autotune(**kw)(_define_kernel())
    return _TUNED


def autotune_enabled(x: Tensor) -> bool:
    return (x.is_cuda and os.environ.get("PAGEDSERVE_INT8_AUTOTUNE", "1") != "0"
            and os.environ.get("TRITON_INTERPRET", "0") != "1")


def int8_gemm(x: Tensor, q: Tensor, scale: Tensor, bias: Tensor | None = None) -> Tensor:
    """`x [M, K]` (fp16/bf16, unit last stride) @ dequant(`q [N, K]` int8, `scale [N]`)^T."""
    import triton

    m, k = x.shape
    n = q.shape[0]
    assert q.shape[1] == k and x.stride(1) == 1 and q.stride(1) == 1
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return out
    bucket = _m_bucket(m)
    splits = split_k(m, n, k) if os.environ.get("PAGEDSERVE_INT8_SPLITK", "1") != "0" else 1
    ws = (torch.empty((splits, m, n), dtype=torch.float32, device=x.device) if splits > 1
          else out)  # unused at SPLIT_K == 1
    b = bias if bias is not None else scale
    args = (x, q, scale, b, out, ws, m, n, k, x.stride(0), q.stride(0), out.stride(0))
    if autotune_enabled(x):
        grid = lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]), splits)  # noqa: E731
        _tuned()[grid](*args, M_BUCKET=bucket, SPLIT_K=splits, HAS_BIAS=bias is not None)
    else:
        cfg = _CONFIGS_SMALL_M[0] if m <= 16 else _CONFIGS_LARGE_M[0]
        bk = min(cfg["BK"], max(16, triton.next_power_of_2(k)))
        grid = (triton.cdiv(m, cfg["BM"]), triton.cdiv(n, cfg["BN"]), splits)
        _kernel()[grid](*args, M_BUCKET=bucket, SPLIT_K=splits, HAS_BIAS=bias is not None,
                        BM=cfg["BM"], BN=cfg["BN"], BK=bk,
                        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    if splits > 1:
        rbm, rbn = (16 if m <= 16 else 32), 128
        _reduce()[(triton.cdiv(m, rbm), triton.cdiv(n, rbn))](
            ws, scale, b, out, m, n, out.stride(0), SPLIT_K=splits, HAS_BIAS=bias is not None,
            BM=rbm, BN=rbn, num_warps=4)
    return out


def warm_int8_kernels(model: nn.Module, max_m: int = 8192) -> int:
    """Run every `Int8Linear` shape once per M bucket so the autotuner's benchmarks happen
    at load time, not inside the first request (or a CUDA-graph warmup). Returns the number
    of launches."""
    shapes: dict[tuple[int, int, bool, torch.dtype, torch.device], Int8Linear] = {}
    for mod in model.modules():
        if isinstance(mod, Int8Linear):
            key = (mod.in_features, mod.out_features, mod.bias is not None,
                   mod.act_dtype, mod.weight_q.device)
            shapes.setdefault(key, mod)
    count = 0
    with torch.inference_mode():
        for (k, _n, _b, dtype, device), mod in shapes.items():
            if not autotune_enabled(mod.weight_q):
                continue
            m = 1
            while m <= max_m:
                mod(torch.zeros((m, k), dtype=dtype, device=device))
                count += 1
                m *= 2
    return count


class Int8Linear(nn.Module):
    """Drop-in for `nn.Linear` holding int8 weights + fp32 per-row scales (+ the fp16 bias)."""

    def __init__(self, weight_q: Tensor, scale: Tensor, bias: Tensor | None,
                 act_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        self.weight_q = nn.Parameter(weight_q, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.in_features = weight_q.shape[1]
        self.out_features = weight_q.shape[0]
        self.act_dtype = act_dtype

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "Int8Linear":
        q, s = quantize_int8_weight(lin.weight)
        bias = lin.bias.detach().clone() if lin.bias is not None else None
        return cls(q, s, bias, act_dtype=lin.weight.dtype)

    @property
    def weight(self) -> Tensor:
        """The dequantized weight (materialized; for inspection and the tests)."""
        return (self.weight_q.float() * self.scale[:, None]).to(self.act_dtype)

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        if kernel_enabled(x2) and x2.dtype in (torch.float16, torch.bfloat16):
            out = int8_gemm(x2.contiguous(), self.weight_q, self.scale, self.bias)
        else:
            out = int8_gemm_torch(x2, self.weight_q, self.scale, self.bias)
        return out.view(*lead, self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8 per-channel"


DEFAULT_SKIP = ("embed_tokens", "kv_b_proj")  # kv_b_proj: MLA reads its weight directly (2 MB)


def quantize_model(model: nn.Module, method: str = "int8", skip: tuple[str, ...] = DEFAULT_SKIP) -> int:
    """Swap every 2-D `nn.Linear` (decoder projections, lm_head) for `Int8Linear`, in
    place, one module at a time. Returns the number of layers converted. `nn.Embedding`,
    the MoE expert stacks and the names in `skip` are left as they are."""
    if method != "int8":
        raise ValueError(f"unknown quantization {method!r} (supported: int8)")
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(module, child_name, Int8Linear.from_linear(child))
                del child
                count += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        warm_int8_kernels(model)
    return count


def linear(module: nn.Module, x: Tensor) -> Tensor:
    """`F.linear` on an `nn.Linear` or the int8 path on an `Int8Linear` (for code that
    calls into a projection's weight directly)."""
    if isinstance(module, Int8Linear):
        return module(x)
    return F.linear(x, module.weight, module.bias)
