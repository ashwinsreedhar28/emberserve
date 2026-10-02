"""Host lists -> device index tensors without a stream synchronization.

`torch.tensor(values, device="cuda")` stages the list in pageable host memory and copies it
with `non_blocking=False`, and PyTorch implements that copy as `cudaMemcpyAsync` followed by
`cudaStreamSynchronize`: the host waits for every kernel already queued on the stream before
it can continue. On the decode path that never happens (inputs go through one pinned block
in `LLMEngine._materialize`), but the mixed-step attention plan (`paged_flash.MixedPlan`)
and the partial-chunk logits selection used to build their index tensors that way, so a
step with a prompt chunk in it drained the pipeline that async scheduling keeps full.

`index_tensor` copies from pinned memory with `non_blocking=True` instead. PyTorch's caching
host allocator records an event on the stream for a pinned block used by a non-blocking
copy and does not reuse the block until it has passed, so the temporary is safe to drop.

`EMBERSERVE_LEGACY_SYNC_COPIES=1` restores the old synchronizing copies, for measuring the
difference on a GPU (read once at import).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

LEGACY_SYNC_COPIES = os.environ.get("EMBERSERVE_LEGACY_SYNC_COPIES", "").strip() == "1"


def index_tensor(values: Sequence[int] | Sequence[float], dtype: torch.dtype,
                 device: torch.device | str) -> torch.Tensor:
    """`torch.tensor(values, dtype=dtype, device=device)` that does not synchronize a CUDA
    stream (see module docstring). On CPU it is exactly that call."""
    device = torch.device(device)
    if device.type != "cuda" or LEGACY_SYNC_COPIES:
        return torch.tensor(values, dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
