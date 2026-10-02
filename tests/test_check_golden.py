"""The golden gate's tie-break rule (scripts/check_golden.py): a mismatch is excused only
when the two disputed tokens are both at the top within the measured noise. It used to
look at the top-2 gap alone, which also excused an unrelated wrong token whenever the top
two happened to be close."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from emberserve.sched.request import SamplingParams
from tests.test_engine import make_engine, prompts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_golden import tie_gap  # noqa: E402


def test_tie_gap_is_about_the_disputed_pair() -> None:
    eng = make_engine()
    ids = prompts(1)[0]
    req = eng.add_request("probe", ids, SamplingParams.greedy(1))
    so = eng.scheduler.schedule()
    with torch.inference_mode():
        logits = eng.step_logits(so)[0].float()
    eng.abort_request(req.request_id)
    eng.backend.free_sequence(req.seq_id)
    order = torch.argsort(logits, descending=True).tolist()
    a, b, far = order[0], order[1], order[-1]
    top2 = float(logits[a] - logits[b])
    assert abs(tie_gap(eng, ids, a, b) - top2) < 1e-5
    assert abs(tie_gap(eng, ids, b, a) - top2) < 1e-5
    # an unrelated token is judged by its own distance from the top, not by the top-2 gap
    assert tie_gap(eng, ids, far, a) > top2 + 1e-3
    assert abs(tie_gap(eng, ids, far, a) - float(logits[a] - logits[far])) < 1e-5
