"""Offline batch API: `LLM(model_dir).generate(prompts, params)`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pagedserve.config import EngineConfig
from pagedserve.engine import LLMEngine
from pagedserve.sched.request import FinishReason, RequestOutput, SamplingParams


@dataclass
class GenerationResult:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    text: str
    finish_reason: FinishReason | None
    metrics: dict = field(default_factory=dict)


class LLM:
    def __init__(self, model_dir: str | Path, engine_config: EngineConfig | None = None,
                 **overrides) -> None:
        self.engine = LLMEngine.from_pretrained(model_dir, engine_config, **overrides)

    @classmethod
    def from_engine(cls, engine: LLMEngine) -> "LLM":
        obj = cls.__new__(cls)
        obj.engine = engine
        return obj

    def generate(self, prompts: list[str] | list[list[int]],
                 sampling_params: SamplingParams | list[SamplingParams] | None = None,
                 ) -> list[GenerationResult]:
        params = sampling_params or SamplingParams()
        if not isinstance(params, list):
            params = [params] * len(prompts)
        assert len(params) == len(prompts)
        reqs = [self.engine.add_request(f"req-{i}", p, sp) for i, (p, sp) in
                enumerate(zip(prompts, params, strict=True))]
        text: dict[str, list[str]] = {r.request_id: [] for r in reqs}
        final: dict[str, RequestOutput] = {}
        while self.engine.has_unfinished_requests():
            for out in self.engine.step():
                text[out.request_id].append(out.text_delta)
                if out.finished:
                    final[out.request_id] = out
        results = []
        for r in reqs:
            out = final[r.request_id]
            results.append(GenerationResult(
                request_id=r.request_id, prompt_token_ids=r.prompt_token_ids,
                output_token_ids=out.output_token_ids, text="".join(text[r.request_id]),
                finish_reason=out.finish_reason, metrics=out.metrics))
        return results

    def chat(self, conversations: list[list[dict]], sampling_params=None) -> list[GenerationResult]:
        tok = self.engine.tokenizer
        if tok is None:
            raise ValueError("chat() needs a tokenizer")
        ids = [tok.apply_chat_template(msgs) for msgs in conversations]
        return self.generate(ids, sampling_params)
