"""Pydantic models for the OpenAI-compatible API surface.

Only the subset emberserve serves is modelled. Unsupported OpenAI fields are rejected by
validators with the same message shape the routes use for 400s; extension fields (`top_k`,
`repetition_penalty`, `ignore_eos`) are plain additions.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from emberserve.sched.request import SamplingParams


# ---- requests ------------------------------------------------------------------------
class SamplingFields(BaseModel):
    """Sampling knobs shared by completions and chat completions."""

    model: str
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=-1)
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    ignore_eos: bool = False

    @field_validator("top_k")
    @classmethod
    def _top_k_disabled_or_positive(cls, v: int) -> int:
        if v != -1 and v < 1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        return v

    @field_validator("n")
    @classmethod
    def _single_choice(cls, v: int) -> int:
        if v != 1:
            raise ValueError("n > 1 is not supported")
        return v

    @property
    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)


class CompletionRequest(SamplingFields):
    prompt: str | list[str] | list[int]
    echo: bool = False

    @field_validator("echo")
    @classmethod
    def _echo_unsupported(cls, v: bool) -> bool:
        if v:
            raise ValueError("echo=true is not supported")
        return v


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(SamplingFields):
    messages: list[ChatMessage] = Field(min_length=1)


def to_sampling_params(req: SamplingFields) -> SamplingParams:
    return SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature,
                          top_k=req.top_k, top_p=req.top_p,
                          repetition_penalty=req.repetition_penalty, stop=req.stop_list,
                          ignore_eos=req.ignore_eos, seed=req.seed)


# ---- responses ----------------------------------------------------------------------
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def of(cls, prompt_tokens: int, completion_tokens: int) -> "Usage":
        return cls(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                   total_tokens=prompt_tokens + completion_tokens)


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None


class ChatDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "emberserve"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: int | str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail

    @classmethod
    def of(cls, message: str, type_: str, code: int | str | None) -> "ErrorResponse":
        return cls(error=ErrorDetail(message=message, type=type_, code=code))


# ---- helpers --------------------------------------------------------------------------
def new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex


def now() -> int:
    return int(time.time())
