"""Tokenizer wrapper with incremental detokenization and stop-string handling.

`transformers` is an optional dependency: the engine works on raw token ids without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class Tokenizer:
    """Thin wrapper over an HF fast tokenizer loaded from the model directory."""

    def __init__(self, model_dir: str | Path) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'pagedserve[hf]' to load a tokenizer") from e
        self._tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.eos_token_id: int = self._tok.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
        return self._tok.apply_chat_template(messages, tokenize=True,
                                             add_generation_prompt=add_generation_prompt)

    @property
    def raw(self):
        return self._tok


@dataclass
class DetokenizerState:
    """Per-request state: text emitted so far, how many ids it covers."""

    text: str = ""
    consumed_ids: int = 0
    _pending: list[int] = field(default_factory=list)


def _stop_prefix_len(text: str, stop: list[str]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of a stop string."""
    best = 0
    for s in stop:
        for n in range(min(len(s) - 1, len(text)), best, -1):
            if s.startswith(text[-n:]):
                best = n
                break
    return best


class IncrementalDetokenizer:
    """Turns a growing list of output token ids into text deltas.

    Decodes the whole output each call (O(n) per step, fine for the lengths we serve) and
    holds back a trailing U+FFFD, which means a multi-byte character is still incomplete.
    Also checks stop strings on the accumulated text and reports where to truncate.
    """

    def __init__(self, tokenizer: Tokenizer | None) -> None:
        self.tokenizer = tokenizer
        self._states: dict[str, DetokenizerState] = {}

    def reset(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def update(self, request_id: str, output_ids: list[int], stop: list[str],
               skip_special_tokens: bool = True, final: bool = False) -> tuple[str, str | None]:
        """Returns (text_delta, matched_stop_string_or_None).

        Text that could still turn into a stop string (a trailing prefix of one) is held
        back so a later match never has to retract emitted text; `final=True` flushes it.
        """
        if self.tokenizer is None:
            return "", None
        st = self._states.setdefault(request_id, DetokenizerState())
        full = self.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)
        if full.endswith("�"):
            full = full[:-1]
        matched = None
        for s in stop:
            idx = full.find(s)
            if idx != -1:
                full = full[:idx]
                matched = s
                break
        if matched is None and not final:
            full = full[:len(full) - _stop_prefix_len(full, stop)]
        if not full.startswith(st.text):
            # Tokenizer merged differently than before (rare, e.g. byte fallback); resync.
            delta = full
        else:
            delta = full[len(st.text):]
        st.text = full
        st.consumed_ids = len(output_ids)
        return delta, matched

    def text(self, request_id: str) -> str:
        st = self._states.get(request_id)
        return st.text if st else ""
