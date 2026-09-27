"""Tokenizer wrapper with incremental detokenization and stop-string handling.

`transformers` is an optional dependency: the engine works on raw token ids without it.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        # tokenize=False then encode: transformers 5.x returns an Encoding from tokenize=True.
        text = self._tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=add_generation_prompt)
        return self._tok.encode(text, add_special_tokens=False)

    @property
    def raw(self):
        return self._tok


@dataclass
class DetokenizerState:
    """Per-request state: text emitted so far, all text decoded so far (before stop-string
    holdback), and the two-offset window the incremental decode works from."""

    text: str = ""
    raw: str = ""
    prefix_offset: int = 0
    read_offset: int = 0


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

    Decodes a sliding window of a few tokens per call (the vLLM / TGI two-offset scheme:
    `prefix_offset..read_offset` is the already-emitted context, `read_offset..` the new
    ids, decoded together so byte-level merges across the boundary come out right), so a
    step costs O(1) tokens per request instead of O(output length). A decode that ends in
    U+FFFD means a multi-byte character is still incomplete; the window then stays put and
    nothing is emitted until it completes. Stop strings are checked on the accumulated
    text; text that could still turn into a stop string (a trailing prefix of one) is held
    back so a later match never has to retract emitted text.
    """

    def __init__(self, tokenizer: Tokenizer | None) -> None:
        self.tokenizer = tokenizer
        self._states: dict[str, DetokenizerState] = {}

    def reset(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def update(self, request_id: str, output_ids: list[int], stop: list[str],
               skip_special_tokens: bool = True, final: bool = False) -> tuple[str, str | None]:
        """Returns (text_delta, matched_stop_string_or_None). `final=True` flushes held text."""
        if self.tokenizer is None:
            return "", None
        st = self._states.setdefault(request_id, DetokenizerState())
        decode = self.tokenizer.decode
        if st.read_offset < len(output_ids):
            prefix_text = (decode(output_ids[st.prefix_offset:st.read_offset],
                                  skip_special_tokens=skip_special_tokens)
                           if st.read_offset > st.prefix_offset else "")
            new_text = decode(output_ids[st.prefix_offset:], skip_special_tokens=skip_special_tokens)
            incomplete = new_text.endswith("\ufffd")
            if incomplete and final:
                new_text = new_text[:-1]
                incomplete = False
            if len(new_text) > len(prefix_text) and not incomplete:
                st.raw += new_text[len(prefix_text):]
                st.prefix_offset = st.read_offset
                st.read_offset = len(output_ids)
        full = st.raw
        matched = None
        if stop:
            for s in stop:
                idx = full.find(s, max(0, len(st.text) - len(s) + 1))
                if idx != -1:
                    full = full[:idx]
                    matched = s
                    break
            if matched is None and not final:
                full = full[:len(full) - _stop_prefix_len(full, stop)]
        if not full.startswith(st.text):
            delta = full  # should not happen: emitted text is always a prefix of `full`
        else:
            delta = full[len(st.text):]
        st.text = full
        return delta, matched

    def text(self, request_id: str) -> str:
        st = self._states.get(request_id)
        return st.text if st else ""
