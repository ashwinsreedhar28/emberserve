"""Tokenizer wrapper with incremental detokenization and stop-string handling.

`transformers` is an optional dependency: the engine works on raw token ids without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def has_tokenizer(model_dir: str | Path) -> bool:
    """Whether a snapshot carries a tokenizer: a fast `tokenizer.json`, or a
    `tokenizer_config.json` (sentencepiece / tiktoken tokenizers, possibly with their own
    code alongside)."""
    d = Path(model_dir)
    return (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists()


class Tokenizer:
    """Thin wrapper over an HF tokenizer loaded from the model directory."""

    def __init__(self, model_dir: str | Path) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'pagedserve[hf]' to load a tokenizer") from e
        try:
            self._tok = AutoTokenizer.from_pretrained(str(model_dir))
        except Exception:  # noqa: BLE001
            # A snapshot that ships its own tokenizer code (Moonlight's tiktoken-based
            # `tokenization_moonshot.py`): the code is already on disk, downloaded with the
            # weights, so run it rather than refuse text prompts.
            if not any(Path(model_dir).glob("tokenization_*.py")):
                raise
            self._tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.eos_token_id: int = self._tok.eos_token_id
        self._backend = getattr(self._tok, "backend_tokenizer", None)
        self._fast_batch = (self._backend is not None
                            and not getattr(self._tok, "clean_up_tokenization_spaces", False))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """HF's default: Llama/Mistral tokenizers prepend BOS, Qwen2's adds nothing. Chat
        templates render their own BOS text and are encoded with `add_special_tokens=False`."""
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, batch: list[list[int]], skip_special_tokens: bool = False) -> list[str]:
        """Many decodes in one call. Goes straight to the Rust `tokenizers` backend (one
        GIL release, no per-call Python) when that is exactly what `decode` would do, i.e.
        no post-processing such as `clean_up_tokenization_spaces`; otherwise a plain loop."""
        if self._fast_batch:
            return self._backend.decode_batch(batch, skip_special_tokens=skip_special_tokens)
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]

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
        return self.update_batch([request_id], [output_ids], [stop], skip_special_tokens,
                                 [final])[0]

    def update_batch(self, request_ids: list[str], output_ids: list[list[int]],
                     stops: list[list[str]], skip_special_tokens: bool = True,
                     finals: list[bool] | None = None) -> list[tuple[str, str | None]]:
        """`update` for many requests with two `decode_batch` calls in total (the prefix
        windows, then the new windows) instead of two `decode` calls per request."""
        if self.tokenizer is None:
            return [("", None)] * len(request_ids)
        finals = finals or [False] * len(request_ids)
        states = [self._states.setdefault(rid, DetokenizerState()) for rid in request_ids]
        # Windows that need decoding this step (a request whose ids did not grow needs none).
        todo = [i for i, (st, ids) in enumerate(zip(states, output_ids)) if st.read_offset < len(ids)]
        prefix_idx = [i for i in todo if states[i].read_offset > states[i].prefix_offset]
        prefix_txt = dict(zip(prefix_idx, self.tokenizer.decode_batch(
            [output_ids[i][states[i].prefix_offset:states[i].read_offset] for i in prefix_idx],
            skip_special_tokens=skip_special_tokens))) if prefix_idx else {}
        new_txt = self.tokenizer.decode_batch(
            [output_ids[i][states[i].prefix_offset:] for i in todo],
            skip_special_tokens=skip_special_tokens) if todo else []
        for i, new_text in zip(todo, new_txt):
            st = states[i]
            prefix_text = prefix_txt.get(i, "")
            incomplete = new_text.endswith("\ufffd")
            if incomplete and finals[i]:
                new_text = new_text[:-1]
                incomplete = False
            if len(new_text) > len(prefix_text) and not incomplete:
                st.raw += new_text[len(prefix_text):]
                st.prefix_offset = st.read_offset
                st.read_offset = len(output_ids[i])
        return [self._emit(st, stop, final) for st, stop, final in zip(states, stops, finals)]

    @staticmethod
    def _emit(st: DetokenizerState, stop: list[str], final: bool) -> tuple[str, str | None]:
        """Stop-string check and holdback on the accumulated text; returns the delta."""
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
