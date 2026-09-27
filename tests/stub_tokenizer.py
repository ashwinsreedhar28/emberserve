"""Byte-level stand-in for `pagedserve.tokenizer.Tokenizer` (no `transformers` needed).

Ids are bytes 0-255, which matches the tiny test model's 256-token vocab. Id 1 is EOS.
"""

from __future__ import annotations

from pagedserve.engine import LLMEngine
from pagedserve.tokenizer import IncrementalDetokenizer

EOS = 1


class StubTokenizer:
    eos_token_id: int = EOS

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode())

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            ids = [i for i in ids if i != EOS]
        return bytes(ids).decode(errors="replace")

    def decode_batch(self, batch: list[list[int]], skip_special_tokens: bool = False) -> list[str]:
        return [self.decode(ids, skip_special_tokens) for ids in batch]

    def apply_chat_template(self, messages: list[dict],
                            add_generation_prompt: bool = True) -> list[int]:
        text = "".join(f"<{m['role']}>{m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text)


def install(engine: LLMEngine) -> LLMEngine:
    """Give a tokenizer-less test engine the stub so text prompts and deltas work."""
    engine.tokenizer = StubTokenizer()
    engine.detok = IncrementalDetokenizer(engine.tokenizer)
    engine.eos_token_id = EOS
    return engine
