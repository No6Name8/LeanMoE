"""Incremental decoding of generated tokens into text that never splits a character.

LeanMoE addition. Standard library only, so its tests run without numpy.
"""

from __future__ import annotations

from typing import Protocol

REPLACEMENT = "\ufffd"


class Decoder(Protocol):
    def decode(self, ids: list[int]) -> str: ...


class IncrementalDecoder:
    """Turns a growing list of token ids into text deltas that never split a character.

    Byte-level tokenizers can end a token in the middle of a multi-byte UTF-8
    character; decoding at that point gives a replacement character. This
    holds back a trailing run of replacement characters until later tokens
    complete them, and only releases text that extends what was released
    before."""

    def __init__(self, tokenizer: Decoder) -> None:
        self._tok = tokenizer
        self.ids: list[int] = []
        self.released = ""

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self._tok.decode(self.ids)
        stable = text.rstrip(REPLACEMENT)
        if stable.startswith(self.released) and len(stable) > len(self.released):
            delta = stable[len(self.released) :]
            self.released = stable
            return delta
        return ""

    def finish(self) -> str:
        """Release whatever is left, including replacement characters that never completed."""
        text = self._tok.decode(self.ids)
        delta = text[len(self.released) :] if text.startswith(self.released) else ""
        self.released = text if text.startswith(self.released) else self.released
        return delta
