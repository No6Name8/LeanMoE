"""Streaming generation: one token at a time, with stopping, limits and cancellation.

LeanMoE addition (not part of ExpertRelay). It runs the same forward pass and
the same greedy (argmax) choice as runtime.generate.generate, so up to the
stopping point it produces the same token ids; generate() itself is unchanged
and still runs a fixed number of tokens for the benchmarks.

    for event in stream_generate(model, prompt_ids, max_new_tokens=200, max_seq=512,
                                 eos_ids={151645, 151643}, decoder=IncrementalDecoder(tokenizer)):
        if event.kind == "token":
            print(event.text_delta, end="", flush=True)
        else:
            print(event.stop_reason, event.time_to_first_token_s)

Rules:
  - The prompt length plus the output allowance is checked against max_seq
    before anything is allocated (ContextTooLong otherwise).
  - Generation stops at the first token in eos_ids (not yielded as text), after
    max_new_tokens tokens, or when the CancelToken is set.
  - Cancellation is checked between forward passes. A forward pass can't be
    interrupted, so a cancel during the prompt's first pass (prefill, which can
    take many seconds on the real model) takes effect after it; the final event
    reports how long the stop took and which phase was running.
  - Per-request resources (the KV cache) are released on completion,
    cancellation, error, or when the caller stops iterating; the expert source
    is left with no expert in use and no queued prefetch.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np

from src.engine.runtime.qwen_moe import KVCache, QwenMoe
from src.engine.runtime.text_stream import IncrementalDecoder

__all__ = [
    "EOS_TOKEN_NAMES", "CancelToken", "ContextTooLong", "IncrementalDecoder", "StreamEvent",
    "check_context", "eos_ids_for", "stream_generate",
]  # fmt: skip

# The Chat model's end-of-turn tokens, as listed in its generation_config.json
# (eos_token_id [151645, 151643]). That file isn't stored with the model, so the
# ids are resolved from these names through the store's tokenizer: the one place
# every caller (dashboard, command-line example, audit) gets them from.
EOS_TOKEN_NAMES = ("<|im_end|>", "<|endoftext|>")


def eos_ids_for(tokenizer) -> frozenset[int]:
    """End-of-sequence token ids for a tokenizer (empty if it has none of the names)."""
    ids = {tokenizer.token_to_id(name) for name in EOS_TOKEN_NAMES}
    return frozenset(i for i in ids if i is not None)


class ContextTooLong(ValueError):
    """The prompt plus the requested output doesn't fit the context size."""


class CancelToken:
    """Thread-safe cancel flag that remembers when it was set."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.requested_at: float | None = None

    def cancel(self) -> None:
        if not self._event.is_set():
            self.requested_at = time.perf_counter()
            self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class StreamEvent:
    kind: str  # "token" or "done"
    index: int = 0  # 0-based position of this token in the output
    token_id: int | None = None
    text_delta: str = ""
    step_seconds: float = 0.0  # this forward pass
    step_stats: dict = field(default_factory=dict)  # expert loads, reads, hits of this forward pass
    # filled on the "done" event
    stop_reason: str | None = None  # "eos", "limit", "cancelled", "error"
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    prompt_tokens: int = 0
    time_to_first_token_s: float | None = None  # the prefill forward pass
    decode_tokens_per_s: float | None = None  # tokens after the first / their forward time
    total_seconds: float = 0.0
    cancel_delay_s: float | None = None  # from cancel request to stop
    cancel_phase: str | None = None  # "before start", "prefill" or "decode"
    error: str | None = None


STEP_STATS = ("expert_loads", "demand_reads", "cache_hits", "prefetch_hits", "prefetch_waits", "expert_read_seconds")


def check_context(prompt_len: int, max_new_tokens: int, max_seq: int) -> None:
    if prompt_len < 1:
        raise ContextTooLong("the prompt is empty")
    if max_new_tokens < 1:
        raise ContextTooLong("max_new_tokens must be at least 1")
    if prompt_len + max_new_tokens > max_seq:
        raise ContextTooLong(
            f"prompt of {prompt_len} tokens + {max_new_tokens} new tokens = {prompt_len + max_new_tokens}, "
            f"over the context size of {max_seq}; shorten the prompt or lower the output limit"
        )


def _release(model: QwenMoe) -> None:
    """No expert in use, no queued prefetch left over from this request."""
    for layer in range(model.c.num_layers):
        model.experts.end_layer(layer)


def stream_generate(
    model: QwenMoe,
    prompt_ids: list[int],
    max_new_tokens: int,
    max_seq: int,
    *,
    eos_ids: set[int] | frozenset[int] = frozenset(),
    decoder: IncrementalDecoder | None = None,
    cancel: CancelToken | None = None,
    between_tokens: Callable[[], None] | None = None,
) -> Iterator[StreamEvent]:
    """Yield one "token" event per generated token, then one "done" event.

    Raises ContextTooLong before allocating anything. Any other exception from
    the forward pass ends the stream with a "done" event whose stop_reason is
    "error" (the exception text is in .error), after releasing resources.
    between_tokens runs after each token, as in generate() (e.g. LiveBudget)."""
    check_context(len(prompt_ids), max_new_tokens, max_seq)
    t_start = time.perf_counter()
    out: list[int] = []
    step_times: list[float] = []
    stop, error, phase = "limit", None, "before start"
    cache = None
    try:
        if cancel is not None and cancel.cancelled:
            stop = "cancelled"
        else:
            cache = KVCache(model.c, max_seq)
            feed = list(prompt_ids)
            for i in range(max_new_tokens):
                phase = "prefill" if i == 0 else "decode"
                t = time.perf_counter()
                logits, timings = model.forward(np.asarray(feed), cache)
                nxt = int(np.argmax(logits))
                step_times.append(time.perf_counter() - t)
                if nxt in eos_ids:
                    stop = "eos"
                    break
                out.append(nxt)
                delta = decoder.push(nxt) if decoder is not None else ""
                stats = {k: getattr(timings, k, 0) for k in STEP_STATS}
                yield StreamEvent("token", index=i, token_id=nxt, text_delta=delta, step_seconds=step_times[-1],
                                  step_stats=stats)  # fmt: skip
                if between_tokens is not None:
                    between_tokens()
                if cancel is not None and cancel.cancelled:
                    stop = "cancelled"
                    break
                feed = [nxt]
    except GeneratorExit:  # the caller stopped iterating
        stop = "cancelled"
        raise
    except Exception as e:  # noqa: BLE001 - reported in the done event, not swallowed silently
        stop, error = "error", f"{type(e).__name__}: {e}"
    finally:
        del cache
        _release(model)
    tail = decoder.finish() if decoder is not None else ""
    decode_times = step_times[1:]
    now = time.perf_counter()
    yield StreamEvent(
        "done",
        index=len(out),
        text_delta=tail,
        stop_reason=stop,
        token_ids=out,
        text=decoder.released if decoder is not None else "",
        prompt_tokens=len(prompt_ids),
        time_to_first_token_s=step_times[0] if step_times else None,
        decode_tokens_per_s=len(decode_times) / sum(decode_times) if decode_times and sum(decode_times) > 0 else None,
        total_seconds=now - t_start,
        cancel_delay_s=(now - cancel.requested_at) if stop == "cancelled" and cancel and cancel.requested_at else None,
        cancel_phase=phase if stop == "cancelled" else None,
        error=error,
    )
