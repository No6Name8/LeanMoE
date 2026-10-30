"""stream_generate (src/engine/runtime/streaming.py) on a tiny synthetic store.

LeanMoE tests. The store is built in a temporary directory by store_helpers;
no model download is needed. Windows-only, like the other engine tests.
"""

from __future__ import annotations

import sys

import pytest
from store_helpers import build_test_store, random_checkpoint

from src.engine.cache.expert_cache import CachedExpertSource
from src.engine.predictor.prefetch_policy import PrefetchPolicy
from src.engine.runtime import streaming
from src.engine.runtime.generate import generate, model_config
from src.engine.runtime.qwen_moe import QwenMoe
from src.engine.runtime.streaming import CancelToken, ContextTooLong, stream_generate
from src.engine.runtime.weights import RamExpertSource, ResidentWeights, UnbufferedExpertSource

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="unbuffered expert reads are Windows-only")

CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 24,
    "norm_topk_prob": False,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
}
PROMPT = [3, 9, 27, 17, 51]
MAX_SEQ = 32
N = 12


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("store")
    build_test_store(d, CONFIG, random_checkpoint(CONFIG, seed=1), mp)
    mp.undo()
    return d


def _ram_model(store) -> QwenMoe:
    return QwenMoe(model_config(store), ResidentWeights.load(store), RamExpertSource(store))


def _cached_model(store, slots: int = 6) -> QwenMoe:
    """A small expert cache (6 slots, smaller than the 3 x 8 experts) with prefetch on."""
    record = UnbufferedExpertSource(store).layout.record_size
    experts = CachedExpertSource(store, capacity_bytes=slots * record, pinned=[], io_threads=2, min_free_slots=3)
    model = QwenMoe(model_config(store), ResidentWeights.load(store), experts, prefetcher=PrefetchPolicy(4))
    model.prefetch_enabled = True
    return model


@pytest.fixture(scope="module")
def reference(store):
    model = _ram_model(store)
    tokens, _ = generate(model, PROMPT, N, MAX_SEQ)
    model.experts.close()
    return tokens


def _run(model, **kw):
    events = list(stream_generate(model, PROMPT, kw.pop("max_new_tokens", N), MAX_SEQ, **kw))
    return [e for e in events if e.kind == "token"], events[-1]


def _no_expert_in_use(model) -> bool:
    slots = model.experts._slots
    return all(s.refs == 0 for s in slots) and not any(s.state == 1 for s in slots)  # 1 = QUEUED


def test_tokens_match_generate_without_eos(store, reference):
    model = _ram_model(store)
    tokens, done = _run(model)
    model.experts.close()
    assert [e.token_id for e in tokens] == reference
    assert done.kind == "done" and done.stop_reason == "limit" and done.token_ids == reference
    assert done.time_to_first_token_s is not None and done.decode_tokens_per_s is not None


def test_cached_model_with_prefetch_matches_generate(store, reference):
    model = _cached_model(store)
    _, done = _run(model)
    model.experts.close()
    assert done.token_ids == reference


def test_stops_at_eos_without_yielding_it(store, reference):
    k = next(i for i in range(2, N) if reference[i] not in reference[:i])  # first new token after index 1
    model = _ram_model(store)
    tokens, done = _run(model, eos_ids={reference[k]})
    model.experts.close()
    assert done.stop_reason == "eos"
    assert done.token_ids == reference[:k] and [e.token_id for e in tokens] == reference[:k]


def test_eos_as_first_token_gives_empty_output(store, reference):
    model = _ram_model(store)
    tokens, done = _run(model, eos_ids={reference[0]})
    model.experts.close()
    assert tokens == [] and done.stop_reason == "eos" and done.token_ids == []


def test_output_limit(store, reference):
    model = _ram_model(store)
    tokens, done = _run(model, max_new_tokens=3)
    model.experts.close()
    assert len(tokens) == 3 and done.stop_reason == "limit" and done.token_ids == reference[:3]


@pytest.mark.parametrize(
    "prompt_len, new, why",
    [(MAX_SEQ - 4, 5, "over the context size"), (0, 5, "empty"), (5, 0, "at least 1")],
)
def test_context_checked_before_any_allocation(store, monkeypatch, prompt_len, new, why):
    model = _ram_model(store)

    def no_alloc(*a, **k):
        raise AssertionError("KV cache allocated before the length check")

    monkeypatch.setattr(streaming, "KVCache", no_alloc)
    with pytest.raises(ContextTooLong, match=why):
        next(stream_generate(model, [1] * prompt_len, new, MAX_SEQ))
    model.experts.close()


def test_cancel_between_tokens(store, reference):
    model = _ram_model(store)
    cancel = CancelToken()
    seen = []
    for e in stream_generate(model, PROMPT, N, MAX_SEQ, cancel=cancel):
        if e.kind == "token":
            seen.append(e.token_id)
            if len(seen) == 3:
                cancel.cancel()
        else:
            done = e
    model.experts.close()
    assert seen == reference[:3] and done.token_ids == reference[:3]
    assert done.stop_reason == "cancelled" and done.cancel_phase == "decode" and done.cancel_delay_s >= 0


def test_cancel_during_prefill_stops_after_it_and_reports_the_phase(store, reference, monkeypatch):
    model = _ram_model(store)
    cancel = CancelToken()
    original = model.forward
    calls = []

    def forward(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            cancel.cancel()  # requested while the prefill pass is running
        return original(*a, **k)

    monkeypatch.setattr(model, "forward", forward)
    tokens, done = _run(model, cancel=cancel)
    model.experts.close()
    assert len(calls) == 1  # no forward pass after the cancelled prefill
    assert [e.token_id for e in tokens] == reference[:1]
    assert done.stop_reason == "cancelled" and done.cancel_phase == "prefill" and done.cancel_delay_s > 0


def test_cancel_before_start_allocates_nothing(store, monkeypatch):
    model = _ram_model(store)
    cancel = CancelToken()
    cancel.cancel()
    monkeypatch.setattr(streaming, "KVCache", lambda *a, **k: (_ for _ in ()).throw(AssertionError("allocated")))
    tokens, done = _run(model, cancel=cancel)
    model.experts.close()
    assert tokens == [] and done.stop_reason == "cancelled" and done.cancel_phase == "before start"


def test_error_mid_layer_is_reported_and_resources_released(store, reference, monkeypatch):
    """Fail inside a forward pass, while an expert is in use and next-layer
    prefetches may be queued: the stream must still leave the cache clean."""
    model = _cached_model(store)
    original_load = model.experts.load
    state = {"armed": False, "loads": 0}

    def load(layer, expert):
        if state["armed"]:
            state["loads"] += 1
            if state["loads"] == 2:  # one expert of this layer already loaded and in use
                raise RuntimeError("simulated read failure")
        return original_load(layer, expert)

    monkeypatch.setattr(model.experts, "load", load)
    events = []
    for e in stream_generate(model, PROMPT, N, MAX_SEQ):
        events.append(e)
        if e.kind == "token" and e.index == 1:
            state["armed"] = True  # fail during the third forward pass
    done = events[-1]
    assert done.stop_reason == "error" and "simulated read failure" in done.error
    assert done.token_ids == reference[:2]
    assert _no_expert_in_use(model)
    monkeypatch.setattr(model.experts, "load", original_load)
    _, again = _run(model)  # the next request works and is unaffected
    model.experts.close()
    assert again.token_ids == reference


def test_stopping_iteration_early_then_next_request_works(store, reference):
    model = _cached_model(store)
    gen = stream_generate(model, PROMPT, N, MAX_SEQ)
    first = [next(gen).token_id for _ in range(2)]
    gen.close()
    assert first == reference[:2]
    assert _no_expert_in_use(model)
    _, again = _run(model)
    model.experts.close()
    assert again.token_ids == reference


def test_consecutive_requests_on_one_model(store, reference):
    model = _cached_model(store)
    results = [_run(model)[1].token_ids for _ in range(3)]
    shorter = _run(model, max_new_tokens=4)[1].token_ids
    model.experts.close()
    assert results == [reference] * 3 and shorter == reference[:4]


def test_between_tokens_hook_runs_after_each_token(store):
    model = _ram_model(store)
    count = []
    _run(model, between_tokens=lambda: count.append(1))
    model.experts.close()
    assert len(count) == N
