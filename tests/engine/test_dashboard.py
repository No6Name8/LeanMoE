"""Dashboard server (src/dashboard/server.py) against the synthetic engine.

LeanMoE tests. The server runs on 127.0.0.1 with a free port and no real model
store; every run uses the tiny synthetic store. Windows-only, like the other
engine tests.
"""

from __future__ import annotations

import http.client
import json
import socket
import sys
import threading
import time

import pytest

from src.dashboard import synthetic
from src.dashboard.server import CachedExpertSource, serve
from src.engine.runtime.generate import generate, model_config
from src.engine.runtime.qwen_moe import QwenMoe
from src.engine.runtime.weights import RamExpertSource, ResidentWeights

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="unbuffered expert reads are Windows-only")

PROMPT = "What is 2 + 2?"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv, engine = serve(0, models_root=tmp_path_factory.mktemp("no-models"))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, engine
    srv.shutdown()
    srv.server_close()
    engine.close()


@pytest.fixture(autouse=True)
def idle(server):
    _, engine = server
    engine.test_token_delay = 0.0
    yield
    wait_idle(engine)


def wait_idle(engine, timeout: float = 30.0) -> None:
    end = time.time() + timeout
    while engine.active is not None and time.time() < end:
        time.sleep(0.02)
    assert engine.active is None, "server still busy"


def port(server) -> int:
    return server[0].server_address[1]


def request(server, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port(server), timeout=60)
    h = {"Content-Type": "application/json"} if body is not None else {}
    h.update(headers or {})
    data = body if isinstance(body, (bytes, str)) or body is None else json.dumps(body)
    conn.request(method, path, body=data, headers=h)
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    try:
        return r.status, json.loads(raw), r
    except ValueError:
        return r.status, raw, r


def stream(server, path, body, on_event=None):
    """POST and read the event stream to the end; returns the list of events."""
    conn = http.client.HTTPConnection("127.0.0.1", port(server), timeout=60)
    conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        payload = json.loads(r.read())
        conn.close()
        return r.status, payload
    events = []
    for raw in r:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data: "):
            ev = json.loads(line[6:])
            events.append(ev)
            if on_event:
                on_event(ev)
    conn.close()
    return 200, events


def body(**kw):
    return {"prompt": PROMPT, "source": "synthetic", "mode": "cached", "max_new_tokens": 12, **kw}


def reference_tokens(tmp_path, prompt: str, n: int) -> list[int]:
    store = synthetic.build_store(tmp_path / "ref")
    model = QwenMoe(model_config(store), ResidentWeights.load(store), RamExpertSource(store))
    tokens, _ = generate(model, synthetic.prompt_ids(prompt), n, 512)
    model.experts.close()
    return tokens


def test_binds_localhost_only(server):
    assert server[0].server_address[0] == "127.0.0.1"


def test_status_without_a_real_store(server):
    status, s, r = request(server, "GET", "/api/status")
    assert status == 200 and s["real_store"]["available"] is False
    assert "LEANMOE_MODELS_ROOT" in s["real_store"]["problem"]
    assert s["recorded"] and all(item["files"] for item in s["recorded"])  # read from committed files
    assert "Content-Security-Policy" in r.headers


def test_real_model_request_is_refused_without_a_store(server):
    status, payload, _ = request(server, "POST", "/api/generate", body(source="real"))
    assert status == 400 and "LEANMOE_MODELS_ROOT" in payload["error"]


def test_synthetic_run_streams_and_matches_the_batch_path(server, tmp_path):
    status, events = stream(server, "/api/generate", body())
    assert status == 200
    kinds = [e["type"] for e in events]
    assert kinds[0] == "accepted" and kinds[1] == "start" and kinds[-1] == "done"
    done = events[-1]
    assert done["stop_reason"] == "limit" and done["output_tokens"] == 12
    assert done["token_ids"] == reference_tokens(tmp_path, PROMPT, 12)
    tokens = [e for e in events if e["type"] == "token"]
    assert len(tokens) == 12 and all("memory" in e and "cache" in e for e in tokens)
    assert done["memory"]["samples"] >= 1


@pytest.mark.parametrize("bad, fragment", [
    ({"prompt": ""}, "prompt"),
    ({"prompt": 7}, "prompt"),
    ({"prompt": "x" * 4001}, "longer than"),
    ({"source": "elsewhere"}, "source"),
    ({"mode": "fast"}, "mode"),
    ({"max_new_tokens": 0}, "max_new_tokens"),
    ({"max_new_tokens": 401}, "max_new_tokens"),
    ({"max_new_tokens": "12"}, "max_new_tokens"),
    ({"max_new_tokens": True}, "max_new_tokens"),
    ({"max_new_tokens": 2.5}, "max_new_tokens"),
    ({"budget_gb": -1}, "budget_gb"),
    ({"start_cache_gb": 100}, "start_cache_gb"),
    ({"prompt": "y" * 500, "max_new_tokens": 64}, "context size"),
])  # fmt: skip
def test_invalid_inputs_are_refused_before_running(server, bad, fragment):
    status, payload, _ = request(server, "POST", "/api/generate", {**body(), **bad})
    assert status == 400 and fragment in payload["error"]
    assert server[1].active is None


def test_malformed_and_foreign_requests(server):
    assert request(server, "POST", "/api/generate", "not json")[0] == 400
    assert request(server, "POST", "/api/generate", "[1, 2]")[0] == 400
    assert request(server, "POST", "/api/generate", json.dumps(body()), {"Content-Type": "text/plain"})[0] == 415
    assert request(server, "POST", "/api/generate", body(), {"Origin": "http://evil.example"})[0] == 403
    assert request(server, "GET", "/api/status", None, {"Host": "evil.example"})[0] == 403


def test_static_and_evidence_paths_cannot_escape(server):
    assert request(server, "GET", "/")[0] == 200
    assert request(server, "GET", "/static/katex/katex.min.js")[0] == 200
    assert request(server, "GET", "/static/../server.py")[0] == 404
    assert request(server, "GET", "/static/..%2Fserver.py")[0] == 404
    assert request(server, "GET", "/evidence/EVIDENCE.md")[0] == 200
    assert request(server, "GET", "/evidence/../README.md")[0] == 404


def _start_slow_run(server, path="/api/generate", **kw):
    """Start a run that takes a while, in the background; return (thread, events, request id)."""
    server[1].test_token_delay = 0.05
    events: list = []
    got_id = threading.Event()

    def on_event(ev):
        if ev["type"] == "accepted":
            events.append(ev)
            got_id.set()

    t = threading.Thread(target=lambda: events.extend(stream(server, path, body(max_new_tokens=60, **kw), on_event)[1]))
    t.start()
    assert got_id.wait(30)
    return t, events, events[0]["request_id"]


def test_overlapping_runs_are_refused(server):
    t, events, rid = _start_slow_run(server)
    status, payload, _ = request(server, "POST", "/api/generate", body())
    assert status == 409 and payload["active_request_id"] == rid
    assert request(server, "POST", "/api/compare", body())[0] == 409
    assert request(server, "POST", "/api/cancel", {"request_id": rid})[0] == 200
    t.join(30)
    assert [e for e in events if e.get("type") == "done"][0]["stop_reason"] == "cancelled"


def test_cancel_is_scoped_to_the_active_request(server):
    assert request(server, "POST", "/api/cancel", {"request_id": "nope"})[0] == 409
    t, events, rid = _start_slow_run(server)
    assert request(server, "POST", "/api/cancel", {"request_id": "someone-else"})[0] == 409
    status, payload, _ = request(server, "POST", "/api/cancel", {"request_id": rid})
    assert status == 200 and payload["accepted"] is True and payload["message"]
    t.join(30)
    done = [e for e in events if e.get("type") == "done"][0]
    assert done["stop_reason"] == "cancelled" and done["cancel_phase"] in ("before start", "prefill", "decode")
    assert done["output_tokens"] < 60


def test_cancel_while_loading_or_in_prefill_says_it_must_wait(server):
    server[1].unload()  # force a fresh load, so the cancel lands before decoding starts
    t, events, rid = _start_slow_run(server)
    status, payload, _ = request(server, "POST", "/api/cancel", {"request_id": rid})
    t.join(30)
    if payload["phase"] in ("loading", "prefill"):
        assert "can't be interrupted" in payload["message"]
    done = [e for e in events if e.get("type") == "done"][0]
    assert done["stop_reason"] == "cancelled"


def test_budget_changes_are_scoped_and_validated(server):
    assert request(server, "POST", "/api/budget", {"request_id": "nope", "budget_gb": 2})[0] == 409
    t, events, rid = _start_slow_run(server)
    status, payload, _ = request(server, "POST", "/api/budget", {"request_id": rid, "budget_gb": 2})
    assert status == 400 and "real-model runs in cached mode only" in payload["error"]
    request(server, "POST", "/api/cancel", {"request_id": rid})
    t.join(30)


def test_cache_is_reset_to_its_starting_size_between_runs(server):
    _, engine = server
    stream(server, "/api/generate", body())
    start_slots = engine.model.experts.num_slots
    record = engine.model.experts.layout.record_size
    engine.model.experts.resize((start_slots + 4) * record)  # as a budget change during a run would
    _, events = stream(server, "/api/generate", body())
    start = [e for e in events if e["type"] == "start"][0]
    assert start["model_reused"] is True and start["cache"]["slots"] == start_slots


def test_mode_switch_unloads_the_previous_model(server):
    _, engine = server
    stream(server, "/api/generate", body(mode="cached"))
    old = engine.model
    assert isinstance(old.experts, CachedExpertSource)
    _, events = stream(server, "/api/generate", body(mode="mmap"))
    assert old.experts._closed and engine.model is not old
    assert engine.loaded_key[:2] == ("synthetic", "mmap")
    assert [e for e in events if e["type"] == "start"][0]["model_reused"] is False
    status = request(server, "GET", "/api/status")[1]
    assert status["loaded"] == {"source": "synthetic", "mode": "mmap"}


def test_compare_runs_both_modes_with_identical_tokens(server, tmp_path):
    _, events = stream(server, "/api/compare", body())
    summary = events[-1]
    assert summary["type"] == "compare_done"
    assert summary["completed_runs"] == ["cached", "mmap"] and summary["token_ids_identical"] is True
    dones = [e for e in events if e["type"] == "done"]
    assert [d["mode"] for d in dones] == ["cached", "mmap"]
    assert dones[0]["token_ids"] == reference_tokens(tmp_path, PROMPT, 12)
    assert server[1].model is None  # compare leaves nothing loaded


def test_closing_the_page_mid_run_cancels_and_frees_the_worker(server):
    _, engine = server
    engine.test_token_delay = 0.1  # 60 tokens would take 6 s or more if nothing cancelled them
    s = socket.create_connection(("127.0.0.1", port(server)), timeout=30)
    payload = json.dumps(body(max_new_tokens=60)).encode()
    s.sendall(b"POST /api/generate HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n" % (port(server), len(payload)) + payload)  # fmt: skip
    buf = b""
    while b'"type": "token"' not in buf:
        buf += s.recv(4096)
    s.close()  # the page goes away
    closed = time.time()
    wait_idle(engine)
    assert time.time() - closed < 3.0, "the run kept going after the page closed"
    _, events = stream(server, "/api/generate", body())
    assert events[-1]["type"] == "done" and events[-1]["stop_reason"] == "limit"


def test_an_engine_error_is_reported_and_the_worker_recovers(server):
    _, engine = server
    stream(server, "/api/generate", body())
    model = engine.model
    original = model.forward

    def broken(*a, **k):
        raise RuntimeError("simulated engine failure")

    model.forward = broken
    try:
        _, events = stream(server, "/api/generate", body())
    finally:
        model.forward = original
    done = events[-1]
    assert done["type"] == "done" and done["stop_reason"] == "error" and "simulated engine failure" in done["error"]
    assert engine.active is None
    _, again = stream(server, "/api/generate", body())
    assert again[-1]["stop_reason"] == "limit"


def test_refuses_a_port_another_program_is_using(tmp_path):
    other = socket.socket()
    other.bind(("0.0.0.0", 0))  # another program listening on all interfaces
    other.listen()
    try:
        with pytest.raises(OSError, match="already in use"):
            serve(other.getsockname()[1], models_root=tmp_path)
    finally:
        other.close()


def test_start_event_echoes_the_request_without_local_paths(server):
    _, engine = server
    _, events = stream(server, "/api/generate", body(prompt="Solve $x^2 = 4$."))
    start = [e for e in events if e["type"] == "start"][0]
    accepted = events[0]
    assert start["request_id"] == accepted["request_id"]
    assert start["prompt"] == "Solve $x^2 = 4$." and start["max_new_tokens"] == 12
    assert start["source"] == "synthetic" and start["mode"] == "cached" and "decoding" in start
    assert set(start["threads"]) == {"openblas", "numba"}
    everything = json.dumps(events)
    for path in (str(engine.store_dir), str(engine.synthetic_dir()), str(engine.store_dir.parent)):
        assert path not in everything and path.replace("\\", "\\\\") not in everything
