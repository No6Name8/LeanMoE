"""Protected backend tests. Real HTTP; synthetic engine in an isolated temporary store."""
import http.client
import json
import sys
import threading
import time

import pytest

from src.dashboard.access import ProxyAccess, COOKIE
from src.dashboard.server import serve

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows engine")
ORIGIN = "https://demo.example.test"
KEY = "test-only-key-" + "x" * 32


@pytest.fixture
def protected(tmp_path):
    access = ProxyAccess(ORIGIN, KEY, frozenset({"alice", "bob"}))
    srv, engine = serve(0, models_root=tmp_path, execution_location="server", access=access)
    # Exercise actual HTTP/streaming/cache cleanup without loading the 14 GB model.
    # Public callers must still submit source=real; only this test's engine is substituted.
    validate = engine.validate
    engine.validate = lambda body, compare=False: validate({**body, "source": "synthetic"}, compare=compare)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv, engine, access
    if engine.active:
        engine.active.cancel.cancel()
    until = time.monotonic() + 15
    while engine.active and time.monotonic() < until:
        time.sleep(.02)
    srv.shutdown()
    srv.server_close()
    engine.close()


def call(env, method="GET", path="/api/status", body=None, user="alice", cookie=None, **headers):
    conn = http.client.HTTPConnection("127.0.0.1", env[0].server_port, timeout=15)
    h = {"X-LeanMoE-Visitor": user, "X-LeanMoE-Proxy-Key": KEY}
    if cookie:
        h["Cookie"] = cookie
    if method == "POST":
        h.update({"Origin": ORIGIN, "Content-Type": "application/json"})
    h.update(headers)
    conn.request(method, path, None if body is None else json.dumps(body), h)
    r = conn.getresponse()
    raw = r.read()
    result = (r.status, json.loads(raw), r.getheader("Set-Cookie"))
    conn.close()
    return result


def session(env, user="alice"):
    status, _, cookie = call(env, user=user)
    assert status == 200 and "Secure; HttpOnly; SameSite=Strict" in cookie
    return cookie.split(";", 1)[0]


def start(env, cookie, on_event=None):
    events = []
    def run():
        conn = http.client.HTTPConnection("127.0.0.1", env[0].server_port, timeout=20)
        conn.request("POST", "/api/generate", json.dumps({"source": "real", "prompt": "private Alice prompt",
                     "max_new_tokens": 50}), {"Content-Type": "application/json", "Origin": ORIGIN,
                     "X-LeanMoE-Proxy-Key": KEY, "X-LeanMoE-Visitor": "alice", "Cookie": cookie})
        response = conn.getresponse()
        assert response.status == 200
        for line in response:
            if line.startswith(b"data: "):
                ev = json.loads(line[6:])
                events.append(ev)
                if on_event:
                    on_event(ev)
        conn.close()
    t = threading.Thread(target=run)
    t.start()
    until = time.monotonic() + 10
    while not events and time.monotonic() < until:
        time.sleep(.01)
    assert events and events[0]["type"] == "accepted"
    return t, events


def test_proxy_auth_cookie_and_origin_are_required(protected):
    assert call(protected, **{"X-LeanMoE-Proxy-Key": "bad"})[0] == 401
    assert call(protected, user="unlisted")[0] == 401
    c = session(protected)
    body = {"prompt": "hello", "source": "real"}
    assert call(protected, "POST", "/api/generate", body)[0] == 401
    assert call(protected, "POST", "/api/generate", body, cookie=c, Origin="https://evil.test")[0] == 403
    assert call(protected, "POST", "/api/generate", body, cookie=c, Origin="")[0] == 403
    assert call(protected, cookie=c, Host="demo.example.test")[0] == 403  # proxy must rewrite Host


def test_other_account_and_browser_cannot_inspect_or_control(protected):
    _, engine, _ = protected
    engine.test_token_delay = .1
    alice, other_browser, bob = session(protected), session(protected), session(protected, "bob")
    t, events = start(protected, alice)
    request_id = events[0]["request_id"]
    try:
        for user, cookie in [("alice", other_browser), ("bob", bob)]:
            _, status, _ = call(protected, user=user, cookie=cookie)
            assert status["busy"] and status["active"] is None and status["loaded"] is None
            assert request_id not in json.dumps(status) and "private Alice prompt" not in json.dumps(status)
            for path in ("/api/cancel", "/api/budget"):
                assert call(protected, "POST", path, {"request_id": request_id, "budget_gb": 2.5}, user=user, cookie=cookie)[0] == 409
            code, busy, _ = call(protected, "POST", "/api/generate", {"prompt": "bob", "source": "real"}, user=user, cookie=cookie)
            assert code == 409 and request_id not in json.dumps(busy) and "No queue" in busy["error"]
        assert call(protected, "POST", "/api/cancel", {"request_id": request_id}, cookie=alice)[0] == 200
    finally:
        engine.active.cancel.cancel() if engine.active else None
        t.join(15)
    assert not t.is_alive() and engine.model is None  # no model/cache reused between sessions


def test_public_input_and_evidence_limits(protected):
    cookie = session(protected)
    for extra in ({"source": "synthetic"}, {"max_new_tokens": 97}, {"budget_gb": 4.1}, {"start_cache_gb": 1.4}):
        protected[2]._rates.clear()
        assert call(protected, "POST", "/api/generate", {"source": "real", "prompt": "x", **extra}, cookie=cookie)[0] == 400
    assert call(protected, "POST", "/api/generate", {"source": "real", "prompt": "x" * 17000}, cookie=cookie)[0] == 400
    assert call(protected, path="/evidence/not-published.json", cookie=cookie)[0] == 404


def test_rate_limit_is_shared_by_browsers_of_one_account(protected):
    a, b = session(protected), session(protected)
    # Invalid jobs count toward admissions too, so this exercises the limit
    # without inference and proves cookie renewal cannot reset it.
    for cookie in (a, b, a):
        assert call(protected, "POST", "/api/generate", {"source": "synthetic", "prompt": "x"}, cookie=cookie)[0] == 400
    assert call(protected, "POST", "/api/generate", {"source": "real", "prompt": "x"}, cookie=b)[0] == 429
    bob = session(protected, "bob")
    assert call(protected, "POST", "/api/generate", {"source": "synthetic", "prompt": "x"}, cookie=bob, user="bob")[0] == 400


def test_deployed_version_is_a_commit_not_arbitrary_environment_text(protected, monkeypatch):
    monkeypatch.setenv("LEANMOE_DEPLOYED_COMMIT", "a" * 40)
    assert call(protected)[1]["build"]["commit"] == "a" * 40
    monkeypatch.setenv("LEANMOE_DEPLOYED_COMMIT", "private/path/or/account")
    assert call(protected)[1]["build"]["commit"] is None


def test_deadline_cancels_stream(protected):
    protected[2].max_seconds = .2
    protected[1].test_token_delay = .1
    t, events = start(protected, session(protected))
    t.join(15)
    assert not t.is_alive()
    done = next(e for e in events if e["type"] == "done")
    assert done["stop_reason"] == "cancelled" and done["output_tokens"] < 50


def test_automatic_headroom_refusal_is_safe_and_releases_public_job(protected, monkeypatch):
    import src.dashboard.resources as resources
    engine = protected[1]
    validate = engine.validate
    engine.validate = lambda body, compare=False: {**validate(body, compare=compare), 'source': 'real', 'profile': 'auto'}
    def refuse(*args):
        raise resources.ResourceUnavailable('Insufficient measured physical RAM; retry.')
    monkeypatch.setattr(resources, 'load', refuse)
    t, events = start(protected, session(protected))
    t.join(15)
    assert not t.is_alive()
    error = next(e for e in events if e['type'] == 'error')
    assert error['resource_error'] and 'Insufficient measured physical RAM' in error['message']
    assert engine.active is None and engine.model is None
    assert not engine.busy.locked()
