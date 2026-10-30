r"""Bounded real-model smoke test of the dashboard server over HTTP.

    $env:LEANMOE_MODELS_ROOT = "D:\leanmoe-models"
    .\.venv\Scripts\python.exe -m eval.dashboard_smoke

LeanMoE addition. Starts the dashboard server on a free 127.0.0.1 port in this
process and talks to it as the page does: two consecutive cached runs (the
second with a live budget change), then one sequential cached/memory-mapped
comparison, all short. Writes results/dashboard_smoke_<timestamp>.json. One
model is loaded at a time; this is a functional check, not a benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src.engine  # noqa: E402,F401  (before numpy)

import http.client  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from src.dashboard.server import serve  # noqa: E402

PROMPT = "What is 487 × 36?"


def main() -> int:
    server, engine = serve(0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def post(path, body):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        return c, c.getresponse()

    def run(path, body, on_event=None):
        c, r = post(path, body)
        if r.status != 200:
            out = {"http_status": r.status, **json.loads(r.read())}
            c.close()
            return out, []
        events = []
        for raw in r:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                events.append(ev)
                if on_event:
                    on_event(ev)
        c.close()
        return {"http_status": 200}, events

    log: dict = {"prompt": PROMPT, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    status = engine.status()
    # no local paths in the result file: only whether a complete store was found
    log["status"] = {"real_store_available": status["real_store"]["available"],
                     **{k: status[k] for k in ("threads", "memory", "limits")}}  # fmt: skip
    if not status["real_store"]["available"]:
        print(status["real_store"]["problem"])
        return 1
    base = {"prompt": PROMPT, "source": "real", "mode": "cached", "max_new_tokens": 24,
            "budget_gb": 3.5, "start_cache_gb": 1.2}  # fmt: skip

    def summary(events):
        d = [e for e in events if e["type"] == "done"]
        s = [e for e in events if e["type"] == "start"]
        b = [e for e in events if e["type"] == "budget"]
        return {"start": s, "done": d, "budget_events": b, "event_count": len(events)}

    print("run 1: cached ...", flush=True)
    r1, ev1 = run("/api/generate", base)
    log["run1"] = {**r1, **summary(ev1)}
    print("run 2: cached, with a live budget change to 2.5 GB after the first token ...", flush=True)
    budget_reply = {}

    def on_event(ev):
        if ev["type"] == "accepted":
            budget_reply["id"] = ev["request_id"]
        if ev["type"] == "token" and ev["index"] == 0:
            c, r = post("/api/budget", {"request_id": budget_reply["id"], "budget_gb": 2.5})
            budget_reply["reply"] = {"status": r.status, **json.loads(r.read())}
            c.close()

    r2, ev2 = run("/api/generate", base, on_event)
    log["run2"] = {**r2, **summary(ev2), "budget_reply": budget_reply.get("reply")}
    print("comparison: cached then memory-mapped ...", flush=True)
    r3, ev3 = run("/api/compare", {**base, "max_new_tokens": 16})
    log["compare"] = {**r3, **summary(ev3), "compare_done": [e for e in ev3 if e["type"] == "compare_done"]}
    log["final_status"] = {k: engine.status()[k] for k in ("busy", "loaded", "memory")}
    server.shutdown()
    server.server_close()
    engine.close()

    out = REPO_ROOT / "results" / f"dashboard_smoke_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    for name in ("run1", "run2"):
        d = log[name]["done"][0] if log[name]["done"] else {}
        print(f"{name}: stop {d.get('stop_reason')}, {d.get('output_tokens')} tokens, first token "
              f"{d.get('time_to_first_token_s')}, model reused {d.get('model_reused')}, cache at start "
              f"{d.get('cache_at_start')}, cache at end {d.get('cache_at_end')}, text {d.get('text')!r}")  # fmt: skip
    print("run2 budget reply:", log["run2"]["budget_reply"], "| budget events:", [b["event"] for b in log["run2"]["budget_events"]])
    print("compare:", log["compare"]["compare_done"])
    print(f"Full log: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
