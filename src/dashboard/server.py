"""LeanMoE local evaluation dashboard: HTTP server and the single model worker.

LeanMoE addition. Standard library only (plus the engine's own dependencies).
Start it with `python -m src.dashboard`; it listens on 127.0.0.1 only.

One worker thread owns at most one loaded model. A run takes the busy lock
before it starts and releases it when it ends; a second run while one is
active is refused with HTTP 409. Loading another source or mode first unloads
and releases the current model. Cancellation and budget changes name the
active request id and are ignored for any other id. Every input is validated
(including the tokenized prompt length and the engine's memory estimate)
before the busy lock is taken.

Prompts and generated text are untrusted text: they are tokenized or decoded,
never executed, and sent to the page as JSON strings, which the page inserts
as text (math through KaTeX with trust disabled).
"""

from __future__ import annotations

import gc
import json
import queue
import re
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

from src.engine import REPO_ROOT
from src.engine.benchmarking import openblas_threads, process_private_bytes
from src.engine.cache.expert_cache import CachedExpertSource
from src.engine.paths import MODELS_ROOT
from src.dashboard.evidence import recorded_evidence

HOST = "127.0.0.1"
DEFAULT_PORT = 8770
STATIC_DIR = Path(__file__).resolve().parent / "static"
RESULTS_DIR = REPO_ROOT / "results"
CONFIG = REPO_ROOT / "configs" / "runtime_cache_chat.json"
STORE_NAME = "qwen1.5-moe-a2.7b-chat-int8"
MAX_SEQ = 512  # context size: prompt + output; its KV cache is in the memory estimate
MAX_NEW_TOKENS = 400
MAX_PROMPT_CHARS = 4000
DEFAULT_BUDGET_GB = 3.5
DEFAULT_START_CACHE_GB = 1.2
SAMPLE_SECONDS = 0.1
HEARTBEAT_SECONDS = 1.0
SOURCES = ("real", "synthetic")
MODES = ("cached", "mmap")
GB = 1e9

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8",
    ".json": "application/json; charset=utf-8", ".md": "text/plain; charset=utf-8",
}  # fmt: skip
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


class BadRequest(ValueError):
    pass


def process_memory() -> dict:
    info = psutil.Process().memory_info()
    return {"private_gb": process_private_bytes() / GB, "working_set_gb": info.rss / GB}


@dataclass
class Request:
    id: str
    kind: str  # "generate" or "compare"
    source: str
    mode: str
    cancel: object = None  # CancelToken
    live: object = None  # LiveBudget, real cached runs only
    phase: str = "loading"
    started: float = field(default_factory=time.perf_counter)


class Sampler(threading.Thread):
    """Process private memory and working set every SAMPLE_SECONDS while a run is active."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self.peak_private = 0.0
        self.peak_ws = 0.0
        self.samples = 0

    def run(self) -> None:
        while not self._stop.is_set():
            m = process_memory()
            self.peak_private = max(self.peak_private, m["private_gb"])
            self.peak_ws = max(self.peak_ws, m["working_set_gb"])
            self.samples += 1
            self._stop.wait(SAMPLE_SECONDS)

    def finish(self) -> dict:
        self._stop.set()
        self.join()
        return {"peak_private_gb": self.peak_private, "peak_working_set_gb": self.peak_ws,
                "samples": self.samples, "interval_s": SAMPLE_SECONDS}  # fmt: skip


class Engine:
    """Owns the loaded model and the busy lock. Only the worker thread touches the model."""

    def __init__(self, models_root: Path | None = None) -> None:
        from src.engine.runtime.generate import RuntimeConfig

        self.busy = threading.Lock()
        self.active: Request | None = None
        self.jobs: queue.Queue = queue.Queue()
        self.model = None
        self.loaded_key: tuple | None = None
        self.start_cache_bytes: int | None = None
        self.load_info: dict = {}
        self._synthetic_dir: tempfile.TemporaryDirectory | None = None
        self.store_dir = Path(models_root or MODELS_ROOT) / STORE_NAME
        self.rt = RuntimeConfig.load(CONFIG)
        self.tokenizer = None
        self.template = None
        self.eos: frozenset[int] = frozenset()
        self.store_problem = self._check_store()
        self.test_token_delay = 0.0  # tests only: slow each token down
        threading.Thread(target=self._worker, name="leanmoe-model-worker", daemon=True).start()

    # ------------------------------------------------------------ setup

    def _check_store(self) -> str | None:
        missing = [f for f in ("store.json", "experts.bin", "resident.bin", "experts_index.json",
                               "resident_index.json") if not (self.store_dir / f).exists()]  # fmt: skip
        if missing:
            return (f"No complete real-model store at {self.store_dir} (missing {', '.join(missing)}). "
                    "Set LEANMOE_MODELS_ROOT to the folder that contains qwen1.5-moe-a2.7b-chat-int8 and "
                    "restart the dashboard; see the README section 'Path 2: the full model'.")  # fmt: skip
        try:
            from src.engine.runtime.streaming import eos_ids_for
            from src.engine.store.chat_template import ChatTemplate
            from src.engine.store.tokenizer import load_tokenizer

            self.tokenizer = load_tokenizer(self.store_dir)
            self.template = ChatTemplate.for_store(self.store_dir)
            self.eos = eos_ids_for(self.tokenizer)
        except Exception as e:  # noqa: BLE001 - reported on the page
            return f"The real-model store at {self.store_dir} could not be prepared: {type(e).__name__}: {e}"
        return None

    def synthetic_dir(self) -> Path:
        if self._synthetic_dir is None:
            from src.dashboard.synthetic import build_store

            self._synthetic_dir = tempfile.TemporaryDirectory(prefix="leanmoe-synthetic-", ignore_cleanup_errors=True)
            build_store(Path(self._synthetic_dir.name) / "store")
        return Path(self._synthetic_dir.name) / "store"

    def close(self) -> None:
        self.unload()
        if self._synthetic_dir is not None:
            self._synthetic_dir.cleanup()
            self._synthetic_dir = None

    # ------------------------------------------------------------ validation (any thread)

    def prompt_ids(self, source: str, prompt: str) -> list[int]:
        if source == "synthetic":
            from src.dashboard.synthetic import prompt_ids

            return prompt_ids(prompt)
        return self.tokenizer.encode(self.template.user_prompt(prompt)).ids

    def validate(self, body: dict, *, compare: bool = False) -> dict:
        from src.engine.runtime.streaming import ContextTooLong, check_context

        def number(name, default, lo, hi, kind=float):
            v = body.get(name, default)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and not float(v).is_integer()):
                raise BadRequest(f"{name} must be a number")
            v = kind(v)
            if not lo <= v <= hi:
                raise BadRequest(f"{name} must be between {lo} and {hi}")
            return v

        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise BadRequest("prompt must be non-empty text")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise BadRequest(f"prompt is longer than {MAX_PROMPT_CHARS} characters")
        source = body.get("source", "real")
        if source not in SOURCES:
            raise BadRequest(f"source must be one of {SOURCES}")
        mode = body.get("mode", "cached")
        if not compare and mode not in MODES:
            raise BadRequest(f"mode must be one of {MODES}")
        if source == "real" and self.store_problem:
            raise BadRequest(self.store_problem)
        max_new = number("max_new_tokens", 64, 1, MAX_NEW_TOKENS, int)
        total_ram = psutil.virtual_memory().total / GB
        budget = number("budget_gb", DEFAULT_BUDGET_GB, 0.5, round(total_ram, 2))
        start_cache = number("start_cache_gb", DEFAULT_START_CACHE_GB, 0.05, 8.0)
        ids = self.prompt_ids(source, prompt)
        try:
            check_context(len(ids), max_new, MAX_SEQ)
        except ContextTooLong as e:
            raise BadRequest(str(e)) from None
        if source == "real":
            from src.engine.runtime.generate import estimate_ram_bytes, model_config

            cfg = model_config(self.store_dir)
            for m in (MODES if compare else (mode,)):
                est = estimate_ram_bytes(self.store_dir, cfg, MAX_SEQ, m, "numpy",
                                         start_cache if m == "cached" else 0.0, "fused") / GB  # fmt: skip
                if est > budget:
                    raise BadRequest(
                        f"{m} mode would need an estimated {est:.2f} GB, over the {budget:.2f} GB budget; "
                        "lower the starting cache or raise the budget"
                    )
        return {"prompt": prompt, "ids": ids, "source": source, "mode": mode, "max_new_tokens": max_new,
                "budget_gb": budget, "start_cache_gb": start_cache}  # fmt: skip

    # ------------------------------------------------------------ requests (handler threads)

    def try_begin(self, kind: str, params: dict) -> Request | None:
        if not self.busy.acquire(blocking=False):
            return None
        from src.engine.runtime.streaming import CancelToken

        req = Request(id=uuid.uuid4().hex, kind=kind, source=params["source"], mode=params["mode"], cancel=CancelToken())
        self.active = req
        return req

    def submit(self, req: Request, params: dict, out: queue.Queue) -> None:
        self.jobs.put((req, params, out))

    def cancel(self, request_id: str) -> dict:
        req = self.active
        if req is None or req.id != request_id:
            raise LookupError("no active run with that request id")
        req.cancel.cancel()
        waiting = req.phase in ("loading", "prefill")
        return {"accepted": True, "phase": req.phase, "message": (
            f"Cancel requested during {req.phase}. That step can't be interrupted; the run stops when it finishes."
            if waiting else "Cancel requested; the run stops after the current token.")}  # fmt: skip

    def set_budget(self, request_id: str, value) -> dict:
        req = self.active
        if req is None or req.id != request_id:
            raise LookupError("no active run with that request id")
        if req.live is None:
            raise BadRequest("live budget changes apply to real-model runs in cached mode only")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BadRequest("budget_gb must be a number")
        facts = req.live.facts
        lowest = (facts.base_ram_bytes + self.model.experts.min_slots * facts.record_bytes) / GB
        highest = psutil.virtual_memory().total / GB
        if not lowest <= value <= highest:
            raise BadRequest(f"budget_gb must be between {lowest:.2f} (the smallest cache this run can use) and {highest:.2f}")
        req.live.set(float(value))
        return {"accepted": True, "budget_gb": float(value), "message": "Applied after the current token."}

    # ------------------------------------------------------------ the worker thread

    def _worker(self) -> None:
        while True:
            req, params, out = self.jobs.get()
            try:
                if req.kind == "compare":
                    self._compare(req, params, out)
                else:
                    self._generate(req, params, out, reuse=True)
            except Exception as e:  # noqa: BLE001 - reported to the page, then state is cleaned up
                out.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
                self.unload()
            finally:
                self.active = None
                out.put(None)
                self.busy.release()

    def unload(self) -> None:
        if self.model is not None:
            self.model.experts.close()
        self.model = None
        self.loaded_key = None
        self.start_cache_bytes = None
        gc.collect()  # releases memory-mapped weight files before anything else loads

    def _load(self, source: str, mode: str, start_cache_gb: float, budget_gb: float) -> dict:
        key = (source, mode, round(start_cache_gb, 4))
        if self.model is not None and self.loaded_key == key:
            return {"reused": True, "load_seconds": 0.0}  # nothing loaded for this run
        self.unload()
        t = time.perf_counter()
        if source == "real":
            from src.engine.runtime.generate import load_model

            rt = replace(self.rt, expert_cache_gb=start_cache_gb, prefetch=True, int8_kernel="fused",
                         max_seq=MAX_SEQ, memory_budget_gb=budget_gb)  # fmt: skip
            self.model, _ = load_model(self.store_dir, mode, MAX_SEQ, budget_gb, None, rt)
        else:
            self.model = self._synthetic_model(mode)
        self.loaded_key = key
        if isinstance(self.model.experts, CachedExpertSource):
            self.start_cache_bytes = self.model.experts.capacity_bytes
        self.load_info = {"load_seconds": time.perf_counter() - t}
        return {"reused": False, **self.load_info}

    def _synthetic_model(self, mode: str):
        from src.dashboard.synthetic import CACHE_SLOTS, MIN_FREE_SLOTS
        from src.engine.predictor.prefetch_policy import PrefetchPolicy
        from src.engine.runtime.generate import model_config
        from src.engine.runtime.qwen_moe import QwenMoe
        from src.engine.runtime.weights import MmapExpertSource, ResidentWeights, UnbufferedExpertSource

        store = self.synthetic_dir()
        if mode == "mmap":
            return QwenMoe(model_config(store), ResidentWeights.load(store, mmap_everything=True), MmapExpertSource(store))
        record = UnbufferedExpertSource(store).layout.record_size
        experts = CachedExpertSource(store, capacity_bytes=CACHE_SLOTS * record, pinned=[], io_threads=2,
                                     min_free_slots=MIN_FREE_SLOTS)  # fmt: skip
        model = QwenMoe(model_config(store), ResidentWeights.load(store), experts, prefetcher=PrefetchPolicy(4))
        model.prefetch_enabled = True
        return model

    def _reset_cache(self) -> None:
        """Back to the starting cache size before every run on a reused model."""
        experts = self.model.experts
        if isinstance(experts, CachedExpertSource) and experts.capacity_bytes != self.start_cache_bytes:
            experts.resize(self.start_cache_bytes)

    def _cache_state(self) -> dict | None:
        experts = self.model.experts
        if not isinstance(experts, CachedExpertSource):
            return None
        return {"slots": experts.num_slots, "gb": experts.capacity_bytes / GB, "cached_experts": len(experts.cached_keys())}

    def _generate(self, req: Request, params: dict, out: queue.Queue, *, reuse: bool, label: str | None = None) -> dict:
        from src.engine.runtime.auto import LiveBudget, store_facts
        from src.engine.runtime.streaming import IncrementalDecoder, stream_generate

        source, mode = params["source"], params["mode"]
        req.mode, req.phase, req.live = mode, "loading", None
        if not reuse:
            self.unload()
        load = self._load(source, mode, params["start_cache_gb"], params["budget_gb"])
        self._reset_cache()
        cache_at_start = self._cache_state()
        if source == "real" and mode == "cached":
            rt = replace(self.rt, expert_cache_gb=params["start_cache_gb"], max_seq=MAX_SEQ, int8_kernel="fused")
            facts = store_facts(self.store_dir, rt, self.model.backend.name)
            req.live = LiveBudget(self.model, facts, psutil.virtual_memory().total)
        decoder = IncrementalDecoder(self.tokenizer) if source == "real" else None
        out.put({"type": "start", "request_id": req.id, "run": label, "source": source, "mode": mode,
                 "model_reused": load["reused"], "load_seconds": load["load_seconds"], "cache": cache_at_start,
                 "prompt_tokens": len(params["ids"]), "max_new_tokens": params["max_new_tokens"], "max_seq": MAX_SEQ,
                 "budget_gb": params["budget_gb"], "live_budget": req.live is not None,
                 "memory": process_memory()})  # fmt: skip
        req.phase = "prefill"
        out.put({"type": "phase", "phase": "prefill", "run": label})
        sampler = Sampler()
        sampler.start()
        seen_events = 0
        totals = {"expert_loads": 0, "demand_reads": 0, "cache_hits": 0, "prefetch_hits": 0}
        done = None

        def between_tokens() -> None:
            if req.live is not None:
                req.live()
            if self.test_token_delay:
                time.sleep(self.test_token_delay)

        try:
            for e in stream_generate(self.model, params["ids"], params["max_new_tokens"], MAX_SEQ,
                                     eos_ids=self.eos if source == "real" else frozenset(),
                                     decoder=decoder, cancel=req.cancel, between_tokens=between_tokens):  # fmt: skip
                if e.kind == "token":
                    req.phase = "decode"
                    for k in totals:
                        totals[k] += int(e.step_stats.get(k, 0))
                    out.put({"type": "token", "run": label, "index": e.index,
                             "text": e.text_delta if decoder else f"[{e.token_id}] ",
                             "step_s": e.step_seconds, "stats": e.step_stats, "totals": dict(totals),
                             "memory": process_memory(), "cache": self._cache_state()})  # fmt: skip
                    if req.live is not None and len(req.live.events) > seen_events:
                        for ev in req.live.events[seen_events:]:
                            out.put({"type": "budget", "run": label, "event": ev, "cache": self._cache_state()})
                        seen_events = len(req.live.events)
                else:
                    done = e
        finally:
            memory = sampler.finish()
            req.live = None
        result = {"type": "done", "run": label, "source": source, "mode": mode, "stop_reason": done.stop_reason,
                  "error": done.error, "prompt_tokens": done.prompt_tokens, "output_tokens": len(done.token_ids),
                  "token_ids": done.token_ids, "text": done.text if decoder else " ".join(f"[{i}]" for i in done.token_ids),
                  "tail": done.text_delta, "time_to_first_token_s": done.time_to_first_token_s,
                  "decode_tokens_per_s": done.decode_tokens_per_s, "total_seconds": done.total_seconds,
                  "load_seconds": load["load_seconds"], "model_reused": load["reused"],
                  "cancel_delay_s": done.cancel_delay_s, "cancel_phase": done.cancel_phase,
                  "totals": totals, "cache_at_start": cache_at_start, "cache_at_end": self._cache_state(),
                  "memory": memory}  # fmt: skip
        out.put(result)
        return result

    def _compare(self, req: Request, params: dict, out: queue.Queue) -> None:
        results = []
        for mode in MODES:
            if req.cancel.cancelled:
                break
            p = {**params, "mode": mode}
            results.append(self._generate(req, p, out, reuse=False, label=mode))
        self.unload()
        same = len(results) == 2 and results[0]["token_ids"] == results[1]["token_ids"]
        out.put({"type": "compare_done", "completed_runs": [r["mode"] for r in results],
                 "token_ids_identical": same if len(results) == 2 else None,
                 "runs": [{k: r[k] for k in ("mode", "stop_reason", "output_tokens", "time_to_first_token_s",
                                             "decode_tokens_per_s", "load_seconds", "memory")} for r in results]})  # fmt: skip

    def status(self) -> dict:
        import numba

        req = self.active
        return {
            "busy": req is not None,
            "active": None if req is None else {"request_id": req.id, "kind": req.kind, "source": req.source,
                                                "mode": req.mode, "phase": req.phase,
                                                "seconds": time.perf_counter() - req.started},  # fmt: skip
            "loaded": None if self.loaded_key is None else {"source": self.loaded_key[0], "mode": self.loaded_key[1]},
            "real_store": {"path": str(self.store_dir), "available": self.store_problem is None,
                           "problem": self.store_problem},  # fmt: skip
            "threads": {"openblas": openblas_threads(), "numba": numba.get_num_threads()},
            "memory": {**process_memory(), "system_available_gb": psutil.virtual_memory().available / GB,
                       "system_total_gb": psutil.virtual_memory().total / GB},  # fmt: skip
            "limits": {"max_seq": MAX_SEQ, "max_new_tokens": MAX_NEW_TOKENS, "max_prompt_chars": MAX_PROMPT_CHARS,
                       "default_budget_gb": DEFAULT_BUDGET_GB, "default_start_cache_gb": DEFAULT_START_CACHE_GB},  # fmt: skip
            "recorded": recorded_evidence(RESULTS_DIR),
        }


def make_handler(engine: Engine, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "LeanMoE"

        def log_message(self, fmt, *args) -> None:  # keep the console readable
            pass

        def _headers(self, status: int, ctype: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            for k, v in {**SECURITY_HEADERS, **(extra or {})}.items():
                self.send_header(k, v)
            self.end_headers()

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8", {"Content-Length": str(len(body))})
            self.wfile.write(body)

        def _host_ok(self) -> bool:
            if self.headers.get("Host") not in allowed_hosts:
                self._json(HTTPStatus.FORBIDDEN, {"error": "this server only answers requests addressed to 127.0.0.1"})
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return
            path = self.path.split("?", 1)[0]
            if path == "/api/status":
                return self._json(HTTPStatus.OK, engine.status())
            if path == "/":
                return self._file(STATIC_DIR / "index.html")
            if path.startswith("/static/"):
                return self._file(STATIC_DIR / path[len("/static/"):], root=STATIC_DIR)
            m = re.fullmatch(r"/evidence/([A-Za-z0-9_.-]+\.(?:json|md))", path)
            if m:
                return self._file(RESULTS_DIR / m.group(1), root=RESULTS_DIR)
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _file(self, path: Path, root: Path = STATIC_DIR) -> None:
            try:
                resolved = path.resolve()
                resolved.relative_to(root.resolve())
            except (ValueError, OSError):
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            if not resolved.is_file() or resolved.suffix not in CONTENT_TYPES:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            data = resolved.read_bytes()
            self._headers(HTTPStatus.OK, CONTENT_TYPES[resolved.suffix], {"Content-Length": str(len(data))})
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802
            # Read the (small) body first, so a refusal below doesn't reset the connection
            # while the client is still sending.
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= 64_000:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "missing or too large request body"})
            raw = self.rfile.read(length)
            if not self._host_ok():
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin not in allowed_origins:
                return self._json(HTTPStatus.FORBIDDEN, {"error": "cross-origin requests are not accepted"})
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "send JSON (Content-Type: application/json)"})
            try:
                body = json.loads(raw or b"{}")
                if not isinstance(body, dict):
                    raise BadRequest("send a JSON object")
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": f"invalid request: {e}"})
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/api/generate", "/api/compare"):
                    return self._run(body, compare=path == "/api/compare")
                if path == "/api/cancel":
                    return self._json(HTTPStatus.OK, engine.cancel(str(body.get("request_id", ""))))
                if path == "/api/budget":
                    return self._json(HTTPStatus.OK, engine.set_budget(str(body.get("request_id", "")), body.get("budget_gb")))
            except LookupError as e:
                return self._json(HTTPStatus.CONFLICT, {"error": str(e)})
            except BadRequest as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _run(self, body: dict, *, compare: bool) -> None:
            try:
                params = engine.validate(body, compare=compare)
            except BadRequest as e:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            req = engine.try_begin("compare" if compare else "generate", params)
            if req is None:
                active = engine.active
                return self._json(HTTPStatus.CONFLICT, {"error": "another run is in progress; wait for it or cancel it",
                                                        "active_request_id": active.id if active else None})  # fmt: skip
            out: queue.Queue = queue.Queue()
            engine.submit(req, params, out)
            self._headers(HTTPStatus.OK, "text/event-stream; charset=utf-8", {"X-Accel-Buffering": "no"})
            connected = True

            def send(text: str) -> None:
                nonlocal connected
                if not connected:
                    return
                try:
                    self.wfile.write(text.encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    connected = False
                    req.cancel.cancel()  # the page went away: stop the run

            send("data: " + json.dumps({"type": "accepted", "request_id": req.id}) + "\n\n")
            while True:
                try:
                    item = out.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    send(": heartbeat\n\n")  # also detects a closed page while prefill runs
                    continue
                if item is None:
                    break
                send("data: " + json.dumps(item) + "\n\n")

    return Handler


class LocalServer(ThreadingHTTPServer):
    """Refuses a port another program already uses. HTTPServer sets SO_REUSEADDR,
    which on Windows lets a second program bind a port that is in use; this asks
    for exclusive use instead."""

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        import socket

        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(port: int = DEFAULT_PORT, models_root: Path | None = None) -> tuple[ThreadingHTTPServer, Engine]:
    """Bind 127.0.0.1:port (0 = any free port) and return the server and its engine."""
    if port:
        import socket

        # Windows lets a program bind 127.0.0.1:port while another program holds the same
        # port on all interfaces; requests would then silently go to one or the other.
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex((HOST, port)) == 0:
                raise OSError(f"port {port} is already in use by another program")
    engine = Engine(models_root)
    server = LocalServer((HOST, port), BaseHTTPRequestHandler)
    server.RequestHandlerClass = make_handler(engine, server.server_address[1])  # the port actually bound
    server.daemon_threads = True
    return server, engine
