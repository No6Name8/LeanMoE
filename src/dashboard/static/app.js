// LeanMoE dashboard page logic. LeanMoE addition.
// All prompt and model text is untrusted: it is inserted with textContent, and
// math segments go through KaTeX with trust disabled. No innerHTML with data.
// Every number shown comes from the server's measurements; nothing is estimated here.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "—" : Number(v).toFixed(d);
  const pick = { source: "real", mode: "cached", "cmp-source": "real" };
  const ui = { status: null, run: null, cmp: null, qwenMissing: false, busy: false };

  // ------------------------------------------------------------ small helpers

  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined && text !== null) e.textContent = String(text);
    if (cls) e.className = cls;
    return e;
  }
  function setKV(dl, rows) {
    dl.textContent = "";
    for (const [k, v] of rows) { dl.append(el("dt", k), el("dd", v)); }
  }
  function setState(id, text, kind) {
    const s = $(id);
    s.textContent = text;
    s.className = "state " + (kind || "idle");
  }
  function setError(id, text) { $(id).textContent = text || ""; }

  function renderMath(box, text, synthetic) {
    box.textContent = "";
    if (!text) { box.appendChild(el("span", "No output.", "placeholder")); return; }
    if (synthetic) { box.textContent = text; return; }
    for (const seg of window.LeanMoEMath.splitMath(text)) {
      if (seg.type === "text") { box.appendChild(document.createTextNode(seg.value)); continue; }
      const span = el(seg.display ? "div" : "span");
      try {
        window.katex.render(seg.value, span, { displayMode: seg.display, throwOnError: false, trust: false,
          strict: "ignore", maxSize: 10, maxExpand: 100, output: "htmlAndMathml" });
      } catch (e) {
        span.textContent = (seg.display ? "$$" : "$") + seg.value + (seg.display ? "$$" : "$");
      }
      box.appendChild(span);
    }
  }

  async function postJSON(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    let j = {};
    try { j = await r.json(); } catch (e) { /* empty */ }
    return { ok: r.ok, status: r.status, body: j };
  }

  async function readEvents(resp, onEvent) {
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        for (const line of chunk.split("\n")) if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
      }
    }
  }

  // ------------------------------------------------------------ tabs and choices

  // The System details popover floats over the page: close it on any outside click,
  // on Escape and on a tab change, so it never covers buttons.
  const sysDetails = document.querySelector(".system details");
  document.addEventListener("click", (e) => { if (sysDetails.open && !sysDetails.contains(e.target)) sysDetails.open = false; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") sysDetails.open = false; });

  const tabs = ["run", "compare", "evidence"];
  function showTab(name) {
    sysDetails.open = false;
    for (const t of tabs) {
      const on = t === name;
      $("tab-" + t).setAttribute("aria-selected", String(on));
      $("tab-" + t).tabIndex = on ? 0 : -1;
      $("panel-" + t).hidden = !on;
    }
    if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  }
  tabs.forEach((t, i) => {
    $("tab-" + t).addEventListener("click", () => showTab(t));
    $("tab-" + t).addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      showTab(next); $("tab-" + next).focus();
    });
  });

  document.querySelectorAll(".segmented button").forEach((b) => {
    b.addEventListener("click", () => {
      if (b.disabled) return;
      const g = b.dataset.group;
      document.querySelectorAll(`.segmented button[data-group="${g}"]`).forEach((o) => o.setAttribute("aria-checked", String(o === b)));
      pick[g] = b.dataset.value;
      updateSelection();
    });
  });

  const SOURCE_SHORT = { real: "Qwen", synthetic: "Engine test" };
  const MODE_SHORT = { cached: "LeanMoE", mmap: "Windows paging" };
  const SOURCE_LONG = { real: "Qwen (Qwen1.5-MoE-A2.7B-Chat, int8)", synthetic: "Engine test (tiny random-weight model; meaningless output)" };
  const MODE_LONG = { cached: "LeanMoE (expert cache + prefetch)", mmap: "Windows paging (memory-mapped baseline)" };

  // The model is Qwen unless "Engine test" is ticked under Advanced. Nothing switches it automatically.
  function updateSelection() {
    for (const [p, key] of [["run", "source"], ["cmp", "cmp-source"]]) {
      const test = $(p + "-engine-test").checked;
      pick[key] = test ? "synthetic" : "real";
      const chip = $(p + "-model-chip");
      chip.textContent = test ? "Engine test · meaningless output" : "Qwen";
      chip.className = "model-chip" + (test ? " test" : (ui.qwenMissing ? " unavailable" : ""));
      $(p + "-missing").hidden = !(ui.qwenMissing && !test);
    }
    const b = $("run-source-badge");
    const syn = pick.source === "synthetic";
    b.textContent = syn ? "Engine test: meaningless output" : "Qwen";
    b.className = "badge" + (syn ? " synthetic" : "");
    $("run-selection").textContent = `Model: ${SOURCE_SHORT[pick.source]} · Method: ${MODE_SHORT[pick.mode]}`;
    document.querySelectorAll(".explain-item").forEach((it) => {
      const on = pick[it.dataset.group] === it.dataset.value;
      it.classList.toggle("selected", on);
      if (on) it.setAttribute("aria-current", "true"); else it.removeAttribute("aria-current");
    });
    refreshButtons();
  }

  function refreshButtons() {
    const runBlocked = ui.qwenMissing && pick.source === "real";
    const cmpBlocked = ui.qwenMissing && pick["cmp-source"] === "real";
    $("run").disabled = ui.busy || runBlocked;
    $("compare").disabled = ui.busy || cmpBlocked;
  }

  for (const p of ["run", "cmp"]) {
    $(p + "-engine-test").addEventListener("change", updateSelection);
    $(p + "-use-test").addEventListener("click", () => {
      $(p + "-engine-test").checked = true;
      $(p + "-advanced").open = true;
      updateSelection();
    });
  }

  function settings() {
    return { max_new_tokens: Number($("max-new").value), budget_gb: Number($("budget").value),
             start_cache_gb: Number($("start-cache").value) };
  }

  // ------------------------------------------------------------ status

  async function loadStatus() {
    let s;
    try { s = await (await fetch("/api/status")).json(); } catch (e) {
      $("store-pill").textContent = "Server not reachable"; $("store-pill").className = "pill no"; return;
    }
    ui.status = s;
    const ok = s.real_store.available;
    $("store-pill").textContent = ok ? "Qwen ready" : "Qwen not installed";
    $("store-pill").className = "pill " + (ok ? "ok" : "no");
    ui.qwenMissing = !ok;
    for (const p of ["run", "cmp"]) $(p + "-missing-detail").textContent = ok ? "" : s.real_store.problem;
    updateSelection();
    $("max-seq").textContent = s.limits.max_seq;
    setKV($("system-details"), [
      ["Qwen model store", ok ? s.real_store.path : "not available"],
      ...(ok ? [] : [["How to set it up", s.real_store.problem]]),
      ["OpenBLAS / numba threads", `${s.threads.openblas} / ${s.threads.numba}`],
      ["System RAM", `${fmt(s.memory.system_available_gb)} GB free of ${fmt(s.memory.system_total_gb)} GB`],
      ["This server process", `private ${fmt(s.memory.private_gb)} GB, working set ${fmt(s.memory.working_set_gb)} GB`],
      ["Loaded now", s.loaded ? `${s.loaded.source}, ${s.loaded.mode}` : "nothing"],
      ["Context limit", `${s.limits.max_seq} tokens (prompt + output)`],
    ]);
    renderRecorded(s.recorded);
  }

  function renderRecorded(items) {
    const ul = $("recorded");
    ul.textContent = "";
    if (!items || !items.length) { ul.appendChild(el("li", "No recorded result files found.")); return; }
    for (const it of items) {
      const li = el("li");
      li.append(el("strong", it.label), el("div", it.value));
      const files = el("div", null, "files");
      for (const f of it.files) {
        const a = el("a", f); a.href = "/evidence/" + encodeURIComponent(f); a.target = "_blank"; a.rel = "noopener noreferrer";
        files.appendChild(a);
      }
      li.appendChild(files);
      ul.appendChild(li);
    }
  }

  // ------------------------------------------------------------ Run tab

  function stopLabel(d) {
    return { eos: ["Finished", "done"], limit: ["Stopped at the token limit", "done"], cancelled: ["Cancelled", "cancel"],
             error: ["Error", "failed"] }[d.stop_reason] || [d.stop_reason, "done"];
  }

  async function startRun() {
    setError("run-error", "");
    const r = { requestId: null, text: "", synthetic: pick.source === "synthetic", decodeTime: 0, decodeTokens: 0, phase: null,
                done: null, start: null, budget: [], steps: [] };
    ui.run = r;
    ["m-ttft", "m-decode", "m-mem"].forEach((id) => { $(id).textContent = "—"; });
    $("run-answer").textContent = "";
    $("run-answer").appendChild(el("span", "Waiting for the first token…", "placeholder"));
    setKV($("run-details"), []);
    $("run-details-empty").hidden = false; $("download-run").disabled = true;
    $("run-confirm").hidden = true; $("run-confirm").textContent = "";
    $("budget-card").hidden = true; $("budget-msg").textContent = ""; $("budget-cache").textContent = "Cache: —";
    setState("run-state", "Starting…", "busy");
    ui.busy = true; refreshButtons();
    let resp;
    try {
      resp = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: $("run-prompt").value, source: pick.source, mode: pick.mode, ...settings() }) });
    } catch (e) { return endRun("Could not reach the server: " + e); }
    if (!resp.ok) {
      let msg = "Request refused (" + resp.status + ")";
      try { msg = (await resp.json()).error; } catch (e) { /* keep */ }
      return endRun(msg);
    }
    $("run-cancel").disabled = false;
    try { await readEvents(resp, onRunEvent); } catch (e) { setError("run-error", "The stream ended unexpectedly: " + e); }
    endRun();
  }

  function endRun(errorText) {
    if (errorText) { setError("run-error", errorText); setState("run-state", "Idle", "idle"); }
    ui.busy = false; refreshButtons(); $("run-cancel").disabled = true;
    $("budget-card").hidden = true;
    if (ui.run) ui.run.requestId = null;
    loadStatus();
  }

  function onRunEvent(ev) {
    const r = ui.run;
    if (ev.type === "accepted") { r.requestId = ev.request_id; setState("run-state", "Loading model…", "busy"); return; }
    if (ev.type === "start") {
      r.phase = "loading";
      if (ev.live_budget) { $("budget-card").hidden = false; }
      if (ev.cache) $("budget-cache").textContent = `Cache: ${ev.cache.slots} experts (${fmt(ev.cache.gb)} GB)`;
      r.start = ev;
      $("run-confirm").textContent = `Live engine run · ${SOURCE_SHORT[ev.source]} · ${MODE_SHORT[ev.mode]}` +
        (ev.source === "synthetic" ? " · meaningless output" : "");
      $("run-confirm").hidden = false;
      showRunDetails(r);
      return;
    }
    if (ev.type === "phase") { r.phase = ev.phase; if (!r.cancelling) setState("run-state", "Reading the prompt (prefill)…", "busy"); return; }
    if (ev.type === "token") {
      r.phase = "decode";
      if (!r.cancelling) setState("run-state", "Generating…", "busy");
      if (ev.index === 0) $("m-ttft").textContent = fmt(ev.step_s, 1) + " s";
      else { r.decodeTime += ev.step_s; r.decodeTokens += 1; $("m-decode").textContent = fmt(r.decodeTokens / r.decodeTime, 2) + " tok/s"; }
      $("m-mem").textContent = fmt(ev.memory.private_gb, 2) + " GB";
      if (ev.cache) $("budget-cache").textContent = `Cache: ${ev.cache.slots} experts (${fmt(ev.cache.gb)} GB)`;
      r.text += ev.text;
      r.steps.push({ index: ev.index, step_s: ev.step_s, private_gb: ev.memory.private_gb, cache_slots: ev.cache ? ev.cache.slots : null });
      renderMath($("run-answer"), r.text, r.synthetic);
      return;
    }
    if (ev.type === "budget") {
      r.budget.push(ev.event);
      const b = ev.event;
      $("budget-msg").textContent = `Applied at token ${b.token}: ${b.slots_before} → ${b.slots_after} experts (${b.reason})`;
      return;
    }
    if (ev.type === "done") {
      r.done = ev;
      r.text = ev.text;
      renderMath($("run-answer"), ev.text, r.synthetic);
      $("m-ttft").textContent = ev.time_to_first_token_s === null ? "—" : fmt(ev.time_to_first_token_s, 1) + " s";
      $("m-decode").textContent = ev.decode_tokens_per_s === null ? "—" : fmt(ev.decode_tokens_per_s, 2) + " tok/s";
      $("m-mem").textContent = fmt(ev.memory.peak_private_gb, 2) + " GB peak";
      const [label, kind] = stopLabel(ev);
      let text = label;
      if (ev.stop_reason === "cancelled" && ev.cancel_delay_s !== null) text += ` (took ${fmt(ev.cancel_delay_s, 1)} s, during ${ev.cancel_phase})`;
      setState("run-state", text, kind);
      if (ev.error) setError("run-error", ev.error);
      showRunDetails(r);
      $("download-run").disabled = false;
      return;
    }
    if (ev.type === "error") { setError("run-error", ev.message); setState("run-state", "Error", "failed"); }
  }

  function showRunDetails(r) {
    const s = r.start, d = r.done;
    if (!s) return;
    $("run-details-empty").hidden = true;
    const rows = [
      ["Request ID", s.request_id],
      ["Prompt (as submitted)", s.prompt],
      ["Model", SOURCE_LONG[s.source]],
      ["Execution", MODE_LONG[s.mode]],
      ["Settings", `max ${s.max_new_tokens} new tokens, context ${s.max_seq}` +
        (s.source === "real" ? `, budget ${s.budget_gb} GB, starting cache ${s.start_cache_gb} GB` : "")],
      ["Decoding", s.decoding],
      ["Threads (OpenBLAS / numba)", `${s.threads.openblas} / ${s.threads.numba}`],
      ["Model load", s.model_reused ? "reused (already loaded)" : `${fmt(s.load_seconds, 1)} s`],
    ];
    if (d) {
      const t = d.totals;
      rows.push(
        ["Generated tokens", `${d.output_tokens} (prompt ${d.prompt_tokens} tokens)`],
        ["Stop reason", d.stop_reason + (d.error ? `: ${d.error}` : "") +
          (d.stop_reason === "cancelled" && d.cancel_delay_s !== null ? ` (took ${fmt(d.cancel_delay_s, 1)} s, during ${d.cancel_phase})` : "")],
        ["First token / decode / total", `${fmt(d.time_to_first_token_s, 2)} s / ${fmt(d.decode_tokens_per_s, 2)} tok/s / ${fmt(d.total_seconds, 1)} s`],
        ["Peak memory (sampled every 0.1 s)", `private ${fmt(d.memory.peak_private_gb)} GB, working set ${fmt(d.memory.peak_working_set_gb)} GB`],
      );
      if (d.cache_at_start) rows.push(["Expert cache", `${d.cache_at_start.slots} → ${d.cache_at_end.slots} experts; ` +
        (d.cache_at_start.cached_experts ? `${d.cache_at_start.cached_experts} were still loaded from an earlier run (warm)` : "started empty")]);
      for (const b of r.budget) rows.push(["Budget change", `${b.budget_gb} GB at token ${b.token}: ${b.slots_before} → ${b.slots_after} experts`]);
      rows.push(["Experts used / cache hits / prefetch hits / blocking reads", `${t.expert_loads} / ${t.cache_hits} / ${t.prefetch_hits} / ${t.demand_reads}`]);
    }
    setKV($("run-details"), rows);
  }

  $("download-run").addEventListener("click", () => {
    const r = ui.run;
    if (!r || !r.start || !r.done) return;
    const record = {
      kind: "leanmoe-dashboard-run",
      note: "Live run on this computer, as reported by the LeanMoE server. Model output is unverified.",
      downloaded_at: new Date().toISOString(),
      request: r.start,
      budget_events: r.budget,
      per_token: r.steps,
      result: r.done,
    };
    const blob = new Blob([JSON.stringify(record, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `leanmoe-run-${r.start.request_id.slice(0, 8)}.json`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  });

  async function cancel(which) {
    const r = which === "run" ? ui.run : ui.cmp;
    if (!r || !r.requestId) return;
    const stateId = which === "run" ? "run-state" : "cmp-state", errId = which === "run" ? "run-error" : "cmp-error";
    const res = await postJSON("/api/cancel", { request_id: r.requestId });
    if (!res.ok) { setError(errId, res.body.error || "Cancel failed"); return; }
    r.cancelling = true;
    const waiting = res.body.phase === "loading" || res.body.phase === "prefill";
    setState(stateId, waiting ? `Cancelling: waits for ${res.body.phase} to finish…` : "Cancelling…", "cancel");
  }

  $("live-budget").addEventListener("input", () => { $("live-budget-value").textContent = $("live-budget").value + " GB"; });
  $("apply-budget").addEventListener("click", async () => {
    const r = ui.run;
    if (!r || !r.requestId) return;
    const res = await postJSON("/api/budget", { request_id: r.requestId, budget_gb: Number($("live-budget").value) });
    $("budget-msg").textContent = res.ok ? `Requested ${res.body.budget_gb} GB: applies after the current token` : res.body.error;
  });

  // ------------------------------------------------------------ Compare tab

  const MODE_NAME = { cached: "LeanMoE", mmap: "Windows paging" };
  function stepState(run, text, cls) {
    const li = document.querySelector(`#cmp-steps li[data-run="${run}"]`);
    li.querySelector(".step-state").textContent = text;
    li.className = cls || "";
  }
  function cmpCard(run) { return document.querySelector(`.side-by-side .card[data-run="${run}"]`); }

  async function startCompare() {
    setError("cmp-error", "");
    const c = { requestId: null, synthetic: pick["cmp-source"] === "synthetic", text: {}, decode: {} };
    ui.cmp = c;
    for (const run of ["cached", "mmap"]) {
      stepState(run, "Waiting");
      const card = cmpCard(run);
      card.querySelectorAll("b").forEach((b) => { b.textContent = "—"; });
      const box = card.querySelector(".answer-text"); box.textContent = ""; box.appendChild(el("span", "Waiting…", "placeholder"));
    }
    $("cmp-match").hidden = true; setKV($("cmp-details"), []);
    $("cmp-confirm").hidden = true; $("cmp-confirm").textContent = "";
    setState("cmp-state", "Starting…", "busy");
    ui.busy = true; refreshButtons();
    let resp;
    try {
      resp = await fetch("/api/compare", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: $("cmp-prompt").value, source: pick["cmp-source"], mode: "cached", ...settings() }) });
    } catch (e) { return endCompare("Could not reach the server: " + e); }
    if (!resp.ok) {
      let msg = "Request refused (" + resp.status + ")";
      try { msg = (await resp.json()).error; } catch (e) { /* keep */ }
      return endCompare(msg);
    }
    $("cmp-cancel").disabled = false;
    try { await readEvents(resp, onCompareEvent); } catch (e) { setError("cmp-error", "The stream ended unexpectedly: " + e); }
    endCompare();
  }

  function endCompare(errorText) {
    if (errorText) { setError("cmp-error", errorText); setState("cmp-state", "Idle", "idle"); }
    ui.busy = false; refreshButtons(); $("cmp-cancel").disabled = true;
    if (ui.cmp) ui.cmp.requestId = null;
    loadStatus();
  }

  function onCompareEvent(ev) {
    const c = ui.cmp, run = ev.run;
    if (ev.type === "accepted") { c.requestId = ev.request_id; setState("cmp-state", "Running both modes, one after the other…", "busy"); return; }
    if (ev.type === "start") {
      c.text[run] = ""; c.decode[run] = { t: 0, n: 0 }; stepState(run, "Loading model…", "active");
      $("cmp-confirm").textContent = `Live engine run · ${SOURCE_SHORT[ev.source]} · now ${MODE_SHORT[ev.mode]}` +
        (ev.source === "synthetic" ? " · meaningless output" : "");
      $("cmp-confirm").hidden = false;
      return;
    }
    if (ev.type === "phase") { stepState(run, "Prefill…", "active"); return; }
    if (ev.type === "token") {
      stepState(run, `Generating (${ev.index + 1} tokens)`, "active");
      const card = cmpCard(run);
      if (ev.index === 0) card.querySelector(".v-ttft").textContent = fmt(ev.step_s, 1) + " s";
      else { c.decode[run].t += ev.step_s; c.decode[run].n += 1; card.querySelector(".v-decode").textContent = fmt(c.decode[run].n / c.decode[run].t, 2) + " tok/s"; }
      c.text[run] += ev.text;
      renderMath(card.querySelector(".answer-text"), c.text[run], c.synthetic);
      return;
    }
    if (ev.type === "done") {
      const card = cmpCard(run);
      renderMath(card.querySelector(".answer-text"), ev.text, c.synthetic);
      card.querySelector(".v-ttft").textContent = ev.time_to_first_token_s === null ? "—" : fmt(ev.time_to_first_token_s, 1) + " s";
      card.querySelector(".v-decode").textContent = ev.decode_tokens_per_s === null ? "—" : fmt(ev.decode_tokens_per_s, 2) + " tok/s";
      card.querySelector(".v-mem").textContent = fmt(ev.memory.peak_private_gb, 2) + " GB";
      stepState(run, stopLabel(ev)[0], "finished");
      if (ev.error) setError("cmp-error", ev.error);
      c[run + "Done"] = ev;
      return;
    }
    if (ev.type === "compare_done") {
      const m = $("cmp-match");
      m.hidden = false;
      if (ev.token_ids_identical === null) { m.className = "match na"; m.textContent = "Token match: not available (a run did not complete)"; }
      else if (ev.token_ids_identical) { m.className = "match same"; m.textContent = "Same tokens in both modes"; }
      else { m.className = "match diff"; m.textContent = "Different tokens in the two modes"; }
      setState("cmp-state", c.cancelling ? "Cancelled" : "Comparison complete", c.cancelling ? "cancel" : "done");
      $("cmp-confirm").textContent = `Live engine run · ${SOURCE_SHORT[pick["cmp-source"]]} · ` +
        ev.completed_runs.map((m) => MODE_SHORT[m]).join(", then ") + (c.synthetic ? " · meaningless output" : "");
      for (const run of ["cached", "mmap"]) if (!ev.completed_runs.includes(run)) stepState(run, "Not run");
      const rows = [];
      for (const r of ev.runs) {
        rows.push([`${MODE_NAME[r.mode]}: model load`, `${fmt(r.load_seconds, 1)} s (fresh load)`]);
        rows.push([`${MODE_NAME[r.mode]}: peak working set (sampled)`, `${fmt(r.memory.peak_working_set_gb)} GB`]);
        rows.push([`${MODE_NAME[r.mode]}: output tokens, stop`, `${r.output_tokens}, ${r.stop_reason}`]);
      }
      setKV($("cmp-details"), rows);
      return;
    }
    if (ev.type === "error") { setError("cmp-error", ev.message); setState("cmp-state", "Error", "failed"); }
  }

  // ------------------------------------------------------------ wiring

  $("run").addEventListener("click", startRun);
  $("run-cancel").addEventListener("click", () => cancel("run"));
  $("compare").addEventListener("click", startCompare);
  $("cmp-cancel").addEventListener("click", () => cancel("cmp"));
  const initial = location.hash.replace("#", "");
  showTab(tabs.includes(initial) ? initial : "run");
  updateSelection();
  if (window.matchMedia("(max-width: 860px)").matches) $("explain").open = false;
  loadStatus();
})();
