// LeanMoE dashboard page logic. LeanMoE addition.
// All prompt and model text is untrusted: it is inserted with textContent, and
// math segments go through KaTeX with trust disabled. No innerHTML with data.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const state = { status: null, requestId: null, running: false, liveBudget: false, outputs: {}, renderQueued: {} };

  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined && text !== null) e.textContent = String(text);
    if (cls) e.className = cls;
    return e;
  }
  const fmt = (v, d = 2, unit = "") => (v === null || v === undefined || Number.isNaN(v)) ? "n/a" : Number(v).toFixed(d) + unit;

  function setMessage(text, isError) {
    const m = $("run-message");
    m.textContent = text || "";
    m.className = "message" + (isError ? " error" : "");
  }

  // ---------------------------------------------------------------- rendering

  function renderOutput(key) {
    const o = state.outputs[key];
    if (!o) return;
    const box = o.textEl;
    box.textContent = "";
    if (o.synthetic) { box.textContent = o.text; return; }
    for (const seg of window.LeanMoEMath.splitMath(o.text)) {
      if (seg.type === "text") { box.appendChild(document.createTextNode(seg.value)); continue; }
      const span = el(seg.display ? "div" : "span");
      try {
        window.katex.render(seg.value, span, {
          displayMode: seg.display, throwOnError: false, trust: false, strict: "ignore",
          maxSize: 10, maxExpand: 100, output: "htmlAndMathml",
        });
      } catch (e) {
        span.textContent = (seg.display ? "$$" : "$") + seg.value + (seg.display ? "$$" : "$");
      }
      box.appendChild(span);
    }
  }
  function queueRender(key) {
    if (state.renderQueued[key]) return;
    state.renderQueued[key] = true;
    requestAnimationFrame(() => { state.renderQueued[key] = false; renderOutput(key); });
  }

  function newOutput(key, title, synthetic) {
    const block = el("div", null, "output-block");
    const head = el("div", null, "output-head");
    head.appendChild(document.createTextNode(title));
    if (synthetic) {
      const tag = el("span", "synthetic: random weights, meaningless token ids", "tag synthetic");
      head.appendChild(tag);
    }
    const text = el("div", null, "output-text");
    block.append(head, text);
    $("outputs").appendChild(block);
    state.outputs[key] = { text: "", textEl: text, synthetic: synthetic };
  }

  function table(rows) {
    const t = el("table");
    for (const [k, v] of rows) {
      const tr = el("tr");
      tr.append(el("th", k), el("td", v));
      t.appendChild(tr);
    }
    return t;
  }

  // ---------------------------------------------------------------- status

  async function loadStatus() {
    try {
      const r = await fetch("/api/status");
      const s = await r.json();
      state.status = s;
      const box = $("status");
      box.textContent = "";
      const store = s.real_store;
      const parts = [
        `Real model store: ${store.available ? "available (" + store.path + ")" : "not available"}`,
        `OpenBLAS threads ${s.threads.openblas}, numba threads ${s.threads.numba}`,
        `Loaded: ${s.loaded ? s.loaded.source + " / " + s.loaded.mode : "nothing"}`,
        `Busy: ${s.busy ? "yes (" + s.active.kind + ", " + s.active.phase + ")" : "no"}`,
        `This process: private ${fmt(s.memory.private_gb)} GB, working set ${fmt(s.memory.working_set_gb)} GB; system RAM free ${fmt(s.memory.system_available_gb)} of ${fmt(s.memory.system_total_gb)} GB`,
      ];
      box.appendChild(el("div", parts.join("  |  ")));
      if (!store.available) box.appendChild(el("div", store.problem, "problem"));
      const realOpt = $("source").querySelector('option[value="real"]');
      realOpt.disabled = !store.available;
      if (!store.available && $("source").value === "real") $("source").value = "synthetic";
      updateSourceNote();
      renderRecorded(s.recorded);
    } catch (e) {
      $("status").textContent = "Could not reach the dashboard server: " + e;
    }
  }

  function renderRecorded(items) {
    const ul = $("recorded");
    ul.textContent = "";
    if (!items || !items.length) { ul.appendChild(el("li", "No recorded result files found in results/.")); return; }
    for (const it of items) {
      const li = el("li");
      li.appendChild(el("strong", it.label + ": "));
      li.appendChild(document.createTextNode(it.value + " "));
      for (const f of it.files) {
        const a = el("a", f);
        a.href = "/evidence/" + encodeURIComponent(f);
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        li.append(document.createTextNode(" "), a);
      }
      ul.appendChild(li);
    }
  }

  function updateSourceNote() {
    const synthetic = $("source").value === "synthetic";
    document.querySelectorAll(".real-only").forEach((n) => { n.style.display = synthetic ? "none" : ""; });
    $("source-note").textContent = synthetic
      ? "Synthetic: a tiny random-weight model built in a temporary folder. It runs the real engine code (store reads, cache, prefetch, cancellation) but its output is meaningless token ids. It is never used in place of the real model."
      : "Real model: answers take about 10 s before the first token and roughly 2 tokens per second on the development laptop.";
  }

  // ---------------------------------------------------------------- running

  function setRunning(on) {
    state.running = on;
    $("run").disabled = on; $("compare").disabled = on; $("cancel").disabled = !on;
    $("live-budget").disabled = !(on && state.liveBudget);
    $("apply-budget").disabled = !(on && state.liveBudget);
  }

  function body() {
    return {
      prompt: $("prompt").value, source: $("source").value, mode: $("mode").value,
      max_new_tokens: Number($("max-new").value), budget_gb: Number($("budget").value),
      start_cache_gb: Number($("start-cache").value),
    };
  }

  async function start(kind) {
    setMessage("");
    $("outputs").textContent = ""; $("metrics").textContent = ""; $("compare-result").textContent = "";
    state.outputs = {}; state.liveBudget = false;
    let resp;
    try {
      resp = await fetch(kind === "compare" ? "/api/compare" : "/api/generate", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()),
      });
    } catch (e) { setMessage("Request failed: " + e, true); return; }
    if (!resp.ok) {
      let msg = resp.status + "";
      try { msg = (await resp.json()).error; } catch (e) { /* keep status */ }
      setMessage(msg, true);
      loadStatus();
      return;
    }
    setRunning(true);
    try { await readStream(resp); } catch (e) { setMessage("Stream ended unexpectedly: " + e, true); }
    setRunning(false);
    state.requestId = null;
    loadStatus();
  }

  async function readStream(resp) {
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data: ")) handle(JSON.parse(line.slice(6)));
        }
      }
    }
  }

  const live = {};
  function handle(ev) {
    const key = ev.run || "single";
    switch (ev.type) {
      case "accepted":
        state.requestId = ev.request_id;
        setMessage("Loading the model if needed ...");
        break;
      case "start": {
        state.liveBudget = !!ev.live_budget;
        setRunning(true);
        const title = `${ev.source === "synthetic" ? "Synthetic" : "Real model"}, ${ev.mode === "cached" ? "cached + prefetch" : "memory-mapped"}`;
        newOutput(key, title, ev.source === "synthetic");
        live[key] = { start: ev, tokens: 0, last: null, budget: [], t0: performance.now() };
        renderLive();
        break;
      }
      case "phase":
        setMessage(ev.phase === "prefill" ? "Prefill: processing the prompt (the first forward pass; it can't be interrupted) ..." : "");
        break;
      case "token": {
        const o = state.outputs[key];
        if (o) { o.text += ev.text; queueRender(key); }
        const l = live[key];
        if (l) { l.tokens = ev.index + 1; l.last = ev; }
        setMessage("Generating ...");
        renderLive();
        break;
      }
      case "budget": {
        const l = live[key];
        if (l) l.budget.push(ev);
        renderLive();
        break;
      }
      case "done": {
        const o = state.outputs[key];
        if (o) { o.text = ev.text; queueRender(key); }
        const l = live[key];
        if (l) l.done = ev;
        renderLive();
        const reason = { eos: "finished (end of answer)", limit: "stopped at the token limit", cancelled: "cancelled", error: "error" }[ev.stop_reason] || ev.stop_reason;
        let msg = `Run ${reason}.`;
        if (ev.stop_reason === "cancelled" && ev.cancel_delay_s !== null) msg += ` The cancel took ${fmt(ev.cancel_delay_s)} s to take effect (requested during ${ev.cancel_phase}).`;
        if (ev.error) msg += " " + ev.error;
        setMessage(msg, ev.stop_reason === "error");
        break;
      }
      case "compare_done": {
        const box = $("compare-result");
        box.textContent = "";
        box.appendChild(el("h3", "Comparison (live, this computer)"));
        const rows = [["Runs completed", ev.completed_runs.join(", ") || "none"],
                      ["Generated token ids identical in both modes", ev.token_ids_identical === null ? "n/a (a run did not complete)" : (ev.token_ids_identical ? "yes" : "NO")]];
        for (const r of ev.runs) {
          rows.push([`${r.mode}: first token / decode / load`, `${fmt(r.time_to_first_token_s)} s / ${fmt(r.decode_tokens_per_s)} tokens/s / ${fmt(r.load_seconds)} s`]);
          rows.push([`${r.mode}: sampled peak private / working set`, `${fmt(r.memory.peak_private_gb)} / ${fmt(r.memory.peak_working_set_gb)} GB`]);
        }
        box.appendChild(table(rows));
        box.appendChild(el("p", "One run of each, one after the other, each from a fresh load. Memory-mapped speed depends on what Windows already has in its file cache, so the order and earlier runs affect it. The recorded benchmark below used repeated, alternating runs.", "note"));
        break;
      }
      case "error":
        setMessage(ev.message, true);
        break;
    }
  }

  function renderLive() {
    const box = $("metrics");
    box.textContent = "";
    for (const key of Object.keys(live)) {
      const l = live[key];
      const s = l.start, last = l.last, d = l.done;
      box.appendChild(el("h3", `${s.source === "synthetic" ? "Synthetic" : "Real model"}, ${s.mode}`));
      const rows = [
        ["Model", s.model_reused ? "reused (already loaded)" : `loaded in ${fmt(s.load_seconds)} s`],
        ["Prompt tokens / output tokens so far", `${s.prompt_tokens} / ${l.tokens} (limit ${s.max_new_tokens}, context ${s.max_seq})`],
        ["Elapsed", `${fmt((performance.now() - l.t0) / 1000, 1)} s`],
      ];
      if (s.cache) rows.push(["Expert cache at start", `${s.cache.slots} experts (${fmt(s.cache.gb)} GB), ${s.cache.cached_experts} already holding an expert`]);
      if (last) {
        rows.push(["Last step", `${fmt(last.step_s, 3)} s`]);
        rows.push(["Process memory now", `private ${fmt(last.memory.private_gb)} GB, working set ${fmt(last.memory.working_set_gb)} GB`]);
        if (last.cache) rows.push(["Expert cache now", `${last.cache.slots} experts (${fmt(last.cache.gb)} GB)`]);
        const t = last.totals;
        rows.push(["Experts used / cache hits / prefetch hits / blocking reads", `${t.expert_loads} / ${t.cache_hits} / ${t.prefetch_hits} / ${t.demand_reads}`]);
      }
      for (const b of l.budget) rows.push(["Budget change", `${b.event.budget_gb} GB at token ${b.event.token}: cache ${b.event.slots_before} -> ${b.event.slots_after} experts (${b.event.reason})`]);
      if (d) {
        rows.push(["Stop reason", d.stop_reason + (d.error ? `: ${d.error}` : "")]);
        rows.push(["Time to first token", `${fmt(d.time_to_first_token_s)} s`]);
        rows.push(["Decode rate", `${fmt(d.decode_tokens_per_s)} tokens/s`]);
        rows.push(["Sampled peak memory during the run", `private ${fmt(d.memory.peak_private_gb)} GB, working set ${fmt(d.memory.peak_working_set_gb)} GB (${d.memory.samples} samples)`]);
      }
      box.appendChild(table(rows));
    }
  }

  async function cancel() {
    if (!state.requestId) return;
    try {
      const r = await fetch("/api/cancel", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ request_id: state.requestId }) });
      const j = await r.json();
      setMessage(r.ok ? j.message : j.error, !r.ok);
    } catch (e) { setMessage("Cancel failed: " + e, true); }
  }

  async function applyBudget() {
    if (!state.requestId) return;
    try {
      const r = await fetch("/api/budget", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ request_id: state.requestId, budget_gb: Number($("live-budget").value) }) });
      const j = await r.json();
      setMessage(r.ok ? `Budget ${j.budget_gb} GB requested. ${j.message}` : j.error, !r.ok);
    } catch (e) { setMessage("Budget change failed: " + e, true); }
  }

  $("run").addEventListener("click", () => start("generate"));
  $("compare").addEventListener("click", () => start("compare"));
  $("cancel").addEventListener("click", cancel);
  $("apply-budget").addEventListener("click", applyBudget);
  $("source").addEventListener("change", updateSourceNote);
  loadStatus();
})();
