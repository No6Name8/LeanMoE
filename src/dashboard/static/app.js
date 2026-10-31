// LeanMoE dashboard page logic. LeanMoE addition.
// All prompt and model text is untrusted: it is inserted with textContent, and
// math segments go through KaTeX with trust disabled. No innerHTML with data.
// Every number shown comes from the server's measurements; nothing is estimated here.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "—" : Number(v).toFixed(d);
  const pick = { source: "real", mode: "cached", "cmp-source": "real", location: "local" };
  const ui = { status: null, run: null, cmp: null, qwenMissing: true, busy: false, connected: false, hostLocation: null, savedSettings: {} };
  const api = (path, location = pick.location) => location === "cloud" && ui.hostLocation !== "server" ? "/cloud" + path : path;
  const locationName = (location = pick.location) => location === "cloud" ? "cloud server" : "this computer";

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

  async function postJSON(url, body, location) {
    const r = await fetch(api(url, location), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
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
    ui.tab = name;
    sysDetails.open = false;
    $("execution-bar").hidden = name === "evidence";
    $("settings").hidden = name === "evidence";
    $("engine-availability").hidden = name === "evidence" || !ui.status?.access?.protected;
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
      if (g === "location" && b.dataset.value !== pick.location) {
        ui.savedSettings[pick.location] = settings();
      }
      document.querySelectorAll(`.segmented button[data-group="${g}"]`).forEach((o) => o.setAttribute("aria-checked", String(o === b)));
      pick[g] = b.dataset.value;
      if (g === "location") {
        const saved = ui.savedSettings[pick.location] || (pick.location === "cloud" ?
          {profile: "auto", max_new_tokens: 64, budget_gb: 4.0, start_cache_gb: 1.3} : null);
        if (saved) {
          $("memory-profile").value = saved.profile; $("max-new").value = saved.max_new_tokens;
          $("budget").value = saved.budget_gb; $("start-cache").value = saved.start_cache_gb;
          profileChanged();
        }
        ui.status = null; ui.connected = false; ui.qwenMissing = true; ui.otherVisitorBusy = false;
        $("execution-help").textContent = pick.location === "cloud" ?
          "Connecting to the public cloud demo. No login or model download is needed." :
          "Local runs on this computer. Cloud runs on the public, bounded server.";
        updateSelection(); loadStatus(); return;
      }
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
    $("run-selection").textContent = `Model: ${SOURCE_SHORT[pick.source]} · Method: ${MODE_SHORT[pick.mode]} · ${locationName()}`;
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
    $("run").disabled = !ui.connected || ui.busy || ui.otherVisitorBusy || runBlocked;
    $("compare").disabled = !ui.connected || ui.busy || ui.otherVisitorBusy || cmpBlocked;
    document.querySelectorAll('[data-group="location"]').forEach((b) => { b.disabled = ui.busy || ui.hostLocation === null || (b.dataset.value === "local" && ui.hostLocation === "server"); });
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
    return { profile: $("memory-profile").value, max_new_tokens: Number($("max-new").value), budget_gb: Number($("budget").value),
             start_cache_gb: Number($("start-cache").value) };
  }
  function profileChanged() {
    const automatic = $("memory-profile").value === "auto";
    $("budget").disabled = automatic; $("start-cache").disabled = automatic;
    $("budget").closest("label").hidden = automatic;
    $("start-cache").closest("label").hidden = automatic;
    if ($("memory-profile").value === "constrained") { $("budget").value = "3.5"; $("start-cache").value = "1.2"; }
    $("profile-help").textContent = automatic ?
      "Automatic measures physical RAM and Windows commit headroom at each model load, reserves runtime allocations and safety headroom, then selects the cache. Effective values appear in Run details." :
      "Explicit application planning budget and expert cache. Constrained adaptation retains the original 3.5 GB / 1.2 GB starting settings. Public cloud manual requests are limited to 4 GB budget and 1.3 GB cache. Neither setting limits Windows file cache or total RAM. Requests that do not fit safe headroom are refused.";
  }
  $("memory-profile").addEventListener("change", profileChanged);
  profileChanged();

  // ------------------------------------------------------------ status

  let statusPending = false;
  let disconnected = false;
  function renderSystem() {
    const v = window.LeanMoESystem.view(ui.status, disconnected);
    setKV($("system-details"), v.rows);
    setKV($("system-technical-values"), v.technical);
    $("system-updated").textContent = v.updated;
    $("system-updated").className = v.warning ? "hint system-warning" : "hint";
  }
  async function loadStatus() {
    if (statusPending) return;
    statusPending = true;
    const requestedLocation = pick.location;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    let s;
    try {
      const response = await fetch(api("/api/status", requestedLocation), {signal: controller.signal});
      if (!response.ok) {
        let detail = "Backend status unavailable.";
        try { detail = (await response.json()).error || detail; } catch (e) { /* empty response */ }
        throw new Error(detail);
      }
      s = await response.json();
      if (!s.system || !s.real_store || !s.limits) throw new Error("invalid status");
      if (requestedLocation === "cloud" && (s.system.execution_location !== "server" || !s.access?.public_demo))
        throw new Error("Cloud backend did not confirm public server execution.");
      disconnected = false;
    } catch (e) {
      disconnected = true;
      if (requestedLocation === pick.location) {
        ui.connected = false; ui.status = null;
        $("store-pill").textContent = requestedLocation === "cloud" ? "Cloud disconnected" : "Local backend disconnected";
        $("store-pill").className = "pill no";
        $("execution-help").textContent = String(e.message || e);
        renderSystem(); refreshButtons();
      }
      return;
    } finally {
      clearTimeout(timeout); statusPending = false;
      if (requestedLocation !== pick.location) queueMicrotask(loadStatus);
    }
    if (requestedLocation !== pick.location) return;
    if (ui.hostLocation === null && requestedLocation === "local") {
      ui.hostLocation = s.system.execution_location;
      if (ui.hostLocation === "server") {
        pick.location = "cloud";
        document.querySelectorAll('[data-group="location"]').forEach((b) => b.setAttribute("aria-checked", String(b.dataset.value === "cloud")));
        $("execution-help").textContent = "Running on the public cloud server. Local execution requires a local installation; see the project README.";
      }
    }
    ui.connected = true;
    ui.status = s;
    if (pick.location === "cloud") $("execution-help").textContent =
      "Public cloud server connected. One active job; no queue. Cloud speedup is not established.";
    else $("execution-help").textContent = "Running on this computer. Cloud runs on the public, bounded server.";
    const protectedSite = Boolean(s.access?.protected);
    ui.otherVisitorBusy = protectedSite && s.busy && !s.active;
    $("engine-availability").hidden = !protectedSite || ui.tab === "evidence";
    $("engine-availability").textContent = ui.otherVisitorBusy ? "Engine busy with another visitor. No queue; retry when it is available." :
      "Public cloud demo. Automatic uses measured server headroom. Manual/constrained limits: 4 GB budget, 1.3 GB cache. One active job; no queue; 96 tokens per mode, 10 minutes per job, three starts per minute globally. Application budgets do not limit Windows file cache or VM RAM.";
    $("max-new").max = s.limits.max_new_tokens;
    if (Number($("max-new").value) > s.limits.max_new_tokens) $("max-new").value = s.limits.max_new_tokens;
    for (const prefix of ["run", "cmp"]) {
      $(prefix + "-advanced").hidden = protectedSite;
      if (protectedSite) $(prefix + "-engine-test").checked = false;
    }
    const ok = s.real_store.available;
    $("store-pill").textContent = ok ? "Qwen ready" : "Qwen store not found";
    $("store-pill").className = "pill " + (ok ? "ok" : "no");
    ui.qwenMissing = !ok;
    for (const p of ["run", "cmp"]) $(p + "-missing-detail").textContent = ok ? "" : s.real_store.problem;
    updateSelection();
    $("max-seq").textContent = s.limits.max_seq;
    const m = s.system || {};
    document.querySelector(".tagline").textContent = m.execution_location === "local" ? "Local engine dashboard" :
      m.execution_location === "server" ? "Server engine dashboard" : "Engine dashboard";
    $("live-location").textContent = m.execution_location === "local" ? "measured on this computer" :
      m.execution_location === "server" ? "measured on the cloud server" : "backend location unavailable";
    $("cmp-live-location").textContent = "Live comparison measured on " + locationName() + ".";
    renderSystem();
    // Avoid replacing open methodology panels when system data is refreshed.
    if (ui.hostLocation !== "local" || requestedLocation === "local") {
      const recordedKey = JSON.stringify(s.recorded);
      if (ui.recordedKey !== recordedKey) { renderRecorded(s.recorded); ui.recordedKey = recordedKey; }
    }
  }

  function renderRecorded(items) {
    $("recorded").textContent = "";
    $("additional-recorded").textContent = "";
    for (const it of items || []) {
      const card = el("article", null, "card experiment");
      const head = el("div", null, "answer-head");
      head.append(el("h3", it.label), el("span", "Recorded", "badge recorded"));
      card.appendChild(head);
      if (it.meaning) card.append(el("div", it.figure, "evidence-figure"), el("p", it.meaning, "figure-meaning"));
      card.appendChild(el("p", it.context || it.value, "result"));
      const details = el("details", null, "details compact");
      details.appendChild(el("summary", "Methodology, limits & reproduce"));
      details.appendChild(el("h4", "Settings"));
      const table = el("table", null, "settings-table");
      table.setAttribute("aria-label", it.label + " settings");
      const body = el("tbody");
      for (const [label, value] of it.settings || []) {
        const row = el("tr"); const th = el("th", label); th.scope = "row";
        row.append(th, el("td", value)); body.appendChild(row);
      }
      table.appendChild(body); details.appendChild(table);
      details.append(el("h4", "Limitations"), el("p", it.limitations, "hint"));
      details.appendChild(el("h4", "Source artifacts"));
      const files = el("div", null, "files");
      for (const [i, f] of it.files.entries()) {
        const a = el("a", it.file_labels?.[i] || "Source artifact"); a.href = "/evidence/" + encodeURIComponent(f); a.target = "_blank"; a.rel = "noopener noreferrer";
        files.appendChild(a);
      }
      details.append(files, el("h4", "Reproduction commands (Windows PowerShell)"),
        el("p", "Use the .venv created by setup. The saved-artifact checker needs no model and runs no inference. Real-model commands need the external int8 store; set LEANMOE_MODELS_ROOT in that shell. They create new result files and can take minutes. Reproduction on a second machine and rebuilding the store from this repository remain unverified.", "hint"));
      details.appendChild(el("pre", it.commands.join("\n")));
      if (it.additional) details.appendChild(el("p", "Run historical thread settings in a separate PowerShell window; close it afterwards to avoid changing your dashboard's thread settings.", "hint"));
      card.appendChild(details);
      $(it.additional ? "additional-recorded" : "recorded").appendChild(card);
    }
  }
  sysDetails.addEventListener("toggle", () => { if (sysDetails.open) loadStatus(); });
  setInterval(() => { if (sysDetails.open) loadStatus(); }, 5000);
  setInterval(() => { if (pick.location === "cloud" && !ui.busy && ui.otherVisitorBusy) loadStatus(); }, 5000);
  setInterval(() => { if (sysDetails.open) renderSystem(); }, 1000);

  // ------------------------------------------------------------ Run tab

  function stopLabel(d) {
    return { eos: ["Finished", "done"], limit: ["Stopped at the token limit", "done"], cancelled: ["Cancelled", "cancel"],
             error: ["Error", "failed"] }[d.stop_reason] || [d.stop_reason, "done"];
  }

  async function startRun() {
    setError("run-error", "");
    const r = { requestId: null, location: pick.location, text: "", synthetic: pick.source === "synthetic", decodeTime: 0, decodeTokens: 0, phase: null,
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
      resp = await fetch(api("/api/generate", r.location), { method: "POST", headers: { "Content-Type": "application/json" },
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
      $("run-confirm").textContent = `Live engine run on ${locationName(r.location)} · ${SOURCE_SHORT[ev.source]} · ${MODE_SHORT[ev.mode]}` +
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
      ["Memory profile", ({auto: "Automatic for this machine", manual: "Manual settings", constrained: "Constrained adaptation experiment"})[s.resource_profile?.profile || "manual"]],
      ["Effective planning budget", s.budget_gb == null ? "OS-managed paging; no RAM cap" : `${fmt(s.budget_gb)} GB`],
      ["Actual cache capacity", s.cache ? `${s.cache.slots} experts / ${fmt(s.cache.gb, 3)} GB` : "OS-managed mappings"],
      ["Safety reserve", s.resource_profile?.reserve_bytes ? `${fmt(s.resource_profile.reserve_bytes / 1e9)} GB` : "unavailable"],
      ["Settings", `max ${s.max_new_tokens} new tokens, context ${s.max_seq}` +
        (s.source === "real" && s.budget_gb != null ? `, budget ${fmt(s.budget_gb)} GB, starting cache ${fmt(s.start_cache_gb, 3)} GB` : "")],
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
      note: "Live run reported by the LeanMoE backend. Model output is unverified.",
      execution_location: ui.status?.system?.execution_location || "unavailable",
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
    let res;
    try { res = await postJSON("/api/cancel", { request_id: r.requestId }, r.location); }
    catch (e) { setError(errId, "Could not reach the selected backend to cancel."); return; }
    if (!res.ok) { setError(errId, res.body.error || "Cancel failed"); return; }
    r.cancelling = true;
    const waiting = res.body.phase === "loading" || res.body.phase === "prefill";
    setState(stateId, waiting ? `Cancelling: waits for ${res.body.phase} to finish…` : "Cancelling…", "cancel");
  }

  $("live-budget").addEventListener("input", () => { $("live-budget-value").textContent = $("live-budget").value + " GB"; });
  $("apply-budget").addEventListener("click", async () => {
    const r = ui.run;
    if (!r || !r.requestId) return;
    let res;
    try { res = await postJSON("/api/budget", { request_id: r.requestId, budget_gb: Number($("live-budget").value) }, r.location); }
    catch (e) { $("budget-msg").textContent = "Could not reach the selected backend."; return; }
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
    const c = { requestId: null, location: pick.location, synthetic: pick["cmp-source"] === "synthetic", text: {}, decode: {} };
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
      resp = await fetch(api("/api/compare", c.location), { method: "POST", headers: { "Content-Type": "application/json" },
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
      $("cmp-confirm").textContent = `Live engine run on ${locationName(c.location)} · ${SOURCE_SHORT[ev.source]} · now ${MODE_SHORT[ev.mode]}` +
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
      $("cmp-confirm").textContent = `Live engine run on ${locationName(c.location)} · ${SOURCE_SHORT[pick["cmp-source"]]} · ` +
        ev.completed_runs.map((m) => MODE_SHORT[m]).join(", then ") + (c.synthetic ? " · meaningless output" : "");
      for (const run of ["cached", "mmap"]) if (!ev.completed_runs.includes(run)) stepState(run, "Not run");
      const rows = [];
      for (const r of ev.runs) {
        rows.push([`${MODE_NAME[r.mode]}: memory profile`, `${r.resource_profile?.profile || "manual"}; cache ${r.cache_at_start ? fmt(r.cache_at_start.gb, 3) + " GB" : "OS-managed"}`]);
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
