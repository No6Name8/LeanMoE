// Presentation only: backend measurements stay intact. Also exercised with Node.
(function (root) {
  "use strict";
  const available = (v) => v === null || v === undefined || v === "" ? "unavailable" : String(v);
  const gb = (v) => typeof v === "number" && Number.isFinite(v) ? `${v.toFixed(2)} GB` : "unavailable";
  const mode = (v) => ({cached: "LeanMoE", mmap: "Windows paging"}[v] || "unavailable");

  function processor(name) {
    if (!name) return "unavailable";
    return String(name).replace(/\b\d+(?:st|nd|rd|th) Gen\s*/gi, "")
      .replace(/\((?:R|TM)\)|[\u00ae\u2122]/gi, "").replace(/\s+@\s+[\d.]+\s*GHz.*$/i, "")
      .replace(/\s+/g, " ").trim();
  }

  function view(status, disconnected = false, now = Date.now()) {
    const m = status?.system || {};
    const timestamp = Date.parse(m.measured_at);
    const seconds = Number.isFinite(timestamp) ? Math.max(0, Math.floor((now - timestamp) / 1000)) : null;
    const stale = seconds === null || seconds >= 15;
    const age = seconds === null ? "Update time unavailable" : seconds < 5 ? "Updated just now" :
      `Updated ${seconds < 60 ? `${seconds}s` : seconds < 3600 ? `${Math.floor(seconds / 60)}m` : `${Math.floor(seconds / 3600)}h`} ago`;
    let state = !status ? "unavailable" : !status.busy ? "Idle" : !status.active ? "Busy with another visitor" : status.active?.phase === "loading" ? "Loading" : "Running";
    if (m.loaded_model && m.loaded_model !== "none") state += ` - ${m.loaded_model} / ${mode(m.execution_mode)}`;
    else if (status?.busy && status.active) state += ` - ${{real: "Qwen", synthetic: "Engine test"}[status.active.source] || "unavailable"} / ${mode(status.active.mode)}`;
    if (disconnected || stale) state = `${disconnected ? "Disconnected" : "Stale"} - last known: ${state}`;
    return {
      updated: `${disconnected ? "Disconnected. " : stale ? "Stale measurements. " : ""}${age}`,
      warning: disconnected || stale,
      rows: [
        ["Running on", {local: "This computer", server: "Cloud server"}[m.execution_location] || "unavailable"],
        ["Processor", processor(m.cpu)],
        ["RAM", `${gb(m.ram_total_gb)} total / ${gb(m.ram_available_gb)} available`],
        ["Engine memory", `${gb(m.process_private_gb)} (private committed)`],
        ["Status", state],
      ],
      technical: [
        ["Backend OS", available(m.os)], ["Logical CPU cores", available(m.logical_cores)],
        ["Windows commit - headroom", gb(m.windows_commit?.headroom_gb)],
        ["Windows commit - limit", gb(m.windows_commit?.limit_gb)],
        ["Process working set - in RAM", gb(m.process_working_set_gb)],
        ["Effective OpenBLAS threads", available(m.threads?.openblas)],
        ["Effective numba threads", available(m.threads?.numba)],
        ["Loaded execution mode", m.loaded_model === "none" ? "none" : mode(m.execution_mode)],
        ["Active request mode", status?.busy ? mode(m.active_mode) : "none"],
        ["Memory units", "GB = 1,000,000,000 bytes; snapshot, not peak. Private committed memory is the engine process allocation, not physical RAM residency. Commit headroom is system-wide allocation capacity, not free RAM."],
      ],
    };
  }
  root.LeanMoESystem = {view, processor};
  if (typeof module !== "undefined" && module.exports) module.exports = {view, processor};
})(typeof window !== "undefined" ? window : globalThis);
