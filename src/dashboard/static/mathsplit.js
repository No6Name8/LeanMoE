// Split untrusted text into plain-text and math segments, for safe rendering.
// LeanMoE addition. Pure function, no DOM: the page renders "text" segments with
// textContent and "math" segments with KaTeX (trust disabled). An unclosed
// delimiter (common while streaming) stays plain text until it is closed.
//
// Delimiters: $$...$$ and \[...\] (display), \(...\) and $...$ (inline).
// Inline $...$ needs a non-space right after the opening $ and right before the
// closing $, no line break inside, and no digit right after the closing $, so
// prices like "$5 and $10" stay text. "\$" is a literal dollar sign.
(function (root) {
  "use strict";

  function findClose(text, from, close) {
    let i = from;
    while (true) {
      const j = text.indexOf(close, i);
      if (j < 0) return -1;
      if (j > 0 && text[j - 1] === "\\" && close[0] === "$") { i = j + 1; continue; }
      return j;
    }
  }

  function splitMath(text) {
    const out = [];
    let buf = "";
    let i = 0;
    const flush = () => { if (buf) { out.push({ type: "text", value: buf }); buf = ""; } };
    while (i < text.length) {
      const c = text[i];
      if (c === "\\" && text[i + 1] === "$") { buf += "$"; i += 2; continue; }
      let open = null, close = null, display = false;
      if (text.startsWith("$$", i)) { open = "$$"; close = "$$"; display = true; }
      else if (text.startsWith("\\[", i)) { open = "\\["; close = "\\]"; display = true; }
      else if (text.startsWith("\\(", i)) { open = "\\("; close = "\\)"; }
      else if (c === "$") { open = "$"; close = "$"; }
      if (open) {
        const start = i + open.length;
        const end = findClose(text, start, close);
        if (end > start) {
          const body = text.slice(start, end);
          let ok = body.trim().length > 0;
          if (open === "$") {
            ok = ok && !/\s/.test(body[0]) && !/\s/.test(body[body.length - 1]) && !body.includes("\n") &&
                 !/[0-9]/.test(text[end + 1] || "");
          }
          if (ok) {
            flush();
            out.push({ type: "math", value: body, display: display });
            i = end + close.length;
            continue;
          }
        }
        buf += open;
        i += open.length;
        continue;
      }
      buf += c;
      i += 1;
    }
    flush();
    return out;
  }

  root.LeanMoEMath = { splitMath: splitMath };
  if (typeof module !== "undefined" && module.exports) module.exports = { splitMath: splitMath };
})(typeof window !== "undefined" ? window : globalThis);
