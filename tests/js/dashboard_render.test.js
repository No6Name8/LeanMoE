// Checks for the dashboard's math splitting and KaTeX settings (LeanMoE).
// Run by tests/test_dashboard_js.py when Node.js is installed:  node tests/js/dashboard_render.test.js
"use strict";
const assert = require("assert");
const path = require("path");
const STATIC = path.join(__dirname, "..", "..", "src", "dashboard", "static");
const { splitMath } = require(path.join(STATIC, "mathsplit.js"));
const katex = require(path.join(STATIC, "katex", "katex.min.js"));

const types = (t) => splitMath(t).map((s) => s.type + ":" + s.value);
let n = 0;
const check = (name, fn) => { fn(); n += 1; };

check("inline dollars", () => assert.deepStrictEqual(types("Solve $x^2 - 5x + 6 = 0$."), ["text:Solve ", "math:x^2 - 5x + 6 = 0", "text:."]));
check("prices stay text", () => assert.deepStrictEqual(types("Price $5 and $10 total"), ["text:Price $5 and $10 total"]));
check("display brackets", () => {
  const s = splitMath("Time: \\[ \\frac{180}{72} \\] h");
  assert.deepStrictEqual(s.map((x) => x.type), ["text", "math", "text"]);
  assert.strictEqual(s[1].display, true);
  assert.strictEqual(s[1].value.trim(), "\\frac{180}{72}");
});
check("inline parens", () => assert.deepStrictEqual(types("so \\(a+b\\) holds"), ["text:so ", "math:a+b", "text: holds"]));
check("double dollars", () => {
  const s = splitMath("I = $$\\int_0^1 x\\,dx$$.");
  assert.strictEqual(s[1].type, "math"); assert.strictEqual(s[1].display, true);
});
check("unclosed stays text while streaming", () => assert.deepStrictEqual(types("half open $x^2 + 1"), ["text:half open $x^2 + 1"]));
check("escaped dollar", () => assert.deepStrictEqual(types("cost \\$5"), ["text:cost $5"]));
check("html is just text", () => assert.deepStrictEqual(types("<script>alert(1)</script>"), ["text:<script>alert(1)</script>"]));

const opts = { throwOnError: false, trust: false, strict: "ignore", maxSize: 10, maxExpand: 100, output: "htmlAndMathml" };
check("katex escapes angle brackets", () => {
  const html = katex.renderToString("a<b", opts);
  assert.ok(!html.includes("<b") || html.includes("&lt;"), "raw < must be escaped");
  assert.ok(html.includes("&lt;"));
});
check("katex refuses links and raw html with trust off", () => {
  for (const tex of ["\\href{javascript:alert(1)}{x}", "\\url{javascript:alert(1)}", "\\htmlData{x=1}{y}", "\\includegraphics{http://example.com/a.png}"]) {
    const html = katex.renderToString(tex, opts);
    assert.ok(!/href=|<img|data-x/i.test(html), "untrusted command produced active HTML: " + tex);
  }
});
check("katex bad input does not throw", () => { katex.renderToString("\\frac{", opts); });

console.log(`dashboard render checks passed: ${n}`);
