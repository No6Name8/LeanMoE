"""Regenerate README's laptop decode chart from the recorded paging artifact.

Run: python docs/readme/decode_chart.py. Uses only the standard library.
The chart uses the arithmetic mean of *per-run decode rates*, like the dashboard.
"""
from collections import defaultdict
from html import escape
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent.parent / "results" / "config_compare_mmap_full_20261030-060448.json"
DEST = HERE / "laptop-decode.svg"


def figures():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert data["design"]["prompts"] == ["fact", "code", "explain"]
    assert data["design"]["tokens"] == 48 and data["design"]["rounds"] == 2
    by_prompt = defaultdict(lambda: defaultdict(list))
    for run in data["runs"]:
        assert run["mode"] in ("cached", "mmap") and run["error"] is None
        assert run["openblas_threads"] == 4 and run["decode_tokens"] == 47
        by_prompt[run["prompt_id"]][run["mode"]].append(run)
    assert len(data["runs"]) == 12
    for prompt in data["design"]["prompts"]:
        assert all(len(by_prompt[prompt][mode]) == 2 for mode in ("cached", "mmap"))
        ids = [r["generated_ids"] for mode in ("cached", "mmap") for r in by_prompt[prompt][mode]]
        assert all(x == ids[0] for x in ids), "mode output changed within a prompt"
    rows = []
    for prompt in (*data["design"]["prompts"], "overall"):
        runs = (data["runs"] if prompt == "overall" else
                [r for mode in ("cached", "mmap") for r in by_prompt[prompt][mode]])
        rows.append((prompt, *(mean(r["decode_tokens_per_s"] for r in runs if r["mode"] == mode)
                               for mode in ("cached", "mmap"))))
    return rows


def render(rows):
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="395" viewBox="0 0 520 395" role="img" aria-labelledby="title desc">',
        '<title id="title">Laptop mean decode throughput by prompt</title>',
        '<desc id="desc">Two runs per mode per prompt. LeanMoE has the higher mean in each prompt, but one warm paging run beats its matched LeanMoE run. Overall means: 1.842 versus 0.891 tokens per second.</desc>',
        '<rect width="520" height="395" rx="16" fill="#f8fafc"/>',
        '<text x="20" y="31" font-family="Arial,sans-serif" font-size="17" font-weight="700" fill="#172554">Laptop · mean decode throughput</text>',
        '<rect x="20" y="46" width="12" height="12" rx="3" fill="#0f766e"/>',
        '<text x="39" y="57" font-family="Arial,sans-serif" font-size="13" fill="#334155">LeanMoE</text>',
        '<rect x="148" y="46" width="12" height="12" rx="3" fill="#475db5"/>',
        '<text x="167" y="57" font-family="Arial,sans-serif" font-size="13" fill="#334155">Windows paging</text>',
    ]
    x0, scale = 152, 120
    for tick in range(6):
        x = x0 + tick * 60
        lines.append(f'<line x1="{x}" y1="78" x2="{x}" y2="330" stroke="#dbe4ee"/>')
        lines.append(f'<text x="{x}" y="349" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#475569">{tick / 2:g}</text>')
    for i, (name, cached, mmap) in enumerate(rows):
        y = 89 + i * 60
        label = {"fact": "Fact", "code": "Code", "explain": "Explain", "overall": "Overall"}[name]
        lines.append(f'<text x="18" y="{y + 23}" font-family="Arial,sans-serif" font-size="14" font-weight="600" fill="#172554">{escape(label)}</text>')
        for j, (value, color) in enumerate(((cached, "#0f766e"), (mmap, "#475db5"))):
            bar_y = y + j * 21
            width = round(value * scale, 2)
            lines.append(f'<rect x="{x0}" y="{bar_y}" width="{width}" height="15" rx="4" fill="{color}"/>')
            lines.append(f'<text x="{x0 + width + 6:.2f}" y="{bar_y + 12}" font-family="Arial,sans-serif" font-size="12" font-weight="600" fill="#172554">{value:.3f}</text>')
    lines += [
        '<text x="152" y="376" font-family="Arial,sans-serif" font-size="13" fill="#334155">Mean decode throughput (tokens/s) →</text>',
        '</svg>',
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    rows = figures()
    DEST.write_text(render(rows), encoding="utf-8")
    print("Wrote", DEST.relative_to(HERE.parent.parent), "from", SOURCE.name)
    for name, cached, mmap in rows:
        print(f"{name}: {cached:.3f} / {mmap:.3f} tokens/s")
