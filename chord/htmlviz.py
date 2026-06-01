"""Standalone HTML renderer for a Benchmark instance.

Produces a single self-contained .html file (no external assets) that shows
the schedule one KU at a time, with:
  - Definitions as colour-tagged DEF / REDEF rows
  - Examples and probes in MLC-style two-column tables, with colored circles
    for atom outputs (matching the original MLC few_shot HTML viz)
  - A header card with metadata + a small ascii-like schedule timeline
  - A SVG of the schedule timeline embedded inline

The colour vocabulary is the one used by `core.COLORS`. Anything that isn't
a known color renders as a black filled circle.
"""
from __future__ import annotations

import base64
import io
import html as _html
from typing import Iterable, Optional

from .core import KU, Probe, COLORS


_COLOR_HEX = {
    "RED":    "#e74c3c",
    "GREEN":  "#27ae60",
    "BLUE":   "#2980b9",
    "YELLOW": "#f1c40f",
    "PURPLE": "#8e44ad",
    "ORANGE": "#e67e22",
    "PINK":   "#ff69b4",
    "BROWN":  "#8b5a2b",
}


def _circle(token: str) -> str:
    color = _COLOR_HEX.get(token, "#222")
    label = _html.escape(token)
    return (f'<span class="circle" style="background:{color}" '
            f'title="{label}"></span>')


def _output_cell(out: str) -> str:
    toks = out.split()
    inner = "".join(_circle(t) for t in toks) if toks else "&nbsp;"
    return f'<span class="circles">{inner}</span>'


def _example_table(rows: Iterable, eval_at: Optional[int] = None) -> str:
    items = []
    for row in rows:
        if isinstance(row, Probe):
            inp, out = row.inp, row.out
            tag = f' <span class="badge">eval_at={row.eval_at}</span>'
        else:
            inp, out = row
            tag = ""
        items.append(
            f'<tr><td class="cmd">{_html.escape(inp)}</td>'
            f'<td class="circles-cell">{_output_cell(out)}</td>'
            f'<td class="meta">→ <code>{_html.escape(out)}</code>{tag}</td></tr>'
        )
    if not items:
        return '<p class="empty">(none)</p>'
    return '<table class="ex">' + "".join(items) + '</table>'


def _def_row(d) -> str:
    badge_class = "redef" if d.kind == "REDEF" else "def"
    arity_label = d.arity
    return (
        f'<div class="def-row">'
        f'<span class="badge {badge_class}">{d.kind}</span>'
        f'<span class="arity">{arity_label}</span>'
        f'<code class="rule">{_html.escape(d.lhs)} → {_html.escape(d.rhs)}</code>'
        f'</div>'
    )


def _ku_section(ku: KU) -> str:
    has_redef = any(d.kind == "REDEF" for d in ku.definitions)
    klass = "ku redef-ku" if has_redef else "ku"
    return (
        f'<section class="{klass}" id="ku{ku.ku_index}">'
        f'<h2>KU {ku.ku_index}{" (REDEF)" if has_redef else ""}</h2>'
        f'<div class="defs">{"".join(_def_row(d) for d in ku.definitions)}</div>'
        f'<h3>Examples</h3>{_example_table(ku.examples)}'
        f'<h3>Probes</h3>{_example_table(ku.probes)}'
        f'</section>'
    )


def _timeline_svg(bench) -> str:
    """Tiny inline SVG timeline: x = KU, y = symbol, dots for DEF / X for REDEF."""
    defs = bench.grammar.defs
    syms = sorted({d.symbol for d in defs})
    n_ku = len(bench.kus)
    if not syms or not n_ku:
        return ""
    w, h, pad_l, pad_t = 720, 28 * len(syms) + 80, 110, 40
    inner_w = w - pad_l - 30
    dx = inner_w / max(1, n_ku - 1) if n_ku > 1 else 0
    dy = 28
    parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" class="timeline">']
    parts.append(f'<text x="20" y="22" class="t-title">Schedule ({_html.escape(bench.kind)})</text>')
    for i, sym in enumerate(syms):
        y = pad_t + i * dy
        parts.append(f'<text x="{pad_l - 10}" y="{y + 4}" text-anchor="end" class="t-ylabel">{_html.escape(sym)}</text>')
        parts.append(f'<line x1="{pad_l}" y1="{y}" x2="{pad_l + inner_w}" y2="{y}" class="t-grid"/>')
    for t in range(n_ku):
        x = pad_l + t * dx
        parts.append(f'<text x="{x}" y="{pad_t + len(syms) * dy + 16}" text-anchor="middle" class="t-xlabel">{t}</text>')
        parts.append(f'<line x1="{x}" y1="{pad_t - 6}" x2="{x}" y2="{pad_t + len(syms) * dy - dy + 6}" class="t-grid-v"/>')
    y_of = {s: pad_t + i * dy for i, s in enumerate(syms)}
    for d in defs:
        x = pad_l + d.ku_index * dx
        y = y_of[d.symbol]
        if d.kind == "DEF":
            parts.append(f'<circle cx="{x}" cy="{y}" r="7" class="t-def"/>')
        else:
            parts.append(f'<g class="t-redef" transform="translate({x},{y})">'
                         '<line x1="-6" y1="-6" x2="6" y2="6"/>'
                         '<line x1="-6" y1="6" x2="6" y2="-6"/></g>')
    parts.append('</svg>')
    return "".join(parts)


def _dag_svg(bench) -> str:
    """Render the matplotlib DAG to an inline SVG string (no external file)."""
    try:
        import matplotlib.pyplot as plt
        from .viz import draw_dag
        n_kus = max(1, len(bench.kus))
        fig, ax = plt.subplots(figsize=(max(9.0, n_kus * 1.8), 5.0))
        draw_dag(bench, ax=ax)
        buf = io.StringIO()
        fig.tight_layout()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg = buf.getvalue()
        return svg[svg.find("<svg"):]
    except Exception as e:
        return f'<p class="empty">(DAG render failed: {_html.escape(str(e))})</p>'


_CSS = """
:root { --bg:#fafafa; --fg:#222; --muted:#666; --card:#fff; --line:#ddd;
        --accent:#2b6cb0; --redef:#c0392b; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); margin: 0; padding: 32px 48px;
       font: 14px/1.5 -apple-system, "SF Pro Text", "Helvetica Neue", Helvetica, Arial, sans-serif; }
header { background: var(--card); padding: 20px 24px; border:1px solid var(--line);
         border-radius: 8px; margin-bottom: 24px; }
h1 { margin: 0 0 8px 0; font-weight: 600; font-size: 22px; }
h2 { margin: 0 0 12px 0; font-size: 17px; font-weight: 600; }
h3 { margin: 18px 0 6px 0; font-size: 13px; font-weight: 600;
     text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.meta-line { color: var(--muted); font-size: 13px; }
section.ku { background: var(--card); padding: 18px 22px; margin-bottom: 18px;
             border:1px solid var(--line); border-radius: 8px; }
section.redef-ku { border-color: var(--redef); }
section.redef-ku h2 { color: var(--redef); }
.defs { margin: 6px 0 0 0; }
.def-row { display: flex; align-items: center; gap: 12px; padding: 6px 0; }
.def-row .arity { color: var(--muted); font-size: 11px; text-transform: uppercase;
                  letter-spacing: 0.08em; width: 60px; }
.def-row code.rule { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
                     font-size: 13px; background: #f4f6f8; padding: 2px 8px;
                     border-radius: 4px; }
.badge { display:inline-block; padding: 2px 8px; border-radius: 999px;
         font-size: 10px; font-weight: 700; letter-spacing: 0.06em; }
.badge.def { background: #e3f2fd; color: #1565c0; }
.badge.redef { background: #fde7e9; color: var(--redef); }
table.ex { width: 100%; border-collapse: collapse; }
table.ex td { padding: 6px 8px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }
td.cmd { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
         font-size: 13px; white-space: nowrap; width: 40%; }
td.circles-cell { width: 25%; }
td.meta { color: var(--muted); font-size: 12px; }
td.meta code { background: #f4f6f8; padding: 1px 6px; border-radius: 3px; font-size: 11px; }
.circles { display: inline-flex; gap: 4px; align-items: center; }
.circle { display: inline-block; width: 16px; height: 16px; border-radius: 50%;
          border: 1px solid rgba(0,0,0,0.15); }
.empty { color: #aaa; font-style: italic; margin: 4px 0; }
.timeline { background: var(--card); border:1px solid var(--line); border-radius: 8px;
            display: block; width: 100%; padding: 8px; margin-bottom: 18px; }
.t-title { font-size: 14px; font-weight: 600; fill: var(--fg); }
.t-ylabel { font-size: 11px; fill: var(--fg); font-family: ui-monospace, monospace; }
.t-xlabel { font-size: 10px; fill: var(--muted); }
.t-grid { stroke: #eee; stroke-width: 1; }
.t-grid-v { stroke: #f4f4f4; stroke-width: 1; }
.t-def { fill: var(--accent); stroke: #1a4e80; stroke-width: 1; }
.t-redef line { stroke: var(--redef); stroke-width: 2.5; }
.dag-card { background: var(--card); border:1px solid var(--line); border-radius: 8px;
            padding: 8px 12px; margin-bottom: 18px; }
.dag-card svg { width: 100%; height: auto; display: block; }
nav.kus { margin: 12px 0 0 0; font-size: 12px; color: var(--muted); }
nav.kus a { color: var(--accent); text-decoration: none; margin-right: 10px; }
nav.kus a:hover { text-decoration: underline; }
"""


def render_html(bench, out_path: str, embed_dag: bool = True) -> str:
    """Write a standalone HTML report for `bench` to `out_path`. Returns the path."""
    n_kus = len(bench.kus)
    n_probes = sum(len(k.probes) for k in bench.kus)
    n_redefs = sum(1 for d in bench.grammar.defs if d.kind == "REDEF")
    nav = " ".join(f'<a href="#ku{k.ku_index}">KU{k.ku_index}</a>' for k in bench.kus)
    sections = "".join(_ku_section(k) for k in bench.kus)
    dag_block = (f'<section class="dag-card">{_dag_svg(bench)}</section>'
                 if embed_dag else "")
    title = f"CHORD instance · {bench.kind} · seed={bench.seed}"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{_html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>{_html.escape(title)}</h1>
  <p class="meta-line">{n_kus} KUs · {n_probes} probes · {n_redefs} REDEF events</p>
  <nav class="kus">{nav}</nav>
</header>
{_timeline_svg(bench)}
{dag_block}
{sections}
</body>
</html>"""
    with open(out_path, "w") as f:
        f.write(html)
    return out_path
