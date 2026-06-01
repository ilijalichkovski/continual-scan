"""Observability: KU pretty-printer (MLC-compatible), DAG plot, schedule timeline, eval trace."""
from __future__ import annotations

from typing import Optional

from .core import KU, TimedGrammar, oracle


# ---------- text ----------

def print_ku(ku: KU) -> str:
    out = [f"=== KU {ku.ku_index} ===", "*DEFINITIONS*"]
    out.extend(str(d) for d in ku.definitions)
    out.append("*EXAMPLES*")
    out.extend(f"IN: {i}  OUT: {o}" for i, o in ku.examples)
    if not ku.examples:
        out.append("(none)")
    out.append("*PROBES*")
    out.extend(f"IN: {p.inp}  OUT: {p.out}   (eval_at={p.eval_at})" for p in ku.probes)
    if not ku.probes:
        out.append("(none)")
    return "\n".join(out)


def print_schedule(bench) -> str:
    header = f"CHORD instance | kind={bench.kind} | seed={bench.seed} | n_kus={len(bench.kus)}"
    parts = [header, "=" * len(header), ""]
    for ku in bench.kus:
        parts.append(print_ku(ku))
        parts.append("")
    return "\n".join(parts)


def trace_solution(tg: TimedGrammar, inp: str, eval_at: int) -> str:
    return f"[t={eval_at}]  {inp}   =>   {oracle(tg, inp, eval_at)}"


# ---------- plots ----------

_ARITY_COLOR = {"atom": "#a8d5e2", "unary": "#f9c784", "binary": "#fc814a", "other": "#cccccc"}
_ARITY_ORDER = {"atom": 0, "unary": 1, "binary": 2, "other": 3}


def _dag_edges_from_probes(bench):
    """Derive dependency edges from the probes actually generated.

    An edge `atom -> fn` means "fn depends on atom" -- some probe applies
    `fn` to `atom`. This keeps the DAG informative (only real dependencies)
    rather than drawing the dense atom-x-fn cartesian product.
    """
    last_t = len(bench.kus) - 1
    latest = bench.grammar.latest(last_t)
    fn_syms = {s for s, d in latest.items() if d.arity in ("unary", "binary")}
    atom_syms = {s for s, d in latest.items() if d.arity == "atom"}
    edges: set = set()
    for ku in bench.kus:
        for p in ku.probes:
            toks = p.inp.split()
            fns_in = [t for t in toks if t in fn_syms]
            atoms_in = [t for t in toks if t in atom_syms]
            for f in fns_in:
                for a in atoms_in:
                    edges.add((a, f))
    return edges


def draw_dag(bench, ax=None):
    """Render the primitive DAG horizontally, ordered by KU sequence.

    - x = KU index where the symbol was introduced (left -> right = the order
      the model sees the KUs).
    - Symbols introduced in the same KU are stacked vertically, ordered by
      arity (atoms low, unary mid, binary high) for readability.
    - Edges (`atom -> fn`) point in the direction of dependency: the arrow
      enters the symbol that *uses* the source. Because later symbols depend
      on earlier ones, edges tend to flow left-to-right.
    - REDEF'd symbols get a red outline; the redefined value appears as a
      separate "shadow" node at its own KU column, with a dashed back-edge.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    n_kus = len(bench.kus)
    redefs = [d for d in bench.grammar.defs if d.kind == "REDEF"]
    redefined_syms = {d.symbol for d in redefs}

    # Main node = the *original* DEF (first introduction) of each symbol.
    # REDEFs are rendered separately as shadow nodes at their own KU column.
    first_def: dict = {}
    for d in bench.grammar.defs:
        if d.kind == "DEF" and d.symbol not in first_def:
            first_def[d.symbol] = d

    G = nx.DiGraph()
    for sym, d in first_def.items():
        G.add_node(sym, arity=d.arity, ku=d.ku_index, body=d.rhs)
    for a, f in _dag_edges_from_probes(bench):
        if a in G.nodes and f in G.nodes:
            G.add_edge(a, f)

    # Horizontal layout: one column per KU. Within a column, stack by arity
    # then alphabetically. Shadow nodes for REDEFs occupy their own KU column.
    by_col: dict[int, list[str]] = {}
    for n in G.nodes:
        by_col.setdefault(G.nodes[n]["ku"], []).append(n)
    # Reserve vertical slots in each column for REDEF shadows of that KU too.
    redef_count_per_ku: dict[int, int] = {}
    for r in redefs:
        redef_count_per_ku[r.ku_index] = redef_count_per_ku.get(r.ku_index, 0) + 1

    max_rows = max(
        (len(syms) + redef_count_per_ku.get(t, 0)) for t, syms in by_col.items()
    ) if by_col else 1
    max_rows = max(max_rows, 1)

    pos: dict = {}
    for t, syms in by_col.items():
        syms_sorted = sorted(syms, key=lambda s: (_ARITY_ORDER[G.nodes[s]["arity"]], s))
        n = len(syms_sorted)
        # Centre each column's nodes around 0.
        for i, node in enumerate(syms_sorted):
            y = (i - (n - 1) / 2.0)
            pos[node] = (t, y)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, n_kus * 1.6), max(4, max_rows * 1.1)))

    node_order = list(G.nodes)
    node_colors = [_ARITY_COLOR[G.nodes[n]["arity"]] for n in node_order]
    edge_colors = ["#cc0000" if n in redefined_syms else "#444444" for n in node_order]
    linewidths = [2.5 if n in redefined_syms else 1.0 for n in node_order]
    labels = {n: f"{n}\n→ {G.nodes[n]['body']}\n[KU{G.nodes[n]['ku']}]" for n in G.nodes}

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>",
                           edge_color="#888888", width=1.2,
                           connectionstyle="arc3,rad=0.08",
                           node_size=2200)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_order,
                           node_color=node_colors,
                           edgecolors=edge_colors, linewidths=linewidths,
                           node_size=2200, node_shape="o")
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8)

    # Render REDEF shadow nodes in their own KU column, stacked below the
    # already-placed nodes in that column.
    redefs_by_ku: dict[int, list] = {}
    for r in redefs:
        redefs_by_ku.setdefault(r.ku_index, []).append(r)
    for t, rlist in redefs_by_ku.items():
        existing = by_col.get(t, [])
        n_existing = len(existing)
        for j, r in enumerate(sorted(rlist, key=lambda d: d.symbol)):
            sy = -((n_existing - 1) / 2.0) - (j + 1)
            sx = t
            ax.scatter([sx], [sy], s=2200, c="#fde0dc", edgecolors="#cc0000",
                       linewidths=2.0, zorder=2)
            ax.text(sx, sy, f"{r.symbol}\n→ {r.rhs}\n[KU{r.ku_index}*REDEF]",
                    ha="center", va="center", fontsize=7)
            if r.symbol in pos:
                ox, oy = pos[r.symbol]
                ax.annotate("", xy=(sx, sy), xytext=(ox, oy),
                            arrowprops=dict(arrowstyle="->", linestyle="--",
                                            color="#cc0000", lw=1.5,
                                            connectionstyle="arc3,rad=0.25"))

    y_lo = min((y for _, y in pos.values()), default=0.0)
    y_hi = max((y for _, y in pos.values()), default=0.0)
    # Extend y_lo to include any redef shadows.
    if redefs_by_ku:
        deepest = min(
            -((len(by_col.get(t, [])) - 1) / 2.0) - len(rlist)
            for t, rlist in redefs_by_ku.items()
        )
        y_lo = min(y_lo, deepest)

    ax.set_xlim(-0.6, n_kus - 0.4)
    ax.set_ylim(y_lo - 0.8, y_hi + 0.8)
    ax.set_xticks(range(n_kus))
    ax.set_xticklabels([f"KU{t}" for t in range(n_kus)])
    ax.set_yticks([])
    ax.set_xlabel("KU index (sequential order seen by the model) →")
    for t in range(n_kus):
        ax.axvline(t, color="#eeeeee", linewidth=1, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"Primitive DAG ({bench.kind})  -- left→right = KU order; "
                 f"edges = dependencies; red = REDEF")
    return ax


def draw_timeline(bench, ax=None):
    """Schedule timeline: x = KU index, y = symbol; DEF = filled blue, REDEF = red X."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    syms = sorted({d.symbol for d in bench.grammar.defs})
    y = {s: i for i, s in enumerate(syms)}
    for d in bench.grammar.defs:
        marker = "o" if d.kind == "DEF" else "X"
        color = "#2b8cbe" if d.kind == "DEF" else "#e34a33"
        ax.scatter(d.ku_index, y[d.symbol], marker=marker, c=color, s=160,
                   edgecolors="black", zorder=3)
    ax.set_yticks(range(len(syms)))
    ax.set_yticklabels(syms)
    ax.set_xticks(range(len(bench.kus)))
    ax.set_xlabel("KU index")
    ax.set_title(f"Schedule timeline ({bench.kind}) -- circle=DEF, X=REDEF")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    return ax
