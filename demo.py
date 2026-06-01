"""End-to-end CHORD demo.

For each of the four schedule kinds:
  1. Build a Benchmark instance.
  2. Print the schedule as MLC-style text blocks.
  3. Run six baselines spanning Protocol A and Protocol B/C:
       - OracleLearner            (cheats; ceiling)
       - SubstitutionInductionLearner  (honest, stdlib-only)
       - MemorizingLearner        (no compositional generalisation)
       - NullLearner              (floor)
       - CumulativeICL + OracleLLM (Protocol B; exercises ICL plumbing)
       - WindowedICL + OracleLLM  (Protocol C; structurally limited)
  4. Render a self-contained HTML report and a matplotlib DAG + timeline PNG.

The ICL learners use OracleLLM by default so this demo runs without API
keys. Swap in `make_openai_llm("gpt-4o-mini")` to evaluate a real model.
"""
import os

import matplotlib.pyplot as plt

from chord import (
    sample_pool, schedule,
    NullLearner, MemorizingLearner, OracleLearner,
    SubstitutionInductionLearner,
    CumulativeICL, WindowedICL, make_oracle_llm,
    metrics, print_schedule, trace_solution,
    draw_dag, draw_timeline,
    render_html,
)


def banner(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)


def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)


pool = sample_pool(n_atoms=4, n_unary=2, n_binary=1, seed=42)
os.makedirs("out", exist_ok=True)

summary_rows = []

for kind in ["topological", "interleaved", "backward", "adversarial"]:
    banner(f"SCHEDULE: {kind}")
    bench = schedule(pool, kind=kind, n_examples=3, n_probes=3,
                     n_redefs=2, seed=42)
    print(print_schedule(bench))

    learners = [
        ("Oracle",           OracleLearner()),
        ("Induction",        SubstitutionInductionLearner()),
        ("Memorizing",       MemorizingLearner()),
        ("Null",             NullLearner()),
        ("ICL-B (oracle)",   CumulativeICL(llm=make_oracle_llm(bench.grammar))),
        ("ICL-C (oracle)",   WindowedICL(llm=make_oracle_llm(bench.grammar))),
    ]

    print("--- learner comparison ---")
    print(f"  {'learner':<18} {'acquisition':>11} {'retention':>10} {'update_fid':>10}")
    for name, learner in learners:
        m = metrics(bench.run(learner))
        acq = fmt(m["acquisition_mean"]) if m["acquisition_mean"] is not None else "n/a"
        ret = fmt(m["retention_mean"]) if m["retention_mean"] is not None else "n/a"
        if m["update_fidelity"]:
            uf = sum(m["update_fidelity"].values()) / len(m["update_fidelity"])
            uf = fmt(uf)
        else:
            uf = "n/a"
        print(f"  {name:<18} {acq:>11} {ret:>10} {uf:>10}")
        summary_rows.append((kind, name, acq, ret, uf))

    if kind == "adversarial":
        print("\n--- oracle trace: REDEF'd symbol at two timestamps ---")
        for d in bench.grammar.defs:
            if d.kind == "REDEF":
                print(trace_solution(bench.grammar, d.symbol, eval_at=0))
                print(trace_solution(bench.grammar, d.symbol, eval_at=len(bench.kus) - 1))
                break

    fig, axes = plt.subplots(1, 2, figsize=(15, 5),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    draw_dag(bench, ax=axes[0])
    draw_timeline(bench, ax=axes[1])
    plt.tight_layout()
    png_path = f"out/{kind}.png"
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    html_path = f"out/{kind}.html"
    render_html(bench, html_path)
    print(f"\nArtifacts: {png_path}   {html_path}")

banner("SUMMARY (acquisition / retention / update-fidelity)")
print(f"  {'schedule':<14} {'learner':<18} {'acq':>5} {'ret':>5} {'uf':>5}")
for kind, name, acq, ret, uf in summary_rows:
    print(f"  {kind:<14} {name:<18} {acq:>5} {ret:>5} {uf:>5}")
