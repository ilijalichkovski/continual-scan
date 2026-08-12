# CHORD — Continual Hierarchical Online Rule Discovery

A symbolic, language-prior-free benchmark for **continual compositional learning**, extending the MLC few-shot setup (Lake & Baroni, *Nature* 2023) with a temporal schedule of Knowledge Units (KUs). See [CHORD_benchmark_design.md](CHORD_benchmark_design.md) for the design.

## Quick start

```
uv sync
uv run demo.py
```

This generates four benchmark instances (one per schedule kind) under `out/` — each as a `.html` self-contained report and a `.png` DAG + timeline figure — and prints a learner comparison table.

## Layout

| File | What it is |
|---|---|
| `chord/core.py` | Typed primitives, time-indexed grammar, symbolic oracle. Reuses MLC's rewrite engine, vendored as `chord/_interpret_grammar.py`. |
| `chord/bench.py` | Schedule generator (`topological`, `interleaved`, `backward`, `adversarial`), `Learner` ABC, three baseline learners, `Benchmark` runner, metrics. |
| `chord/learners.py` | `SubstitutionInductionLearner` (Protocol A, stdlib-only honest baseline) and the ICL family: `CumulativeICL` (Protocol B), `WindowedICL` (Protocol C), plus `make_oracle_llm` and `make_openai_llm` adapters. |
| `chord/viz.py` | DAG plot (layered layout, edges from probes, REDEF shadow nodes), schedule timeline, KU pretty-printer, oracle evaluation trace. |
| `chord/htmlviz.py` | Standalone HTML renderer per benchmark instance — MLC-style colored circles, per-KU sections, inline CSS, embedded SVG timeline + DAG. |
| `demo.py` | End-to-end walkthrough. |

## Protocols supported

- **A (parametric)** — `Learner.train_on_ku` performs weight updates between KUs. Ship: `OracleLearner`, `MemorizingLearner`, `NullLearner`, `SubstitutionInductionLearner`.
- **B (ICL-cumulative)** — `CumulativeICL(llm=...)`. The context grows with each KU. Natural setup for long-context LLMs.
- **C (ICL-windowed)** — `WindowedICL(llm=...)`. Context is reset to just the latest KU. Impossible by design on probes that need earlier KUs; the gap is what real CL methods must bridge.

Both ICL learners take any `llm: Callable[[str], str]`. The demo uses `make_oracle_llm(bench.grammar)`, which routes through the symbolic oracle — so the demo runs without API keys. Swap in `make_openai_llm("gpt-4o-mini")` (stdlib HTTP, reads `OPENAI_API_KEY`) to evaluate a real LLM.

## Schedule kinds

- **topological** — atoms first (one KU), then unaries, then binaries.
- **interleaved** — uniform shuffle (first KU forced to contain an atom).
- **backward** — one starter atom bundled with *all* functions in KU 0; later KUs introduce remaining atoms. Probes apply earlier-learned functions to newly-introduced atoms — the genuine-abstraction test.
- **adversarial** — topological, then a tail of explicit `REDEF` events. Every overwrite is a token-level visible event; probes carry `eval_at=t` so ground truth is unambiguous.

## What's deferred from the design doc

Higher-order modifiers, set-cover minimisation, the dense-ICL calibration oracle, and a public frozen v1 corpus. All attach cleanly to the existing surface.
