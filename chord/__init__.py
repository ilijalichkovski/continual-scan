"""CHORD: a continual compositional benchmark on the MLC rewrite-rule substrate."""
from .core import (
    Definition, Probe, KU, TimedGrammar,
    sample_pool, oracle, COLORS,
)
from .bench import (
    Learner, NullLearner, OracleLearner, MemorizingLearner,
    Benchmark, schedule, metrics,
)
from .learners import (
    SubstitutionInductionLearner,
    CumulativeICL, WindowedICL,
    make_oracle_llm, make_openai_llm,
)
from .viz import (
    print_ku, print_schedule, trace_solution, draw_dag, draw_timeline,
)
from .htmlviz import render_html

__all__ = [
    "Definition", "Probe", "KU", "TimedGrammar",
    "sample_pool", "oracle", "COLORS",
    "Learner", "NullLearner", "OracleLearner", "MemorizingLearner",
    "Benchmark", "schedule", "metrics",
    "SubstitutionInductionLearner",
    "CumulativeICL", "WindowedICL",
    "make_oracle_llm", "make_openai_llm",
    "print_ku", "print_schedule", "trace_solution", "draw_dag", "draw_timeline",
    "render_html",
]
