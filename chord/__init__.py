"""CHORD: a continual compositional benchmark on the MLC rewrite-rule substrate."""
from .core import (
    Definition, Probe, KU, TimedGrammar,
    sample_pool, oracle, COLORS, OPERATORS,
)
from .bench import (
    Learner, NullLearner, OracleLearner, MemorizingLearner,
    Benchmark, schedule, metrics,
)
from .learners import (
    SubstitutionInductionLearner,
    CumulativeICL, WindowedICL,
    make_oracle_llm, make_openai_llm,
    format_ku, PROMPT_HEADER,
)
from .viz import (
    print_ku, print_schedule, trace_solution, draw_dag, draw_timeline,
)
from .htmlviz import render_html
from .dataset import DatasetConfig, generate_dataset, load_config, iter_records

__all__ = [
    "Definition", "Probe", "KU", "TimedGrammar",
    "sample_pool", "oracle", "COLORS", "OPERATORS",
    "Learner", "NullLearner", "OracleLearner", "MemorizingLearner",
    "Benchmark", "schedule", "metrics",
    "SubstitutionInductionLearner",
    "CumulativeICL", "WindowedICL",
    "make_oracle_llm", "make_openai_llm",
    "format_ku", "PROMPT_HEADER",
    "print_ku", "print_schedule", "trace_solution", "draw_dag", "draw_timeline",
    "render_html",
    "DatasetConfig", "generate_dataset", "load_config", "iter_records",
]
