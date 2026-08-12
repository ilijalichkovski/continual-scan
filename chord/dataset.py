"""Config-driven dataset generation for ICL / perplexity pretraining evals.

Each record is a cumulative (Protocol B) context up through timestep t, paired
with a held-out probe completion. Score perplexity on `target` given `context`.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator, List, Optional

from .bench import Benchmark, schedule
from .core import OPERATORS, sample_pool
from .learners import PROMPT_HEADER, format_ku


@dataclass
class DatasetConfig:
    """Generator configuration.

    Attributes
    ----------
    n_primitives:
        Number of atomic primitives introduced across the schedule.
    operations:
        Operator names from `OPERATORS` (e.g. twice, thrice, after, surround).
    n_timesteps:
        Exact number of Knowledge Units in each schedule instance.
    primitive_length:
        Character length of synthesized atom names and output tokens. Use
        unfamiliar strings so the eval does not leak English/color priors.
    redefinitions:
        If True, reserve `n_redefs` timesteps for explicit REDEF events.
    n_redefs:
        Number of REDEF events when `redefinitions` is True.
    n_instances:
        How many independent seeded schedules to emit.
    n_examples / n_probes:
        Teaching demos and held-out probes per timestep.
    seed:
        Base RNG seed; instance i uses seed + i.
    output:
        Destination JSONL path (set by CLI if omitted in file).
    """
    n_primitives: int = 5
    operations: List[str] = field(default_factory=lambda: ["twice", "thrice", "after"])
    n_timesteps: int = 10
    primitive_length: int = 3
    redefinitions: bool = False
    n_redefs: int = 2
    n_instances: int = 100
    n_examples: int = 3
    n_probes: int = 3
    seed: int = 0
    output: str = "data/chord_eval.jsonl"

    def __post_init__(self) -> None:
        if self.n_primitives < 1:
            raise ValueError("n_primitives must be >= 1")
        if self.n_timesteps < 1:
            raise ValueError("n_timesteps must be >= 1")
        if self.primitive_length < 1:
            raise ValueError("primitive_length must be >= 1")
        if not self.operations:
            raise ValueError("operations must be a non-empty list")
        unknown = [op for op in self.operations if op not in OPERATORS]
        if unknown:
            raise ValueError(f"unknown operations {unknown}; known={sorted(OPERATORS)}")
        if self.redefinitions and self.n_redefs < 1:
            raise ValueError("n_redefs must be >= 1 when redefinitions is True")
        min_slots = 1 + (self.n_redefs if self.redefinitions else 0)
        if self.n_timesteps < min_slots:
            raise ValueError(
                f"n_timesteps={self.n_timesteps} too small for "
                f"redefinitions={self.redefinitions} (need >= {min_slots})"
            )


def load_config(path: str | Path, overrides: Optional[dict] = None) -> DatasetConfig:
    """Load a TOML or JSON config file, applying optional key overrides."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".toml":
        data = tomllib.loads(raw.decode("utf-8"))
    elif path.suffix == ".json":
        data = json.loads(raw.decode("utf-8"))
    else:
        raise ValueError(f"unsupported config type: {path.suffix} (use .toml or .json)")

    # Allow a top-level [dataset] table in TOML, or flat keys.
    if "dataset" in data and isinstance(data["dataset"], dict):
        data = data["dataset"]

    allowed = {f.name for f in fields(DatasetConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")

    merged = {**data, **(overrides or {})}
    return DatasetConfig(**merged)


def _build_benchmark(cfg: DatasetConfig, instance_seed: int) -> Benchmark:
    pool = sample_pool(
        n_atoms=cfg.n_primitives,
        operations=list(cfg.operations),
        primitive_length=cfg.primitive_length,
        seed=instance_seed,
    )
    return schedule(
        pool,
        n_timesteps=cfg.n_timesteps,
        redefinitions=cfg.redefinitions,
        n_redefs=cfg.n_redefs,
        n_examples=cfg.n_examples,
        n_probes=cfg.n_probes,
        primitive_length=cfg.primitive_length,
        seed=instance_seed,
    )


def _context_and_target(bench: Benchmark, t: int, probe) -> tuple[str, str, str]:
    """Protocol-B cumulative prompt split into context / target / full text."""
    kus = [format_ku(bench.kus[i]) for i in range(t + 1)]
    body = "\n\n".join(kus)
    prefix = (
        PROMPT_HEADER
        + "\n" + body
        + f"\n\n*QUERY* (evaluate at t={probe.eval_at})\n"
        + f"IN: {probe.inp}  OUT:"
    )
    target = probe.out
    text = f"{prefix} {target}"
    return prefix, target, text


def iter_records(cfg: DatasetConfig) -> Iterator[dict[str, Any]]:
    """Yield JSON-serializable records for every instance × timestep × probe."""
    for inst in range(cfg.n_instances):
        instance_seed = cfg.seed + inst
        bench = _build_benchmark(cfg, instance_seed)
        for t, ku in enumerate(bench.kus):
            # After observing KU t, score every probe from KUs 0..t (acquisition
            # on the diagonal, retention off-diagonal) — matches Protocol B.
            for src in range(t + 1):
                for p_i, probe in enumerate(bench.kus[src].probes):
                    context, target, text = _context_and_target(bench, t, probe)
                    yield {
                        "id": f"inst{inst:04d}_t{t:02d}_src{src:02d}_p{p_i}",
                        "instance": inst,
                        "seed": instance_seed,
                        "timestep": t,
                        "probe_source_ku": src,
                        "probe_index": p_i,
                        "n_timesteps": len(bench.kus),
                        "kind": bench.kind,
                        "input": probe.inp,
                        "eval_at": probe.eval_at,
                        "context": context,
                        "target": target,
                        "text": text,
                        "definitions_at_t": [str(d) for d in ku.definitions],
                        "config": {
                            "n_primitives": cfg.n_primitives,
                            "operations": list(cfg.operations),
                            "n_timesteps": cfg.n_timesteps,
                            "primitive_length": cfg.primitive_length,
                            "redefinitions": cfg.redefinitions,
                            "n_redefs": cfg.n_redefs,
                        },
                    }


def generate_dataset(cfg: DatasetConfig, output: str | Path | None = None) -> Path:
    """Write a JSONL dataset and a sibling manifest JSON; return the JSONL path."""
    out = Path(output or cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for rec in iter_records(cfg):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    manifest = {
        "config": asdict(cfg),
        "n_records": n,
        "output": str(out),
        "known_operations": sorted(OPERATORS),
    }
    man_path = out.with_suffix(out.suffix + ".manifest.json")
    if out.suffix == ".jsonl":
        man_path = out.with_name(out.stem + ".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out
