#!/usr/bin/env python3
"""CLI: generate a CHORD ICL / perplexity-eval dataset from a config file.

Examples
--------
  python generate_dataset.py --config configs/pretrain_10t5p3o.toml

  python generate_dataset.py --config configs/pretrain_10t5p3o.toml \\
      --n-instances 20 --output data/smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chord.dataset import DatasetConfig, generate_dataset, load_config
from chord.core import OPERATORS


def _parse_ops(s: str) -> list[str]:
    return [p.strip() for p in s.split(",") if p.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate CHORD datasets for ICL / perplexity pretraining evals.",
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to a TOML or JSON config file.")
    p.add_argument("--n-primitives", type=int, default=None)
    p.add_argument("--operations", type=str, default=None,
                   help=f"Comma-separated ops. Known: {', '.join(sorted(OPERATORS))}")
    p.add_argument("--n-timesteps", type=int, default=None)
    p.add_argument("--primitive-length", type=int, default=None)
    p.add_argument("--redefinitions", action=argparse.BooleanOptionalAction,
                   default=None)
    p.add_argument("--n-redefs", type=int, default=None)
    p.add_argument("--n-instances", type=int, default=None)
    p.add_argument("--n-examples", type=int, default=None)
    p.add_argument("--n-probes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    overrides = {}
    if args.n_primitives is not None:
        overrides["n_primitives"] = args.n_primitives
    if args.operations is not None:
        overrides["operations"] = _parse_ops(args.operations)
    if args.n_timesteps is not None:
        overrides["n_timesteps"] = args.n_timesteps
    if args.primitive_length is not None:
        overrides["primitive_length"] = args.primitive_length
    if args.redefinitions is not None:
        overrides["redefinitions"] = args.redefinitions
    if args.n_redefs is not None:
        overrides["n_redefs"] = args.n_redefs
    if args.n_instances is not None:
        overrides["n_instances"] = args.n_instances
    if args.n_examples is not None:
        overrides["n_examples"] = args.n_examples
    if args.n_probes is not None:
        overrides["n_probes"] = args.n_probes
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output is not None:
        overrides["output"] = args.output

    if args.config:
        cfg = load_config(args.config, overrides=overrides)
    else:
        cfg = DatasetConfig(**overrides) if overrides else DatasetConfig()

    out = generate_dataset(cfg)
    # Summarize from the manifest we just wrote.
    man = Path(out).with_name(Path(out).stem + ".manifest.json")
    meta = json.loads(man.read_text())
    print(f"wrote {meta['n_records']} records -> {out}")
    print(f"manifest -> {man}")
    print("config:", json.dumps(meta["config"], indent=2))


if __name__ == "__main__":
    main()
