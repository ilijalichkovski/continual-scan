"""Core types: Definition / KU / Probe, time-indexed grammar, symbolic oracle.

The substrate is MLC's rewrite-rule engine (Lake & Baroni, Nature 2023). We
reuse it as-is and layer a temporal index on top: every Definition carries the
KU at which it was emitted, and `TimedGrammar.at(t)` materialises the concrete
grammar in force at the end of KU t, with most-recent-definition-wins semantics
for symbols that have been REDEF'd.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

from ._interpret_grammar import Grammar as _MLCGrammar, Rule as _MLCRule


COLORS = ["RED", "GREEN", "BLUE", "YELLOW", "PURPLE", "ORANGE", "PINK", "BROWN"]
_ATOM_NAMES = ["dax", "lug", "wif", "zup", "fep", "blicket", "kiki", "tufa"]
_UNARY = {"thrice": "[u1] [u1] [u1]", "twice": "[u1] [u1]"}
_BINARY = {"after": "[x2] [x1]", "surround": "[x1] [x2] [x1]"}
# Unified operator catalog: name -> (arity_tag, lhs_template, rhs_template)
OPERATORS: dict[str, tuple[str, str, str]] = {
    "twice": ("unary", "u1 twice", "[u1] [u1]"),
    "thrice": ("unary", "u1 thrice", "[u1] [u1] [u1]"),
    "after": ("binary", "x1 after x2", "[x2] [x1]"),
    "surround": ("binary", "x1 surround x2", "[x1] [x2] [x1]"),
}
_NAME_ALPHABET = "bcdfghjklmnpqrstvwxz"
_OUT_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ"


@dataclass(frozen=True)
class Definition:
    """One symbol introduction. `kind` is 'DEF' (first time) or 'REDEF' (overwrite)."""
    kind: str
    lhs: str
    rhs: str
    ku_index: int

    @property
    def arity(self) -> str:
        n = len(self.lhs.split())
        return {1: "atom", 2: "unary", 3: "binary"}.get(n, "other")

    @property
    def symbol(self) -> str:
        """The new symbol this rule introduces: the atom itself or the function name."""
        parts = self.lhs.split()
        return parts[0] if self.arity == "atom" else parts[1]

    def __str__(self) -> str:
        return f"{self.kind} {self.lhs} -> {self.rhs}"


@dataclass
class Probe:
    """A held-out (input, output) pair, resolved against the grammar at `eval_at`."""
    inp: str
    out: str
    eval_at: int
    introduced_at: int = 0
    depends_on: Tuple[str, ...] = ()


@dataclass
class KU:
    """A Knowledge Unit: definitions emitted at this step, demos for them, and probes."""
    ku_index: int
    definitions: List[Definition]
    examples: List[Tuple[str, str]]
    probes: List[Probe]

    @property
    def new_symbols(self) -> List[str]:
        return [d.symbol for d in self.definitions]


@dataclass
class TimedGrammar:
    """Append-only stream of Definitions; can be resolved to an MLC Grammar at any t."""
    defs: List[Definition] = field(default_factory=list)

    @property
    def input_symbols(self) -> List[str]:
        return sorted({d.symbol for d in self.defs})

    def latest(self, t: int) -> dict:
        """Most-recent Definition per symbol with ku_index <= t."""
        out: dict = {}
        for d in self.defs:
            if d.ku_index <= t:
                out[d.symbol] = d
        return out

    def at(self, t: int) -> _MLCGrammar:
        rules = [_MLCRule(d.lhs, d.rhs) for d in self.latest(t).values()]
        # Iconic-concat fallback — same default as MLC's generate_datasets.py.
        rules.append(_MLCRule("u1 x1", "[u1] [x1]"))
        return _MLCGrammar(rules, self.input_symbols)


def oracle(tg: TimedGrammar, inp: str, eval_at: int) -> str:
    """Symbolic ground truth: rewrite `inp` using the grammar in force at `eval_at`."""
    try:
        return tg.at(eval_at).apply(inp)
    except Exception:
        return ""


def _random_token(rng: random.Random, length: int, alphabet: str,
                  forbidden: set[str]) -> str:
    if length < 1:
        raise ValueError("primitive_length must be >= 1")
    for _ in range(10_000):
        tok = "".join(rng.choice(alphabet) for _ in range(length))
        if tok not in forbidden:
            return tok
    raise RuntimeError(f"could not sample a unique token of length {length}")


def sample_pool(n_atoms: int = 4, n_unary: int = 2, n_binary: int = 1,
                seed: int = 0, *,
                operations: List[str] | None = None,
                primitive_length: int | None = None) -> List[Definition]:
    """Sample a pool of primitive Definitions (ku_index left as -1 for scheduling).

    Parameters
    ----------
    n_atoms:
        Number of atomic primitives.
    n_unary / n_binary:
        Used only when `operations` is None (legacy demo path).
    operations:
        Explicit operator names drawn from `OPERATORS` (e.g. twice, thrice, after).
        When set, arity counts are ignored.
    primitive_length:
        If set, atom names and their output tokens are random strings of this
        length (unfamiliar vocabulary for ICL / perplexity evals). If None,
        use the classic MLC nonsense words + color outputs.
    """
    rng = random.Random(seed)
    used_names: set[str] = set(OPERATORS)
    used_outs: set[str] = set()

    if primitive_length is None:
        if n_atoms > len(_ATOM_NAMES):
            raise ValueError(f"n_atoms={n_atoms} exceeds built-in name list "
                             f"({len(_ATOM_NAMES)}); set primitive_length to synthesize names")
        atoms = rng.sample(_ATOM_NAMES, n_atoms)
        colors = rng.sample(COLORS, n_atoms)
        atom_defs = [Definition("DEF", a, c, ku_index=-1)
                     for a, c in zip(atoms, colors)]
    else:
        atom_defs = []
        for _ in range(n_atoms):
            name = _random_token(rng, primitive_length, _NAME_ALPHABET, used_names)
            used_names.add(name)
            out = _random_token(rng, primitive_length, _OUT_ALPHABET, used_outs)
            used_outs.add(out)
            atom_defs.append(Definition("DEF", name, out, ku_index=-1))

    op_defs: List[Definition] = []
    if operations is not None:
        unknown = [op for op in operations if op not in OPERATORS]
        if unknown:
            raise ValueError(f"unknown operations: {unknown}; "
                             f"known: {sorted(OPERATORS)}")
        if len(set(operations)) != len(operations):
            raise ValueError(f"duplicate operations: {operations}")
        for name in operations:
            _arity, lhs, rhs = OPERATORS[name]
            op_defs.append(Definition("DEF", lhs, rhs, ku_index=-1))
    else:
        unaries = rng.sample(list(_UNARY.items()), n_unary)
        binaries = rng.sample(list(_BINARY.items()), n_binary)
        op_defs += [Definition("DEF", f"u1 {n}", b, ku_index=-1) for n, b in unaries]
        op_defs += [Definition("DEF", f"x1 {n} x2", b, ku_index=-1) for n, b in binaries]

    return atom_defs + op_defs


def fresh_outputs(n: int, primitive_length: int | None, used: set[str],
                  seed: int) -> List[str]:
    """Sample `n` fresh atom-output tokens not in `used` (for REDEF events)."""
    rng = random.Random(seed)
    outs: List[str] = []
    if primitive_length is None:
        pool = [c for c in COLORS if c not in used]
        if len(pool) < n:
            # Fall back to synthesized tokens if the classic palette is exhausted.
            while len(pool) < n:
                pool.append(_random_token(rng, 4, _OUT_ALPHABET, used | set(pool)))
        outs = pool[:n]
    else:
        for _ in range(n):
            tok = _random_token(rng, primitive_length, _OUT_ALPHABET, used | set(outs))
            outs.append(tok)
    return outs
