"""Schedules, Learner interface, Benchmark runner, metrics.

Protocol A only (parametric, per-KU training). Four schedule kinds:
  topological  : atoms first (all in one KU), then unaries, then binaries.
  interleaved  : random order, but first KU is forced to contain an atom
                 (otherwise the cumulative grammar is empty and probes can't
                  be generated).
  backward     : ALL functions are introduced in KU 0 together with one
                 starter atom; subsequent KUs introduce more atoms. Probes
                 at KU >= 1 exercise an earlier-learned function on a newly
                 introduced atom — the genuine-abstraction test.
  adversarial  : topological, then a tail of explicit REDEF events for some
                 atoms. REDEFs are token-level visible to the learner, and
                 probes carry eval_at so ground truth is unambiguous.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .core import Definition, KU, Probe, TimedGrammar, oracle, fresh_outputs


# ---------- schedule construction ----------

def _order(defs: List[Definition], kind: str, seed: int) -> List[List[Definition]]:
    rng = random.Random(seed)
    atoms = [d for d in defs if d.arity == "atom"]
    unaries = [d for d in defs if d.arity == "unary"]
    binaries = [d for d in defs if d.arity == "binary"]

    if kind == "topological":
        return [atoms, *[[u] for u in unaries], *[[b] for b in binaries]]

    if kind == "interleaved":
        flat = defs[:]
        rng.shuffle(flat)
        # Force the first KU to contain at least one atom so the grammar is non-empty.
        if flat[0].arity != "atom":
            i = next(i for i, d in enumerate(flat) if d.arity == "atom")
            flat[0], flat[i] = flat[i], flat[0]
        return [[d] for d in flat]

    if kind == "backward":
        starter, *rest = atoms
        return [[starter, *unaries, *binaries], *[[a] for a in rest]]

    raise ValueError(f"unknown schedule kind: {kind}")


def _candidates(tg: TimedGrammar, t: int, max_depth: int = 2,
                max_len: int = 4) -> List[Tuple[str, str]]:
    """Enumerate (input, output) pairs from the grammar in force at t, up to depth/len caps."""
    latest = tg.latest(t)
    atoms = [s for s, d in latest.items() if d.arity == "atom"]
    unaries = [s for s, d in latest.items() if d.arity == "unary"]
    binaries = [s for s, d in latest.items() if d.arity == "binary"]
    inputs: Set[str] = set(atoms)
    frontier = list(atoms)
    for _ in range(max_depth):
        nxt: List[str] = []
        for x in frontier:
            for u in unaries:
                s = f"{x} {u}"
                if len(s.split()) <= max_len:
                    nxt.append(s)
            for b in binaries:
                for y in atoms:  # keep one operand atomic to bound the blow-up
                    s = f"{x} {b} {y}"
                    if len(s.split()) <= max_len:
                        nxt.append(s)
        inputs.update(nxt)
        frontier = nxt
    pairs: List[Tuple[str, str]] = []
    for inp in inputs:
        out = oracle(tg, inp, t)
        if out:
            pairs.append((inp, out))
    return pairs


def _examples_and_probes(tg: TimedGrammar, t: int, new_syms: List[str],
                         n_examples: int, n_probes: int,
                         rng: random.Random, used: Set[str]
                         ) -> Tuple[List[Tuple[str, str]], List[Probe]]:
    cands = _candidates(tg, t)
    # Prefer pairs that reference a symbol introduced at this KU. Composition-only
    # KUs (empty new_syms) fall back to any novel pair from the cumulative grammar.
    if new_syms:
        touching = [c for c in cands
                    if c[0] not in used
                    and any(s in c[0].split() for s in new_syms)]
    else:
        touching = [c for c in cands if c[0] not in used]
    rng.shuffle(touching)
    examples = touching[:n_examples]
    probe_pairs = touching[n_examples:n_examples + n_probes]
    probes = [Probe(inp, out, eval_at=t, introduced_at=t,
                    depends_on=tuple(inp.split()))
              for inp, out in probe_pairs]
    return examples, probes


def _pack_defs(defs: List[Definition], n_slots: int,
               seed: int) -> List[List[Definition]]:
    """Pack definitions into exactly `n_slots` groups (some may be empty).

    Guarantees the first non-empty group contains at least one atom when any
    atoms exist. Empty trailing groups are composition-only timesteps.
    """
    if n_slots < 1:
        raise ValueError("n_timesteps must be >= 1")
    rng = random.Random(seed)
    atoms = [d for d in defs if d.arity == "atom"]
    others = [d for d in defs if d.arity != "atom"]
    rng.shuffle(others)
    ordered = (atoms[:1] + others + atoms[1:]) if atoms else others[:]

    groups: List[List[Definition]] = [[] for _ in range(n_slots)]
    if not ordered:
        return groups

    # Spread defs as evenly as possible across slots; leftover slots stay empty.
    n_active = min(n_slots, len(ordered))
    # Round-robin into the first n_active slots so early KUs get content.
    for i, d in enumerate(ordered):
        groups[i % n_active].append(d)

    # Ensure KU 0 has an atom when possible (needed for a non-empty grammar).
    if atoms and not any(d.arity == "atom" for d in groups[0]):
        for i in range(1, n_slots):
            atom_i = next((j for j, d in enumerate(groups[i]) if d.arity == "atom"), None)
            if atom_i is not None:
                groups[0].append(groups[i].pop(atom_i))
                break
    return groups


def schedule(defs: List[Definition], kind: str = "topological",
             n_examples: int = 3, n_probes: int = 3,
             n_redefs: int = 2, seed: int = 0,
             n_timesteps: int | None = None,
             redefinitions: bool | None = None,
             primitive_length: int | None = None) -> "Benchmark":
    """Build a full Benchmark from a pool of Definitions and a schedule kind.

    When `n_timesteps` is set, definitions are packed into exactly that many
    KUs (empty KUs become composition-only). `redefinitions` overrides the
    legacy `kind == "adversarial"` switch: True injects REDEF KUs into the
    timeline (consuming slots from `n_timesteps` when set, else appending).
    """
    rng = random.Random(seed)
    do_redefs = (kind == "adversarial") if redefinitions is None else redefinitions
    n_redef_events = n_redefs if do_redefs else 0

    if n_timesteps is not None:
        def_slots = max(1, n_timesteps - n_redef_events) if do_redefs else n_timesteps
        groups = _pack_defs(defs, def_slots, seed)
    else:
        base_kind = "topological" if kind == "adversarial" else kind
        groups = _order(defs, base_kind, seed)

    if do_redefs and n_redef_events > 0:
        atoms = [d for d in defs if d.arity == "atom"]
        if not atoms:
            raise ValueError("redefinitions require at least one atom primitive")
        used_outs = {d.rhs for d in atoms}
        fresh = fresh_outputs(n_redef_events, primitive_length, used_outs,
                              seed=seed + 17)
        targets = rng.sample(atoms, k=min(n_redef_events, len(atoms)))
        # If we still need more redefs than unique atoms, cycle with replacement.
        while len(targets) < n_redef_events:
            targets.append(rng.choice(atoms))
        redef_groups = [
            [Definition("REDEF", tgt.lhs, fresh[i], ku_index=-1)]
            for i, tgt in enumerate(targets[:n_redef_events])
        ]
        if n_timesteps is not None:
            # Replace trailing empty composition slots first, then append if needed.
            for rg in redef_groups:
                placed = False
                for i in range(len(groups) - 1, -1, -1):
                    if not groups[i]:
                        groups[i] = rg
                        placed = True
                        break
                if not placed:
                    groups.append(rg)
            # Trim / pad to exact n_timesteps.
            if len(groups) > n_timesteps:
                groups = groups[:n_timesteps]
            while len(groups) < n_timesteps:
                groups.append([])
        else:
            groups.extend(redef_groups)

    tg = TimedGrammar(defs=[])
    kus: List[KU] = []
    used_inputs: Set[str] = set()
    for t, group in enumerate(groups):
        stamped = [Definition(d.kind, d.lhs, d.rhs, ku_index=t) for d in group]
        tg.defs.extend(stamped)
        new_syms = [d.symbol for d in stamped]
        ex, pr = _examples_and_probes(tg, t, new_syms, n_examples, n_probes,
                                      rng, used_inputs)
        used_inputs.update(inp for inp, _ in ex)
        used_inputs.update(p.inp for p in pr)
        kus.append(KU(ku_index=t, definitions=stamped, examples=ex, probes=pr))

    out_kind = kind
    if n_timesteps is not None:
        out_kind = "configured"
        if do_redefs:
            out_kind = "configured+redef"
    return Benchmark(kus=kus, grammar=tg, kind=out_kind, seed=seed)


# ---------- learner interface ----------

class Learner(ABC):
    """Protocol A: parametric per-KU training. Sees one KU at a time, then predicts."""

    def reset(self) -> None:
        pass

    @abstractmethod
    def train_on_ku(self, ku: KU) -> None: ...

    @abstractmethod
    def predict(self, probe: Probe) -> str: ...


class NullLearner(Learner):
    """Always answers ''. Floor baseline."""
    def train_on_ku(self, ku: KU) -> None: pass
    def predict(self, probe: Probe) -> str: return ""


class OracleLearner(Learner):
    """Cheats by returning the ground truth. Ceiling sanity-check."""
    def train_on_ku(self, ku: KU) -> None: pass
    def predict(self, probe: Probe) -> str: return probe.out


class MemorizingLearner(Learner):
    """Memorises (input -> output) across all training examples it has ever seen.

    Solves probes whose input appeared in a prior KU's examples; fails on novel
    compositions. Useful contrast to OracleLearner: the gap between them on
    held-out probes is the compositional-generalisation signal.
    """
    def __init__(self) -> None:
        self.memory: Dict[str, str] = {}

    def reset(self) -> None:
        self.memory.clear()

    def train_on_ku(self, ku: KU) -> None:
        for inp, out in ku.examples:
            self.memory[inp] = out  # REDEFs overwrite naturally.

    def predict(self, probe: Probe) -> str:
        return self.memory.get(probe.inp, "")


# ---------- runner + metrics ----------

@dataclass
class Benchmark:
    kus: List[KU]
    grammar: TimedGrammar
    kind: str = "topological"
    seed: int = 0

    def run(self, learner: Learner) -> Dict:
        """Train sequentially, evaluate retrospectively. Returns an accuracy grid.

        grid[t][s] = accuracy on KU s's probes after training on KUs 0..t.
        None means KU s had no probes.
        """
        learner.reset()
        n = len(self.kus)
        grid: List[List[float | None]] = [[None] * n for _ in range(n)]
        for t in range(n):
            learner.train_on_ku(self.kus[t])
            for s in range(t + 1):
                probes = self.kus[s].probes
                if not probes:
                    continue
                correct = sum(1 for p in probes
                              if learner.predict(p).strip() == p.out.strip())
                grid[t][s] = correct / len(probes)
        return {"grid": grid, "kus": self.kus}


def metrics(result: Dict) -> Dict:
    grid = result["grid"]
    kus = result["kus"]
    n = len(grid)

    acq = [grid[t][t] for t in range(n) if grid[t][t] is not None]
    retention = []
    for s in range(n):
        later = [grid[t][s] for t in range(s + 1, n) if grid[t][s] is not None]
        if later:
            retention.append(sum(later) / len(later))

    redef_kus = [t for t, ku in enumerate(kus)
                 if any(d.kind == "REDEF" for d in ku.definitions)]
    update_fidelity = {t: grid[t][t] for t in redef_kus if grid[t][t] is not None}

    return {
        "acquisition_mean": (sum(acq) / len(acq)) if acq else None,
        "retention_mean": (sum(retention) / len(retention)) if retention else None,
        "update_fidelity": update_fidelity,
        "n_redefs": len(redef_kus),
    }
