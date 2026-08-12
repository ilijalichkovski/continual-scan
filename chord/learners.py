"""Concrete learners.

Parametric (Protocol A):
  - SubstitutionInductionLearner -- honest mid-tier baseline, stdlib only.
      Learns atom -> output mappings from atom-only examples, induces
      compositional "skeletons" from multi-token examples, then predicts
      by skeleton-matching + atom substitution. Won't solve probes whose
      skeleton was never demonstrated; will solve probes whose skeleton
      matches a known one even when the atoms in the probe weren't in any
      example of that skeleton (the genuine compositional generalisation
      signal).

In-context (Protocols B / C):
  - CumulativeICL  -- Protocol B: prompt contains all KUs seen so far.
  - WindowedICL    -- Protocol C: prompt contains only the latest KU.

Both ICL learners are agnostic to the LLM: they take a callable
`llm: Callable[[str], str]`. Two shipped:
  - OracleLLM: uses the symbolic oracle. Lets you exercise ICL plumbing
    without any API keys.
  - openai_llm(model=...): a stdlib HTTP client. Activates only if
    OPENAI_API_KEY is set; raises a clear error otherwise.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Callable, List, Optional, Tuple

from .bench import Learner
from .core import KU, Probe, TimedGrammar, oracle


# ---------- 1. SubstitutionInductionLearner ----------

Skeleton = Tuple[str, ...]   # e.g. ("<A0>", "thrice")
Recipe = Tuple[int, ...]     # output token i comes from atom_order[recipe[i]]


class SubstitutionInductionLearner(Learner):
    """Simple induction baseline. ~50 lines of stdlib Python.

    Training:
      For each example (inp, out):
        - if inp is a single token, record atom_map[inp] = out.
        - else, factor inp into (skeleton, atom_order) where atom_order lists
          the distinct atoms in left-to-right order and the skeleton replaces
          atom positions with <A0>, <A1>, ... . Try to express out as a
          permutation of atom outputs; if so, record (skeleton, recipe).

    Prediction:
      Factor probe.inp the same way. If the skeleton is known, apply its
      recipe to the probe's atoms; else return ''.

    Honest: it cannot solve novel skeletons. It can solve novel atoms within
    a known skeleton -- which is exactly the compositional-generalisation
    case the benchmark is built to probe.
    """

    def __init__(self) -> None:
        self.atom_map: dict[str, str] = {}
        self.skeletons: dict[Skeleton, Recipe] = {}

    def reset(self) -> None:
        self.atom_map.clear()
        self.skeletons.clear()

    @staticmethod
    def _factor(inp_toks: List[str], atom_map: dict) -> Optional[Tuple[Skeleton, List[str]]]:
        atom_order: List[str] = []
        pos: dict = {}
        skel: List[str] = []
        for tok in inp_toks:
            if tok in atom_map:
                if tok not in pos:
                    pos[tok] = len(atom_order)
                    atom_order.append(tok)
                skel.append(f"<A{pos[tok]}>")
            else:
                skel.append(tok)
        if not atom_order:
            return None
        return tuple(skel), atom_order

    def train_on_ku(self, ku: KU) -> None:
        # Pass 1: atom -> output mappings (single-token examples).
        for inp, out in ku.examples:
            ti, to = inp.split(), out.split()
            if len(ti) == 1 and len(to) == 1:
                self.atom_map[ti[0]] = to[0]  # REDEFs naturally overwrite.

        # Pass 2: skeleton induction (multi-token examples).
        for inp, out in ku.examples:
            ti = inp.split()
            if len(ti) < 2:
                continue
            factored = self._factor(ti, self.atom_map)
            if factored is None:
                continue
            skel, atom_order = factored
            # Try to express each output token as the output-of-one-of-the-atoms.
            atom_outs = {self.atom_map[a]: i for i, a in enumerate(atom_order)}
            recipe: List[int] = []
            ok = True
            for ot in out.split():
                if ot in atom_outs:
                    recipe.append(atom_outs[ot])
                else:
                    ok = False
                    break
            if ok and recipe:
                self.skeletons[skel] = tuple(recipe)

    def predict(self, probe: Probe) -> str:
        toks = probe.inp.split()
        if len(toks) == 1:
            return self.atom_map.get(toks[0], "")
        factored = self._factor(toks, self.atom_map)
        if factored is None:
            return ""
        skel, atom_order = factored
        recipe = self.skeletons.get(skel)
        if recipe is None:
            return ""
        out_toks = [self.atom_map[atom_order[i]] for i in recipe]
        return " ".join(out_toks)


# ---------- 2. ICL learners ----------

LLM = Callable[[str], str]


def format_definitions(defs) -> str:
    return "\n".join(str(d) for d in defs)


def format_examples(rows) -> str:
    return "\n".join(f"IN: {i}  OUT: {o}" for i, o in rows)


def format_ku(ku: KU) -> str:
    return (
        f"--- KU {ku.ku_index} ---\n"
        f"*DEFINITIONS*\n{format_definitions(ku.definitions)}\n"
        f"*EXAMPLES*\n{format_examples(ku.examples)}"
    )


# Back-compat aliases for earlier private names.
_format_definitions = format_definitions
_format_examples = format_examples
_format_ku = format_ku


PROMPT_HEADER = """You are evaluating a symbolic rewrite-rule puzzle.

Rules are introduced over time in Knowledge Units (KUs).
Each definition is tagged DEF (new symbol) or REDEF (overwrite of an existing
symbol). The "rules in force at time t" are the most recent DEF/REDEF, per
symbol, with ku_index <= t. An iconic concatenation rule `u1 x1 -> [u1] [x1]`
is always implicitly present as a low-priority fallback.

Apply the rules in force at the requested evaluation time. Output ONLY the
sequence of result tokens separated by single spaces. No commentary.
"""
_PROMPT_HEADER = PROMPT_HEADER


class _ICLBase(Learner):
    """Base for ICL learners: keeps a list of KUs seen, formats a prompt."""

    def __init__(self, llm: LLM, max_examples_per_ku: Optional[int] = None) -> None:
        self.llm = llm
        self.max_examples_per_ku = max_examples_per_ku
        self.seen: List[KU] = []

    def reset(self) -> None:
        self.seen = []

    def _ku_for_prompt(self, ku: KU) -> KU:
        if self.max_examples_per_ku is None:
            return ku
        return KU(
            ku_index=ku.ku_index,
            definitions=ku.definitions,
            examples=ku.examples[: self.max_examples_per_ku],
            probes=[],
        )

    def _context_kus(self) -> List[KU]:  # overridden
        raise NotImplementedError

    def predict(self, probe: Probe) -> str:
        ctx = "\n\n".join(_format_ku(self._ku_for_prompt(k))
                          for k in self._context_kus())
        prompt = (
            _PROMPT_HEADER
            + "\n" + ctx
            + f"\n\n*QUERY* (evaluate at t={probe.eval_at})\n"
            + f"IN: {probe.inp}  OUT:"
        )
        return (self.llm(prompt) or "").strip()


class CumulativeICL(_ICLBase):
    """Protocol B: context = all KUs seen so far. The natural long-context LLM setup."""

    def train_on_ku(self, ku: KU) -> None:
        self.seen.append(ku)

    def _context_kus(self) -> List[KU]:
        return self.seen


class WindowedICL(_ICLBase):
    """Protocol C: context = only the most recent KU. Stateless ICL.

    By design this fails on probes that require any KU other than the latest;
    that gap is what genuine continual-learning machinery has to bridge.
    """

    def train_on_ku(self, ku: KU) -> None:
        self.seen = [ku]

    def _context_kus(self) -> List[KU]:
        return self.seen


# ---------- 3. LLM client adapters ----------

def make_oracle_llm(grammar: TimedGrammar) -> LLM:
    """A perfect ICL participant: uses the symbolic oracle to answer.

    Useful for verifying the ICL plumbing end-to-end without an API key. Read
    the prompt only to find the `IN: ... OUT:` line and the `t=<int>` marker.
    """
    import re
    in_re = re.compile(r"IN:\s*(.+?)\s+OUT:\s*$")
    t_re = re.compile(r"evaluate at t=(\d+)")

    def llm(prompt: str) -> str:
        m_in = in_re.search(prompt)
        m_t = t_re.search(prompt)
        if not m_in or not m_t:
            return ""
        return oracle(grammar, m_in.group(1), int(m_t.group(1)))

    return llm


def make_openai_llm(model: str = "gpt-4o-mini",
                    temperature: float = 0.0,
                    api_key_env: str = "OPENAI_API_KEY",
                    base_url: str = "https://api.openai.com/v1/chat/completions",
                    timeout: float = 60.0) -> LLM:
    """Minimal OpenAI Chat Completions client using only stdlib http.

    Returns a callable suitable for `CumulativeICL(llm=...)`. Requires
    OPENAI_API_KEY in env. No `openai` package needed.
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        def _err(prompt: str) -> str:
            raise RuntimeError(
                f"{api_key_env} is not set; cannot call OpenAI. "
                f"Set it or use make_oracle_llm(...) instead."
            )
        return _err

    def llm(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            base_url, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return f"<HTTPError {e.code}: {e.read().decode('utf-8', 'ignore')}>"
        except Exception as e:  # noqa: BLE001
            return f"<Error: {e}>"
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            return ""

    return llm
