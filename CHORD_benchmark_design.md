# A Principled Continual Learning Benchmark from Compositional Primitives

*Sister benchmark to COLD. Method-independent. No language prior required.*

---

## 1. Motivation

COLD evaluates continual learning on text-based knowledge units. It is the most useful benchmark *for current LLMs*, but its KUs presuppose that the learner understands English. This is a baked-in inductive prior that makes COLD silent on a deeper question: **what is the fundamental capability of continual learning, independent of any particular learner's priors?**

ARC-AGI feels fundamental partly because its puzzles can in principle be tackled by any learner with the right perceptual scaffolding — humans, neurosymbolic systems, LLMs. We want the same property here, but for *continual* learning specifically: the ability to acquire, retain, compose, and update knowledge over a sequence.

This benchmark — provisional name **CHORD** (Compositional Hierarchical Online Rule Discovery) — ports the COLD methodology onto a symbolic substrate. It inherits COLD's core principles (procedural generation, dependency-aware KUs, ICL-calibrated difficulty) and removes the language prior. The substrate is the algebraic rewrite-rule paradigm of SCAN (Lake & Baroni, 2018), COGS (Kim & Linzen, 2020), and Meta-Learning for Compositionality (Lake & Baroni, *Nature*, 2023), extended for the first time with sequential temporal structure.

The promise: **a fully procedural continual learning benchmark — DAG construction, primitive construction, example construction, and adversarial schedule construction all generated from first principles.**

---

## 2. Position relative to existing work

| Work | Compositional? | Procedural? | Sequential / CL? | Method-independent? |
|---|---|---|---|---|
| SCAN | Yes | Limited | No | Yes |
| COGS | Yes | Limited | No | Yes |
| MLC (Lake 2023) | Yes | Yes | No | Yes |
| ESCAIP | Yes | Yes | No | Yes |
| TRACE / TiC-LM | No | No | Yes | No (LLM-bound) |
| COLD | Yes (text KUs) | Yes | Yes | No (LLM-bound) |
| **CHORD** | **Yes** | **Yes** | **Yes** | **Yes** |

The novelty hook is the upper-right cell. Compositional generalization benchmarks have universally used a *fixed* grammar per problem. Sequential benchmarks have universally been built on natural language. **An evolving-grammar, sequentially-introduced, dependency-aware compositional benchmark is virgin territory.**

---

## 3. The substrate: algebraic rewrite rules

CHORD inherits the formal grammar from MLC and ESCAIP. A puzzle is defined by:

- An alphabet of **primitive symbols** Σₛ (e.g. `$, &, %, @`).
- An **interpretation function** ι : Σₛ → Σ\* mapping each symbol to a concrete base string (e.g. `$ → "abc"`).
- A set of **operator** and **modifier** functions over strings.
- A **target expression** built from these primitives via a context-free grammar with deterministic left-to-right evaluation.

A KU corresponds to introducing one of the following:

- **Atomic primitive**: `$ → "abc"` — a new symbol-to-string mapping.
- **Unary function**: `f(x) = reverse(x)` — applied to one argument.
- **Binary function**: `f(x, y) = concat(x, y)` — applied to two arguments.
- **Higher-order modifier**: `μ(f) = f composed with reverse` — transforms a function.

The COLD methodology — minimal teaching signal, ICL calibration, dependency-aware difficulty — applies cleanly to this substrate.

---

## 4. Core design: the temporal DAG

In single-shot benchmarks (SCAN, ESCAIP), each problem is a self-contained DAG of dependencies among the primitives, operators, and target expressions. The set-cover algorithm picks the minimal definition set so the target is uniquely solvable.

CHORD extends this in one critical way: **the dependency DAG is unrolled across time.**

A *benchmark instance* in CHORD consists of:

1. A target expression *E* of bounded compositional complexity.
2. A DAG of primitives, functions, and intermediate compositions whose union is sufficient to evaluate *E*.
3. A **temporal schedule** — a topological ordering of the DAG into KUs KU₁, KU₂, …, KUₙ, with an evaluation probe attached to each KU.

The schedule is not arbitrary. It is constructed to systematically test specific CL capabilities:

### 4.1 Schedule types

- **Pure topological** — primitives introduced first, then unary functions over those primitives, then binary functions, then higher-order compositions. Tests basic forward dependency tracking.
- **Interleaved** — KUs from different DAG levels mixed together. Tests resistance to interference between unrelated primitives introduced in adjacent slots.
- **Backward composition** — a function is introduced before all its eventual operand primitives. When a new primitive is introduced later, the learner must retroactively recognize it as a valid argument to the earlier-learned function. Tests genuine abstraction (the function was learned as a *function*, not as a finite lookup table).
- **Adversarial / redefinition** — a KU explicitly redefines a previously-introduced primitive (`$ → "abc"` at KU₃, then `$ → "xyz"` at KU₁₇). The probe at KU₁₇ tests the new definition; probes for *prior* tasks involving `$` test whether the learner correctly updates downstream compositions. This is the symbolic analog of the "lessons not statistics" updating that the text COLD benchmark tests, and it's the schedule type with no precedent in the SCAN/COGS/MLC lineage.

The temporal schedule is the load-bearing innovation. Without it, this is just MLC; with it, this is the first procedural benchmark for compositional continual learning.

### 4.2 Schedule construction is itself procedural

CHORD does not ship a fixed schedule. Schedules are generated from a configurable specification:

- DAG depth, branching factor, primitive count, operator count
- Schedule type (topological / interleaved / backward / adversarial / mixed)
- Adversarial fraction (what % of KUs redefine prior content)
- Probe density (how many evaluation tasks per KU, drawn from the cumulative knowledge so far)

This makes CHORD a *benchmark generator*, not a frozen artifact. A frozen v1 corpus is published for leaderboard comparability; the generator is the actual product.

---

## 5. Knowledge units: what the learner sees

Each KU in the schedule is a tuple `(definitions, examples, probe)`:

- **Definitions** — a small set of explicit symbol/operator/modifier specifications introducing the new content of this KU.
- **Examples** — *k* (input, output) demonstration pairs drawn from the new rule's distribution. *k* is selected by the same set-cover-based verified-definition procedure used in ESCAIP: the minimum number of examples such that the rule is uniquely identifiable among all rules in the function library. This is the symbolic analog of COLD's minimal-teaching-signal principle.
- **Probe** — a held-out evaluation task. The probe may require *only* the new KU's content, or it may require composition with content from *any* prior KU.

Two values of *k* are exposed:
- *k*_sparse — the actual benchmark setting; minimal exposure.
- *k*_dense — a calibration setting with abundant exposure, used as an oracle to verify that the rule is in-principle learnable from the demonstrations alone. This is the direct symbolic translation of COLD's ICL-calibration methodology: if the dense-ICL learner cannot solve a probe given the full context of all prior KUs, the eval is malformed and gets filtered out before scoring.

---

## 6. Evaluation protocols

A single CHORD instance can be scored under three protocols. The same model can be evaluated under all three; different methods naturally favor different protocols. Reporting all three is what makes the benchmark method-independent.

### Protocol A — Parametric

The learner is fine-tuned (or otherwise weight-updated) sequentially on KU₁, KU₂, …, KUₙ. After each KU, the learner is evaluated on probes from all prior KUs. This is the standard CL protocol used by TRACE, TiC-LM, and the parametric track of COLD.

What it measures: catastrophic forgetting, retention, forward transfer, plasticity-stability tradeoffs.

### Protocol B — ICL-Cumulative

At step *t*, the learner's context contains the demonstrations from KU₁ through KU_t concatenated. The learner is evaluated on probes from all prior KUs. This protocol measures whether the model can compose information across a long, structured context — essentially asking whether long-context ICL is a *substitute* for genuine continual learning.

What it measures: long-context compositional retrieval; whether ICL scales as a CL strategy.

### Protocol C — ICL-Windowed

At step *t*, the learner's context contains *only* KU_t's demonstrations. The learner is evaluated on probes that may require KUs from earlier in the schedule. This protocol is **impossible by design** for a stateless ICL learner; that's the point. It exposes the gap that genuine continual-learning machinery — RAG, episodic memory, summarization, parameter updates — must bridge.

What it measures: which methods successfully bridge the no-context gap, and how.

The headline metric is not a single number but a **profile** across protocols: a method's CHORD scorecard reports A, B, and C separately, capturing the different shapes of CL capability.

---

## 7. Metrics

For each protocol, the following are computed across the KU sequence:

- **Acquisition**: probe accuracy at the KU where the relevant content was introduced.
- **Retention**: probe accuracy at later KUs for content introduced earlier. Decomposed into:
  - *Standard retention* — does the learner still know what it learned?
  - *Composition retention* — can it still compose old primitives with new ones?
- **Forward transfer**: speedup or accuracy gain on KU_t when prior KUs include compositionally relevant content vs. when they don't (controlled via the schedule generator).
- **Backward composition**: success rate on probes requiring a primitive introduced *after* the function it's used with.
- **Update fidelity**: on adversarial schedules, accuracy on probes for the *new* definition of a redefined primitive, alongside accuracy on probes for prior compositions that must be updated.
- **Robustness profile**: variance across multiple seeded schedule realizations of the same difficulty class.

A single CHORD score is a vector, not a scalar. Aggregation into a leaderboard number is a deliberate choice we make at publication time, but the underlying evaluation produces all six axes.

---

## 8. What's procedurally generated

Everything. To make this concrete, the full pipeline:

1. **Sample a target DAG**: depth, branching factor, primitive count, operator count drawn from the configuration.
2. **Run set-cover on the DAG**: determine the minimal set of primitives, operators, and modifiers needed to evaluate the target.
3. **Topologically order the DAG**: produce one or more valid orderings consistent with dependency.
4. **Apply the schedule type**: rearrange / interleave / inject adversarial nodes per the schedule specification.
5. **For each KU**: generate verified-minimal demonstrations (ESCAIP-style) for the new content, plus held-out probes.
6. **Run ICL calibration**: dense-context oracle pass to filter malformed probes.
7. **Emit**: the schedule as a sequence of `(definitions, examples, probe)` tuples plus ground-truth answers.

No human authoring is involved at any stage. This is what makes CHORD a *methodology* and not just a dataset. Researchers running CL on a custom domain — a particular code dialect, a robotics primitive library, anything that admits compositional structure — can plug in their own primitives and operators and inherit the rest of the pipeline.

---

## 9. The ICL-calibration story translates cleanly

In COLD, ICL calibration solves the "is this probe answerable in principle" problem. The same logic applies here, with one improvement: in the symbolic setting, we can additionally verify answerability by running a *symbolic solver* (which has perfect access to the rule library) on the probe. If the symbolic solver succeeds and the dense-context ICL learner fails, that's a strong signal that the learning method — not the eval — is what's being measured. If the symbolic solver fails, the probe is malformed and gets dropped.

This dual oracle (symbolic + dense-ICL) is something COLD cannot have, because there's no symbolic ground-truth solver for natural-language KUs. It makes CHORD's filtering more rigorous than its sibling.

---

## 10. The exploration question, deferred

A natural extension is to let the learner *actively* probe each rule — request inputs, observe outputs, decide when it has learned enough. We considered this and explicitly rejected it for the base benchmark: requiring active exploration raises the minimum agent complexity (planning, budget management, termination criteria), which damages method-independence. A pure SGD fine-tuner cannot "explore"; it just consumes data.

The middle path — fixed exposure budget per KU, with sparse and dense settings — captures most of the value while preserving universality. An active-exploration track ("CHORD-Interactive") may ship later as a sibling track for agentic methods.

---

## 11. Why this works for the ARC-AGI conversation

The talking points are:

1. **CHORD and COLD are siblings, not v1/v2.** COLD is the maximally-useful version for current LLMs. CHORD is the maximally-fundamental version, free of any language prior. Same methodology, two arrows at the target.
2. **CHORD is the natural temporal extension of MLC.** Lake's group showed that meta-learning over compositional grammars matches human compositional generalization in the one-shot case. CHORD asks the obvious next question — what about the sequential case? — and provides the first principled benchmark for it. This is the kind of credibility hook that makes a benchmark feel fundamental rather than just hard.
3. **It is method-independent.** Humans, neurosymbolic systems, LLMs, RL agents with weight updates, RAG-augmented models — all can be evaluated on CHORD. The three-protocol scorecard makes the comparison apples-to-apples in a way no existing CL benchmark allows.
4. **Procedural top to bottom.** Researchers can generate their own CHORD instances tuned to their needs. The frozen v1 corpus is the leaderboard; the generator is the standard.

---

## 12. Open design questions worth thinking through

- **DAG complexity calibration.** What's the right axis along which to scale difficulty — DAG depth? Operator count? Schedule adversarial fraction? Probably a multi-dimensional difficulty profile rather than a scalar.
- **Adversarial KU semantics.** When a primitive is redefined, do prior probes test the *original* or the *updated* definition? The cleanest answer is: probes are timestamped, and a probe attached to KU_t at the moment of its introduction tests against the rules in force at time *t*. But this requires careful spec design.
- **Cross-protocol fairness.** Protocols A and B differ in compute and memory cost; comparing methods naïvely across them favors the protocol that was cheap for that method. The scorecard should normalize for this somehow — possibly by reporting compute-per-KU alongside accuracy.
- **Human baseline.** Should we collect human performance on a small CHORD subset? The ARC-AGI move is "humans solve it easily, machines don't" — if humans crush CHORD, that's a useful headline; if they struggle on adversarial schedules, that's an interesting finding in its own right.

---

## References

- Lake, B. M. & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. *ICML*. [SCAN]
- Kim, N. & Linzen, T. (2020). COGS: A compositional generalization challenge based on semantic interpretation. *EMNLP*.
- Lake, B. M. & Baroni, M. (2023). Human-like systematic generalization through a meta-learning neural network. *Nature* 623, 115–121. [MLC]
- Lichkovski, I. (2024). ESCAIP: Evaluating Symbolic Compositionality from Aligned Inductive Priors. [Compositional Puzzles dataset]
- [COLD reference — Lichkovski et al., forthcoming]
