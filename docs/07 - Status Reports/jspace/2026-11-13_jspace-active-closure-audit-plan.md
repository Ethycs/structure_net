# Active-position closure audit — exploratory replay

This follow-up was designed after observing the main CAV/head results. It uses
the already evaluated worlds and unchanged heads. It is neither fresh accuracy
confirmation nor a new independent model replication.

Two questions:

1. How do the frozen heads score on the existing query-donor and answer-only
   donor natural examples? Report all five heads at both sites on all three
   variants. The 192 variants are correlated observations of 64 base worlds;
   do not pool them as 192 independent units or select another model.
2. Can two model states with bitwise identical retained coordinates yield
   different next retained coordinates or first-token decisions when the
   surrounding state differs?

Freeze the first four evaluation worlds in the original order for which both
source and answer-only donor were naturally correct. Re-run both prompts and
require clean correctness. Record blocks 24,25,29,30,35. For block 24 copy the
donor's 20 question states except the final position into the source; for 29
copy its final state. Use direct assignment, recording bitwise equality with
the donor, exact source-boundary replay and unchanged outside positions. Compare
the next retained state, full first logits and completed answer with the natural
donor. At block 35, final-state copying must exactly reproduce the donor's full
first logits, serving as a positive implementation control.

An exact retained-state match with a different next state is a concrete
counterexample to autonomous reduced dynamics **on this intervention-augmented
domain**. A changed first-token argmax additionally demonstrates task-output
dependence on omitted state. The hybrid state need not lie on the natural
activation manifold; these witnesses cannot prove impossibility of useful
approximate compression or task-specific early-exit heads on natural inputs.
They do not establish closure failure specifically for the paper's full J-space.

Use root locked Pixi; CPU preparation/scoring with CUDA hidden. GPU replay
requires a live 3060 reservation before any CUDA query/load. Archive producer,
protocol, input and model identities before replay, retain all raw states and
logits, enforce the allocation, and release on child exit. No new model fitting,
evaluation tuning or deployment occurs.
