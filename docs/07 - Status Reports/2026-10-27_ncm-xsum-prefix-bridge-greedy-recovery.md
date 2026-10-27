# Preserve the released greedy policy without CUDA cumulative sampling

The v1 GPU pilot exited 1 (session 92104, chunk 96a48b) in the released
sampler's CUDA cumulative sum, which strict deterministic PyTorch rejects.
Fit-only normalization and pilot optimizer replay had run, but native generation
did not finish and no pilot result or full training was produced. Preserve
the original producer, normalizers, first-update snapshot, failure and receipt.

For the fixed temperature=1, top_k=1, top_p=0 policy, the released sampler first
keeps maximal logits, sorts them, and retains only the first sorted maximum.
Its categorical distribution therefore has one surviving entry. Version 2
uses the same top-k threshold, masked logits and sort, then returns that first
index. Preserve sort tie ordering; ordinary argmax need not choose the same
tied index. Reject all other sampling settings and nonfinite logits. This is
an equivalent policy implementation, not a temperature change or a relaxation
of deterministic execution. The unused categorical RNG draw is omitted;
bridge optimization has no stochastic forward operation and uses a separate
seeded source permutation.

Patch the sampler only on the model instance during generation and restore it
afterward. Do not edit vendor files or any model weights. Compare against the
actual released CPU sampler on random, extreme and tied finite logits before
a fresh GPU pilot. The original native/identity-prefix prediction controls
and exact optimizer replay remain required.

Use fresh prepare/pilot/train/evaluate/audit/results v2 roots and schema
`nal.ncm-xsum-prefix-bridge.v2`. Require byte-identical prepared cases to v1,
SHA256 `04cfc18fdaca0aea88bff865a17860815f75799deab9f85981f4f8e7507ede09`.
All models, input splits, optimizer settings, numerical and preservation gates
are unchanged. The main epoch4 trainer continues unmodified on GPU1.
