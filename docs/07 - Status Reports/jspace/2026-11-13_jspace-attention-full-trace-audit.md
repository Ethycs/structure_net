# Posthoc attention perturbation energy audit

Declared after JSPACE-ATTENTION-TRANSPORT-01 evaluation and analysis. This is a
diagnostic reuse of all 128 evaluation worlds, not an independent replication or
an altered selection/accuracy endpoint. The main result is immutable.

Observed compact country coordinates lose most of their standardized perturbation
at layer 27's attention update. That does not establish that the full hidden-state
perturbation disappears: it could move outside the eight directions or be hidden
by positional pooling. This audit distinguishes those possibilities.

Replay all 128 source, donor and selected layer-26 projected-J interventions with
unchanged prompts, dictionary, method and strengths. Assert identical complete
generated token IDs against frozen evaluation records. Collect the full prefill
attention write, post-addition residual and post-MLP residual for one world at a
time. Source and edited runs are paired. Never retain the full corpus of activations.

At each of the 75 boundaries, aggregate sum-squared full hidden differences,
their projections into that boundary's rank-eight J span, and unmodified state
energy. Report full difference L2, relative L2, and J projected energy fraction.
Zero difference has undefined projected fraction. Do not confuse a layer-specific
coordinate projection or pooled statistic with total perturbation energy.

Attention-output outer hooks run before the inner edit; replace the selected
attention snapshot with the existing runtime's explicitly post-edit snapshot.
Generation readback guards against hook side effects. Exact mathematical rank
constraints apply before BF16 rounding, so measure actual applied differences.

Freeze this protocol/script and hash the original result before collection.
Use root locked Pixi ncm, 3060=10240 MiB reserved for at most 30 minutes, existing
Stage guard before CUDA, periodic lease checks and append-only artifacts/manifests.
No new statistical pass/fail threshold or causal claim. confirmed=false.
