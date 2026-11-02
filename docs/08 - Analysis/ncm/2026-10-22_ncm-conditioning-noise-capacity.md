# Conditional NCM noise: capacity gate failure before full training

Date: 2026-10-22. `ncm-conditioning-noise-capacity-v1` is complete, classified
**all_level_noise_capacity_failed**. The proposed original full residual-learning
comparison is stopped before training. The95% complete-summary goal remains
unmet; no task accuracy or speedup is established by this diagnostic.

| Noise recipe | Correct non-PAD target tokens | Exact target texts | Capacity gate >=95% |
| --- | --- | --- | --- |
|clean|4440/4,452=99.7305%|116/128|yes|
|root_only|4429/4,452=99.4834%|106/128|yes|
|all_levels|4226/4,452=94.9236%|26/128|no|
|mild_fine|4420/4,452=99.2812%|100/128|yes|

These are **target-aware reconstruction** measurements on128 hash-selected
existing training summaries, using the frozen polished codec. Each finer expert
knows the correct target latent and requantizes against actual corrupted prior
codes. They are not generated-summary pass rates or a held-out model evaluation.

The root-only recipe replaces half the coarse codes in expectation with one of
eight nearest non-identical vectors. All-level mode additionally replaces finer
codes uniformly, with probabilities linearly.5to0. Mild mode keeps the same
root replacements and thins the fine masks to probability.05 at the first fine
level, falling linearly to0. Last-level codes remain uncorrupted. Common draws,
nested masks and identical replacement IDs control the comparison. Every
declared mode is reported; none is chosen using development task outcomes.

No-noise labels exactly match vendor quantization. Codec state, code reload,
finiteness and GPU1 lifecycle checks passed. Quantization is FP32, neighborhood
distances FP64, decoder FP16, batch32. Minimum distinct-neighbor squared distance
was1.30769. Own peak allocation was855,412,736bytes. One seed and128 training
targets limit the interpretation; no statistical guarantee about unseen text
or student quality is claimed.

The original v1 conditional-residual learning pilot completed two real updates
per arm with matched initialization/noise, nonzero query and adapter gradients,
causal coarse inputs, no same/future fine-label leakage, unchanged codec and
exact checkpoint reload. Those are systems checks, not the1,024-update mechanism
endpoint. The128-target capacity measurement then failed its predeclared95%
gate for v1's all-level noise, so full v1 training was never launched. Its passed
pilot and original source copies from commit13fe31e remain intact.

The explicit amendment creates a fresh v2 study with mild fine noise, qualified
at99.2812% oracle reconstruction, and requires a matching new pilot. Its full
comparison is running separately; **no v2 training result is included here**.
Even its eventual mechanism pass would only support a separately declared
broader training run with actual source-disjoint summary grading. Original-body
and codec weights remain frozen; inference receives no reference text or codes.

The failed original producer's runner path has intentionally advanced to v2.
Finalization checks its preserved source bytes against the original producer
hash, and checks every unchanged source/input normally. It does not rewrite
the old hash or confuse new code with the old run. The amendment and verification
receipt are part of the aggregate.

Aggregate: `data/experiments/ncm_paper_regime/20261022_noise_capacity_results_v1/campaign_results.json`; SHA256
`c7016166f7cc9f4d0e1efa38ec37f3371556409aedbda5261d0ac85ae81dce8c`. Reproduce the capacity measurement
with `experiments.structure_net.ncm_conditioning_noise_capacity`, following its
immutable plan and recorded moba runtime/GPU mapping. Registry records retain
confirmed=false, underpowered=true and explicit unexecuted main-training status.
The final256-case task test stays sealed.
