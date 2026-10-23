# Jacobi Forcing is a closer training reference for the frozen-native model

Date: 2026-10-23. Exploratory source audit, not a local upstream benchmark.
Pinned official revision: `028dbd6283fdc6fa356f0b109ce04c5ca538e49f`.
Snapshots and SHA256 manifests are under
`data/experiments/ncm_paper_regime/20261023_jacobi_forcing_reference_v1`.

[Jacobi Forcing](https://github.com/hao-ai-lab/JacobiForcing/tree/028dbd6283fdc6fa356f0b109ce04c5ca538e49f)
publishes models, trajectory generation, training and inference implementations.
It retains a causal pretrained model and trains on imperfect parallel word
states. Its reported coding/math results motivate an experiment; they do not
establish summarization quality or speed on our hardware.

The earlier [repository comparison](2026-10-23_ncm-upstream-repository-comparison.md)
still applies: official NCM has released inference/checkpoints but no trainer in
the inspected tree; SpecForge has concrete native-vocabulary training but its
speculative target verifier changes the system being evaluated.

| Mechanism | Released Jacobi Forcing implementation | Current local adaptation |
| --- | --- | --- |
| Language path | Intact pretrained causal model | All28native Llama3B blocks retained; original weights frozen |
| Trainable weights | Example launcher uses full-model training; optional LoRA branch | Only new query attention adapters, concept heads, hint and level biases;8,637,696parameters |
| Inputs | Native tokens in paired noisy/final trajectory blocks | Five residual concept levels, then native-token parallel word states |
| Histories | Prepared model trajectories with progressive noise | Source-only generated coarse history; word state from0–3current student passes; noisy suffix grows8→128 |
| Loss | Clean AR cross-entropy×10 plus detached clean-branch soft targets on noisy positions | Native-center word loss×10; hard labels plus offline top32original-teacher probabilities; coarse conditional-residual objectives |
| Attention | Packed clean/noisy branches with structural block masks and separate prior histories | Source-only frozen KV plus causal query blocks; no gold targets in source cache |
| Primary decoding | Iterative causal block decoding; repository also provides multiblock/rejection recycling | Fixed five concept plus four word passes, no AR fallback, external verifier or convergence rescue |
| Quality evidence | Authors' coding/math benchmarks | Four-update GPU lifecycle passed; trained student development accuracy still pending |

The actual example launch calls
[`soft_flexattn_train_cllm_multiblock.py`](https://github.com/hao-ai-lab/JacobiForcing/blob/028dbd6283fdc6fa356f0b109ce04c5ca538e49f/JacobiForcing/train/soft_flexattn_train_cllm_multiblock.py).
Its objective is in
[`soft_flexattn_cllm_trainer_multiblock.py`](https://github.com/hao-ai-lab/JacobiForcing/blob/028dbd6283fdc6fa356f0b109ce04c5ca538e49f/JacobiForcing/train/soft_flexattn_cllm_trainer_multiblock.py),
not merely the legacy `cllm_trainer.py`. Inspection confirms the current trainer
also weights AR cross-entropy by10. It compares noisy-branch logits to detached
clean-branch distributions and masks duplicate prefixes/padding. The inspected
n64launcher uses four GPUs, BF16, full-model training,16Kpacked sequences and
learning rate1e-6. Our single3060adapter run is a materially different recipe.

The actionable changes are preserving native language inputs, training against
the histories used at inference, preserving a clean language objective and
measuring complete generated summaries. These address plausible interface and
exposure mismatches; no ablation yet isolates which caused the previous0/64
failure. The upstream trainer has been inspected, not run locally.

The local revision passed CPU causality/target-leakage checks and the real GPU
mixed-task/cache/reload pilot. Its preregistered2048updates use143accepted summary
labels and1024raw language windows. After training, judge all64original unseen
development summaries, require at least61passes from qualified Terra and then
Sonnet, and keep final256cases sealed until that gate passes. This remains a
short-source prototype; representative full-range final evaluation is required.

If the fixed nine-pass output fails, report the complete failure. More training
loss reduction, teacher123/128quality or99.22%teacher-forced numerical agreement
cannot substitute for student summary accuracy. Measure latency and tokens/s
only after quality passes; adding parallelism or adapters alone proves neither
compression nor acceleration.
