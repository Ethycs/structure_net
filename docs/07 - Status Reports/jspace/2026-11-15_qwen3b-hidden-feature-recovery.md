# Prospective hidden-feature diagnostic and recovery comparison

Status: preparation only. No GPU run, training change, new acceptance gate, or held-out access is authorized by this document. The frozen staged campaign continues unchanged. All proposed results are exploratory and `confirmed=false`.

The current joint objective is `.7 × T² KL(accepted packed parent || student) + .3 CE`, T=2, averaged over four output positions. There is no joint hidden-feature loss. Dense entrance prealignment ends before joint recovery; its target is the original next native boundary in the accepted parent's entrance coordinates. These objectives are implemented in `jspace_qwen_parallel_train.py:167–246` and `jspace_3b_entrance_distill.py:124–156`.

## First diagnostic, after the unchanged cut-23 baseline completes

Require completed matching train and fresh-verification manifests before choosing its selected checkpoint. Authenticate campaign/stage plans, checkpoint hashes, source archives, case identities, split/offset tables, geometry and native caches. Keep these references separate:

- Accepted twelve-block parent: cut24 selected checkpoint `db53db32597ebb4947e38f7324e31e6eb9c7686f491a9e8b9745b0c43bca8292`, with its own trained norm; native boundary24 is its input.
- Original native model: original `boundary_36.npy` before final RMSNorm and original full-vocabulary `teacher_logits.npy`, using the explicit original row mapping. These are not accepted-parent outputs.
- Cut23 raw: `pilot/initializer.pt`; prealigned: that exact raw stack/norm plus `prealign/selected_entrance.pt` applied only to block0; selected: `train/selected.pt`. All three run from the identical native boundary23 on their own resulting states.

First observe the SAME accepted-parent packed weights in inference mode and input-autograd Torch mode. Record actual compact outputs after inherited layers24–35, before their outgoing bridges, plus final reconstructed output. This measures the packed-versus-differentiable arithmetic floor; it is not recovery error. CPU tests cannot establish the CUDA/Triton floor. Also observe the equivalent unpacked parent to distinguish folding differences from the fused-kernel difference.

Then compare raw, prealigned and selected student trajectories to the accepted-parent trajectory. Report per-layer equal-case/valid-token relative RMS, last-valid-token (answer-position) relative RMS, absolute errors and per-case values. Keep the original-native projected final-feature comparison separate. Use existing64 development cases descriptively, without refitting/selecting from these diagnostics; capture a fixed training subset for prospective feature-loss normalization and gradient-scale checks. Do not infer that joint training destroyed entrance alignment without directly measuring it.

## Exact coordinate convention and observer

`CompactQwenStack` encodes `z = stack._geometry(h, U.T)` and exits with `h = stack._geometry(z, U, anchor)`. Its original anchor is `mean − (mean @ U) @ U.T`. Thus `h_native36 @ U35` is the source-consistent projected native target; no additional anchor subtraction or pseudoinverse is silently introduced. Original FP32 orthogonality makes this the corresponding coordinate in real arithmetic. Stored BF16 bases and rounded reconstruction make reconstruction→projection only approximate. Measure that roundtrip floor explicitly.

Accepted-parent feature targets must instead be the actual post-block coordinates. Do not reconstruct/project them or recompute bridges from FP32 fits. Preserve absolute layer indices, positional anchors, causal/key-padding masks and stored BF16 bridge matrices/shifts. The final basis remains stored after packing, although interior bases may be stripped.

New helper `experiments/structure_net/jspace_3b_hidden_feature_diagnostic.py` observes the existing `_run_reduced` calls inside the unchanged forward. Ordinary block hooks do not fire on this path. Its observer is serialized and restricted to the calling thread and supplied stack; the function binding is restored in `finally`, including on exceptions. It requires a frozen eval model, does not change weights/flags or run backward, and requires a reservation guard for CUDA. Returned detached tensors support bounded batchwise disk streaming. It performs no source loading, registration, selection or artifact writes. Callers own provenance and leases. This observer is diagnostic-only and must not be retained through a training backward pass.

## Two distinct prospective recovery questions

**A: missing final-feature anchor.** Keep current accepted-parent KL/CE and add normalized MSE between the student's actual final compact answer state and the frozen accepted parent's actual state at the same position. Student execution remains end-to-end through its own entrance and suffix, without teacher-state substitution. This isolates adding a feature anchor. Matching every inherited layer is a further intervention, not the same final-feature ablation.

**B: original-native teacher.** Use projected original-native final features and original-native logits in place of both accepted-parent targets. Keep the same CE and all other settings for a matched A/B comparison. This changes the teacher as well as providing the projected-feature-plus-KL mechanism; it is not evidence for A alone, nor an exact replication of an objective that omits CE. Native and accepted norms/readouts must not be interchanged when producing references.

Neither arm changes residual rank, heads, polynomial channels, geometry, architecture or inference work. Preserve the existing raw/warm candidate protection, native quality budget and packed fresh verification. A shared feature normalization and coefficient must be declared using training data/gradient-scale evidence before any new development comparison; they are not frozen here. No additional feature-error acceptance gate is proposed.

Native boundaries00–36 already exist for all512training/64development cases. Current accepted-parent captures contain only four-position full-vocabulary logits, not its inherited hidden trajectory. Fresh accepted-parent capture would run the pinned parent from native boundary24 in the existing split-specific batches of4. Training targets alone need approximately1MiB for final-answer coordinates,12MiB for all12 inherited answer states, or1.58GiB for all69,130 valid training tokens across those12 states, in BF16. Capture any development diagnostic targets separately; never fit normalization or losses on them. Full-state capture is unnecessary for the smallest final-answer feature arm.
