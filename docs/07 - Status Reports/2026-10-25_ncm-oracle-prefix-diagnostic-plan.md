# Locate the conditional NCM generation failure with forced concept prefixes

2026-10-25. Exploratory hypothesis `ncm-level-conditioning-oracle-prefix-diagnostic-v1`.
The new fixed 1,024-update native655 checkpoint has already produced ten empty
outputs among 128 archived source-only predictions. It cannot satisfy the
unchanged 95% gate. Training updates, protected codec geometry and exact reload
have been verified; useful generalization has not.

Before new training, use exactly the first and longest-source cases of the
existing 64-case component cohort, with the already audited complete FP32
original-body features and 641 oracle codes from the qualified teacher targets.
Choose longest by source token count, then ID, with no selection using model
outputs. These are diagnostic cases, not the sealed final 320 cases.

Load the entire new `3911eab7...` checkpoint strictly. Freeze all parameters,
disable TF32 and use FP32 inference on physical GPU2. No parent body is loaded,
no gradients, optimizer steps, remote calls or semantic cache mutations occur.
Maximum allocated memory is 3 GB. Preserve every raw code and 128-word output.

For each case force the first K hierarchy levels to the oracle codes, for
K = 0, 1, 3, 6, 12, then greedily generate every remaining level with the
unchanged native generator. Implement an instance-local temporary sampling
override only; restore it even on exceptions. Require exactly one sampler call
per native level, exact forced prefix preservation, and equality between K=0
and ordinary source-only generation. K=12 must equal the complete oracle codes.
Repeat K=0 exactly before other conditions. All output decoding uses the
unchanged source-conditioned word decoder and native CLS renderer.

Record per-level cross entropy to oracle codes along each mixed history, final
latent mean squared error to the oracle latent, active non-PAD word-ID match,
word cross entropy and complete rendered outputs. At K=12 additionally decode
without source attention using the same decoder weights and oracle latent.
Verify identical model state, clean source contexts and no parameter gradients
at exit. Freeze source, plan, checkpoint and selected feature/code provenance.

These are causal diagnostics with privileged target information. Code matches
and word matches are not semantic accuracy; two cases do not establish a
generalization result. No forcing level may be selected as a deployable model
or used to rescue the failed candidate. Preserve a failed execution and use a
new versioned root for any changed producing code. This diagnostic does not
bypass the held semantic lineage or qualify a replacement judge.

Root: `20261025_level_conditioning_oracle_prefix_diagnostic_v1` under
`data/experiments/ncm_paper_regime/`. A separate CPU audit will verify the raw
prefixes, counts, native rendering, checkpoint bindings and producer hashes.
