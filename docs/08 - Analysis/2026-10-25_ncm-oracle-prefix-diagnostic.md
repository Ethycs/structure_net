# Correct codes restore decoding; correcting the root alone does not

The frozen new NCM checkpoint produced substantially more coherent text on
two designated diagnostic cases when supplied all correct target codes.
Supplying only the correct root did not fix either output. This localizes
failure to prediction across the concept hierarchy, without establishing
semantic accuracy or a successful training recipe.

Hypothesis: `ncm-level-conditioning-oracle-prefix-diagnostic-v1`.
The exploratory plan fixed the first and longest-source cases of the existing
64-case component cohort, the new checkpoint, and forcing levels before this
diagnostic ran. No model weights changed, and all source positions were kept.
These cases are separate from training; the final 320-case test was not read
or hashed. All privileged outputs are diagnostic and cannot qualify deployment.

| Diagnostic case | No supplied levels | First 1 | First 3 | First 6 | All 12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| First source: active word-ID matches / 71 | 4 | 4 | 2 | 13 | 66 |
| Longest source: active word-ID matches / 88 | 4 | 1 | 6 | 19 | 88 |

**These positional word matches are not semantic accuracy.** The raw complete
outputs, code predictions, per-level conditional log losses, word losses and
latent errors are preserved. The no-forcing outputs are incoherent on visual
inspection. The all-oracle outputs largely reconstruct the teacher text;
neither has been assigned a new semantic pass by a judge. Root-only forcing
even produced an empty output on the longest case. The all-oracle source-off
decoder control also produced substantially coherent text.

Even with correct earlier levels, later code predictions remain uncertain or
wrong: this is broader than an initial-root error propagating into an otherwise
qualified generator. We cannot infer the sole cause from two cases. Limited
source-to-code training, weak transfer from an unconditional generator and
training on histories unlike inference remain hypotheses, not established
explanations. More decoder-only repair is not supported by these observations.

The next proposed mechanism is a separate source-conditioned attention pass
over the completed predicted code hierarchy. It should learn from actual
generator errors, with the generator, codec and parent frozen. Correcting all
levels jointly and supervising the reconstructed latent can penalize
incompatible coarse/fine residual choices. This adds work; it is not itself a
speedup claim. Any learned correction must ultimately improve complete-summary
semantic quality on source-disjoint evaluation, with its latency measured.

Producer commit: `355411a`. GPU2, FP32, TF32 disabled. Peak allocated memory:
1,297,719,296 bytes. Exactly two cases and all five forcing conditions completed.
The zero-forcing code/word predictions matched ordinary generation exactly;
all-oracle codes matched the complete target hierarchy; sampling overrides and
source contexts were restored. Full model hashes remained unchanged and no
parameter gradients were created.

Checkpoint SHA-256:
`3911eab74db6ad13104f87765196e4212893a3e179ba0283e103e65e78273bec`.
Output SHA-256:
`4324dcb9d4fe3e958ca7ef5522afc93fc630aafd74f86768b62eb84eefbd4e69`.

Raw evidence under `data/experiments/ncm_paper_regime/`:

- `20261025_level_conditioning_oracle_prefix_diagnostic_v1/`
- `20261025_level_conditioning_oracle_prefix_audit_v1/`
- `20261025_level_conditioning_oracle_prefix_results_v1/`

The diagnostic worker and independent CPU binding/rendering audit exited 0.
Finalization checks actual tool receipts and every audit-bound artifact, then
registers this two-case exploratory result conservatively. This does not
complete the held semantic studies, repair their transport failures, or
establish the 95% goal.
