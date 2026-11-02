# Author word-selection policy on the fixed source-only NCM

Date: 2026-10-24. Hypothesis
`ncm-lm1b-native-word-readout-source-only-DEV-v1`. Exploratory, non-rescuing
follow-up after the fixed4096 model scored0/128 under the qualified semantic
Terra judge, all valid. Its failed campaign remains unchanged and registered
(SHA0e6e3706...). The full95% source-disjoint task goal remains unmet.

The native LM1B evaluation script applies strong word-frequency and adjacency
penalties. Our completed control used independent greedy word argmax. Isolate
this difference before training further: use the same final4096 attention
checkpoint SHA bb239a48..., same pretrained generator/domain codec, same FP32
features and all unchanged128 DEV cases. Recompute source-only generation from
features/mask only; require every concept ID and every unpenalized word argmax
to equal the previous128 outputs exactly. Never use gold words or concepts.

Compile only the unchanged author pick_tokens function from the pinned source
AST. Use its exact allowance from generator.pt:inference.token_unigram, frequency
penalty11.5, deficit2 and adjacency10/window4/decay0.5. The selector uses FP64;
all neural computation remains FP32 and TF32off. Do not tune those parameters,
rerank candidates, repair text, force boundaries or change concept sampling.
The existing predicted-leading-CLS frame renderer remains unchanged. This is
one word-readout control, not the full native unconditional sampling policy.

Use the existing representative batch1 CUDA lifecycle; run a fresh2-case
determinism/parity pilot before all128. Keep every raw128-word frame and641-code
hierarchy. Save all FP32 word logits and file/tensor digests for independent CPU
replay of every selected word. Full inference must match the retained pilot
exactly. Complete model hashes must equal the prior fixed model before/after.
No further weights are initialized or trained. Physical GPU0, one model only.

Inherit all2600 canonical first judgments from the registered failed baseline,
cache SHA65d6a4fa.... Preserve every old and new valid negative and exhausted
invalid. Same4-case validity-only judge pilot; all128 Terra. Sonnet is admitted
only if EACH complete64-case cohort has64valid and>=61Terra passes. Success
requires>=61of the SAME64 cases passing BOTH qualified judges in EACH cohort.
Invalids and all failures remain in denominators; no selected-case success.
Use the unchanged128-storage-token faithfulness/salience/readability rubric.

Fresh `20261024_lm1b_native_word_readout_*_v1` roots. Reuse only the immutable
source-only metadata and features; preserve all old artifacts. Independent CPU
audit replays the word selector on every saved logit frame, verifies unchanged
neural predictions, then replays source identities, first judgments, role
admission and same-case gates. Report and register conservatively. The final320
selected sources remain closed/unhashed unless the entire new DEV gate passes.
No speed claim and no overall goal completion from this development control.
