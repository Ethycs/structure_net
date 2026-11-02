# Native NCM CUDA lifecycle v2: measured floating-point parity allowance

2026-10-24. Hypothesis `ncm-lm1b-native-finetuning-lifecycle-v2`.
The first pilot7bb259e exited1 before optimization, preserving its producer and
same-history logits: max difference6.87e-5 and zero argmax disagreements. It
stopped at the separate initial before/after-unfreezing bitwise concept check.

Fresh zero-update diagnostic83e9049 (actual68871/199559/0) isolated the condition:
concept logits differ by at most7.62939453125e-5, allclose(rtol1e-4,atol1e-4),
zero concept argmax changes; word logits are bitwise identical and hard generated
codes/words are identical. Complete weight/protected-codec hashes, trainable
count127732736and eval modes match. No optimizer step occurred. Do not assert
an exact kernel cause or call numerical allclose bitwise equality.

Preserve v1 and diagnostic evidence unchanged. Use fresh v2 producer/auditor and
roots. All original lifecycle conditions in the v1 plan remain, except initial
concept-logit equality is now the measured FP32allclose1e-4 condition. Initial
word logits and hard generation MUST still be bitwise exact; weights and frozen
encoder/VQ hashes remain exact. Save both complete logit tensors and detailed
parity measurements BEFORE applying the guard so any future failure is visible.
Require actual successful diagnostic receipt and frozen producer verification.

Same fixed c1f1b7c3... initializer, first/longest TRAINsources, no weight
initialization, native generator/decoder/source attention127732736parameters,
two BF16microbatch4 updates, seeds112142/111142, identical full objective and
8GBpeak limit. Keep the TRAINsame-history stepwise/joint comparison and abort
before updates if allclose1e-4 fails. No development scores choose any setting.
Freeze source/plan before GPU use. Require actual exit0, independent CPU replay,
full-state/optimizer/allRNG/hard-source-only reload and registered systems result.
Original Llama body stays unloaded/frozen, encoder/VQ fixed. No semantic judge
calls, final access, task-quality or speed claim. Do not use this two-update
checkpoint as a full-training initializer. The95%task objective stays unchanged.
