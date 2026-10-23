# Codec capacity under conditional-target training noise

Date: 2026-10-22. EXPLORATORY `ncm-conditioning-noise-capacity-v1`.
Before launching the full conditional-residual learning comparison, test whether
its proposed all-level noise leaves enough codec capacity to reconstruct text.
The previous successful oracle perturbed nearby coarse codes only. Do not infer
that arbitrary fine-code errors are equally recoverable.

Freeze the polished codec. Select128 existing training targets by
SHA256(`ncm-noise-capacity-v1:`+id); first four are a separate systems pilot.
No generator training, new labels, private upload, held-out task evaluation or
final-test access. Use GPU1, FP32 quantization and FP16 decoder, batch32, seed42,
augmentation3042 and eight nearest non-identical coarse neighbors in FP64.

Compare clean, root-only probability.5, all-level probability linearly.5to0
(proposed learning v1), and mild fine noise. Mild mode retains the same root
draws and thins all-level fine masks by.05/.4375, giving first fine probability
.05 and a linear decrease to0. Use identical replacements and nested masks
across modes. At every level, dynamically recompute the target expert against
the actual corrupted history, then apply that level's sampled error. Decode
the final conditioned hierarchy; the last level is not corrupted in any mode.

Report non-PAD token reconstruction and exact-text counts for all modes. A
noise recipe is capacity-qualified for this training-mechanism test only if
non-PAD reconstruction is at least95%. This is not95% summary-task accuracy.
The current full training plan will not proceed unexamined if its all-level
recipe lacks capacity; preserve its completed systems pilot and explicitly
amend or replace the proposed full run before any main training outcomes.

First pass the four-case real GPU1 lifecycle pilot, no-noise vendor parity,
frozen-state identity, exact result reload, finite values and memory headroom.
Full measurement requires matching pilot source hashes. Stop on OOM, changed
inputs, nonfinite values or failed invariants, without CPU fallback. Keep all
outcomes and original producers. Artifacts:
`20261022_noise_capacity_{pilot,measure}_v1` under the paper root. This is a
target-aware representation diagnostic; no speed or summarizer success claim.
