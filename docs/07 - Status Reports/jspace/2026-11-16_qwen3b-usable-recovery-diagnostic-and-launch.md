# Qwen3B usable recovery: diagnosis and execution

The priority is a compressed model that produces useful completed answers.
Multiple-choice accuracy is a retention check, not proof of recovered generation.
The current broad-corpus progressive and ABLATION32 producers remain unchanged.
This companion experiment first establishes recovery at the existing three-block
depth before transferring a successful recipe to deeper compressed checkpoints.

## What the previous recovery established

The prior progressive joint-recovery stage repeatedly used 736 TRAIN examples
(25,172 response/EOS tokens), rather than the entire available inventory.
Its three-block update46 checkpoint has response DEV NLL 2.319574 and MC 197/256,
against native MC 195/256. The earlier selected update19 scored one additional MC
answer but had worse NLL 2.465718. Neither MC result establishes usability:
the selected two- and three-block candidates failed to terminate on all 22
generation probes, including cases with a correct answer followed by repetition.

Update46 was selected prospectively as this experiment's initializer. It was
not selected on the new diagnostic outcomes. The 128 DEV examples and the
six canaries plus 16 generation cases are exposed selection material, not
untouched confirmation data. A promotion still requires a broader generation
panel and independent confirmation.

## Actual inference-path diagnostic

Completed artifact:
`data/experiments/jspace_3b_usable_recovery/20261116_v1/diagnostics/update_0046_v2/run/result.json`.

The reserved plain RTX2060 run took 109.47 seconds and peaked at
1,920,991,232 bytes of CUDA reserved memory. On 28 fixed token histories it used
the same actual original-model prefix for the native suffix, trained unpacked
student, and freshly packed Torch student. Packed reload and the production
generation entry point reproduced their respective outputs exactly.

- Packed/unpacked greedy agreement: 27/28; mean KL 0.000200655, maximum
  0.000474146. The single argmax difference was a near BF16 tie after “42”.
- After the correct “42” prefix, native EOS probability was about 0.999916;
  compressed probability was about 0.001458, ranked 27th. Both compressed
  execution paths failed here.
- On the four native-completion histories, native selected EOS four times,
  while both compressed paths selected EOS twice. Errors also occurred at
  the first substantive answer token, so this is not solely termination.

These observations do not support packing as the explanation for the tested
failures. They support further model recovery. They do not qualify the Triton
backend: this Turing card used explicitly declared packed Torch execution.
Cross-GPU native activation RMS differences were 2.84% at the boundary and
5.47% after final normalization relative to the recorded RTX3060 cache.
The suffix comparisons above share the same actual RTX2060 prefix; no
cross-hardware bit-exact equivalence is claimed. Earlier RTX3060 free-generation
failures remain independent evidence that the problem predates this diagnostic.

## Recovery now implemented

Stage A trains on all 4,441 TRAIN examples from 4,393 source groups exactly once:
105,089 response/EOS tokens across eight task families, with zero TRAIN/DEV
source-group overlap. It starts from update46 afresh, freezes the fitted
geometry and original teacher, and jointly trains the existing three compressed
blocks and final normalization. It uses exact full-vocabulary forward KL,
temperature 1, LR 2e-5, effective accumulation 16, BF16 live parameters and
FP32 optimizer masters. This is a new recovery setting, not an optimizer resume.

The initializer and updates 71, 134, 199 and 278 are retained. Every candidate
gets generation evaluation and blinded semantic judging, plus executable checks
where available. Selection prioritizes useful completed tasks, then generation
reliability, then DEV loss. The existing MC limit is at most five percentage
points overall loss and ten per family. If no candidate qualifies, the best
observed candidate remains explicitly unqualified.

The controller authenticates completed phases, reserves each GPU phase,
overlaps one CPU judge with subsequent generation, and evaluates all five
candidates regardless of poor quality. It does not silently repeat epochs or
claim an incomplete judge result as success. All GPU phases use the reservation
system and locked/offline Pixi. Plain RTX2060 allocation is 3,584 MiB with a
3,328 MiB allocator cap, alongside ABLATION32's separate 2,048 MiB allocation.

Implementation: `jspace_3b_usable_recovery_data.py`,
`jspace_3b_generation_path_diagnostic.py`, `jspace_3b_usable_recovery.py`, and
`jspace_3b_usable_recovery_controller.py`, under `experiments/structure_net`.
The four focused suites have 30 passing tests in total, including actual
structural restoration, inference-path replay, source-group coverage, parameter
updates, checkpoint selection and controller lifecycle behavior.

The real discarded pilot completed under lease `e3df0084e022`. Both update
windows had finite gradients, all three blocks changed in BF16, live parameters
matched their FP32-master casts, the packed export reloaded exactly, and a fresh
initializer remained unchanged. Peak reserved memory was 2,067,791,872 bytes
(about 1.93 GiB). Its training-only extrapolation is 2,975 seconds, about 50
minutes; generation and checkpoint evaluation are additional work.

The detached controller started at 20:24 UTC, PID 2526634. It authenticated the
completed pilot and launched the scientific pass under lease `f9601f0c071b`,
plain RTX2060 allocation 3,584 MiB. The scientific initializer is unmodified by
the pilot. Stage A fingerprint:
`2f7a29fa28d29b5af106a4d5c9e32ec403908679881f5d5c71e95d62733d0574`.
Progress is recorded under
`data/experiments/jspace_3b_usable_recovery/20261116_v1/controller_v1/` and
`stage_a/run/`. Launch is not a quality result. At this point the separate
progressive controller has completed its four-block recovery phase and continues
toward 36 blocks; ABLATION32's joint basis search remains active.

## Next decision

Reference-prefix loss is only a surrogate for free generation. If this full pass
still produces poor answers, the next targeted intervention is short
student-rollout distillation: score the student's own generated histories with
the frozen original teacher, refreshing trajectories as the student changes.
The protocol proposes 256 balanced TRAIN prompts with a 32-token cap, retaining
real EOS and never inventing EOS at truncation. Stage B is planned separately;
it is not implemented or running in this Stage A worker.

This addresses the distribution mismatch described by
[GKD](https://arxiv.org/abs/2306.13649) and the post-compression generation
collapse discussed in the [ShortOPD v2 abstract](https://arxiv.org/abs/2607.13124v2).
Those sources motivate the comparison, not a claim that this compression has
already recovered. Full-pass generated quality, deployed speed and full-model
size reduction remain to be measured for any promoted candidate.
