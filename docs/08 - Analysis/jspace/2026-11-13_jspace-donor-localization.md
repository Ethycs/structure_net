# Donor effects change position between measured blocks 24 and 29

Date: 2026-11-13. Study: `JSPACE-DONOR-LOCALIZATION-01`.
Status: completed prospective single-checkpoint pilot; all validity gates passed.
Classification: `distributed_question_state_supported`. Confirmed: false.

Dense donor substitution now has a working positive control. On 63 pairs where
both clean questions were answered correctly, replacing the final state at
block 24 redirected **0/63** answers, while replacing the other question states
redirected **63/63**. At block 29 those results reversed: **63/63** for the final
state and **0/63** for the other question states. Replacing the whole question
suffix worked at every tested layer and exactly reproduced the donor's first
logits, not just its top answer.

This resolves a limitation of the preceding **dense positive control**. It does
not establish that the learned CAVs or J components work: the prior learned-CAV
edits also failed when applied across the question suffix. Dense substitution
and edits along learned concept directions remain different interventions.

- [Frozen protocol](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-donor-localization-plan.md).
- [Aggregate](../../../data/experiments/jspace_donor_localization/20261113_v1/campaign_results.json).
- [Direct result](../../../data/experiments/jspace_donor_localization/20261113_v1/result.json).
- Fingerprint: `40dced33132cbe3a7875fc44dff890e013935eaf9266a38725b6b3967e7ad81c`.
- Result SHA-256: `fdedbb4a0eb6f224b3d9098f678e390935e5c03460133f2a05780184f0c5fd57`.
- Aggregate SHA-256: `87caf61a67927173adaa598af331cd66335c0ed09f54c6fe6755fe6ebeb1d040`.

![Counterfactual switches by layer and replacement scope](../../../data/experiments/jspace_donor_localization/20261113_v1/donor_localization.png)

Only marked blocks were measured. Connecting lines do not measure or establish
a gradual transition through blocks 25–28. The [SVG](../../../data/experiments/jspace_donor_localization/20261113_v1/donor_localization.svg)
and the artifact's `plot_results.py` / `plot_manifest.json` preserve the figure
and its link to the frozen aggregate.

## Scope and validity

The model was frozen Qwen2.5-3B-Instruct, BF16, revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. There was no model training,
compression, CAV fitting, or J-space projection in this experiment.

From 512 generated evaluation candidates, 249 had aligned source/donor token
lengths and identical pre-question tokens. A frozen selection chose 64 pairs,
four per original country × colour cell. All named worlds were new relative
to the relation and causal-component studies. Source and donor retained
identical facts and changed the queried person. This matched-length population
is deliberately narrower than an unrestricted language task. Four separately
generated shakedown pairs supplied systems checks only.

Clean source accuracy was **63/64 (98.44%)**; clean donor accuracy was
**64/64 (100%)**. The 63 pairs correct on both questions became the primary
denominator before any patch scoring. These clean-answer scores and the
counterfactual-switch scores below are distinct measurements. This is not a
claim of 98% summarization accuracy or achievement of the broader quality goal.

The following checks passed:

- All hidden prefixes before the question were bitwise identical for each
  matched source/donor pair at every tested block.
- Final-block self-copy preserved the source's full first-step logits and
  generated tokens in all 64 pairs.
- Final-block donor-copy reproduced the donor's full first-step logits exactly
  in all 64 pairs and gave the intended answer on all 63 eligible pairs.
- Full-question-suffix donor-copy reproduced the donor's full first-step
  logits exactly in every layer/case cell: 448/448 comparisons, zero maximum
  logit discrepancy. This is expected with identical hidden prefixes and an
  exact suffix replacement, and validates the implementation/alignment.
- The patch copied the specified BF16 states exactly, preserved every unedited
  position, ran only during prefill, and removed all hooks afterwards.

The complete-word parser and eight-token cap were unchanged. Separate first-step
logits avoid confusing a copied first token with a correctly completed answer.

## Counterfactual switch results

Higher counts below mean more successful transfers to the **intended donor
answer**, not merely degradation of the original answer. All cells use the
same 63 eligible paired worlds.

| Zero-based block | Final token only | Whole question suffix | Question suffix excluding final token |
| --- | ---: | ---: | ---: |
| 5 | 0/63 | 63/63 | 63/63 |
| 11 | 0/63 | 63/63 | 63/63 |
| 18 | 0/63 | 63/63 | 63/63 |
| 24 | 0/63 | 63/63 | 63/63 |
| 29 | 63/63 | 63/63 | 0/63 |
| 34 | 63/63 | 63/63 | 0/63 |
| 35 | 63/63 | 63/63 | 0/63 |

For these sample sizes the Wilson intervals are approximately 0–5.75% for
0/63 and 94.25–100% for 63/63. The primary block-24 paired contrast is
**+100 percentage points** for the question-excluding-final mask: 63 pairs
succeeded only under that mask and none only under final-token copying.
Paired exact McNemar p = `2.1684e-19`. This is a narrow diagnostic on one
checkpoint and generated task family, not a population claim across models.

At block 24, final-token copying retained the original answer in 62/63 cases
and produced a third answer in one case. At blocks 5, 11 and 18 it retained all
63 originals. The late question-excluding-final mask retained all originals.
Thus the unsuccessful masks generally preserved the source decision rather
than producing unrelated text. All-case results, including the one clean-source
error, are separately recorded in `all_case_metrics`.

## What this establishes, and what comes next

At the earlier tested blocks, changing question states other than the final
position can transfer the answer, while changing the final position alone
cannot. At the later tested blocks, the opposite mask is sufficient. This is
consistent with information relevant to the answer becoming influential at the
final position between measured blocks 24 and 29. The experiment does not
identify which intervening block performs that computation.

The classification's word “distributed” has a restricted operational meaning:
the causal effect is accessible outside the final position at block 24.
It does **not** prove that multiple other positions are necessary, that a
particular concept is encoded there, or that the paper's global-workspace claim
has been replicated. A single untested question position could suffice.
Late final-token copying could also transfer an already computed colour answer,
rather than the intermediate country; these must be distinguished.

The next useful work is now concrete:

1. Measure blocks 25–28 and smaller subsets of question positions, using new
   cases or explicitly exploratory reuse, to narrow the transition.
2. At the already effective block-24 question scope, decompose the actual
   source/donor states and compare the full donor change, its J-derived change,
   and its remainder, with norm-matched controls. Decompose source and donor
   separately before differencing if using nonnegative representations; an
   activation difference can require both positive and negative changes.
3. Fit shared country concept directions at effective positions, with country
   and final colour independently balanced. Test whether edits follow the
   country through different colour assignments rather than copying a fixed
   output colour. Preserve the dense donor arm as a positive control.

Only successful concept-level interventions would justify returning to
continuous polynomial relations and their Gröbner transport. This run does not
measure compression, speedup, or task preservation after dropping a remainder.

## Execution

Nine CPU tests passed, including tiny real-Qwen terminal/suffix logit equality,
mask invariants, hook cleanup, alignment rejection, deterministic balanced
selection, and known-outcome grid aggregation. Real GPU shakedown checked
identity/donor controls, saved-state roundtrip and memory.

One reservation, `7e84641c00d1`, allocated 10,240 MiB on the RTX 3060. The capture
stage took **285.36 seconds**, including all 1,344 grid cells (64 terminal cells
reused once from the control pass). Peak allocated/reserved memory was
5.88/6.03 GiB. Stage time is not a throughput benchmark. Root locked Pixi `ncm`
used Torch 2.4.1 and CUDA 12.1. Preparation/analysis and plotting ran with CUDA
hidden. The lease was released and no GPU bookings remained. No regular model
service was started. There was one run and no scientific retry.

Raw pairs store BF16 suffix states and source/donor logits in `capture/pairs/`;
terminal patched logits are in `capture/terminal_logits/`. Human-readable
natural outputs, controls, full grid, events and hash manifests are preserved.

```bash
CUDA_VISIBLE_DEVICES='' pixi run --locked -e ncm python -m experiments.structure_net.jspace_donor_localization prepare --output NEW_ROOT
/usr/bin/python3 /usr/local/bin/inference-ctl lease run --allocation 3060=10240 --minutes 60 --reason 'Donor localization' -- pixi run --locked -e ncm python -m experiments.structure_net.jspace_donor_localization capture --output NEW_ROOT
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_donor_localization analyze --output NEW_ROOT
```
