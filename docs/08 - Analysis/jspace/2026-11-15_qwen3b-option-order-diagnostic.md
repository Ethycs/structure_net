# Compressed Qwen has strong option-position bias on the exposed WinoGrande cases

The controlled option swap supports a severe positional bias in the accepted
fourteen-block checkpoint. Both arms first reproduced all 32 original saved
answers and logits exactly, then scored the same questions with A/B choices
exchanged and gold labels remapped.

| Measurement | Original Qwen | Compressed Qwen |
|---|---:|---:|
| Original order correct |22/32|11/32|
| Swapped order correct |20/32|24/32|
| Mean accuracy across both orders |65.625%|54.6875%|
| Same underlying answer after remapping |24/32|5/32|
| A selections, original order |12/32|25/32|
| A selections, swapped order |20/32|28/32|
| Both orientations correct |17/32|4/32|

The compressed model largely follows the A position when the option text moves.
Its apparent improvement to 24/32 in the swapped orientation is therefore not
evidence that reasoning has been repaired. Likewise, answering either
orientation correctly on 31/32 pairs is not a capability score: a constant-A
binary classifier would achieve that on all 32 pairs. The paired orientations
are repeated measurements of 32 questions, not 64 independent examples.

This diagnostic does not establish why the bias emerged or prove that all
underlying semantic competence survives. The native model also shows some
order sensitivity. These are the already exposed sanity cases; no new holdout,
training, checkpoint selection, precision change or speed measurement occurred.
The native boundary control was bit-exact, and both strict checkpoint/replay
checks passed. The reserved final 512 examples remain sealed.

The result motivates the already specified recovery comparison with expanded
training and balanced correct-label positions. Its arms, data and budget remain
unchanged; we did not select a favorable prompt order to replace the failed
original benchmark result.

Evidence: [result.json](../../../data/experiments/jspace_3b_option_order/20261115_v1/result.json).
Fingerprint: `64aa4fd3b44ca41b6de13a6e91ff0f61a4fc7130dff3d1297201bd8e17503eb3`.
Producer/protocol commit: `e5ea13f`; 13 focused CPU tests passed before launch.
Normal controller leases `dca8804e9656` and `31748198f18b` each reserved
11,264 MiB on the 3060 and were released after completion. `confirmed=false`.
