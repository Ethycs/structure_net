# Locate the failed native NCM word-generation interface

Exploratory mechanism study, fixed before its outputs. Hypothesis ID:
`ncm-native-word-mechanism-v1`. The completed 2048-update, nine-pass model
scored 0/64 whole-summary passes, including 0/64 readable, on development.
This study cannot change that endpoint or establish task accuracy.

Use only the first four fixed-order summary TRAIN cases and first four raw
language diagnostic cases from the existing immutable CPU-cache manifest.
No development targets, sealed sources, new teacher calls, or training.
Restore the final failed checkpoint and retain its frozen 28-block body,
books, query adapters and precision. Four crossed arms:

| Arm | Coarse plan | Previous native words |
| --- | --- | --- |
| A | Source-only predicted five-level plan | Rounded plan, then four parallel refinements |
| B | Target-derived clean encoded plan (oracle) | Same four parallel refinements |
| C | Source-only predicted plan | True previous target words, one teacher-forced pass |
| D | Target-derived clean encoded plan (oracle) | True previous target words, one teacher-forced pass |

Targets masked after their valid length are filled with native EOS, exactly as
in training. B and D intentionally reveal target information. C and D measure
teacher-forced recovery, not free generation. Retain every raw 128-position
prediction, active IDs through first configured stop, text, plan codes, valid
token matches and denominator. Never call token recovery summary accuracy.
Record masked-position CE for C/D and the final free-state logits in A/B.
Record per-pass outputs in A/B to expose repetition/convergence behavior.

First run focused CPU scoring/arm-contract tests. Then a separate CUDA pilot
on the first two summary training cases, with all four arms. Require finite
logits, unchanged body/books/new weights/CPU cache, and exact A-arm replay
against the existing training reload probe, including codes and raw words.
Only after its actual terminal-zero receipt run the fixed eight-case study,
batch two, physical GPU0, BF16 body/FP32 new weights, TF32 disabled. No model
selection, altered decoding, extra iterations, reruns for better results, AR
fallback, external verifier, throughput or compression claims.

Poor A but good B would implicate plan prediction. Poor A/B but strong C/D
would implicate conversion from parallel noisy word states to fluent words,
despite retained teacher-forced language ability. Poor C/D would implicate the
adapted language interface too. These are exploratory contrasts, not causal
proof from eight selected cases or evidence that NCM itself cannot work.
Archive all failures and hash producing code, inputs and results. A scientific
negative is a valid completed experiment; a systems failure stops the study.
