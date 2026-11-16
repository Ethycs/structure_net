# Applying complete-response recovery to both Qwen students

Study `JSPACE-QWEN3B-RESPONSE-RECOVERY-01` is executing the
[new recovery protocol](../../07%20-%20Status%20Reports/jspace/2026-11-16_qwen3b-complete-response-recovery.md).
This report records launch evidence, not recovered quality. The old sparse-MC
runs retain their original sources and results.

## Models and transition

The progressive scheduler stopped after its verified 16-block stage; the already
leased block17 boundary capture finished without advancing to fitting/training.
Its selected update144 checkpoint scores 187/256 on the reused MC DEV set,
versus native 195/256. Its SHA256 is
`e4bd8882d68eed06242977460a9e86ff48251a940553105a520575873c9316a2`.

All36 was interrupted after its update144 checkpoint and DEV record were saved.
Update144 scored 100/256, while the earlier update72 scored 109/256. Following
the user's preference for early high-accuracy checkpoints, new recovery starts
from update72, SHA256
`5fa29c5c5a4b1d2b6f1b8d6c0d623028c87ce596bcc107be7b51b6fd55853d58`.
The latest state remains preserved. These are old-DEV initializer choices.

The [transition receipt](../../../data/experiments/jspace_3b_response_recovery/20261116_v1/transition.json)
records the controller/worker signals and checkpoint boundaries. Old GPU leases
released through their wrappers; Phi remains paused. New runs preserve geometry,
readout and the progressive native prefix, while changing recovery coverage and
the objective. They do not recompute a new common Gröbner basis.

## Broad task mixture

The [admitted corpus](../../../data/experiments/jspace_3b_response_data/20261116_v3/manifest.json)
contains 4,441 TRAIN examples and 105,089 response/EOS tokens across Dolly, XSum,
ARC, BoolQ, HellaSwag, WinoGrande, GSM8K and MBPP. DEV has 128 examples, 16 per
family, and 4,875 response/EOS tokens. The sampled tasks use full answers,
continuations, completed sentences, reasoning, code and summaries. Nothing is
silently truncated; maximum observed complete length is 383 tokens within the
416-position geometry. This does not qualify longer application inputs.

The first deterministic mixed pass uses 3,648 distinct TRAIN examples, 228
updates and 100,204 supervised tokens per model. Retained checkpoints are
updates 0, 46, 90, 152 and 228. Pure forward KL at temperature 1 covers every
response token including EOS. Original-teacher hidden targets preserve BF16
bits; full-vocabulary normalization is reconstructed with the tied readout.
Optimizer masters and moments are FP32 on CPU. Pilot updates are discarded.

The native cache contains 4,569 complete sequences / 710,537 input tokens and
approximately 5.82 GB of lossless hidden/prefix arrays. Capture plus native
generation controls completed in 626.68 seconds under lease `3b224fd4abed`.
Teacher immutability and full-vocabulary native-path checks passed.

## Execution and quality controls

Actual-model probes found a distinction that earlier choice-only checks missed:

| Path check | Progressive16 | All36 early72 |
| --- | --- | --- |
| Historical packed reference versus response adapter | Failed full-logit tolerance; 3/3 greedy choices matched | Failed full-logit tolerance; 3/3 greedy choices matched |
| Independently cloned unpacked HF reference versus response adapter | Passed all 151,936 logits over prompt plus two generated prefixes | Passed the same check |

The tolerance stayed `0.03125 + abs(reference)/128`. Learned tensors, fixed
geometry and parameter identities were unchanged. Unpacked checks qualify the
actual recovery execution path. Packed deployment remains unqualified; neither
these probes nor concurrent training timings establish inference speed gains.
Receipts are under `jspace_3b_response_systems/20261116_v1` and `20261116_v2`.

All 16 DEV MBPP reference programs passed their original tests in isolated
bubblewrap processes. Generated code receives the same bounded execution.
GSM8K gets final-number checks; summary quality uses source-grounded semantic
judging rather than exact wording. The configured judge route
`codex_sdk/gpt-5.6-sol` passed a reviewed known-answer numerical probe. Its usage
counters are unavailable, not evidence of zero token use.

The native control passed 2/6 strict generation canaries. Its failures include
JSON type/wrapper/date-format deviations and a wrong arithmetic yes/no answer.
Native failures and token-capped ordinary responses stay visible. This is a
small engineering screen, not general accuracy; compressed quality must not be
inferred from this native baseline or from teacher agreement alone.

At every retained checkpoint, the workers preserve complete-response DEV
losses, MC retention, canaries, two generated cases per source family, executable
task outcomes and blinded semantic judgments. CPU judging overlaps subsequent
GPU updates using immutable checkpoint outputs. Every required judgment resolves
before selection. Failed or unavailable evidence cannot produce an eligible
checkpoint; all checkpoints remain available for diagnosis.

Prepared run fingerprints:

- Progressive16: `c7aecdead01dce6507a6c2280bf5c4afdb37a514ed0dc8cddff7abaa77e9f74f`.
- All36: `147df48ef13592397397be887c28b36c004158e9262d1b2022b8b76a2606bd2b`.
- Shared cache: `ec256f8bb3d6e1e0e229279804d1b37d396ec4866e2fabd94e01182f31a94061`.

Representative recovery pilots launched concurrently on the 3060 and 2060
SUPER, under leases `fc58c8b6208b` and `cf47bed1e165`. Their receipts and the
subsequent scientific progress live under
`data/experiments/jspace_3b_response_recovery/20261116_v1/`.

Both pilots completed successfully: every compact block received gradients,
changed its FP32 masters and changed actual BF16 parameter storage. Fixed state
was unchanged and saved pilot checkpoints replayed full-vocabulary logits
exactly. The normalization FP32 master changed while its BF16 storage did not;
that declared diagnostic remains visible. Pilot weights are not used for science.

| Pilot | Peak CUDA reserved | Update-window times | CPU masters/moments plus gradient bank |
| --- | ---: | ---: | ---: |
| Progressive16 | 5,752,487,936 bytes | 34.19s, 33.91s | 7,478,603,776 bytes |
| All36 | 5,152,702,464 bytes | 83.39s, 84.10s | 16,824,113,152 bytes |

The scientific progressive worker launched under lease `d0f328ac1a12` on the
3060, and all36 under `0d378d7d602e` on the SUPER. At 2026-11-16 07:46 UTC,
progressive was evaluating its fresh initializer and all36 was initializing.
This snapshot does not count pilot updates as scientific recovery progress.

The final focused integration suite passed **65 tests**, including actual
bubblewrap isolation. An independent read-only audit checked both frozen plans,
full-response/EOS coverage, teacher binding, disjoint groups, fresh scientific
initialization, asynchronous checkpoint-specific judgments and selection.
Implementation, tests and initial records are committed as `9b368d1`.

The progressive scientific initializer's new complete-response DEV NLL is
6.96739 versus native 2.17769 over 4,875 target tokens; teacher KL is 5.98475.
Its actual unpacked MC path scores 186/256, one fewer than the archived packed
187/256 measurement. This runtime distinction is recorded rather than treating
the two paths as interchangeable. At 07:50 UTC both scientific workers were
still performing their initial quality evaluations, with no scientific update
yet reported; the real pilot updates remain separate.

## Data-access incident

Superseded corpus v1 decoded the reserved final512 task JSON to collect question
exclusions; labels were also decoded into memory but unused, and no model was
evaluated on that file. This violated the identity-only preparation boundary.
The [correction receipt](../../../data/experiments/jspace_3b_response_data/20261116_v1/access_correction.json)
preserves the incident. V3 rejects the reserved task path and uses identity-only
exclusions, with independently checked TRAIN/DEV separation. A new untouched
confirmation set is required; the old set must not be described as unopened.
