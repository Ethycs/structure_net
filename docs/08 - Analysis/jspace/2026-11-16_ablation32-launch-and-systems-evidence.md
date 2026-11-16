# Qwen3B ABLATION32: launch and systems evidence

Status at 2026-11-16 18:46 UTC: the real all-36-block discarded-basis update
pilot passed. The complete paired experiment controller is launched. Scientific
basis fitting waits for the completed shared native activation bank, then runs
automatically. There is no measured compression-quality or speedup result yet.

Protocol: [joint finite-removal search](../../07%20-%20Status%20Reports/jspace/2026-11-16_qwen3b-ablation32-all-at-once.md).
The implementation follows the user's pasted recipe; its Library report and ZIP
were unavailable, so this is a Qwen adaptation, not an audited reproduction of
the original Fashion-MNIST code.

## What is running

- Teacher: frozen original Qwen2.5-3B-Instruct, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, BF16.
- Joint search: all 36 residual bases, width 2048 to 1024, eight epochs,
  2,500 fitting examples and 500 group-disjoint selection examples, full-response
  and EOS teacher KL at temperature 2. Epoch zero participates in selection.
- Controls: the learned bases and their identical initial random bases, with
  equal compiled architecture and fitting budgets. Both whole models compile
  before student distillation starts.
- Student recovery: 36 sequential stages per arm, 12 epochs per stage, using
  3,938 training examples outside the selection groups. Each stage uses its
  actual selected student prefix and the exact frozen native suffix for
  look-ahead. No joint recovery phase is added.
- Evaluation: fresh packed reload, DEV128, MC256, generated responses,
  separate semantic judging, actual stored bytes and matched warm timings.
  These exposed development checks are not an untouched final benchmark.

The progressive campaign keeps its RTX 3060 reservation. This experiment uses
the RTX 2060 SUPER for the teacher and the plain RTX 2060 for FP64 QR, through
`inference-ctl lease run`. CPU orchestration waits without holding GPUs.
Scientific quality failures are reported; they do not cancel subsequent stages.

## Measured pilot

The pilot used eight actual training records, one per task family, covering
494 supervised response/EOS tokens. Its zero anchors are explicitly systems
only. The pilot update is discarded, and a fresh initializer plus empty
optimizer is required before scientific training with authenticated anchors.

| Measurement | Result |
|---|---:|
| All residual interfaces updated | 36/36 |
| Frozen teacher unchanged | Passed |
| Fresh raw checkpoint and projector replay | Exact |
| Teacher GPU peak allocated | 6,887,843,840 bytes |
| Auxiliary QR GPU peak allocated | 126,124,544 bytes |
| Host peak RSS | 8,593,817,600 bytes |
| Eight-example optimizer window | 25.52 s |
| Frozen model forward/backward portion | 16.26 s |
| QR forward / backward | 3.87 s / 4.23 s |
| Complete pilot including loading and checks | 223.53 s |

Linear extrapolation of this deliberately long-record pilot gives approximately
12–15 hours of basis-search computation, including selection overhead, before
compilation and student fitting. This is provisional: it is not a measured
end-to-end ETA. The 72 student stages can take substantially longer, and their
duration must be measured separately. No epoch/data budget was shortened.

An unprojected BF16 teacher/cache control showed finite numerical differences
across devices: temperature-scaled KL 0.00097–0.00724 on these eight examples.
This is recorded as a numerical baseline, not counted as compression quality.

All 56 focused CPU tests passed in 15.85 s, covering geometry, data separation,
full-model intervention gradients, compilation, worker updates, checkpoint
authentication, packing, evaluation and controller sequencing. Agents also
cross-reviewed the worker/evaluator/controller interfaces. These checks plus
the GPU pilot establish executable machinery, not scientific success.

## Deliberate adaptations and remaining limitations

Qwen uses RMSNorm. The retained basis preserves the all-ones direction and
normalization retains the original 2048-channel denominator. Each block's basis
is applied at entry, after the attention residual, and after the MLP residual,
so search and compilation implement the same interventions.

Native attention head dimensions, softmax and MLP widths remain unchanged in
this first control. It isolates residual-subspace choice; it is not the same
architecture budget as prior attention/head-compressed studies. Positionwise
affine anchors and folded numerator tables remain FP32. Those numerator tables
alone are roughly 1.47 GB, so halving residual width does not imply halving total
storage. The evaluator includes these buffers and the tied vocabulary embedding
in actual model-byte measurements. Optimizing these tables is a separate
follow-up after this comparison.

## Durable artifacts

Root: `/data/structure_net_experiments/jspace_3b_ablation32/20261116_v1`.

- Data fingerprint:
  `31c3109252bbd95744392c59ca597173454d9e2255e67075fe0df24f94e96f10`.
- Active prepared campaign: `campaign_v2`, fingerprint
  `6fa8be1ddaf038bcd7e0364a5c8cc217ed717f989063af631947682589e486fe`.
- Passed pilot: `campaign_v2/pilot/result.json`; detailed measured update:
  `campaign_v2/pilot/systems/result.json`.
- Pilot reservation `67c9a9fa9ffe` completed and released.
- Full controller: `controller_v1`; launcher PID 2166136. Its progress and
  failures are recorded under that directory and in `controller-launch.log`.
- The earlier `campaign` preparation was superseded before GPU execution to
  correct its source-inventory closure. It remains preserved with a
  `superseded_before_gpu.json` record.

Do not promote the pilot weights, label preparation as completed training, or
report the original Fashion-MNIST accuracy as a Qwen result.
