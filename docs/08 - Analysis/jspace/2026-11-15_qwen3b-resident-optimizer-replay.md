# Qwen2.5-3B: resident optimizer replay

The twelve-block replay completed **12 epochs and 384 updates**, selecting epoch 7 at **51/64 (79.6875%)**. Fresh-process verification reproduced its packed predictions, logits and metrics exactly. This qualifies the resident optimizer for this tested configuration; it does not establish a general accuracy improvement or inference speedup. Results use the existing 512 training and 64 development cases; `confirmed=false`.

Root: `data/experiments/jspace_resident_adam_replay/20261115_v1`; fingerprint `59881a497c5868a2f129c828767d48222923c72e7d2c744d1c3c13ae4f5c0e94`.

This replays **cut 24, twelve compressed blocks**, with the original BF16 weights, fixed geometry, teacher, objective, learning rate, batch order and protected-candidate selection. It does not replace the separately accepted thirteen-block cut 23 checkpoint, which remains 46/64. Only optimizer state placement changes: FP32 masters and moments move to the RTX 2060 SUPER, using bounded host transfers; model computation remains on the RTX 3060. CPU/CUDA FP32 update rounding can differ.

| Measurement | CPU optimizer | Resident optimizer | Observed ratio |
|---|---:|---:|---:|
| Matched discarded pilot: synchronized optimizer update | 2.3056 s | 0.6530 s | 3.53× |
| Same pilot: forward/backward/update window | 4.2574 s | 2.3371 s | 1.82× |
| Complete training phase, historical CPU comparison | 1,469.3419 s | 872.8108 s | 1.68× |

The pilots used identical warm records, metrics and initial master fingerprints, with one update each. Their timing ratios are single observations. **No new full CPU replay ran**: the final row compares against the earlier frozen CPU training phase, a 40.6% lower elapsed time. Low-priority feature preparation and short CPU tests overlapped part of the resident run, so it was not an isolated paired throughput trial. Optimizer counters/timers exclude generic master-fingerprint downloads and evaluation; complete phase time includes those operations. None of these measurements are token-generation throughput.

All 109 parameter states fitted on the spare card: **4,807,680,000 bytes** of FP32 state, zero CPU state spill, with an 83,886,080-byte work allowance inside the 6 GiB state/work budget. Measured CUDA reservation peaks were **5.498 GiB on the 3060 and 4.621 GiB on the SUPER**, below their respective allocator caps. Larger suffixes require their own capacity checks.

Candidate accuracy was raw 49, aligned 50, then `49,49,47,46,49,50,51,49,50,50,50,50`. Epochs 2 and 3 were ineligible; epoch 7 was the best eligible candidate. Its CE 1.562915 and original-native KL 0.583184 both passed the protected raw thresholds. Native accuracy is 48/64, the fixed minimum 45/64, and immediate parent 52/64.

Historical CPU selection scored 50/64. Resident selection gained three answers and lost two across five changed predictions; family counts are ARC 14, BoolQ 15, HellaSwag 13 and WinoGrande 9, each out of 16. This small development difference is descriptive, not evidence of statistical superiority. Fixed-state checks and all-block update participation passed. Held-out confirmation and inference benchmarking remain pending.

Authenticated evidence:

- Selected checkpoint: `04305c6130838d997ddfc59e82982aeeaace83bdbc35157bef906bc218fb0a17`.
- `resident/train/result.json`: `cc2623256407a170db0823f05e34dc7731224939f11d295ce1e4c78ca6983beb`.
- `resident/train/manifest.json`: `ba021c966072a8d929a696321a8c1f33f9f1fc0a4c7512c1a44bf33e7e4d2e9e`.
- `resident/verify/manifest.json`: `ec14788ae13e609dd0ff256cfed0ba014ce4cccb17ce72789526eff4aeb5fd5e`.
