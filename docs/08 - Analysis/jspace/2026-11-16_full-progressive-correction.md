# Full progressive compression: evidence and correction

The newer broad-corpus progressive rebuild reached **1 of 36 blocks**; the older progressive branch had reached **16 of 36**. Neither completed the requested full progressive compression. The separate all-at-once branch completed its recovery schedule, with failed quality measurements.

The newer progressive controller stopped because response negative log-likelihood (NLL) exceeded 1.10 times native. That was an inappropriate stopping rule for the user's full-compression objective and permitted five-percentage-point task-accuracy loss. It used distributional loss to terminate progression before directly testing generation. The appropriate conclusion at that point was **generation quality not established**. The original failed classification remains preserved; the new study changes the advancement policy prospectively.

The completed one-block rebuild produced the following **packed** measurements on exposed development data:

| Candidate | MC correct / 256 | Response DEV NLL | MC logit vectors changed from preceding candidate |
| --- | ---: | ---: | ---: |
| Original native model | 195 | 2.17769 | — |
| Raw compressed initializer | 198 | 3.45472 | — |
| After local alignment | 198 | 3.34657 | 256 / 256 |
| Joint update 10 | 198 | 3.28203 | 251 / 256 |
| Joint update 19 | 198 | 3.19257 | 250 / 256 |
| Joint update 46 | 198 | 2.95058 | 240 / 256 |

All five compressed candidates made the same MC predictions: six native errors were corrected and three native correct answers became errors. Their 77.34375% MC accuracy exceeded native's 76.171875% by 1.171875 percentage points on this reused set. The unchanged accuracy did not mean the saved models were ignored: their actual logits changed, and the training receipt records BF16 changes in the block parameters and final norm. The inspected manifests, outputs and update receipts provide **no evidence of tampering**. They also provide no basis for equating this small MC result with broad generation accuracy.

These values come from the [completed one-block result](/data/structure_net_experiments/jspace_3b_progressive_response_rebuild/20261116_v2/blocks_01/recovery/run/result.json) and its candidate `multiple_choice.json` records. The result SHA256 is `1ac6c576a506d73b3681abae67f62b5ae1e95796c85d462a79dc247ba4c36ef1`; the completion-manifest SHA256 is `537fd9b6a1c75334a68f3b7fadccab81e6c699777f58a59386209486423c4ef3`.

The separate all-at-once recovery completed **228 updates and 100,204 response tokens**. Its recorded executor was the unfused BF16 training path; these numbers should not be silently relabelled as packed deployment results.

| All-at-once update | MC correct / 256 | Response DEV NLL |
| --- | ---: | ---: |
| 0 | 109 | 13.46914 |
| 46 | 91 | 7.59897 |
| 90 | 89 | 6.90505 |
| 152 | 90 | 6.84611 |
| 228 | 87 | 6.99165 |

Here there is measured MC task failure relative to native, despite substantial NLL improvement. Failed or unavailable semantic requests must still be distinguished from measured wrong answers. Preserve this completed branch; it is not the parent of the new progressive chain. Its [completion result](/data/structure_net_experiments/jspace_3b_response_recovery/20261116_v1/all36/train/result.json) has SHA256 `1a2bd1ff9c022690622c15dbcb1de0b3071f1f571c034983498125fc0b6c8bac`.

The [new full-progressive protocol](../../07%20-%20Status%20Reports/jspace/2026-11-16_qwen3b-full-progressive-compression.md) starts from the one-block update-46 checkpoint, then schedules native layers **34 through 0**, reaching 36 compressed blocks. Each stage fits the new geometry, aligns only its new block, and jointly recovers the compact suffix on complete responses. Continuation selects the best observed packed MC score, then NLL, then the earliest checkpoint. NLL remains visible but does not stop the schedule or suppress generation. Actual task qualification is reported separately; systems or provenance failures require diagnosis and repair.

The two teacher roles must remain explicit. Distillation targets come from the **original native model's** final hidden states and frozen vocabulary readout. Functional basis fitting differentiates through the **native new entrance followed by the frozen compact parent suffix**. That is not a claim to have computed the full native model's logit Jacobian at every stage.

Before launch, the expanded chain was prepared and the fixed seed task probe was in progress. The probe subsequently completed all22 generations: its canary batch is **0/6**, versus native's2/6, and CPU semantic/executable grading is running. Observed repetition is a real limitation beyond MC accuracy. The new full-progressive execution includes native-prefix/cache controls to distinguish implementation errors from model behavior. No final semantic accuracy, full-36 quality pass, speedup or deployment claim follows from the current evidence.

At **2026-11-16 18:00:23 UTC**, the full controller was launched as PID2074966
under locked/offline Pixi. Its first phase acquired3060 lease`81cc64ea8ba8`
for11264MiB (11008MiB allocator cap) and started capturing all native boundaries.
The immutable controller plan lists every remaining cut34through0 and
`quality_failure_stops_progression:false`; it automatically proceeds into fitting,
discarded systems pilots, local alignment and joint recovery after capture.
The controller fingerprint is
`430ffd045c38ef771b5d87c5d19f4c06f3dd5a40348bd14349ca493ac675fb80`.
The data fingerprint is
`025fd7048eec86d8bb81fb501a96b9679febac4ac77ddbabdcac633c8cca7f64`.
The focused final suite passed **41 tests**, including full-depth scheduling,
actual packed reloads, native-prefix controls, retry boundaries, shared serial
judge locking and graceful process-group cleanup. The seed judge shares
`.cpu_judge.lock` with subsequent stage judges while GPU phases advance.

Artifacts and ongoing progress are in
[the full run](/data/structure_net_experiments/jspace_3b_full_progressive/20261116_v1/controller/progress.jsonl)
and [the seed probe](/data/structure_net_experiments/jspace_3b_progressive_task_probe/20261116_v1).
This is an active full-depth campaign, **not a completed36-block model**.
