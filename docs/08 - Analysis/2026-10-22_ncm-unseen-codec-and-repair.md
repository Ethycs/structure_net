# The adapted codec fails on unseen text, before summary-code prediction

Date: 2026-10-22. `ncm-unseen-codec-control-v1` is complete, classified
**unseen_codec_capacity_failed**. The current NCM attachment remains a failed
summarizer; its prior development result is0/64 under Terra. No new task accuracy
or speedup is established here, and no broader student training is admitted.

| Frozen codec | Cohort | Correct non-PAD target tokens | Exact target texts |
| --- | --- | --- | --- |
|released|seen|647/672=96.2798%|4/20|
|released|novel|695/717=96.9317%|8/20|
|adapted|seen|671/672=99.8512%|19/20|
|adapted|novel|664/717=92.6081%|4/20|

Both codecs receive their own full **correct target codes**, produced by encoding
the reference text. The adapted codec fits seen text but misses its predeclared
98% novel-token gate. This is a defect before source-to-code prediction. One
adapted reconstruction changes Luxembourg to Denmark and Grand Duke Henri to
Grand Commun Basil. Thus even correct generated codes would not reliably recover
novel names or text with this codec. This does not establish that codec error
alone explains the much worse generated summaries. The original release also
falls below98% on these texts, so merely restoring it is not a demonstrated fix.

The20 novel targets are the completed, format-qualified public teacher-pilot
labels, disjoint by ID, session, group and source hash from all1,999 existing
training pairs. Their summary quality was not audited; the experiment asks only
whether their exact text can be encoded and reconstructed. The20 seen cases were
hash-selected before observing outcomes. Quantization and decoding are FP32,
batch4, on GPU1. Model states stayed unchanged, full code tensors reloaded
exactly, public sources and untruncated target-token identities verified. Own
peak GPU allocation was416,337,920bytes. The sample is a
diagnostic, not a95% complete-summary success estimate or a confidence guarantee.

## Residual learning works on its separate training-only mechanism endpoint

`ncm-conditional-residual-learning-v2` completed1,024 updates in each matched arm
and **mechanism_gate_passed**. Both arms start from the same failed attachment,
use identical sample/noise/dropout schedules, freeze the codec and original body,
and train on the same32 old training targets. Static uses clean fixed finer
labels; repair recomputes finer expert labels after actual corrupted history.
Both use root noise0.5 and mild fine noise0.05 falling to0. The all-wrong-root
evaluation deliberately supplies target-derived nearby wrong roots; it is not
ordinary summary inference.

| Arm | Final training diagnostic | Correct target tokens | Exact target texts |
| --- | --- | --- | --- |
|static|free|1175/1176=99.9150%|31/32|
|static|gold_root|1175/1176=99.9150%|31/32|
|static|one_wrong|1162/1176=98.8095%|19/32|
|static|all_wrong|1102/1176=93.7075%|5/32|
|repair|free|1175/1176=99.9150%|31/32|
|repair|gold_root|1175/1176=99.9150%|31/32|
|repair|one_wrong|1175/1176=99.9150%|31/32|
|repair|all_wrong|1169/1176=99.4048%|25/32|

Repair's advantage after four wrong roots is5.6973percentage points,
above the5point mechanism threshold. Its free-training reconstruction change
from the shared baseline is1.1905points, within the declared tolerance.
The final free-training output is identical in accuracy between the two arms.
This supports learning conditional finer-code repair on known examples. It
does not demonstrate unseen source understanding, better summary accuracy or
95% performance. It cannot rescue the codec's unseen-text failure or admit a
broader student run. No new development summaries were graded in this experiment.

No-noise quantization parity, causal coarse inputs, no same/future fine-label
leakage, nonzero query/bridge gradients, frozen codec and exact prediction
save/reload all passed. Root-neighbor minimum squared distance is1.30769.
The original all-level-noise v1 stop and all prior failures remain preserved.

Aggregate: `data/experiments/ncm_paper_regime/20261022_transfer_controls_results_v1/campaign_results.json`; SHA256 `57be437d5aa091bdd233ae00c61e222e1055d494c41e652b368c1b7a333aa216`. Exact raw outputs,
producer/source snapshots, schedules, optimizer checkpoints, lifecycle pilots
and input seals remain in their append-only roots. Both registry records retain
confirmed=false and task_accuracy_measured=false. The final task test is sealed.
