# Complete-summary evaluation for the fixed task-supervision comparison

This implements the existing `ncm-task-supervision-expansion-v1` evaluation
plan without changing its training, semantic gates, or final-test policy.
At implementation time, public epochs 1 and 2 remain 0/128 on semantic DEV,
epoch 4 is running, and the additional teacher collection remains incomplete.
No new student quality result exists.

The separate CPU `offline` qualification can run before training finishes. It
exercises the actual completed judgment ancestry, all128 DEV identities, the
existing1,376 reviewed-source and public-data guards, and replays the historical
epoch2 decisions without new requests or model inference. It does not admit
expanded data, full evaluation or training, and cannot create a new accuracy
measurement. This systems artifact has its own root, separate from preflight.

Use `ncm_task_supervision_development preflight --bindings <completed consumer
preflight root>` only after both full training arms and their independent
training audits complete. The binding identifies actual completed catalog,
frozen-codec targets and training roots. Require all original 1,376 cases plus
at least 900 admitted new cases, exactly 2,048 updates and 65,536 presentations
per arm, the fixed epoch-2 initializer, unchanged frozen tensors, actual
successful receipts, and strict final checkpoint reload. Read the new trainer's
`strict_final_reload_passed` field directly; never rewrite historical records.

The CPU preflight preserves the complete author-probe judgment lineage and
every completed intervening public milestone. A partially judged milestone
blocks this preflight rather than losing its first responses. Both 64-case DEV
cohorts remain unchanged. Recheck full reviewed TRAIN/DEV text overlap and the
complete public-data guard. The sealed final 320 are not opened.

For each arm run stages `inference_pilot`, `outputs`, `judge_pilot`, `grade`,
then `audit`, recording actual terminal receipts between stages. Complete and
audit the existing arm before starting the expanded arm's evaluation. The
expanded arm inherits every existing-arm first judgment. No new public or other
semantic grading job may overlap this sequence. If an intervening public
judgment changes the frozen base lineage, stop and preserve all firsts; a fresh
explicit lineage version is required. Subsequent public evaluation must in turn
inherit the completed task-supervision judgments.

Inference uses locked Pixi ncm, physical GPU0 RTX3060, FP32, deterministic
algorithms, TF32 off, and at most 4.9GB allocated GPU memory. The original source
feature matrices are loaded one case at a time. Predictions receive source-only
rows; they never receive target text or target codes. Reuse the established
generator, word decoder and renderer. Strictly load each final model, check its
tensor hash and frozen tensors, repeat pilot outputs exactly, and match pilot
outputs in the full run. The RTX2060 public trainer remains untouched.

CPU grading reuses the unchanged semantic protocol and qualified Terra/Sol
judges. Terra grades all128 anonymous source/summary packets; Sol is admitted
only with all Terra decisions valid and at least61 passes in each64-case cohort.
Qualification requires at least61 same-case joint passes in each cohort.
Preserve valid paraphrases, invalid requests and all first responses. A file
lock prevents two task-supervision grading processes from overlapping; the
operator must also keep other grading entry points serialized.

The independent CPU audit reconstructs text from raw output IDs and replays
every judgment from retained raw first responses, including pilot calls,
component scores, complete request coverage and same-case joint decisions.
The paired comparison reports the exact same source IDs, all unresolved cases,
and expanded-minus-existing semantic passes. Seven additional valid Terra
passes establishes only the planned descriptive benefit. Final-model selection
requires DEV qualification, favors more joint passes, and breaks ties toward
existing. Passing DEV still does not establish the95% sealed-final goal.

Use `CUDA_VISIBLE_DEVICES=''` for CPU stages and
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` for inference, with
`CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`, running
`pixi run --locked -e ncm python -m experiments.structure_net.ncm_task_supervision_development`.
The paired report and registry publication follow audited real results; no
unexecuted stage or passing unit test is model-quality evidence.
