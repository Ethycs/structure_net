# Public curriculum milestone audit, report and registration

This completes the prospective result-verification plan while epoch1 is still
training. No public milestone DEV predictions or judgments have been inspected.
The eight-epoch curriculum, model, training code and quality gates are unchanged.

For each scheduled epoch1,2,4,8, use fresh roots
`20261026_public_curriculum_epoch{epoch}_{training_audit,semantic_replay,results}_v1`.
The selected completed training attempt is explicit. Independently verify its
committed checkpoint chain back to update0 with the qualified CPU replay helper.
Completed ancestor segments require successful actual receipts. Interrupted
ancestors require a nonzero `failure_actual_tool_receipt.json` binding their
producer and raw trace. Preserve all raw tails; audit only the updates committed
in the exact resumed checkpoint and never count a discarded tail twice.
Require complete cumulative public/anchor exposure and unchanged reconstruction.

After this audit, run the qualified `ncm_public_development_lineage` inference
and semantic stages. Independently replay outputs and all first judgments, then
report passes out of128, valid judgments, both cohort counts and the same-case
Terra/Sol qualification. Carry every earlier milestone forward. Continue only
after valid failed earlier milestones; select the earliest qualifying milestone.
DEV qualification never establishes the final95% goal. Final320 stays sealed.

`finalize_ncm_public_curriculum {audit,replay,report} --epoch E --attempt A`
produces each stage, with an actual tool receipt required before its dependent
stage. Reports use `docs/08 - Analysis/2026-10-26_ncm-public-epoch{epoch}.md`.
`register_ncm_public_curriculum {register,readback} --epoch E` records the fresh
milestone hypothesis `ncm-public-curriculum-epoch{epoch}-v1` and verifies exact,
semantic and local-JSON readback. Use locked Pixi `ncm` for CPU audits/reporting
and the default Pixi environment for the existing Chroma registry integration.
Set CUDA visibility to empty for these commands and retain deterministic CUBLAS.

No audit reruns training, regrades an existing first response, substitutes
code agreement for task accuracy, or supplies evidence of inference speed.
Archive failures and use a fresh producing version for an actual remedy.

Before a full milestone exists, qualify the reporting path on the first64
published updates of the running epoch1 attempt1. Capture their exact complete
JSONL prefix in `20261026_public_training_prefix64_audit_v1`, verify it again
after replay, and retain the producing training fingerprint. This checks8,192
recorded presentations without reading DEV or requiring a model checkpoint.
It is explicitly neither a complete-epoch audit nor checkpoint qualification.
Use `finalize_ncm_public_curriculum prefix --epoch 1 --attempt 1` on CPU.
