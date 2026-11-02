# Audit the fixed source concept bootstrap before downstream training

Date: 2026-10-23. Prospective audit of `ncm-source-hierarchy-bootstrap-v1`;
the fixed run is currently live. This changes no training configuration, gate,
checkpoint selection or evaluation rule.

After actual zero-exit receipts for the repeated systems pilot and the complete
16,384-update training worker, replay every sampler draw, full source frame,
clean/corrupted branch, uniformly corrupted code-history hash and learning rate
from the fixed seeds. Require exactly the planned number of finite update
records, including all eight level losses. Verify the selected checkpoint hash,
configuration, final step, optimizer steps and final sampler/noise RNG states.
Compare copied word modules and the unused finest head directly against their
registered initialization. Require frozen-state hashes to match the pilot,
exact saved model/source-only reload checks and the unchanged memory ceiling.

Finalize in fresh root `20261023_source_hierarchy_bootstrap_results_terminal_v2`
using `experiments.neural_architecture_lab.finalize_ncm_source_hierarchy_bootstrap`.
Hash all worker and supervisor evidence, checkpoint/recovery snapshots, producer
identities, this audit and the original plans. Write the corresponding measured
analysis report only after the terminal audit passes. Register conservatively
with exact JSON and semantic read-back; keep confirmed false and task accuracy
unmeasured. No partial trace or component training fit can complete this audit
or the overall 95% complete-summary goal. All earlier failures remain visible.
