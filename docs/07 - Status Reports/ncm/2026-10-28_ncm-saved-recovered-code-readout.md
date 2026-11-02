# Improved codec on the four saved recovered TRAIN controls

The latest 6,416-update run's four preselected recovered-data controls still
produce repetitive, factually poor summaries through the existing source-bound
readout. Their saved reference-code agreement is low; agreement itself is not
semantic accuracy. The live paired-readout diagnostic uses four older anchor
controls, so it does not directly answer whether the improved codec changes
these newly inspected examples.

Decode the unchanged saved initial and final source-predicted codes for all four
recovered controls through the fixed step6212 standalone codec. Load the released
encoder/VQ and only the fixed trained decoder; require exact final codec hash
`44b48863faed5238af13593856b44f41cfc26ab6d959107c670fd10838ec536d`.
No generator call, model update, checkpoint selection, judge request or held-out
evaluation is included. Keep the before/after code streams separate and never
substitute reference codes into them.

Add a separately labeled reconstruction control for each reference summary.
Its encoded hierarchy must exactly match the singleton-produced stored target
codes. This target-aware control can diagnose representation/readout capability,
but cannot establish source-only summary accuracy. Repeat every decode exactly,
require finite full-vocabulary logits, and verify unchanged codec tensors.

Run FP32, TF32 disabled, batch1 on physical GPU1 RTX2060 after checking at least
1.5GB free. Stop if this process exceeds 1.5GB allocated GPU memory, an input
binding differs, or repeated outputs disagree. The frozen native codec path
already has completed real-model readout qualifications; require two new
saved-code selection contracts before this twelve-output diagnostic. Use locked
Pixi ncm and fresh append-only root
`20261028_saved_recovered_codes_new_readout_v1`. Preserve failures, exact source
identities and actual terminal receipts. Final320 remains sealed.
