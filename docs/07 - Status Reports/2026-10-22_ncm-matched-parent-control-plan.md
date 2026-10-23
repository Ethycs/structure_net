# Full parent reference on the current NCM development cohort

Date:2026-10-22. EXPLORATORY `ncm-matched-parent-control-v1`.
Earlier parent quality measurements used a different64-case cohort. The current
NCM development sources are fresh and source-disjoint from its training groups.
Establish a parent reference on exactly these64sources so the next NCM result
can be interpreted against the original frozen model, rather than borrowing a
pass rate from another cohort. Parent performance is a reference, not a logical
upper bound on a task-adapted head. No NCM failure is reclassified.

Use the exact cached Llama-3.2-3B-Instruct weights and original32-word chat prompt
used to obtain the frozen body features. Preserve full28layers, their MLPs,
original language head, BF16 weights/forward and SDPA. Greedy generation,
max_new_tokens96, batch4, source prompt <=512tokens without truncation, original
EOSsettings and tokenizerEOSasPAD. This cap avoids arbitrary truncation to the
NCM codec's48GPT2canvas. Explicitly report the output-budget mismatch; both are
graded against the application's128cl100ktoken limit. Do not use this control
to claim matched inference speed or equal output-token cost.

Run on physicalGPU0 after the active NCM training worker is confirmed terminal.
Never contend with it or change resident services. Require at least7.5GBfree
device memory before loading. A separate4-case CUDA generation pilot must pass
finite/valid output, repeated greedy equality, memory and original body digest
`75cb5e87d431fa688ec1a68dc868636cf3cfbf2b421601ea56dd21848284d38c`.
Source/config/weight hashes must match the full64-case control. Stop on OOM,
changed weights/source, source overflow or unequal repeated greedy output; no
CPU fallback or silent batch/precision changes.

Then save all64raw parent outputs/token IDs and frozen-body/weight identities.
Grade with both qualified original judges and the unchanged one-candidate
source-grounded rubric, including faithful/salient/readable/nonempty/<=128token
pass and all missing/invalid judgments. Preserve all valid failures; no semantic
retries. Report all numerator/denominators and Wilson intervals. The final256
test is not opened for this control. No training occurs and no speed is measured.

Roots:`20261022_parent_pilot_v1` and`20261022_parent_outputs_v1` under
`data/experiments/ncm_paper_regime`, followed by distinct role-specific grade
roots. This is one frozen checkpoint, not multiple training seeds. It does not
replace the requirement that the frozen-body NCM candidate itself earn95%.
