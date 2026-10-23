# Correct the long diagnostic count before word-final model preparation

Date: 2026-10-23. Amendment to the word-final frozen-parent protocol, before
any model execution in that study. Its first metadata preflight exited1:
252TRAIN and64short-development sources exist, but the inherited long diagnostic
has16cases, not64. The failed source/plan snapshots and terminal observation
are preserved in `20261023_word_final_preflight_count_failure_v1`. No prepared
dataset root, checkpoint, generation or quality result was produced.

Preserve all16existing long diagnostic sources and all64sealed longer sources.
Expand the diagnostic to64 by selecting12additional source groups per each of
the four512–2048cl100k bands. Select only from the SHA-bound original eligible
public-source index. Exclude all old pool TRAIN/diagnostic/sealed identities,
all252word-final TRAIN and64short-development identities, and all prior unpaired
codec-fit source groups. Verify IDs, sessions, groups and source hashes.

The availability check found remaining group counts3556/408/12/65 by band after
the inherited pool exclusion, unchanged after excluding codec-fit groups.
Select the scarce1024–1535band first, then1536–2048,768–1023,512–767. Within
each band order by SHA256(`ncm-word-final-long-diagnostic-v1:`+caseID), retain
the first12nonconflicting identities and update exclusions across bands.
No selection by output, summary label, model score or judgment. Fail rather
than reduce a quota if availability or disjointness does not hold.

Recover the48complete sources from their original train-file line numbers,
checking public spans, SHA, cl100k length and native prompt metadata. Add no
labels. Keep original16in their original order and append the new48. The
new64cohort has16cases per band and is development data, not a sealed final.
Neither sealed cohort text file is opened. Input fragments remain complete.

Root `20261023_word_final_long_diagnostic_v1`; entry point
`python -m experiments.structure_net.ncm_word_final_long_diagnostic`.
The subsequent word-final metadata stage uses this expanded cohort and binds
this amendment, its sources and the failed preflight. All original512-update,
quality61/64per-development-cohort, frozen-parent and no-speed-claim requirements
remain unchanged. This fixes an erroneous inherited count; it does not shrink
the95% task objective or retrospectively qualify any failed model.
