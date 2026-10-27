# Reserved-summary code prediction at public epoch2

Exploratory hypothesis `ncm-public-epoch2-reserved-code-probe-v1`: the source-to-code
generator predicts reserved summary codes well when supplied correct earlier
levels, but errors accumulate during free generation. The current application
DEV score is already 0/128 and the sixteen TRAIN controls have 98.36% rollout
code agreement. Neither result will be replaced by this diagnostic.

Use all 64 previously reserved component summaries, in the codec roundtrip
preparation's original order, with their original native frames and oracle
codes. These examples have been inspected in earlier component experiments;
they are not a sealed test. Revalidate source identities against all 1,376
reviewed TRAIN pairs and 162,353 admitted public pairs. Reuse the independently
audited public full-text exclusion only after checking that its producer binds
the exact original component source file. Independently repeat the normalized
text-overlap check against reviewed TRAIN. Do not open final320.

Load only the completed epoch2 checkpoint, strictly and by recorded hashes.
Run matched-source and cyclically shuffled-source conditions with fixed weights.
Capture teacher-forced argmax codes from the existing diagnostics function,
which also performs free generation. Report both code agreements, their gap,
agreement per level, root agreement, and native-word agreement. Check that
teacher-forced and freely generated root codes agree for every source condition.
Re-encode every native target with the frozen codec and require the original
641 oracle codes exactly. Target codes are supplied only to the teacher-forced
diagnostic; free generation remains source-only.

The descriptive exposure-error pattern requires at least 90% matched teacher
code agreement and a teacher-minus-rollout gap of at least 20 percentage points.
If matched teacher agreement is below 90%, classify teacher prediction itself
as weak on this population. Otherwise classify the pattern as unresolved.
These thresholds describe an error pattern, not semantic summary quality or
proof of a causal mechanism. Different valid summaries can have different
codes; exact-code agreement cannot establish or rule out semantic competence.

Use locked Pixi ncm, physical GPU0 RTX3060, FP32, TF32 off, deterministic
algorithms, seed42, CUBLAS :4096:8 and a 256MiB feature cache. Keep the live
epoch4 training unit on physical GPU1 unchanged. No backward pass, optimizer,
new semantic judgment, final-test access or speed measurement is permitted.
Peak allocated GPU memory must remain at most 4.9GB.

Before GPU work, qualify selection, split checks, raw metric replay, root
consistency and classification in focused CPU tests. A separate two-case pilot
uses the first reserved case and the longest source, determined from the
feature manifest before inference. Repeat both matched predictions exactly and
verify full model hashes before and after inference. Then run all 64 sources
in both conditions; matched pilot outputs must repeat exactly. Save every raw
code, word, per-level count and loss, source identity, model/input/source hash,
and actual process receipt. An independent CPU replay precedes a measured report
and registry readback. Preserve every failure without widening tolerances.

Fresh roots are `20261027_reserved_code_probe_<pilot,run,audit,results>_v1`
under `data/experiments/ncm_paper_regime`. The producer is
`experiments.structure_net.ncm_reserved_code_probe`. Completed ancestry and
the live trainer remain unmodified.
