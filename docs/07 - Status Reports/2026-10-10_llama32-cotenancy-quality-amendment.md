# Prospective correction of an invalid extractive quality assay

The original Q4_K_M validation produced 8/12 exact gold fragments, 8/12 supported
lines and 2/4 complete summaries. This fails the preregistered baseline
competence gate (90% coverage, 95% supported lines). Keep that assay classified
invalid. Its candidate generations are diagnostics only and cannot establish
compression quality. Packing, reconstruction, memory and runtime measurements
are independent of this quality failure and continue under the original protocol.

At this amendment no tensorized model quality outcome has been inspected.
The original Q4 outputs sometimes returned bare facts instead of full source
sentences and sometimes selected a superseded fact. Introduce one fixed worked
example demonstrating the required verbatim extraction format. Keep the task,
scoring, 96-token generation cap and all numerical/quality thresholds unchanged.
Use fresh validation seeds 2963 and 2964, one work and one personal example each.
The worked example uses seed 9988, family work, split demonstration. The sealed
ten final source examples retain seeds 7, 17, 29, 41 and 53 in each family, with
the new fixed prompt format; none has been generated or scored by a model yet.

Hash the new fixture file before executing its Q4 baseline. Q4 must first pass
the original competence gate. If it fails, stop the corrected quality assay as
invalid without running candidate quality under this revision. If it passes,
evaluate unmodified FP16 and all four already-fixed tensorizations. Apply the
original validation promotion criteria and packing controls. Only eligible
candidates advance to the ten sealed cases. Do not tune ranks or alter factors
based on either quality assay.

Report both assay outcomes. A corrected-assay result must identify this prompt
revision and cannot retroactively make the first assay valid. Even a pass is
evidence only for this small synthetic exact-source task, not production summary
quality or 95% application accuracy.
