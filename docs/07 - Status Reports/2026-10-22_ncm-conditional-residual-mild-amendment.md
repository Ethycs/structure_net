# Replace unrecoverable noise before the full residual-learning experiment

Date: 2026-10-22. New exploratory `ncm-conditional-residual-learning-v2`.
The original v1 two-update/two-arm systems pilot passed gradient, causality,
matched-schedule and reload checks. Its full1,024-update comparison never ran.
The separately declared128-target capacity measurement now shows all-level
noise at4,226/4,452=94.9236% reconstruction, below its95% gate. Root-only gives
99.4834%; mild fine noise gives4,420/4,452=99.2812%. These are target-aware codec
diagnostics, not task accuracy. No main learning outcome or held-out quality
has been inspected for either new recipe.

Close the unlaunched v1 full comparison as stopped after noise-capacity failure.
Preserve its original passed systems pilot and original source snapshot from
commit13fe31e. The new v2 uses the capacity-qualified mild recipe: root noise.5,
first fine noise.05 declining linearly to0. Sample the original masks and
replacements, then thin fine masks by.05/.4375 using the same augmentation RNG
in both arms. Root draws remain unchanged. This is the exact recipe tested in
the capacity measurement. Both arms otherwise retain every original budget,
data, initialization, objective comparison, seed, gate and reporting limitation.

Use fresh `20261022_residual_learning_{pilot,train}_v2` roots. Require completed
capacity evidence with mild_fine qualified, exact original-code parity, all
original lifecycle invariants and a matching new two-update pilot before full
training. The runner source changes version; its v1 source copy and fingerprints
remain immutable. No expected source hash in an old artifact is updated.
Record capacity/stop receipts as v2 producer inputs. The full v2 mechanism gate
remains5points improvement under all-wrong roots with at most2points loss in
free training reconstruction. Passing remains insufficient for the95% summary
goal; broader training and source-disjoint actual quality are still required.
