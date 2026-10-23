# Full-data NCM coarse curriculum with a fixed-code decoder

Date: 2026-10-22. EXPLORATORY `ncm-full-curriculum-v1`.
The32-case mechanism test passed:6,379/6,400free codes,128/128coarse codes,
matched versus shifted coarse advantage99.22percentage points. The decoder-only
phase passed its diagnostic gate:69,515/69,682non-PAD tokens and1,856/1,999exact
training texts, with every non-decoder state/code assignment unchanged. Neither
is held-out summary accuracy. Original noisy-pair outputs remain visibly poor;
full independent development grading is running after both original judges
recovered. Previous studies and all their failures remain unchanged.

Question: do the successful competence changes improve complete-summary accuracy
on the fixed64source-disjoint development passages when trained on all1,999
teacher pairs? This is one bundled engineering intervention, not an ablation
claim for each component. Main endpoint is actual whole-summary accuracy under
both original qualified judges, same rubric and128cl100ktoken limit. Require
>=61/64independently under both for admission to a separately declared sealed
256-case final-test study. No success claim from training fit, reconstruction,
one judge, token CE or missing/invalid judgments. No speed claim in this study.

Continue from `20261022_noisy_native_v1/model.pt`, which already received the
512-update unpaired stage and2,048conditional updates. Reset the new global-body
bridge to its declared zero initialization. Use the polished decoder with an
explicit compatibility receipt: load/check the original codec, compare every
non-decoder state byte, and verify identical FP32codes for all1,999targets after
loading the new codec. Retain original/new codec and source generator SHA.
Construct tied heads with unchanged books; freeze the entire new codec. Do not
silently alter the original checkpoint's codec-pair metadata. Save a new pair.

Use all1,999immutable teacher labels, with their known30/32audit and label-noise
status disclosed. Native GPT2source embeddings plus a zero-initialized global
residual from the frozen Llama final valid hidden state; original body frozen.
Released optional coarse AR, causal first4target block after load;12generator
evaluations. Preserve actual training/development source IDs, hashes and frozen
feature digests. Extract the final valid state correctly from both left-padded
training and right-padded development feature banks. No truncation or label
remapping. Development sources do not receive reference labels or gradients.

Fixed continuation budget:5,120updates, first2,048coarse-only and remaining3,072
all-level equal-weight(alpha1). Batch8x accumulation2(effective16), dropout.1,
AdamW(.9,.95), LR3e-4 for generator/bridge, WD.01, clip1, warmup128, cosine
floor.1. Clean ARpast labels, clean targets, finer conditioning corruption.50
to.10. Seeds42/52/2042/3042. This deliberately spends additional continuation
compute; it is not a matched-compute improvement over the earlier recipe.

Before full training, a new matching two-update CUDA lifecycle pilot must pass
both objectives, nonzero query/bridge gradients, correct feature padding,
zero-residual parity, no future coarse label leakage, identical code language,
frozen codec, finite values and exact reload. One worker on physicalGPU0; all
resident services preserved. Stop on OOM, nonfinite loss, changed dependencies
or failed invariants. No CPU fallback. Source/config/input snapshots before run.

Save every1,024updates plus the fixed final checkpoint. Retain32training
diagnostics and all64development generations at each checkpoint. Predeclared
primary development checkpoint is the finalstep5,120; intermediate quality is
diagnostic and does not replace a failing final endpoint. Record the shifted
source control on32training cases at the end. No seed retries to obtain success.
Final256test remains sealed unless a new protocol admits a qualified model.

Artifacts: `data/experiments/ncm_paper_regime/20261022_full_{pilot,train}_v1`.
Append aggregate/report and conservative registry after quality stabilizes.
One model seed is underpowered; report actual numerator/denominator and every
failure. The prototype still covers only512GPT2source tokens and48target tokens,
not the complete application input/batching range.
