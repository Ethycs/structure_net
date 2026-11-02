# NCM decoder adaptation to generated codes

Hypothesis `ncm-generated-code-decoder-v1`. Exploratory, one training seed,
underpowered for architecture claims. The fixed parent scored 0/128 complete
summaries on source-disjoint DEV. Its decoder recovered all 941 digit-bearing
TRAIN tokens with reference codes, but only 278 with generated codes. This study
tests whether adapting the existing decoder to actual generated codes improves
complete-summary quality. It preserves the failed parent and every earlier result.

The intervention changes only existing text-decoder weights. Original Llama
features, encoder, VQ codebooks, generator, concept/source adapters and word/source
adapters stay frozen. No architecture, inference pass, output repair, vocabulary
mask or copy mechanism is added. The parent checkpoint SHA256 is
`37d8d70f14d81a63ecd63ca6b9a3f5bf1ac9301d866af1135e24404224e795ad`.

Use all 732 unchanged engineering TRAIN pairs and their complete source features.
Use the parent's saved source-only generated codes as decoder inputs. Verify the
saved output hash and exact identities; independently regenerate the fixed 16
diagnostic cases and the longest source and require exact code/output agreement
before training. Gold codes never enter the decoder training path. Labels are the
existing native 128-position token frames, including EOS101 and ignoring PAD0.
These stored summaries are not independently verified semantic gold.

Run CPU gradient/loss/input-routing tests and a two-update CUDA lifecycle pilot,
then 1,024 updates on the RTX2060 physical GPU1 in locked Pixi `ncm`. Both stages
start from the same parent; pilot updates are not carried forward. Microbatch1,
accumulation4, shuffled without replacement, sampler seed171142, dropout172142.
AdamW CPU FP32 master/state, betas0.9/0.95, epsilon1e-8, weight decay0.01, clip1.
FP32 weights, FP16 training, FP32 deterministic evaluation, TF32off. Warm for32
updates to1e-4, cosine decay to1e-5. The only objective is per-example word CE.
Retain the existing lossless FP32 disk loader and256MiB feature cache. GPU peak
allocation must stay below4.9GB. No concurrent study worker on this GPU.

Require finite losses/gradients, decoder output-projection gradients, no frozen
gradients, nonzero decoder updates and unchanged protected tensor hashes. Save
full model, optimizer, scaler, sample permutation and all RNG states. Strictly
reload into a fresh parent and reproduce diagnostic outputs. Failed lifecycle or
fingerprint gates stop the stage. Do not select checkpoints by observed scores,
add updates after failure, or replace unfavorable valid judgments.

The primary endpoint is the existing source-disjoint128 DEV semantic protocol:
64short and64long cases, whole-summary faithfulness/salience/readability and the
128stored-token limit. Preserve all3667 prior first judgments and outstanding
historical uncertain-request allowances. Terra runs first; >=61/64 each cohort
admits Sol. >=61same-case joint passes per cohort qualify DEV. The sealed final320
remains unopened until that gate passes; passing DEV alone is not completion of
the95% final task goal. All128 outputs must be scored after successful full
training, regardless of TRAIN agreement. Code/token exact match is diagnostic.

A positive but subthreshold semantic score is partial progress. Zero passes is a
failed intervention. Invalid transport is unresolved quality, not zero accuracy.
The reused parent is a before/after baseline with no matched extra oracle-training
arm, so improvement cannot uniquely isolate generated-code conditioning from
additional decoder training. Engineering TRAIN has only eight source components;
broader-data generalization remains a limitation. No speedup claim is authorized.

Fresh roots use `20261025_generated_code_decoder_*_v1` under
`data/experiments/ncm_paper_regime`. Freeze source/input/environment fingerprints,
retain raw steps and all outputs, independently replay sample counts, losses,
frozen state, checkpoint and semantic judgments, then register the conservative
result. Python commands use `pixi run --locked -e ncm` from the repository root.
