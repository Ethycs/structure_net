# Native codec decomposition on the64 new teacher summaries

2026-10-24. Hypothesis `ncm-sol64-frozen-codec-component-v1`.
Exploratory target-aware representation control; no training and no task
accuracy claim. The raw64 labels already qualified62/64 jointly. The BERT
text-rendering semantic study is currently running; its outcomes are unknown.
These64 are additional TRAIN-designated sources held out from prior gradients,
not sealed final cases. Preserve every first target and all failed studies.

Generate two oracle reconstructions of every complete original native128
frame: released LM1B codec, and the same encoder/VQ with the fixed domain
decoder checkpoint SHA0a6b7d4d... previously trained on626 other task sources.
Verify both the original690 TRAIN and128 DEV identity disjointness. Encode
each original target once to all641 codes; decode that SAME latent through
the two fixed decoders. No target truncation, vocabulary restriction, output
repair, source lookup, candidate selection or teacher-content postprocessing.
Retain all128 raw word IDs, predicted BOS/EOS, code IDs and rendered text.
The native CLS renderer and full30528-vocabulary argmax remain unchanged.

Use physicalGPU1 RTX2060/logical0, FP32 with TF32 disabled. This is a small
standalone codec inference cell: never load Llama or the NCM generator. Require
at least1.8GB free at startup and enforce1.5GB peak allocated. CPU preparation
and contract tests come first; run first/longest-target CUDA pilot, repeat
outputs and state hashes, then a separate complete64 inference process. Full
outputs must match retained pilot outputs. The same model must generate both
conditions without any parameter updates. Preserve and stop on mismatches or
memory failure, without switching CPU/precision/models as a fallback.

Fresh roots `20261024_sol64_codec_component_*_v1`. Freeze code/tests/plan,
teacher cases and raw audit, native tokenizer/config/released checkpoint and
fixed domain checkpoint. Actual child exits and whole-tool exit are required.
Rendering/code counts and word agreement are diagnostics, never semantic
task accuracy. Source-conditioned word decoding with oracle codes is a
different later control; these standalone results do not establish its ceiling.

This producer collects numeric outputs only while the BERT semantic campaign
runs. No external judgments are authorized by this entry point. After that
campaign completes, freeze a separate semantic evaluator inheriting its latest
entire canonical cache. Reuse exact raw/BERT packets and grade each codec arm
on all64 secondary cases; primary is admitted per arm only at64valid/>=61
secondary passes. Require>=61 SAME cases passing both qualified judges with
all valid. Never branch a second semantic campaign from an older cache.
Sealed final sources remain unopened and unhashed; quality precedes speed.
