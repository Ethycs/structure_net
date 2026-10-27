# Keep the code representation fixed for the new reviewed summaries

Preparation hypothesis `ncm-expanded-reviewed-targets-v1`. No student quality
or speed claim is made. The fixed teacher collection is still running and the
catalog lifecycle is being qualified. Model application DEV remains 0/128.

Use the same repaired BROAD codec and exact original tokenizer/frame policy as
the public curriculum. Require the checkpoint and codec tensor hashes already
fixed by `ncm_public_code_targets` and `ncm_public_xsum_codec_recovery`. Keep
all 1,376 original code targets unchanged. Admit new code targets only from
unchanged teacher firsts accepted by the expanded catalog; never tokenize a
shortened or edited target to make it fit.

Run CPU contract tests, then a separate GPU pilot over all newly admitted
eight-case catalog-pilot targets and the existing 16 TRAIN controls. Re-encode
all 16 controls and require exact equality with archived hierarchies. Encode
every new target twice and require exact equality. Each hierarchy contains
641 integer codes in [0,8192), bound to its exact full target text, source and
128-position native frame. Source inputs and references remain unchanged.

Physical GPU0 RTX3060, FP32, TF32 off, seed42, deterministic algorithms and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; peak allocation must be at most4.9GB.
The live RTX2060 trainer remains independent. The codec stays frozen with no
gradients. No full generator or Llama body needs to run for target preparation.

Independently replay target/frame/code bindings on CPU after the pilot. Full
preparation requires the full 1,024-outcome collection, admitted union catalog
and its complete feature audit; never substitute the pilot for full admission.
Reuse original targets, encode every newly admitted target, and require exact
pilot/full equality for every pilot target. A separate full CPU audit must
reconstruct the complete ordered union and all original/new bindings before
training targets are admitted. The CPU audit does not independently repeat GPU
encoding; the repeated GPU encodes and frozen controls are separate evidence.
The pilot and full catalog filenames/row offsets may differ; compare every
other field exactly, including the original raw draft/review attempt paths,
source, target, native frame and codes. Retain both catalog-provenance records.

Entry point `experiments.structure_net.ncm_expanded_teacher_codes` supports
`encode {pilot,full}` and `audit {pilot,full}`. Fresh roots are
`20261027_expanded_teacher_codes_<pilot,full,pilot_audit,full_audit>_v1`.
Use repository-root locked Pixi `ncm`; CPU audit sets `CUDA_VISIBLE_DEVICES=''`.
Record actual successful receipts before dependent stages. No teacher/judge
request, student training, final-test access or accuracy measurement occurs.
