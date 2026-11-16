# Phi-4 cached-source pilot launch

The user redirected the RTX3060 from Qwen progressive expansion to Phi-4 and
pointed out the local model cache. The source is `/data/models`, distinct from
the repository's `data/models` directory.

- GGUF: `/data/models/Phi-4-GGUF/phi-4-Q4_K_M.gguf`, 9,053,114,816 bytes.
- Editable source: `/data/models/Phi-4-GPTQ-int4`, two safetensors shards totaling
  9,143,429,560 bytes. No model was downloaded.
- GPTQ tensor payload: 9,143,347,200 bytes; decoded BF16 parameter payload:
  29,319,014,400 bytes. The latter is not a full-precision original checkpoint.

The experiment uses the quantized source decoded into BF16 native Phi blocks,
one resident layer at a time. One decoded block is 681,594,880 bytes; the
embedding and untied readout are each 1,027,604,480 bytes. This avoids loading
the full floating model onto the 12GB card. The loader was checked on actual
cached projection slices against the AutoGPTQ eager decoding formula, including
the source's signed scales. This does not assert bit identity with GGUF or
ExLlama kernels.

The prospective study is recorded in
`docs/07 - Status Reports/jspace/2026-11-15_phi4-single-block-pilot.md`.
Its first target is block39 of40, residual rank2560, attention head width64,
and1664 MLP channels represented through the fitted polynomial relations.
All40 original blocks execute for the reference; only the final block is
compressed in this pilot. Whole-model compression, generation recovery and
speed remain unestablished.

Data preparation is complete at
`data/experiments/jspace_phi_data/20261115_v1`. All4864 available TRAIN/DEV
examples retokenize without truncation. The pilot has128 TRAIN and64 DEV,
balanced across ARC-Challenge, BoolQ, HellaSwag and WinoGrande. Complete
supervision is the observed answer letter and template EOS, not a general
generation corpus. Prompt-only native trajectories are separately executed
for calibration and task evaluation, avoiding BF16 shape differences between
prompt inference and teacher-forced contexts.

Native production is frozen at
`data/experiments/jspace_phi14b/20261115_v1`, plan fingerprint
`ccfe92ea04595903e52770b5802ddb85aeac445b588efc38f2eff3fc494d6268`.
Reservation `0643b92b3b0f` allocates11264MiB on RTX3060; the PyTorch cap is
11008MiB. The independent Qwen all36 run on RTX2060 remains active.
Native layer completion was observed on the real3060, approximately9–10seconds
per layer including source decoding, both context streams and cache writes.
This is preparation timing, not an inference throughput benchmark.

The first launcher invocation used system Python3.10, which lacks
`hashlib.file_digest`, and exited before a lease or worker was created. Its
failure is preserved under `launch_native/launch_error.json`. The unchanged
launcher was then run in its tested locked/offline Pixi `ncm` Python environment;
the active attempt has its own `launch_native_pixi` log and receipts. No failed
GPU experiment was overwritten.

The subsequent stages are native-receipt verification, functional calibration,
basis fitting, compiled-interface capture, relation fitting, a disposable
optimizer smoke test, three early recovery epochs, and fresh selected-checkpoint
reload. Checkpoint selection prefers highest DEV correct count then earliest
epoch. A pass permits at most three net lost answers out of64 relative to this
Phi native reference. No Qwen native score is reused. This small selected DEV
screen is not an independent quality certification.

## User-directed pause after native completion

The native phase completed in394.78seconds, with103/128 TRAIN and58/64 DEV
correct (90.625% DEV) and1,052,770,304 peak reserved bytes. These are the
quantized-parent reference scores; there is no compressed Phi score yet.
Reservation0643b92b3b0f was released normally. The complete banks and receipt
remain under the native root above.

The user then explicitly prioritized Qwen progressive compression and asked
to pause Phi. No Phi fitting, recovery, or campaign controller was launched.
The tested Phi implementation is retained for a later explicit resumption.
The Qwen15→36 campaign was prepared at
`/data/structure_net_experiments/jspace_3b_progressive36/20261115_v1`, with
fingerprint`8e683e920281da8a9324b8b22c0584b23595af7c6b6e4b0d31ec4de3c82a4138`,
and its reservation-backed controller was launched. The independent all36
run on RTX2060 remains separate and active.
