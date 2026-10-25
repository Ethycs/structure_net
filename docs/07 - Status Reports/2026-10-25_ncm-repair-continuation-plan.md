# Continue the same TRAIN control before scaling data

V3's 512-update control completed and strictly reloaded. Mean word CE fell from
7.7043 over the first 64 updates to 0.78354 over the last 64. All 16 outputs
changed and none was empty. Nevertheless all 16 decoded pairs were inspected
and still include consequential substitutions and mangled identifiers. The
numeric threshold passed; the decoded-text review failed. Full 732-case v3
training stays unadmitted. Its 596/672 active word-ID agreement is a TRAIN
diagnostic, never a semantic accuracy claim. Preserve every v1/v2/v3 failure.

Test whether more optimization resolves the remaining errors before adding
more data or altering the architecture. Continue the exact completed v3
capacity checkpoint from global step 512 to fixed step 2,048 on the same 16
TRAIN cases. Keep its AdamW moments, CPU/CUDA/dropout and sampler RNG states,
microbatch one, accumulation four, clipping, loss, dropout, classifier scale
16, FP32 and all frozen original tensors. Keep the final step-512 learning
rate 3e-5 constant; no restart, warmup, sampler reseeding or checkpoint choice.
This is an explicitly longer optimization control, not a revised v3 outcome.

First run two genuine resumed updates (513–514) in a fresh pilot root, with
complete initial-output parity and strict final reload. The full continuation
starts independently from the original step-512 checkpoint, must reproduce
both pilot updates exactly (model, optimizer and all RNG state), and then
continues to step 2,048. Pilot weights do not initialize it. Save the actual
terminal receipts separately. No full-data training or semantic evaluation
is implemented by this control. Inspect final decoded TRAIN pairs and compare
the same target-aware native-code oracle before deciding the next experiment.

No heldout source or target is added. Original 128 DEV, sealed 320 final and
additional 156 engineering test scope remain unchanged. All source/body/MLP
and original NCM tensors remain frozen. No judge call, semantic 95% claim,
paper-replication claim or speedup follows from fitting the TRAIN control.
