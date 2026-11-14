# Single compressed-layer distillation before further compression

Study: `JSPACE-QWEN-SINGLE-LAYER-RECOVERY-01`, exploratory.
The user corrected the recovery order on2026-11-14: distill one compressed layer
at a time. The planned all36-layer semantic-recovery runner is not launched.
The earlier all36 joint trial completed96updates with no development-accuracy
improvement; epoch0 remained the selected checkpoint. It is failed recovery.

## Prospective first gate

Replace only native block0 with the training-derived enriched rank1024,
full128-head softmax residual-carry block. Keep native2048-wide residuals,
native11008-wide feed-forward activation,16query/2KVheads and the original scale.
The remaining35 blocks, embedding, readout and original teacher stay frozen.
Only block0's compact weights/offsets are trainable; geometry remains fixed.

The native model scores50/64 on the existing development set. The enriched block
before distillation scores46/64, compared with27/64 for the earlier half-rank
basis. These are development observations, not fresh confirmation.

Use the original512 training cases and cached native block input/output pairs.
Fit12epochs in FP32 with learning rate1.2e-4, effective batch16, gradient clipping1,
and no weight decay. A real longest-context memory pilot precedes training.
Microbatch may be reduced solely to fit memory while preserving effective batch.

The objective averages per-token native-coordinate output squared error,
`||student(h0)-h1||²`, normalized by the teacher update energy `||h1-h0||²`.
Fit a numerical floor from training data only:1e-4 times median per-token teacher
update energy. Average valid tokens within each case, then average cases,
excluding padding. This weights update recovery rather than the size of an
unchanged residual. Teacher targets/normalizers remain detached; no independent
student normalization discards amplitude. Report ordinary feature MSE, relative
error, and separate first/content/answer errors. None substitutes for task accuracy.

After initialization and every epoch, place the one compressed block into the
otherwise native model and score the same64 development questions in matched
FP32 arithmetic. Select by task accuracy, with relative feature loss breaking
ties. Preserve per-epoch states/predictions and verify the selected state in a
fresh reconstruction. No other compressed layer is introduced during this fit.

The development gate is at least47/64 correct: at most3 net lost answers versus
the original50/64, within the user's five-percentage-point allowance. Passing
this small, repeatedly inspected set only permits the next staged experiment.
The untouched512-case confirmation population stays closed until a final
compressed stack has been selected.

For a subsequent layer, retain the accepted compressed prefix, train only the
new compact layer against the original teacher targets, and leave the remaining
suffix native. Test the complete assembled model after each addition. The
five-point allowance applies to the whole stack against the original native
baseline; it is not a separate loss allowance for every layer. A failing layer
is repaired before extending the compressed prefix.

All execution uses locked Pixi and `inference-ctl lease run`. Changed producer
code or objectives require new experiment roots. Completed failures remain
preserved. Size and speed remain separate measurements after quality recovery.
