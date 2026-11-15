# Progressive Qwen2.5-3B suffix recovery

**7/36 blocks accepted; 51/64 reused development answers correct.** Classification: `progressive_execution_failure`. Confirmed: false.

Qwen2.5-3B BF16 progressive suffix recovery from the verified seven-block seed. One new preceding block is calibrated against each accepted parent; the complete suffix and final norm are jointly trained for12epochs on512 cases and selected on64 reused development cases. Native prefix boundaries are cached. No untouched confirmation, speed or general-generation evidence.

The primary metric counts verified accepted compressed blocks. Accuracy, the immediate-parent and original-native five-percentage-point gates, and storage are reported separately. A resource or execution stop is not a completed quality failure.

| Cut | Blocks | Parent | Initial | Selected | Epoch | Minimum | Gate | Whole tensor bytes |
|---:|---:|---:|---:|---:|---:|---:|---|---:|

Last accepted model: cut 29, 5,720,377,456 persistent tensor bytes versus 6,171,877,632 native bytes (7.32% reduction). Includes fixed geometry and tied E once; excludes activations, KV cache and optimizer.

All36 recovery is reported only after verified passing cut0. The initial seven-block seed is51/64; the matched original BF16 teacher is48/64. Parent-relative budgets are local, while every stage also retains the original-native budget. A passing stage does not establish untouched-test accuracy or speed.

The accepted teacher uses the unchanged two-group/eight-probe full-vocabulary Jacobian estimator. Each twelve-epoch stage recomputes the entire student suffix, keeps geometry/readout fixed and adapts all suffix parameters plus final norm. Epoch0 remains eligible; accuracy selects first, full-vocabulary CE breaks ties.

Some parent/native/seam controls are authenticated aggregate checks from the frozen producer; absent executed raw rows cannot be independently recomputed. All selected-epoch and fresh-verification rows are retained and independently checked.

Stop/progress: `{"cut": 28, "error": "RuntimeError('Differentiable teacher cache changed packed parent predictions: {\"passed\": false, \"predictions_exact\": false, \"logits_pass\": true, \"max_abs_logit_delta\": 0.25, \"correct\": 50, \"count\": 64, \"accuracy\": 0.78125, \"atol\": 0.03125, \"rtol\": 0.0078125}')", "kind": "execution_failure", "phase": "capture"}`.

Evidence: `data/experiments/jspace_qwen3b_progressive/20261115_v2/qwen3b_progressive_analysis.json`.
# Diagnostic follow-up before recovery

The one changed development decision was BoolQ identity
`4b700704f50cd05c95825666dda3bdc742d2b078555fff8c0f1ed1e54e3db71f`.
Packed-parent label logits were `[31.375, 31.5]`; the differentiable capture
produced `[31.5, 31.5]`, whose argmax tie-break selects the other label. Thus
the parent's 0.125 margin became a BF16 tie. All choice logits still passed
the predeclared numerical tolerance. No compressed eight-block student was
trained or scored in this attempt.

The next version preserves these complete gradient/statistics artifacts,
but obtains its separate full-vocabulary distillation targets from the
accepted parent's packed deployment using the same four-case batches as its
recorded scoring path. It retains the original differentiable logits as a
diagnostic and keeps the parent/native accuracy gates unchanged.
