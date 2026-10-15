# Prefill epilogue revision before model evaluation

The v2 operator run passed its 82 numerical cases and twelve irregular-shape
controls. Decode improved with the preserved load layout and fused epilogue;
prefill gains were smaller. No model quality output has been observed.
Preserve all v1/v2 producers, timings and selections.

The final pre-model implementation round, `20261014_operators_v3`, separates
the small FP16 core's two GEMMs from the Q4 prefill GEMM. The latter now reads
the already-rounded FP16 core output and fuses only the final tail-rounding
and addition. This reduces the GEMM kernel's live operands and shared-memory
requirements. It retains the original projection's rounding boundaries.
Decode and transient expansion retain v2 execution. The v3 finite prefill
list includes K=32/64/128 tiles recorded in its producer; FP16 tensor partials
still accumulate across K in FP32 for the staged family.

Use v3 numerical/timing selection and all three declared families for the
model pilot/development stage. Do not pool earlier versions' timing cells
with v3. Quality, final seeds, controls, thresholds and stop rules remain
unchanged. This is a pre-quality engineering change, not a retry of any model
quality outcome.
