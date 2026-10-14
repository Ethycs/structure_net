# Q4 residual operator tuning before model evaluation

Exploratory amendment to `llama32-3b-fp16-core-q4-tail-v1`. The original
operator pilot passed all nine numerical checks, but its eager Q4 operations
were slower than cuBLAS FP16, especially at M=512. No compressed-model quality
or development outcome has been observed. Preserve that pilot unchanged.

Before model evaluation, benchmark a fixed set of tile and split-K choices
under CUDA graphs to remove unequal Python launch overhead. Load each
64-channel scale once per K tile. For split-K, materialize FP32 partial output
activations and sum them in FP32; never materialize a dense residual weight.
Both implementations use the same dequantized FP16 weights as the numerical
control. Require the existing RMS/cosine/finiteness gate for every choice.

Use the two actual MLP shapes, M=1/9 for decode and M=512/6300 for prefill,
seed51014, three event-timed repetitions of twenty replays after warmup and
capture. Select one decode configuration by smallest sum of median times
over both shapes and both decode widths, and one prefill configuration over
both shapes and prefill widths; ties use lexicographic configuration order.
The finite candidate list is hashed in the tuning producer. This selection
uses only operator timing, not summary quality. Use a new `operator_tuning_v2`
root and freeze its selected configurations before any development response.

The original model quality, size and speed gates remain unchanged. Eager
operator speed is retained as a diagnostic and cannot establish model speed.
