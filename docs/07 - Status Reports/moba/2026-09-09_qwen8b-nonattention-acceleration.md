# Frozen Qwen prefix: accelerate work outside MoBA

User request: accelerate the remaining work. This is an exploratory optimization
study following the measured idle prefix latency: approximately 951 ms in five
MLPs, 309 ms in attention projections, 326 ms in official MoBA and 201 ms in
remaining work at 16K. Preserve all previous evidence and the original weights.
No optimizer updates, layer removal or MLP retraining are authorized by this
implementation. Official MoBA remains block1024/top2 across six attention layers.

First screen one actual intermediate MLP input on the Wikipedia validation
window. Compare original FP16, packed gate/up FP16, cuBLASLt preference,
TorchInductor compilation, and bitsandbytes INT8 with outlier threshold 6 (and
threshold 0 as an explicitly less protected diagnostic). Retain errors and
unsupported backend failures. Use real CUDA, warmup outside timing, five paired
samples, exact source/environment identity, and new roots.

Then benchmark useful execution variants in the full six-layer prefix. FP16
optimizations may pack Q/K/V or gate/up weights without changing values and fuse
normalization, rotary embedding, activation and readout operations. Check
compiled/fused results against the original graph: mean head/output cosine
>=0.9999, relative L2 <=0.005, change Pearson >=0.999, change F1 >=0.95. INT8
copies are approximations, distinct from unchanged original checkpoint tensors:
require mean head/output cosine >=0.995, relative L2 <=0.05, change Pearson
>=0.995 and change F1 >=0.95 against original FP16 MoBA. Also retain all original
full-attention comparison endpoints. Application accuracy is not measured.

Use the existing paired Wikipedia/BNC validation windows at 8K/16K to select by
measured prefix speedup >=1.05 on both sources with the relevant fidelity gate.
Select by mean speed among passing variants at 16K, then fidelity, then listed
order. Check the selected execution on the existing five final input windows
per source as a numerical/runtime regression population; these are reused
windows, not a newly sealed application-accuracy evaluation. Count each gate
per window and preserve misses. All ten must pass for execution support.

Every run records exact GPU process state, input schedules, source snapshots,
compile/setup time separately from steady-state latency, peak memory, artifact
hashes and before/after original-weight hashes. Use an idle RTX 3060 and new
`data/experiments/qwen_moba_prefix/20260909_nonattention_<stage>_v1` roots.
Performance claims exclude compilation/setup and must state that limitation.

If the application mostly appends summaries, a separate causal cache experiment
may eliminate repeated prefix work. It must compare an appended-summary update
against recomputing that same longer sequence and must be labeled incremental,
not presented as a faster initial full-context pass. This branch depends on
the user's workload clarification and does not replace the full-pass screen.

## Full-prefix execution list sealed after the MLP pilot

The completed MLP pilot found little benefit from cuBLASLt, gate/up packing or
compiling just the MLP. INT8 threshold 6 was faster but changed that MLP's output
by 7.8% relative L2; threshold 0 changed it by 24%. Before full-prefix evaluation,
fix the candidate order to: fused FP16; fused execution with all first-five MLP
linears INT8 at threshold 6; the same at threshold 3; and INT8 gate/up at threshold
6 with FP16 down projections. Threshold 3 retains more outliers in FP16 and the
last candidate isolates quantization of the down projection. Original checkpoint
parameters stay frozen and retained in every case. This does not change the
fidelity gates in the original protocol.

FP16 execution packs Q/K/V, compiles pointwise operations and MLP activation,
and moves the sixth output projection after segment pooling. The last transform
is equivalent in exact arithmetic but changes FP16 rounding; it is subject to
the full FP16 fidelity gate. A one-window 8K fused-FP16 lifecycle pilot precedes
the screen. Measure five rotating, alternating samples of full attention,
original MoBA, and each candidate; one numerical forward per arm is the warmup.
Report the numerical/setup forward separately. The selection requires both
16K validation sources to pass; 8K results are a separate crossover diagnostic.
