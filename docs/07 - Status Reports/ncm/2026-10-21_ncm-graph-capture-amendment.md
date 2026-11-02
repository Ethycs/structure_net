# Capture-safe scalar allocation correction

The first graph systems pilot stopped before graph output or timing evaluation.
The pinned NCM `generate` method calls `torch.tensor(scalar, device='cuda')`
for minus infinity and zero while building its prefix-padding bias. The CPU
copy inside these calls is not permitted during CUDA graph capture. Preserve
`data/experiments/ncm_llama_graph/20261021_pilot_v1`, its frozen source, full
error log and `failure.json`; it is an implementation failure, not a quality
or speed threshold miss.

Version2 uses a local copy of only the official `generate` method, with exactly
two replacements: CUDA `torch.tensor(scalar, device=dev)` becomes CUDA
`torch.full((), scalar, device=dev)`. Values, shape, dtype, masks, level schedule,
native sampler and weights remain the same. The pinned external repository is
unchanged. A CPU source-comparison test must establish that these are the only
function-body changes, and the pilot must compare against the original method.
Use new root `20261021_pilot_v2`; keep the original parity/quality/speed gates.
No graph numerical or speed outcome has been observed at this amendment.
