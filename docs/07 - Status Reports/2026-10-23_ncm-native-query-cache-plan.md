# Preserve the native pretrained query path and share its source cache

Date: 2026-10-23. EXPLORATORY systems-only `ncm-native-query-cache-v1`.
Known: the five-block learned-bridge readout failed0/64unseen summaries; intact1B
is readable64/64but only50/64whole summaries pass under the old profile. Neither
is a standalone trained NCM success. The new label pipeline passed10/10pilot
finals and its fixed128-case sample is still running. No training on that sample
is admitted before its audit. The existing intact1B correction remains a tested
candidate; this separate control does not silently promote it as the final NCM.

Test a shared-weight alternative using all28original3B blocks, raw native prompt
tokens, native word embeddings and tied LM head. Cache only the source prefix
excluding the final prompt token. That exact final prompt token anchors the
first query; subsequent teacher-forced queries use previous native words.
This preserves native predecessor-token alignment instead of inserting a BOS
token or replacing the source with an untrained learned projection. Source
prefill never sees targets and runs with new query adapters disabled.

Only query Q/K/V/O projections receive rank8zero-output LoRA; all original
embeddings, norms, attention and MLP weights stay frozen. Disabled adapters must
return the base operation directly. Zero-output adapters also return the base
operation directly under no-grad inference; their training branch retains
gradients. Each query call forks a fresh DynamicCache from the immutable source
cache, never retaining other refinement levels' keys/values. Query attention is
causal. Correct native words should reproduce the stock teacher-forced logits;
this is a numerical oracle control, not free-generation accuracy.

CPU: tiny native model with left-padded sources, full/cached logit tolerance,
source cache/anchor alignment, source-cache immutability, repeated calls, query
adapter gradients and no original-weight gradients. No current/future target
word may enter the source cache. The adapter context must clean up on errors.

GPU pilot: first four old independently accepted training cases, physicalGPU0,
BF16/SDPA, batch2, seed42. Verify original3B body hash before wrapping. Compare
stock prompt+teacher-prefix logits against cached native queries at initialization;
relativeL2<=0.02 and top1agreement>=0.95. Two AdamW updates at1e-4/clipping1,
full128-target activation ceiling, finite losses/new gradients, no base gradients,
source cache unchanged, native frozen hash unchanged and exact adapter reload.
Require7.5GBfree; stop on OOM, invalid numerics/cache/identity/gradient checks.
Use `20261023_native_query_cache_pilot_v1`, archive numerical metrics and seed,
hash source/config/plan/model/accepted-label audits, record peak memory.

A pass only establishes a usable native conditional-query interface. A future
NCM may use native residual concept banks, noisy/self-generated earlier levels,
conditional expert labels and parallel word refinement through this interface,
with its own protocol. That future training/free-rollout evaluation has not run.
No additional1B decoder is required by this control. Reusing weights does not
by itself imply lower FLOPs, faster decoding,95%accuracy or paper replication.
