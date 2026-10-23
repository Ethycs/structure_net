# Released NCM on its published XSum inputs

Date: 2026-10-22. EXPLORATORY `ncm-released-xsum-control-v1`.
The untouched released checkpoint has not yet been tested on the author's own
XSum inputs in this task-transfer campaign. Current conversation results are
negative; testing the released inference path on its intended domain is a
missing control, not a replacement for the95% conversation-summary objective.

Download only `xsum/passages.pickle` and `xsum/reference_summaries.pickle` from
the official `hazyresearch/ncm-datasets` revision
`6580ba41a4b0a976dde876fbe84d25327617b0e6`. Save public metadata and verify each
download against its published LFS SHA256. Parse primitive pickle containers
with a restricted unpickler. No training or private sources are involved.

Use the first four rows for a real GPU0 systems pilot, then the first32 for the
control. Run the vendored author's `scripts/eval_shard.py` unchanged, batch4,
the original released config/codec/generator and FP32 inference. Preserve its
first512 source tokens, exactly as its evaluator specifies; this truncation is
explicitly limited to the published-domain control. No Llama bridge, adapted
codec, coarse AR or finetuning. Run the local loading/generation path on the
same prefix IDs, FP32, cache disabled, released penalties and prior. Require
all author/local output texts to agree exactly after whitespace stripping.

Freeze source/config/checkpoint/data hashes before each run. Full control
requires a passed matching pilot. Stop on parity failure, OOM, nonfinite logits
or changed dependencies. Preserve failures, no CPU fallback or new seed retry.
Record every output beside the released reference, exact reference matches
and nonempty counts. These are inference-path diagnostics, not complete-summary
accuracy or a full reproduction of paper ROUGE/perplexity. Do not call parity
proof that the summaries are faithful, or use news results to admit the sealed
conversation test. No speed claim, original-body changes or final-test opening.

Artifacts: `20261022_released_xsum_{data,pilot,control}_v1` in the paper root.
[Official authors' evaluator and dataset links](https://github.com/HazyResearch/ncm).
