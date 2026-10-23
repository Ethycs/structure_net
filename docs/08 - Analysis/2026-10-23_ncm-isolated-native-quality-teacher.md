# Isolated native quality teacher: complete development evaluation

Date: 2026-10-23. Hypothesis `ncm-isolated-native-quality-teacher-v1`.
Classification **isolated_quality_teacher_development_failed**. Aggregate `data/experiments/ncm_paper_regime/20261023_isolated_quality_results_v1/campaign_results.json`,
SHA256 `b3fbdc13b5847d61f5f70a92e68e757997ddf5e1b110a0a75bc0a3f410148c15`.

The clean native quality teacher scored **56/64 whole-summary
passes (87.50%)** under qualified Terra. This is an isolated native-language
teacher control preparing NCM distillation, not a working NCM replacement or
goal completion. The original nine-pass NCM remains0/64; the converged shared
adapter hybrid remains41/64; unchanged3B and1B baselines remain52/64 and47/64.

| Role | Whole summary | Valid | Faithful | Salient | Readable |
| --- | --- | --- | --- | --- | --- |
| secondary | 56/64 | 64/64 | 56/64 | 63/64 | 64/64 |

Whole-summary success jointly requires faithfulness, salience, readability,
nonempty output and at most128stored cl100k tokens. Terra found
0 overflows and 0 empty outputs.
Its Wilson95% interval is `[0.7722543817887005, 0.9352775733666955]`. Independent Sonnet was
not admitted after the Terra gate failed.
The teacher development gate **failed**. Every
failed case and cited judgment is retained in `data/experiments/ncm_paper_regime/20261023_isolated_quality_results_v1/case_failures.json`.

Training used exactly512updates, batch one and accumulation eight, on252vetted
complete summaries:143old short labels plus109longer-source labels. Inputs
were complete, with a maximum2,276native prompt in a2,304-token frame. Saved
single-atom snippets were not used as if they summarized complete passages.
Only4,587,520fresh word-adapter parameters trained; original28blocks/MLPs,
embeddings, native head, old concept bank, heads, hints, biases and books stayed
frozen. Word-quality queries intentionally used no concept hint/bias. No coarse
loss, noisy-word loss, raw-language replay or intermediate checkpoint selection.

The actual longest-input cache/gradient pilots passed before full training.
Their earlier rank2-ID/rank3-embedding validation bug and correctedv2runs
remain separate preserved artifacts. Full training started from fresh zero-B
word adapters, not pilot weights. Frozen parameters/books, all CPU source cache
values and exact optimizer/RNG/checkpoint reload passed integrity checks.

After training, the first two development outputs from the plain implementation
were retained exactly. Static word-adapter inference had identical initial
full-vocabulary logits and all128word histories, including after checkpoint
reload. Its only implementation change avoids repeated GPU zero checks for
unchanged matrices; no speed measurement was made. All64outputs used128fixed
causal word updates and one exact fixed-point check, with EOS initialization.
No oracle plan, gold history, external verifier, repair or alternative selection.

All256prior first judgments, including negatives, were audited and preserved.
Generation terminated before judging. Complete histories, source identities,
token counts and raw first-valid decisions were replayed offline. Development
sources are disjoint from training by ID/session/group/hash, but are reused
development cases and short fragments. Original256and newer64final sources
remain sealed. A teacher gate does not establish final/full-range NCM accuracy.
Later acceleration training must use the qualified native teacher's own fixed
points/histories and retain complete-summary quality checks. No compression,
latency, tokens/s improvement or NCM success is established here.
