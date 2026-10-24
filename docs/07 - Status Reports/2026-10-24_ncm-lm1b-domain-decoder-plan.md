# Source-disjoint task-domain NCM decoder adaptation

Date: 2026-10-24. Exploratory, underpowered hypothesis
`ncm-lm1b-source-disjoint-domain-decoder-v1`. The full objective remains a frozen
Llama body with NCM summary generation achieving at least 95% complete-summary
task accuracy on source-disjoint evaluation, before any speed claim.

Known outcomes: native PAD0 and CLS codecs each passed 45/64 semantic probes;
generic author-TRAIN decoder adaptation improved token agreement by 197 positions
but fell to 39/64 semantic passes. Its checkpoint remains rejected. Original
targets pass 64/64 both judges, BERT roundtrip 59/64 Terra. These are component
development probes; the last conditional student remains 0/128. All final sources
remain sealed. The latest canonical cache contains 2,266 first decisions.

Question: can fitting the released decoder to the application's text domain
improve meaning preservation on 64 source sessions excluded from its gradients?
The prediction is improvement over the unmodified release, with qualification
still requiring 61 of the same 64 cases passing both qualified judges. This is
exploratory development on previously inspected cases, not a fresh final test.

Start from exact original author LM1B weights, never the rejected generic repair.
Keep encoder, codebooks, EMA and normalization state frozen in eval mode. Train
only the same 18,870,656 decoder parameters. The original Llama and native NCP
generator are absent. Keep native 128-position geometry, all 12 clean concept
levels, FP32, TF32 off, unchanged raw argmax and PAD0 framing. No optimized target
latent, code corruption, fine-level dropout, output repair or fallback.

Hold out exactly the earlier 64 hash-selected semantic cases, all their source
sessions, groups, source hashes and turn hashes. Verify no overlap with the
remaining 626 cases. Only those 626 approved summaries and complete raw sources
enter task-domain reconstruction pools. BERT-tokenize all text, split into
complete lexical chunks of at most 126 words and add CLS boundaries and PAD0.
Do not clip or discard long text. Record every source, text hash and window.

The fixed 16,384-frame training mixture contains 3,276 author TRAIN replay frames,
6,554 summary frames and 6,554 raw-source frames. Select author frames without
replacement from the verified previous 16,384-frame author-TRAIN subset; precompute
the previous 20% full / 80% short lexical-prefix augmentation for those frames.
Sample task frames uniformly with replacement within their complete candidate
pools. Corpus seed 83142, precomputation seed 85142. Runtime sampling seed 84142,
dropout seed 86142. Runtime frames remain unchanged. This jointly changes corpus
and length distribution; it is not a single-factor causal estimate of domain.

Reuse the frozen generic trainer's objective and lifecycle: word CE ignoring
PAD0, AdamW (.9,.95), eps 1e-8, weight decay .01, clip 1; microbatch 8,
accumulation 4, 1024 updates, warmup 64 to 2e-5, cosine to 2e-6. Save step 512
for recovery only and select the fixed final 1024. A fresh two-update CUDA pilot
must prove real gradients only in the decoder, exact optimizer/RNG/output reload,
unchanged frozen state and peak at most 4.7 GB before full training. Full training
resets to the original release, reproducing the pilot's initial outputs.

Use new `20261024_lm1b_domain_decoder_{data,pilot,train,workflow}_v1` roots. Preserve
all source snapshots, input hashes, every sampled row, finite loss/gradient trace,
checkpoints and actual child/whole-tool terminal receipts. No old run is resumed.
A separate prospectively frozen evaluator will reconstruct all 690 targets with
fit-versus-heldout counts and grade only the same 64 held-out cases with canonical
first judgments. Better token fit cannot rescue a semantic failure. Even a
passing codec is not a source-only summarizer; conditional training, full 128-case
development qualification and sealed final evaluation remain required.
