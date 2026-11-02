# All 888 engineering sources have complete model geometry

The 732 TRAIN and 156 held-out engineering sources were fully tokenized and
aligned in a CPU-only run with actual exit 0 (session 36665). Source fields were
whitelisted before tokenizer access; changing target labels cannot affect this
interface. The two focused interface tests passed (session 23667), and original
first/longest TRAIN prompt/alignment geometry matched exactly.

The prepared inputs retain 1,735,277 GPT2 source positions, with a maximum of
5,187 per input. The longest native Llama prompt is 2,276 tokens. The complete
FP32 aligned-feature estimate is 21,323,083,776 bytes. No truncation or reduced
precision was applied. Prompts keep the original training date, 23 Oct 2026,
so date changes do not silently alter the frozen-body representation.

Evidence: `data/experiments/ncm_paper_regime/20261025_engineering_source_metadata_v1/`.
Index SHA256: `f34d9832db4848130036c132c8b1b48d799fbe67137f56fddaf2d7ba83e12989`.
The CPU stage loaded no parent weights, extracted no features and updated no
model. A separate GPU workflow performs repeated extraction, original-reference
parity, frozen-body hash checks and lossless FP32 storage before the next training
experiment. Existing DEV and sealed final cohorts remain unchanged.
