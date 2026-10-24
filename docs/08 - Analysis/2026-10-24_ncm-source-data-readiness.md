# Larger TRAIN source metadata and lossless FP32 storage controls

Status **completed data controls; student quality unproven**.
Hypotheses `ncm-additional10000-source-tokenization-v1` and `ncm-lossless-FP32-source-feature-storage-control-v1`.
Campaign `data/experiments/ncm_paper_regime/20261024_source_data_readiness_results_v1/campaign_results.json`, SHA256 `b3718206614496f224878943ae6ca9a3ce813cfbc324a358624e11ef9a33f948`.

All 10,000 reserved public TRAIN sources were independently retokenized and
their full character alignments rebuilt on CPU. Original first/longest
TRAIN prompt/date/alignment geometry matched exactly. No new teacher
labels or selected final sources entered preparation.

The corpus has 7,777,015 GPT-2 source positions: 95.56 GB of raw FP32 features.

Both lossless layouts were independently decoded on all 16 fixed TRAIN
feature tensors. Every one of 32 payloads reproduced the original FP32 bytes,
including every source position/channel; no quantization or arithmetic.

Byte-shuffled zlib at level 1 stored 68,204,240 bytes versus 214,523,904 raw bytes (31.79%).
Applying measured per-band sample ratios projects 31.32 GB for the new corpus. That is an estimate; its actual full feature cache has not been computed.

This compresses training-data storage. It does not shrink model weights,
measure model token throughput or establish 95% summary accuracy. Current
student training on 655 examples and its held-out evaluation remain separate.
Full feature extraction is gated after its completed DEV and runs only
if further training is needed. Final 320 sources stay closed and unhashed.
