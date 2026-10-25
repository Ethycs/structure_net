# Larger original frozen-body source feature cache

Status **all10000_lossless_FP32_source_cache_payloads_and_bindings_replayed**. Data preparation only. Hypothesis `ncm-additional10000-frozen-body-lossless-FP32-features-v1`.
Campaign `data/experiments/ncm_paper_regime/20261024_additional10000_source_features_results_v1/campaign_results.json`, SHA256 `2b5cb94998e058147e4ac200eb9c086095b5d486dbe11a6bc50d4cbfbbcb9b84`.

All10,000reserved source-disjoint TRAIN sources retain every source position. The unchanged28layer BF16Llama body produced7,777,015character-aligned FP32source positions, each with3072channels. Original first/longest TRAINfeature references matched exactly before each worker; every pilot was reused. All10,000stored payloads were independently decompressed and verified against source/prompt/alignment/body bindings and originalFP32checksums.

Actual raw features: 95,563,960,320bytes. Actual stored headers+payloads: 32,842,107,878bytes (34.37%of raw). Byte-shuffled zlib1is reversible storage compression, not quantization.

This compresses data storage, not model weights or inference computation. Teacher-label creation and future model training are separate. The native655student still failed0/128DEV; this cache establishes neither95%summary quality nor a speedup. No new canonical semantic judgments were made and final320sources remain closed/unhashed.
