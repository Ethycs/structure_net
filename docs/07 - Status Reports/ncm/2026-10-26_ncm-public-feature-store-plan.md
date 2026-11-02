# Once-only original-body features for all admitted public sources

The source producer measured 162,353 sources, 91,344,717 GPT2 source positions,
and 125,616,400 native prompt positions. Maximum lengths are 4,124 and 2,275,
respectively. The previous nine-source feature pilot reached only 2,257 source
positions. Its measurements do not qualify the full population.

The root filesystem has about 241 GiB available, but the mounted writable
`/data2` filesystem has about 3 TiB. Use a dedicated new directory there for
complete disk-backed feature storage. The full uncompressed FP32 estimate is
1,122,443,882,496 bytes. This supersedes the earlier inference that features
must be recomputed because only root-filesystem space was available. Keep
bounded host memory; do not load all feature tensors into RAM.

Require the completed exhaustive metadata audit before either stage. Select
the union of the original nine public probes, the four longest GPT2 sources
and the four longest native prompts, ties by ID. Run the unchanged original
28-layer Llama body in BF16 on physical GPU0 RTX3060, TF32 off; retain the
unchanged FP32 character-overlap alignment. Repeat each selected extraction
and compare every FP32 byte. The nine prior probes must match their saved raw
feature hashes, and both historical TRAIN feature controls must reproduce.
Require peak allocation at most 8.6 GB and unchanged body tensors/no gradients.

Store the existing byte-shuffled zlib-level1 representation in bounded
256 MiB shards with individual payload hashes, original body/source/prompt/
alignment bindings, source metadata record hashes and raw FP32 hashes. Use
four CPU encoding threads and at most eight pending feature matrices. Every
encoded record must decode byte-exactly before writing. Pilot disk readback
must reproduce every feature and verify complete shard/record coverage.
Pilot concurrency uses these same four workers and bounded queue.

After observed successful pilot exit and a separate CPU replay of its disk
records, run all 162,353 sources once, in catalog order. Reuse none of the
pilot matrices: extract them again and require the same raw hashes. Retain
all full source positions. Keep at least 100 GiB disk headroom, checked every
1,024 records. Preserve partial shards if interrupted; no partial store is
admitted. Hash completed shards as they are sealed. A later exhaustive CPU
readback must validate every record before training use, with four decoding
threads and at most eight pending records.

Repository evidence roots are `20261026_public_feature_store_{pilot,full}_v1`
under the existing NCM experiment directory. Feature shards live in
`/data2/structure_net/ncm_public_features_20261026_v1/{pilot,full}`. Run from
the repository root with locked Pixi `ncm`, physical GPU0. No training,
semantic evaluation or application speed claim occurs in this preparation.
All public references remain pretraining labels, not verified factual gold.
The original 95% task endpoint and sealed final protocol remain unchanged.
