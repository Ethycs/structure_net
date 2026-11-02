# Singleton native codes for recovered paraphrase targets

Prepare the complete target hierarchy for every retained pair in the recovered
paraphrase catalog. This is frozen encoding, not training or summary evaluation.
Deduplicate identical complete native128 frames, preserve all pair-to-frame
bindings, and encode every distinct frame separately in FP32 with TF32 disabled.
The previous128-case diagnostic found one batch-sensitive reference; do not reuse
the earlier batch-eight corpus codes or infer parity from a small batch pilot.

Use the exact released LM1B codec, whose encoder/VQ agrees with the repaired
parent. Decoder adaptation does not alter these targets. Require the full codec
hash `9aace450a49a7dfb44ba1eb00a19974a50d67062ad9f2202a6a2361ea4344e32`.
Each output must contain641 integer IDs in[0,8192). No target truncation,
alternate renderings, fitted weights, or output selection.

Pilot: choose32 distinct TRAIN frames covering first/last and shortest/longest
frames; verify each singleton twice, plus reproduce all128 saved singleton
reference code vectors. Diagnostic controls are not added to TRAIN. Full pass:
re-encode all eligible unique frames, reproduce every pilot code, and verify
every persisted record's offset, hash, range, frame hash, and pair binding.

PhysicalGPU2 RTX2060 SUPER, locked Pixi, deterministic runtime,3.5GB own allocated
GPU memory ceiling and4GB host-RSS ceiling. Check free headroom first. Use fresh
roots `20261028_recovered_paraphrase_codes_{pilot,full}_v1` and binary storage
under `/data2/structure_net/ncm_recovered_paraphrase_20261028_v1/codes/`.
Freeze producer and inputs before encoding. Full pass requires actual successful
pilot completion. Stop on any mismatch; preserve failed roots. Existing tree
training, task evaluations, and reserved final sources remain untouched. Prompt
features and a fixed subsequent noisy-supervision training protocol are still
required; these targets alone do not admit training or establish accuracy.
