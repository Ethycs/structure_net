# Qwen3B native-boundary and packing diagnosis

**The native-boundary failure is caused by mismatched batch and padding shapes. The selected eleven-block model has a separate recovery-quality problem.** This is a read-only diagnostic on the existing 64 development cases; no training, held-out scoring or speed benchmark was performed.

Artifacts: `data/experiments/jspace_3b_seam_diagnostic/20261115_v1`. Fingerprint: `b754aa3bc6372614d13c7e23153573588e58ad79fd7b1dcece0319af4ea1f9b2`. Result SHA256: `b9278390c56150d3aae07323fc664e47bdf3f9941e797711d3d2539870959551`.

The same frozen layer24 and accepted compact suffix25–35 were used throughout. Native layer24 was recomputed in three layouts; each result was then fed to the same packed parent in its original four-case readout grouping.

| Native recomputation | Cached boundary exact? | Boundary maximum difference | Downstream correct /64 | Downstream maximum choice-logit difference |
|---|---|---:|---:|---:|
| Unpadded singleton | True | 0.0 | 46 | 0.0 |
| Singleton padded to the original group width | False | 0.5 | 46 | 0.25 |
| Original padded batch of four | False | 0.5 | 45 | 0.625 |

Unpadded singleton replay matches every valid cached boundary element exactly and reproduces all downstream logits. The prior mixed-batch seam changed one WinoGrande answer and failed the unchanged numerical tolerance. The next control should preserve singleton native execution, then reassemble the original parent batch. This fixes an execution comparison; it does not relax the accuracy or numerical gates.

Packed Torch and packed Triton agree exactly across all full-vocabulary logits on these cases. Unpacked evaluation and unpacked autograd execution both score46/64, with KL divergence0.00023901 relative to packed inference and maximum choice-logit difference0.25, inside the existing tolerance. Their teacher-target KL remains about0.415, agreeing with the degraded packed checkpoint. The large recovery loss is therefore not explained by export or fused-kernel differences in this measurement.

All model weights and buffers were hashed before and after and remained unchanged. The cached-parent reference reproduced the selected checkpoint exactly. No optimizer was created and no configuration or tolerance was changed.

Reservation `2e9393ceecf0` allocated11264MiB on the RTX3060, with11008MiB allocator cap. Runtime 55.74s; peak reserved 8,046,772,224 bytes. These are diagnostic timings, not model token throughput.
