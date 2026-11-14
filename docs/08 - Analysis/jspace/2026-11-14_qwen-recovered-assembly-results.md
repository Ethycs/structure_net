# Recovered Qwen assembly results

Study: **JSPACE-QWEN-RECOVERED-ASSEMBLY-01**. Classification: `hybrid_assembly_development_gate_failed`. **Confirmed: false.**

The primary hybrid scored **24/64** against **50/64** native and **failed** the fixed47/64 development gate. All36 compact blocks scored **27/64**. The accepted two-block control reproduced **49/64**. These are actual complete model paths on the same64 repeatedly inspected development questions.

| Arm | Correct /64 | Accuracy | MatchedFP32 tensor bytes | Reduction vs nativeFP32 |
|---|---:|---:|---:|---:|
| native | 50 | 78.12% | 12,343,755,008 | — |
| accepted_two | 49 | 76.56% | 12,052,439,304 | 2.36% |
| all_compact | 27 | 42.19% | 7,100,072,336 | 42.48% |
| hybrid | 24 | 37.50% | 7,391,388,040 | 40.12% |

The hybrid uses the selected compact blocks except12 and15, which remain native because their independent tests missed the isolated gate. The all-compact result is a separate diagnostic and cannot replace the declared primary outcome. Every arm began with original token embeddings, executed all36 blocks, and used the original tied vocabulary readout. No cached internal states bypassed composition.

| Arm | ARC /16 | BoolQ /16 | HellaSwag /16 | WinoGrande /16 | Gains vs native | Losses vs native |
|---|---:|---:|---:|---:|---:|---:|
| native | 15 | 13 | 14 | 8 | 0 | 0 |
| accepted_two | 15 | 13 | 14 | 7 | 0 | 1 |
| all_compact | 5 | 7 | 7 | 8 | 7 | 30 |
| hybrid | 5 | 7 | 4 | 8 | 7 | 33 |

Both controls passed exact prediction replay and logit tolerance5e-4+1e-5×|reference|. Maximum logit differences were 0.000347137451 (native) and 0.000175476074 (accepted two). Selected checkpoints, all epoch-report hashes, their fresh-verification receipts, per-layer fits and the frozen protocol were pinned before assembly. Block0 retained its separate first-block enrichment fit.

Storage uses the actual arm inventories, including fixed bases/geometry, packed parameters, final norm and rotary buffers, and tied embedding/readout counted once. All compared tensors are FP32. **These reductions are not comparisons against the original BF16 stored model.** Activations, KV cache, temporary label-row copies and allocator reserve are excluded.

The evaluator streamed one block at a time. Its elapsed time includes loading and disk work and is not inference throughput. No speed benchmark or untouched confirmation evaluation occurred. A development pass still requires separately frozen confirmation and deployment measurements; a miss shows that isolated recovery did not preserve the tested assembled behavior.

Evidence: `data/experiments/jspace_recovered_assembly/20261114_v1/assembly_analysis.json`, SHA-256 `7cb2f988bdf279cb263b04541fe56f3c3b5acab665355364658089ade786f5ed`. Plan, aggregate, four arm manifests/results, reference rows, live/archived producers and small inputs rehashed. Large selected states retain frozen producer SHA256 plus current immutable file-identity checks; assembly rehashed each selected state immediately before loading it.
