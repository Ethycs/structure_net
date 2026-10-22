# NCM training on the shared memory summaries

The existing NCM completed an initial adaptation using a dataset of
**45,136 original-text / saved-summary pairs**, with
**4,988 compatible pairs held out by source conversation.
The fixed training budget samples from that dataset; it is not a full epoch.
This continues the existing model weights. The final checkpoints are saved.

Date: 2026-10-22. Hypothesis: `ncm-memory-summaries-v1`.
Classification: `shared_summary_training_completed_quality_unqualified`.
One training seed; underpowered for an architecture comparison.
Aggregate: `data/experiments/ncm_memory/20261022_training_v1/campaign_results.json`.
SHA256: `5c984628ea61d66f8a3ef3dc3cb4befcc5653d7b0fbae78b51aca2c4164eeff5`.

## Data

Restored all 13 shared bundles and verified all 1,500 member hashes. Input
repository: `Ethycs/memory_condense`, Git commit
`fc25ea0d3cccc589346ddc13747218b8dc7a1221`, lakeFS commit `82d974caec273e113dd864f9da352c003540e9225fbcce3766a81c103e005dab`.

| Split | All unique atomic pairs | Source sessions | Native-compatible pairs |
| --- | ---: | ---: | ---: |
| Train | 204,967 | 17,283 | 45,136 |
| Test | 22,777 | 1,912 | 4,988 |

The deterministic 90/10 split assigns source-session groups, not individual
rows. Repeated non-bypass input text is grouped together. Source-session overlap
and eligible exact-input overlap are both zero. This partition uses LongMemEval
as training data and is **not the untouched LongMemEval benchmark**. Saved
summaries are training labels, not independently human-verified references.

Targets are unchanged stored atomic summaries. Every raw span was recovered
from the original turn and checked against its saved SHA256. All atomic pairs
are exported to `train.jsonl` and `test.jsonl`; compatible training inputs are
`native_train.jsonl` and `native_test.jsonl` under `data/experiments/ncm_memory/20261022_data_v1`.

This run retains the model's 512-GPT-2-token source and 48-position output
canvas (47 text tokens plus EOS). It excludes application bypass inputs below
192 cl100k_base tokens, over-capacity sources/targets and targets over the
128-cl100k_base-token stored limit. Nothing is truncated. Exclusion counts,
which can overlap, are in the dataset manifest. Hierarchical/parent sections
remain in the restored stores and were not training targets in this run.

## Training and held-out diagnostics

The encoder and multiscale codebooks stayed frozen. The text decoder completed
512 AdamW updates, batch 16, LR 2e-4. The generator completed 1,024 updates,
batch 8, peak LR 1e-4 with warmup/cosine decay. We used BF16 computation and
FP32 weights/code assignment on physical GPU0, an RTX3060. Final checkpoints
were fixed in advance; no held-out checkpoint selection or retuning occurred.
Full optimizer/RNG continuation checkpoints and loss traces are retained.

The decoder sampled 7,473
unique training pairs; the generator sampled
7,484. The first
systems check stopped before any optimizer update because a JSONL reader split
on Unicode paragraph separators inside strings. A physical-line reader and
regression test fixed it; the second CUDA pilot passed. Both roots are retained.

Both original and trained models were evaluated on the same 256
test examples selected by a fixed ID hash after optimization finished.

| Diagnostic | Original checkpoint | Trained checkpoint |
| --- | ---: | ---: |
| Code cross entropy | 2.9702 | 1.2698 |
| Codec text cross entropy | 0.3562 | 0.2637 |
| Codec token accuracy (excludes PAD) | 94.26% | 95.46% |
| Mean word-unigram F1 | 0.0840 | 0.2839 |
| Exact saved-summary matches | 0 | 0 |
| Empty generated summaries | 3 | 0 |

Word-unigram F1 uses lowercased regex word counts, with both missing and extra
words penalized; it does not measure factual support or semantic equivalence.
Codec token accuracy measures reconstruction from the target's own codes,
not generation from the source. These diagnostics are not calibrated final task
accuracy. The 95% quality target and a speed improvement remain unestablished;
no speed comparison was run here. Per-case outputs remain in the raw artifacts.

## Artifacts and reproduction

- Codec: `data/experiments/ncm_memory/20261022_training_v1/codec.pt`,
  SHA256 `905e6563394f37f539ff55a0466620589d7f5feab9a0aa2c7eb2dbd9e3b05eb6`.
- Generator: `data/experiments/ncm_memory/20261022_training_v1/generator.pt`,
  SHA256 `9a27a3e214915d30221640b0ff0c7fb6745a3240f1f5fbfe3b83b3015cc9de66`.
- Data and training entry points: `experiments/structure_net/ncm_memory_data.py`
  and `experiments/structure_net/train_ncm_memory.py`.
- The dated training plan records exact inputs, budgets and boundaries.

Run the data entry point's `restore` and `prepare` commands first, then the
training entry point's `pilot` and `train` commands against fresh output roots.
Use the existing MoBA Python environment, `LD_LIBRARY_PATH=''`,
`HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=2`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`CUDA_VISIBLE_DEVICES=0` and `PYTHONPATH=src:.`. Preparation additionally uses
the separately installed tiktoken 0.11.0 dependency. Exact producing source,
configuration and data/checkpoint hashes are saved in `producer.json`.
