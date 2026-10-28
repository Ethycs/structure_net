# Source-hierarchy training qualification and recovered codec corpus

The source hierarchy is implemented as shared source memory read by each of
the twelve NCM generator blocks. Token-level attention remains available;
eight-way pooled ancestors provide broader context. The hierarchy has not yet
been shown to improve complete-summary accuracy or speed. The latest completed
application semantic DEV result remains 0/128; the final cohort remains sealed.

Two training-qualification failures exposed concrete implementation constraints.
Both were stopped before any optimizer update and retained as evidence.

| Lifecycle attempt | Observation | Consequence |
| --- | --- | --- |
| Original dense activations, v1 | CUDA OOM during FP32 batch-four gradient comparison on RTX2060 | Add activation recomputation; preserve batch and objective |
| Activation checkpointing, v2 | Fit within 3,907,890,688 allocated bytes; clean gradient relative error 0.01830263 exceeded the unchanged tolerance | Repair per-source hierarchy-depth masking |

The v2 gradient discrepancy was concentrated in the twelve new tree-output
projections. A 197-position source has three ancestor levels, but sharing a
batch with a 2,273-position source gave it a fourth, duplicate root. The initial
zero output projections concealed this in forward-only parity checks. Padding
therefore affected the training signal even though initial generated codes and
words matched exactly.

The correction masks inactive levels separately for each source. It preserves
all original token positions and each source's actual hierarchy. CPU regressions
cover lengths 1, 3, 17 and 65 in one batch, with nonzero tree outputs and level
weights. Individual and batched outputs and gradients agree; the original
unmasked implementation fails the regression. Checkpoint recomputation also
preserves exact outputs, gradients and RNG state with and without dropout.
Four new tests passed in locked Pixi. The fresh GPU lifecycle v3 then passed:

| Corrected real-checkpoint check | Result |
| --- | --- |
| Initial source-only codes and words on four controls | Exact |
| Clean FP32 batch/individual relative gradient error | 0.0000112150, passed unchanged tolerance |
| Noisy FP32 batch/individual relative gradient error | 0.00000636154, passed unchanged tolerance |
| Argmax disagreements, both gradient comparisons | 0 |
| FP16 optimizer updates | Two full 32-example accumulated updates |
| Resumed second update, model/optimizer/scaler/RNG and source-only outputs | Exact |
| Protected codec and word-source tensors | Unchanged |
| Peak own allocation on RTX2060 | 3,907,891,200 bytes (3.91 GB) |
| Newly added tree output projections updated | 12/12 |
| Newly added token-attention output projections updated | 9/9 |

The post-checkpoint inspection also found nonzero level-router weights in five
blocks after these two warmup updates; the other seven routers remained at their
initial uniform mixture. This does not prevent their nonzero tree-output updates.
There were 64 scheduled TRAIN presentations plus one replayed verification
update. These are systems checks, not a full trained hierarchy or quality result.

Separately, all 99,367 reserved-training native text frames now have their full
641-code targets cached. The unchanged released encoder/codebooks produced the
cache in batches of eight. All 32 fixed individual-versus-batched controls
matched exactly; every written code row was read back and checked. Encoding and
verification took 426.17 seconds and peaked at 791,160,320 allocated GPU bytes.
The binary code cache occupies 127,388,494 bytes; frames occupy 25,437,952 bytes.
This is text-reconstruction training material, not admission of these records
as high-quality source-to-summary pairs.

The decoder-only adaptation pilot passed. It used the full 32-domain-plus-eight
retention-example update, retained the repaired initialization, and preserved
the encoder/VQ tensors. After two scheduled updates, an uninterrupted third
update exactly matched the same update after checkpoint reload, including
model parameters, optimizer and RNG state. Those two extra verification updates
are pilot-only and are not part of the full training exposure. Peak own GPU
allocation was 1,216,039,424 bytes. The full run started afresh from the parent
and is scheduled for two complete epochs, 6,212 updates, including both tails.
It streams domain frames/codes from disk and trains only the codec decoder.

The public-data curriculum also reached its four-epoch milestone at update
6,768 with successful process exit, strict checkpoint reload and unchanged
reconstruction path. Its new checkpoint SHA-256 is
`d19f3d2287ac0d4e653ed69e17fd8efe00bce07730d5ffd4c4f8f42421320213`.
Its independent training-chain audit passed: 649,412 public and 216,472 anchor
presentations across all 6,768 committed updates. The unchanged 128-case
source-only semantic evaluation must finish before assigning it an accuracy
result.

Evidence roots under `data/experiments/ncm_paper_regime`:

- `20261027_source_tree_training_lifecycle_v1`: actual exit 1, session 45830,
  chunk `192698`, no optimizer updates.
- `20261027_source_tree_training_lifecycle_v2`: actual exit 1, session 53876,
  chunk `67f7c3`, no optimizer updates.
- `20261027_source_tree_padding_CPU_v1.json`: four passed, session 37732,
  chunk `48e007`.
- `20261027_source_tree_training_lifecycle_v3`: actual exit zero, session 86753,
  chunk `fbdcdd`; successful corrected lifecycle.
- `20261027_source_tree_trained_branches_v1`: actual exit zero, session 30102,
  chunk `390d92`; inspection of the saved two-update checkpoint.
- `20261027_recovered_codec_codes_v1`: terminal systemd exit zero observed in
  chunk `d0ce28`; unchanged codec and complete readback.
- `20261027_recovered_decoder_pilot_v1`: actual exit zero, session 83627,
  chunk `81db13`.
- `20261027_recovered_decoder_train_v1`: durable training service, prospective
  budget and artifact identities recorded before updates.
- `20261026_public_epoch4_attempt1_service_terminal_v1`: actual observer exit
  zero, session 95785, chunk `7b147a`; original training invocation exited zero.
- `20261026_public_curriculum_epoch4_training_audit_v2`: actual exit zero,
  session 11256, chunk `f2d442`; complete committed exposure replay.

All implementations use locked Pixi. Historical failures and active producers
remain unchanged. Decoder repair, source-hierarchy qualification and full
application semantic accuracy are separate measurements.
