# Source-conditioned root learning on16TRAINcases

Date:2026-10-23. Hypothesis `ncm-word-final-source-root-capacity-v1`.
Classification **source_root_TRAIN_capacity_failed**. Aggregate `data/experiments/ncm_paper_regime/20261023_word_final_root_capacity_results_v1/campaign_results.json`,
SHA256 `ba0dd4fba3612aeb04ef1fd2bb55639f760099fc7f94fc48e1cdc02cef6fce73`.

This is a TRAIN-only root-classification capacity check. It does not measure
summary accuracy. The preceding word-final model remains0/64short and0/64long
complete-summary passes. The95%task goal remains unmet; both final cohorts
remain closed and no speedup is established.

| State | Active-block root labels | All root labels | Active labels with wrong sources |
| --- | --- | --- | --- |
| initial | 6/176 | 78/256 | 6/176 |
| final | 121/176 | 200/256 | 6/176 |

The fixed capacity gate failed: at least95%active
and95%all-root recovery, with at least50percentage points lost when sources are
cyclically shuffled. Only the final256-update checkpoint was selected. Source
shuffling scores each case's labels against the next complete source's
predictions; it is a negative control, not held-out generalization.

The model started from the same failed final512checkpoint as its two-update
pilot. Both reproduce all16initial source-only root sequences exactly. The
fixed selection contains four TRAINcases in each of four source-length bands.
Only the new source bridge, shared generator, root queries and root head trained;
original Llama weights/MLPs, codec, word projection/decoder and other concept
heads stayed frozen. The original parent was not loaded. Full source features
were retained with explicit masks and no source or target clipping.

Training used256updates, batch1, active-root CE+.25padded-root CE, FP32master
weights and AdamW with BF16autocast/FP32loss. The16-update warmup reached3e-4,
then cosine decay ended at3e-5. Scope, weighting, learning rate and data size all
differ from the earlier joint run; this does not isolate one causal change.
No word loss or target history enters root prediction. All256source samples,
frames and learning rates, full raw root IDs, counts and shuffled controls
replayed. Frozen parameter/codec hashes and model/optimizer/RNG/source-only
reload passed. Peak training allocation was3,517,923,328
bytes. The pilot exercised the2,816-position maximum source frame.

A capacity pass can inform a separately frozen full-hierarchy training
experiment. It does not qualify the later concept levels, word decoder,
source-to-summary quality, task speed or model generalization. Preserve the
earlier full128-case failure and all its first judgments unchanged.
