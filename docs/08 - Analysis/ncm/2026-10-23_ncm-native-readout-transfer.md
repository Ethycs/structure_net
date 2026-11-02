# Five-block native readout fails all64 unseen summaries

Date: 2026-10-23. `ncm-native-readout-transfer-v1` is complete,
**native_AR_readout_transfer_failed**. Whole-summary accuracy is **0/64=0.0%**
under qualified Terra; all64 judgments are valid. Component pass counts:
faithful0/64, salient0/64, readable0/64.
The95%task objective remains unmet. This is an AR reference, not NCM success.

The same fixed final checkpoint reproduced32/32 accepted training summaries
and995/995 native target tokens. Source shuffling reduced recovery to25/995;
the separate replay matched original outputs/metrics exactly. These controls
establish fit and source use, not generalization. The development outputs are
nonsensical fragments, such as “The products, moving.” for Badlands travel tips.
This is a substantive application failure, not exact-reference disagreement.

All64 development sources are disjoint from training by ID, source session,
group and source hash. Same source prompt, no truncation, greedy seed42,
BF16/SDPA/GPU0, batch2,128-native-token generation ceiling and128-cl100k storage
cap. No further updates or output repair. All four pilot judgments were valid
negatives and were reused;60 new judgments completed the64-case endpoint.
No Sonnet stage ran after the failed61/64 secondary gate.

The original3B body/FC weights and five native readout base blocks stayed frozen.
Copying the vocabulary and selected blocks did not preserve usable language
generation in this trained configuration. These results do not isolate pruning,
bridge initialization and limited data as separate causes. They do reject a
claim that the training fit solves the summarization task.

The new native-parallel NCM pilot passed numerical systems checks, but its
planned full training-only capacity run is deferred by the explicit intact-
language-control amendment. It has no held-out quality result. The next language
component control keeps an entire pretrained small decoder and requires an
initial no-op correction path; it is not a rescue of this0/64 model.

Aggregate `data/experiments/ncm_paper_regime/20261023_native_readout_transfer_results_v1/campaign_results.json`; SHA256`8f39f51dfa2ab10fb2206d76bde77b002e53a8e0d5632b14876ceffa14bd4d65`. Fixed checkpoint, source
features, raw64outputs, first requests/responses and source snapshots remain
append-only. Final256task cases stay sealed, registry confirmed=false, no speed
or NCM-success claim.
