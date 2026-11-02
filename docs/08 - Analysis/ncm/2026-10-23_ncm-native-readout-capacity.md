# Native readout fits accepted training summaries; transfer remains separate

Date: 2026-10-23. `ncm-native-readout-capacity-v1` is complete,
**capacity_gate_passed**. The fixed checkpoint reproduces32/32 training
texts and995/995 native target tokens including EOS.
Source shuffling gives25/995tokens and0/32 exact
texts. This is training memorization and source dependence, not held-out
summary accuracy or an NCM result. The95%task goal remains unestablished.

The readout retains native tied word embeddings/head, final norm and five
frozen Llama blocks0,7,14,21,27. Per-token original-body states1,14,28 enter a
trainable fusion projection. Only32,656,384 fusion/LoRA parameters trained;
897,352,704 native readout base parameters and all original body/FC weights
stayed unchanged. There is no learned text codec and no parent verifier.

Training used the same32 previously independently accepted summaries for1,024
updates, batch2 and accumulation4, AdamW1e-4. The final checkpoint alone is the
endpoint. Native token round trips, causal masking, gradient routing, real
GPU/BF16 memory, cached/full checks and exact trainable reload passed. Peak
allocation6795678208bytes includes full-body extraction.
Training timing on shared hardware is not a speed benchmark.

The producing runner retained only scalar shuffled metrics. A separately
declared no-training replay restored the frozen final checkpoint, matched all
original free token sequences and both metric dictionaries exactly, and saved
the missing raw shuffled predictions. Original artifacts were not rewritten.
This permits audit of the counterfactual without changing the original gate.

The four unseen transfer-pilot outputs are systems checks, not quality evidence.
The separate64-case development evaluation must establish any generalization;
the native AR readout is not itself a coarse-to-fine NCM. Broader training still
requires newly qualified labels. Final256 cases remain sealed.

Aggregate `data/experiments/ncm_paper_regime/20261023_native_readout_results_v1/campaign_results.json`; SHA256`310ba610522453547f2ab640f1c25bd69f17687d4d23b7388ae362c7da0a4f66`. Models, source features,
first positive label judgments, raw free/shuffled replay and producer snapshots
remain append-only. Registry confirmed=false, task_accuracy_measured=false.
