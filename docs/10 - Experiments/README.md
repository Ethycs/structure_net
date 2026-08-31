# A/B/C Semantic Routing Experiment Program

**Status:** EXPERIMENTS 01–10 AND WAVELET-ROUTER FOLLOW-UP COMPLETE  
**Date:** 2026-08-16  
**Primary task:** PAWS-Wiki paraphrase identification

## Goal

Distill one shared semantic task into three capacity levels, learn where each
model succeeds in prompt-embedding space, route each example to the cheapest
adequate model, and construct task-specific wavelets on an operational
Whitney-style competence stratification.

| Role | Model | Interface |
| --- | --- | --- |
| A | TinyLLM d8 51M BabyLM checkpoint | local PyTorch |
| B | SmolLM2-360M-Instruct | local SafeTensors |
| C | Qwen3 8B Q4_K_M | LiteLLM `qwen3-8b` |

The primary label is human PAWS ground truth. Qwen is the distillation teacher,
not the correctness oracle.

## Fixed task contract

Input:

```text
Sentence A: <sentence1>
Sentence B: <sentence2>
Do these sentences have the same meaning?
Answer only PARAPHRASE or DIFFERENT.
```

`label=1` maps to `PARAPHRASE`; `label=0` maps to `DIFFERENT`. Any other output
is incorrect. Prompt templates must be frozen before the confirmatory test.

Dataset details and integrity rules are in
[`paws-wiki-dataset.md`](paws-wiki-dataset.md).

## Experiment sequence

### 01 — Dataset and verifier integrity

Validate hashes, TSV parsing, labels, official split isolation, pair grouping,
prompt serialization, token lengths for all three tokenizers, and exact output
parsing. Produce immutable example IDs and split manifests.

### 02 — Undistilled model baselines

Measure A, B, and C independently on frozen development subsets. Record exact
accuracy, balanced accuracy, malformed-output rate, latency, and cost. Do not
infer nested competence from parameter count.

### 03 — Qwen teacher annotation

Run C on the training split with a frozen deterministic prompt. Preserve hard
answer, optional rationale in a separate field, latency, failure status, and
request fingerprint. Ground truth remains authoritative when C is wrong.

### 04 — SmolLM distillation

Fine-tune B on training prompts using ground-truth classification plus a
declared teacher-fidelity objective. Select checkpoints on development only.
Compare label-only and teacher-assisted arms.

### 05 — TinyLLM distillation

Fine-tune A from the existing d8 BabyLM checkpoint with the same examples and
update budget policy. Because A and B use different tokenizers, matching means
the same semantic examples and order, not identical token tensors.

### 06 — Held-out competence atlas

Evaluate frozen A/B/C on a disjoint routing-calibration partition. Store the
three-bit success signature and separately estimate each model's conditional
success probability. Report all non-nested signatures.

### 07 — Embedding-proximity router

Establish fixed baselines using lexical features and a frozen sentence-pair
embedding. Compare unweighted and distance-weighted k-nearest neighbors.
Choose the cheapest model above a validation-calibrated lower confidence bound;
otherwise route to C.

### 08 — Operational Whitney strata

Construct a continuous feature carrier containing sentence embeddings,
lexical-overlap statistics, word-order displacement, length, and competence
logits. Partition it by constant label, generation family when recoverable,
and A/B/C competence signature. Audit boundary rank, frontier incidence, and
local tangent stability. This is an empirical stratification, not a proof of a
Whitney stratification of language.

### 09 — Custom competence wavelets

Build the task graph from lexical/structural counterfactual adjacency while
preserving the strata incidence graph. Weight edges using calibrated label and
per-model competence distributions, construct the normalized Laplacian and
diffusion operator, then derive a truncated orthonormal multiscale basis.
Measure reconstruction, boundary recall, and competence-signal compressibility.

### 10 — End-to-end routing campaign

Freeze all models, basis construction, calibration rules, and thresholds.
Evaluate cost, accuracy, routing regret, escalation rate, calibration, OOD
fallback, and latency on untouched PAWS test data. Run lexical-overlap,
word-order, and later WiC transfer checks without using them for selection.

## Evidence boundary

Experiments 01–09 may select the construction using training and development
data. Experiment 10 alone reads the official test labels. Every training,
calibration, and test artifact must retain dataset hashes, model identities,
prompt fingerprints, and producing-code fingerprints.
