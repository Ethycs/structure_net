# PAWS-Wiki Dataset Contract

**Status:** ACQUIRED AND AUDITED  
**Date:** 2026-08-16  
**Source revision:** `sharad/paws-wiki@dfa6cca8c6cf178ba296bf8bb95ce19e6c18040a`  
**Upstream:** Google Research, *PAWS: Paraphrase Adversaries from Word Scrambling*

The original Google Storage archive returned HTTP 403 during acquisition. The
project therefore uses a commit-pinned mirror of the original labeled files.
The files are TSV-formatted despite their `.csv` suffix.

## Files

| Split | Rows | Label 0 | Label 1 | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| train | 49,401 | 27,572 | 21,829 | `f8ac90c04483a5b4b2c3583aad5e355122702f6066dcdf18787051f2ac2e7c98` |
| dev | 8,000 | 4,461 | 3,539 | `069c5c604db421f5d3513e932e9b1025456f1ffa7321ddb8336189bbc627bd67` |
| test | 8,000 | 4,464 | 3,536 | `51fc5f45bf33e7991a0c09b38bb4ff1ee482afd170128ae5ae5d8c4ba1d250cf` |

Local root: `data/datasets/paws-wiki/labeled/`.

## Integrity findings

- Columns are `id`, `sentence1`, `sentence2`, and `label`.
- No field is missing.
- There are no identical full rows.
- Exact sentence reuse across train/dev, train/test, or dev/test is zero.
- Exact ordered pair reuse across official splits is zero.
- Within-split duplicate ordered pairs exist: train 52, dev 6, test 18.

All rows sharing an ordered pair, reversed pair, or normalized unordered pair
must remain in one sampling unit. Experiment 01 found normalized/reversed groups
with conflicting labels. They are retained in the raw data and immutable
manifests with `eligible=false`, but excluded from training, routing
calibration, and headline evaluation rather than silently resolved.

## Usage rules

1. Preserve the official train/dev/test boundary.
2. Derive routing calibration from train or a preregistered subdivision of
   development; never calibrate on test.
3. Do not use Qwen output as the correctness label.
4. Keep rationales separate from classifier targets.
5. Report lexical overlap and word-order statistics by label so a router cannot
   be mistaken for a semantic model when it has learned a superficial cue.
6. Preserve raw text and create normalized derivatives as separate,
   fingerprinted artifacts.
7. Read the test labels only in the frozen end-to-end campaign.

## Geometry fields to derive

- unigram and bigram overlap;
- token-count difference;
- aligned-token displacement and inversion count;
- edit distance and longest common subsequence;
- frozen sentence-pair embedding;
- A/B/C success signature and calibrated competence logits;
- local support radius and graph component identity.

These fields define an empirical lexical/structural carrier for routing and
wavelet construction. They do not establish a canonical manifold of meaning.
