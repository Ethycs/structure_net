# Vocabulary-balanced reconstruction with a frozen code language

Date: 2026-10-22. EXPLORATORY `ncm-vocabulary-balanced-codec-v1`.
Known: natural-text broad decoder training failed, with94.8396% novel and
96.4953% raw validation reconstruction. The old adapted codec's44/53 novel
content errors involve IDs absent from old summaries;8 involve IDs absent even
from the broader natural bank. Its coverage is33,635 distinct content IDs out
of50,256. The dictionary inverse pilot and full run both failed codec gates;
the unquantized inverse control was exact, but nearest-dictionary inversion of
quantized latents was worse than the neural decoder even with true statistics.
That does not establish irreversible information loss.

Test one final decoder-only data intervention before changing the codec/head
architecture: give every valid GPT-2 content-token ID positive reconstruction
supervision. Start from the untouched released codec, freeze all nondecoder
state, use exactly the existing2,048-update broad-decoder training engine,
optimizer, seeds, batch64, FP32 quantization, BF16 decoder and final-checkpoint
rule. Preserve the failed natural-only experiment; do not relabel it success.

Reuse all57,258 exact public natural training windows and1,999 old summary
anchors from the frozen broad-codec dataset. Append twice57,258 synthetic native
token sequences to the raw bank, interleaving one natural entry and two
synthetic entries so the pilot exercises both kinds. Synthetic lengths are independently sampled
uniformly8..47 with CPU RNG seed5042. Fill content slots by successive shuffled
permutations of IDs0..50,255 using separate seed6042, append EOS50,256 and PAD
50,257 to48. All content IDs receive positive bank exposure. Synthetic sequences
are token-space exercises, may not form valid natural-language text, are not
summaries and are never semantic-evaluation examples. Do not decode/re-tokenize
them as natural labels. No private data, reference summaries, new API labels or
final task-test inputs are involved.

The unchanged engine samples48 entries from the augmented raw bank and16 old
summary anchors per update, giving an expected32 synthetic/16 natural/16 anchor
batch. Preserve every actual synthetic/native ID. Before main training replay
the exact sampler seed2042 and draw order for all2,048 updates, requiring positive
target exposure for every content-token ID. This is a preflight validity check,
not an outcome or a seed-search opportunity; stop if it fails.

Reuse raw256 validation, novel20 and seen20 targets byte-for-byte. They and their
source groups remain excluded from natural training; generic synthetic symbol
exposure is the intended intervention and is not held-out source memorization.
The20 novel outcomes are known development evidence; no fresh blind task claim.
Primary joint codec gate remains99.5% non-PAD reconstruction and90% exact texts,
separately on novel20 and raw256. Report baseline and every intermediate/final
result without choosing a passing checkpoint. Summary task accuracy remains
unmeasured by this study, and the current attachment remains0/64 under Terra.

Use a versioned wrapper to redirect the existing immutable trainer's data,
pilot, main roots and plan; preserve/hash both wrapper and engine. The training
loop and original engine file are not edited. The wrapper records synthetic
scope and complete data policy in the producer. A new two-update real-batch
CUDA pilot must pass all old gradient, frozen-state, code-cache and prediction
reload checks before main training. Focused CPU tests cover permutation coverage,
EOS/PAD/length validity, determinism and restoration of wrapper globals.

Roots: `20261022_balanced_codec_{data,pilot,train}_v1`. Commands:
`python -m experiments.structure_net.ncm_vocabulary_balanced_codec
{prepare|pilot|train}` in the moba runtime, offline HF, empty LD_LIBRARY_PATH,
OMP_NUM_THREADS=2, GPU0 selected before pilot/train startup. Record strict
manifests, synthetic seeds/ID hashes, actual-schedule exposure counts, all old
controls and fixed endpoint. Stop on OOM, nonfinite loss/logits, data/hash drift,
changed nondecoder state or reload failure. No altered seed or LR retries.

If this codec gate also fails, stop frozen-XSum-decoder data sweeps. The next
study must change the representation or final token readout and retain the
same actual summary objective. Passing this codec gate alone still cannot
admit or establish95% full-summary accuracy or any speedup.
