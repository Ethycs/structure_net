# Test code information with an explicit inverse of the token embedding encoder

Date: 2026-10-22. EXPLORATORY `ncm-codec-inverse-control-v1`.
Known: the adapted codec reconstructs92.6081% novel tokens; original96.9317%
in FP32. Broad decoder-only training failed: final94.8396% novel,96.4953% raw
validation,99.8512% seen tokens. The whole summarizer remains0/64 under Terra.
Do not repeat scaled student training or claim that more data solved the codec.

The released codec encoder has no transformer/conv blocks or downsampling:
token embedding plus positional embedding, then8-group normalization without
affine weights. Quantization reconstructs an approximation to that normalized
sequence. Can explicit inversion distinguish information lost by quantization
from errors made by the learned neural text decoder?

Evaluate the released and current adapted codecs, unchanged, on the fixed20
seen and20 novel targets of the unseen-codec control, plus the256 disjoint raw
validation targets from broad-decoder data preparation. All are development
diagnostics; no final summary-test data are opened. All full target IDs include
EOS and PAD to48 with no truncation.

For each codec/cohort, use FP32 and batch4 on GPU1. Compare four fixed readouts:
(1) original neural raw argmax; (2) target-blind dictionary inverse, initialized
from that raw argmax, then exactly four alternating steps; (3) quantized latent
with oracle normalization statistics from the correct target; and (4) original
unquantized normalized latent with oracle target statistics, an implementation
control that must reconstruct every token ID exactly, including padding.

Each dictionary inverse denormalizes the latent using per-sequence/group means
and standard deviations, subtracts the learned positional embedding, then
finds the nearest original token embedding by FP32 squared Euclidean distance
over exactly50,258 valid GPT-2/EOS/PAD IDs. The target-blind variant estimates
statistics solely from its current token predictions; it never reads target IDs
or target-derived statistics. Oracle variants are labelled target-aware and
cannot be used as inference candidates. No seed, iteration count, dictionary,
decoder or distance-metric search after inspecting outcomes.

Primary diagnostic: non-PAD target-token reconstruction and exact decoded texts
for each declared readout/cohort, with full IDs and intermediate iteration
predictions retained. A target-blind replacement is only codec-qualified if it
reaches99.5% non-PAD tokens and90% exact texts separately on novel20 and raw256.
Oracle success with target-blind failure is evidence that recovering the
normalization/decision rule remains unresolved; oracle failure suggests
quantization/encoder representational limitations, without proving an absolute
information-theoretic limit. Neither measurement establishes summary quality.

Require focused CPU normalization/nearest-dictionary tests and a real two-case
per-cohort GPU lifecycle pilot, exact serialize/reload, immutable models,
unquantized-oracle100% ID parity, finite values and source/input hashes. Full run
requires the same sources/inputs and a passed pilot. Stop on control failure,
OOM, hash drift or nonfinite output; preserve failures without configuration or
seed retry. Run readouts in a shared batch with common codes; verify reversing
the readout order gives identical target-blind predictions in the pilot.

Roots: `20261022_codec_inverse_{pilot,measure}_v1`. Reproduce with
`python -m experiments.structure_net.ncm_codec_inverse_control {pilot|measure}`,
existing moba runtime, GPU1 selected before startup, offline HF, empty
LD_LIBRARY_PATH, OMP_NUM_THREADS=2. Save strict JSON, source/input seals, models'
frozen hashes, all outputs, raw code and prediction reload. Finalize/register
conservatively. Original body, codec and generator weights stay unchanged.
