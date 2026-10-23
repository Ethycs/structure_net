# Broader decoder training with the released code language frozen

Date: 2026-10-22. EXPLORATORY `ncm-broad-decoder-transfer-v1`.
Known before launch: current adapted codec seen reconstruction99.8512%, novel
92.6081%; released codec novel96.9317%, on20 novel teacher-pilot texts. Both
fail the98% diagnostic. The current full attachment remains0/64 summaries under
Terra. Training-only conditional finer-code repair passed separately; it cannot
repair a text codec that mistranscribes even known correct codes.

Question: can broader unpaired text improve reconstruction while preserving
the released encoder, codebooks and code semantics? Start from the original
released XSum codec. Train only its decoder; freeze every other parameter and
state buffer. This deliberately avoids the codebook/encoder changes in the
failed adaptation. It is an exploratory remedy, not a paper replication or a
claim that codec error explains the entire summary-generation failure.

Use only the already frozen8,192 public training sources in
`20261022_expanded_data_v2/train_sources.json`, plus the existing1,999 training
summary labels. Do not consume the still-running expanded teacher job. Sort
the8,192 sources by SHA256(`ncm-broad-decoder-split-v1:`+ID); reserve the first
1,024 whole source/session/groups for codec validation, train on the other7,168.
Require disjointness from the20 novel teacher-pilot sources and old training
groups. Never read the final task-test examples or labels.

Construct exact source substrings using whitespace word boundaries. Consider
8/16/24/32-word windows starting every16words, require GPT-2 IDs+EOS at most48
without truncation, then pad with PAD. Select at most eight windows per source
by SHA256(`ncm-broad-decoder-window-v1:`+ID+start/end). Deduplicate target text
within each partition and exclude training text equal to a validation target
or novel target. The validation bank is one hash-selected window from each of
the first256 eligible reserved sources; no repeated source units. Require at
least30,000 raw training windows before training. Save all exclusions and text
span/token identities. Training summaries remain a separate anchor bank.

Fixed training:2,048 AdamW updates, effective batch64 (48 raw windows,16 old
summary anchors), LR1e-4,64-step warmup then cosine decay, betas.9/.95,
weight decay.01, decoder dropout inherited from the release, gradient clip1.
Use independent fixed initialization42, dropout52 and sampler2042 seeds.
Quantization and all cached code assignment FP32. Decoder forward BF16 with
FP32 cross entropy on GPU0. Cache each bank's known correct codes under the
unchanged released encoder/codebooks. No nested dropout or corruption; this is
a full-code reconstruction remedy. Code caches do not establish inference speed.

Before the full fixed run, focused CPU tests and a two-update lifecycle pilot
must pass. Pilot uses the real batch64, real FP32 quantization batch128 (repeat
small bank if needed), optimizer update, nonzero decoder gradients, exact
checkpoint/prediction reload and all nondecoder-state invariants. Pilot outputs
are systems-only. Main run uses the same hashes/settings, full banks and a fresh
root. Check GPU0 headroom with resident services; stop on OOM, nonfinite values,
changed encoder/codebooks, malformed data, hash drift or reload failure. No
seed retries, unplanned LR changes or CPU fallback.

Measure at baseline and steps512/1024/1536/2048: raw256 validation and all20
novel target texts; also20 fixed old training targets. Record non-PAD correct
tokens, totals, exact texts and every reconstruction. Final2,048-step checkpoint
is the sole endpoint; intermediate metrics cannot select a different reported
winner. Diagnostic success requires at least99.5% non-PAD reconstruction and
90% exact texts separately on raw256 validation and novel20 summary texts.
Report baseline deltas and failures in full. Novel20 results are already known
to motivate this remedy, so this is development evidence, not a fresh blind
test. No target text or source from those20 enters training.

Even a pass only qualifies a codec for a separately declared source-to-code
training experiment; it is not95% summary accuracy, model compression, final
task admission or a speed result. Preserve the original failed codec and all
previous source/generator weights. Source-to-code generation still needs its
own unseen-quality demonstration.

Fresh roots: `20261022_broad_decoder_{data,pilot,train}_v1`.
Commands: `python -m experiments.structure_net.ncm_broad_decoder_transfer
{prepare|pilot|train}` using the moba runtime, offline HF, empty LD_LIBRARY_PATH,
OMP_NUM_THREADS=2, GPU0 selected before CUDA startup for pilot/train. Freeze all
source, plan, data, tokenizer and checkpoint hashes; save source snapshots,
strict JSON, optimizer/RNG checkpoints, code cache/reload, full final outputs,
completed report and conservative registry read-back.
