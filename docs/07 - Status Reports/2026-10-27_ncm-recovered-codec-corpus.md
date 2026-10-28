# Recovered-domain text corpus for codec adaptation

The new128-case check confirms the earlier reserved64 result: high average token
agreement still permits material reconstruction errors. Reverting to the released
codec is worse overall on this sample. The next useful codec intervention is
broader application-domain reconstruction training, not another tiny fit test.

Prepare all usable unique native frames from the audited104,503-pair recovered
review pool for this separate representation objective. Saved-label semantic
quality is not required to reproduce a text, but these texts must not thereby be
admitted as source-to-summary supervision. Keep the original128-case sample as a
codec development set: exclude every one of its128 source components, and exclude
any exact native frame present in that sample even when found under another source.
Retain existing DEV/component/final source protections from the parent pool.

Re-tokenize complete targets with the qualified native tokenizer and compare
lengths to the audited export. No special/unknown IDs or truncation. Use the
established128 frame with CLS/BOS, lexical tokens, CLS/EOS and PAD. Deduplicate by
the SHA-256 of exactly256 little-endian uint16 frame bytes, retaining all contributing
pair IDs and source components. Source content is not an input to codec training.

Use CPU-only locked Pixi, a disk-backed SQLite staging table and streaming output,
with the fresh root `20261027_recovered_codec_corpus_v1`. Export a memory-mappable
uint16 frame file and an ordered JSONL index containing exact frame hashes,
offsets, representative target text and provenance. Independently read every frame
back, check the hash, source exclusions and held-out frame exclusions. Record
actual full counts, exclusions and duplicate frames; do not assume the entire
104,503-pair pool is independent or unique training text.

This preparation does not train a model. Before a later fixed-budget decoder-only
run, freeze its optimizer, sample exposure and checkpoint selection and qualify
the actual gradient, memory and restart path. Keep codec encoder, codebooks,
Llama body and source generator unchanged. Reuse existing author TRAIN text as
a separately declared retention control if selected in that future protocol.
No post-hoc sample filtering, model-accuracy claim, teacher request or final access.
