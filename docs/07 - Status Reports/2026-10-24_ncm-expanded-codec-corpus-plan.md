# Expanded source-disjoint raw-text codec corpus

2026-10-24. Hypothesis `ncm-expanded9936-source-disjoint-codec-corpus-v1`.
Data preparation only. Preserve the existing standalone decoder52/64->60/64
component result, which is below the fixed61/64 admission threshold. A future
decoder-only repair may use broader raw text without more teacher API calls.
It must have a separate frozen optimizer/precision/budget protocol and systems
pilot; this corpus does not itself authorize a particular trained checkpoint.

Use all10,000original reserved public TRAIN sources minus the entire64source
families in the new component controls, yielding9936sources. Check case,
session, group, source-hash and constituent-turn-hash disjointness against those
64. Check IDs/session/group/source hashes against both permanent DEV cohorts.
The sealed final320files are never opened or hashed. Verify each selected
passage's exact spans against the published LongMemEval-S release. No summary
target text or semantic score enters the corpus or its selection.

Tokenize complete raw source text with the pinned native BERT tokenizer,
`add_special_tokens=False,split_special_tokens=True`. Literal special-token
strings in technical source text must become lexical pieces rather than
embedded frame boundaries. Preserve every resulting lexical ID in consecutive
windows of at most126IDs, with CLS101at each end and PAD0to128. Reject an
unrepresentable structural frame as a data error, never silently drop sources
or windows. Record all UNK100counts; BERT normalization is not lossless text
storage. This is training data for the existing native codec, not a new claim
of exact text preservation. Keep complete source/window offsets for replay.

Retain the existing16,384verified author TRAIN frames as a separately bound
replay pool. Do not access author validation data. Prospective training can
choose its mixture and learning rate in a separate plan before gradients.
Fresh CPU data root `20261024_expanded9936_codec_corpus_v1`; independent CPU
replay reconstructs every frame and source boundary after actual tool exit0.
All earlier negative judgments and the running FP16student trial stay fixed.
