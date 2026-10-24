# TinyStories byte tokenizer: component capacity control

Date: 2026-10-24. Exploratory component control
`ncm-tinystories-pretoken-capacity-control-v1`. This does not test or pass the
95% source-only summarization goal. The current LM1B training and queued DEV
evaluation remain unchanged. No neural model or recovery checkpoint is loaded.

The existing LM1B BERT normalization control altered some case-sensitive facts;
its previous semantic results and all failures remain recorded. The pinned author
TinyStories configuration instead uses an8192-entry byte BPE vocabulary and a
128-position neural codec. Test whether this alternative preserves text bytes
and how many current training targets fit its lexical budget. This is not a
neural VQ reconstruction test, language evaluation or source-conditioned model.

Use all690 exact current full-training target texts, including the35 existing
capacity revisions. No exclusions, rewriting, clipping or new teacher/judge calls.
Fetch only the two small author BPE assets at the already pinned HF revision
d342ce8aaea58830eef3b8ad473664673e41f692. Require both byte lengths and SHA256s
to match the prior release inventory. Load plain vocabulary/merge data with a
restricted unpickler and use the unchanged author BPETokenizer implementation.

Record every target hash, all native BPE IDs, exact text round-trip, lexical length,
and counts at<=128,<=127and<=126 lexical positions. Those are capacity bounds
with zero, one or two boundary positions; author framing has NOT been qualified.
Record the corresponding BERT lexical lengths and literal text round-trip, with
an explicit warning that exact byte agreement is not semantic accuracy. Check
case/Unicode fixtures. Keep every failing case in the record.

No TinyStories neural checkpoint is downloaded. Its native generator root policy
differs from the active LM1B model, so this is not a drop-in replacement claim.
The pinned OpenWebText config uses1024positions and its pinned release inventory
contains a TODO; do not infer current availability or a working shorter model.
Keep all DEV/final sources, labels and predictions outside this control. Do not
change the canonical semantic cache. Fresh append-only root
`20261024_tinystories_pretoken_capacity_v1`; source/tests/plan frozen before run.
