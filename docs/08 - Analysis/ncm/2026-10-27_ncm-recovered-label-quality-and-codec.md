# Recovered labels: protocol mixture and imperfect reconstruction

The fixed 128-case saved-label review passes **54/128 (42.19%) jointly** under
GPT-5.6 Terra and GPT-5.6 Sol. This measures the saved targets against normalized
raw source text under the current complete-summary rubric. It is not NCM model
accuracy, a uniformly weighted corpus estimate, or a measurement of the original
retrieval application's success. All 256 decisions are valid and independently
replayed; no unfavorable judgment was retried. The pilot's eight first decisions
were reused. The cache now preserves all 5,519 canonical first judgments.

| Fixed sample | Terra passes | Sol passes | Same-case joint passes |
|---|---:|---:|---:|
| Long sources | 33/64 | 32/64 | 29/64 |
| Short sources | 30/64 | 30/64 | 25/64 |
| All | **63/128 (49.22%)** | **62/128 (48.44%)** | **54/128 (42.19%)** |

Both judges find 113/128 faithful, 65/128 salient and 120/128 readable. Missing
important source content dominates. There are 17 pass/fail disagreements. The
qualified profile accepts paraphrases and up to 128 stored tokens; it does not
require exact summary wording or a 32-word limit.

Inspection finds a task mismatch inside the broad candidate pool. Of 104,503
review candidates, **59,125 targets are exact substrings** of their raw inputs;
45,378 are not. In the fixed sample, 71 are substrings: only 8 pass both judges.
Of the other 57, 46 pass both. These are post-hoc diagnostic strata, not a newly
selected success rate or proof that every non-substring target is good.

Examples of failed targets include an introductory sentence promising Italian
citizenship application steps without giving the steps, and “Here are some ideas
to help you optimize your bathroom cabinet organization:” without the advice.
Their source–target bindings are real; they are inadequate as complete summaries.

The original application source explains why extraction records can occur here:
`runtime/local.py:629` replaces raw-summary prompts with a routing-excerpt contract
requiring one exact contiguous quotation of at most 32 words. The accompanying
`runtime/semantic_summary.py` explicitly distinguishes the frozen benchmark's
extractive validator from the live proxy's paraphrasing semantic route. A field
named `summary` therefore does not establish a complete-summary training objective.
Individual request provenance still needs classification; exact substring matching
alone cannot prove which protocol produced every record. Useful extractive
summaries should not be rejected solely for copying source wording.

The census export also omits role/date/spine conditioning retained in its source
artifacts. Some attribution failures may involve that missing context. Preserve
these first judgments under the exact packets actually reviewed; any restored
context is a separate, explicitly bound diagnostic, not a silent correction to
the 42.19% result. Do not present this result as a universal quality score for the
original caches.

## Codec reconstruction on the same 128 targets

The repaired codec remains better overall than the original release, but neither
reconstructs these targets reliably enough to equate token fit with complete
summary fidelity. Both have exactly the same encoded 641-code hierarchies for all
128 targets, isolating the decoder contrast. No weights were updated.

| Frozen decoder | Exact normalized summaries | Content-token agreement | Digit-sequence retention |
|---|---:|---:|---:|
| Original released | 79/128 (61.72%) | 5,249/5,306 (98.93%) | 27/27 |
| Current repaired | **101/128 (78.91%)** | **5,264/5,306 (99.21%)** | 26/27 |

Relative to the repaired decoder, the original release fixes 5 cases and breaks 27;
74 are exact under both and 22 under neither. Both have zero empty or out-of-range
outputs. Peak own CUDA allocation was 0.80 GB in deterministic FP32 on RTX 2060 SUPER.
Raw output IDs, rendering and metrics were independently replayed for both arms.

Some repaired-decoder changes are materially wrong: `1896` becomes `1912`,
`arrays` becomes `zoning`, and `4 GHz` becomes `4×`. The last example passes a
digits-only check while losing its unit. Exact reconstruction here compares with
the native tokenizer's normalized text; it is a preservation diagnostic, not
semantic task accuracy or a hard ceiling for the complete source-conditioned
decoder. This confirms the earlier reserved 64 codec limitation rather than
establishing a newly discovered problem.

## Consequences for the next training experiment

Keep extraction supervision distinct from complete-summary supervision and
recover the original conditioning and generation protocol before treating this
pool as semantic gold. The fixed negative judgments remain in the evidence.
The 45,378 non-substring pairs are candidates for further provenance/context
review, not an automatically qualified replacement dataset.

For codec representation learning, text reconstruction does not require a good
source summary. A separate disk-backed preparation now provides **99,367 unique
native 128 frames** after excluding all 128 diagnostic source components and exact
diagnostic frames. It retains 102,224 pair provenance links, excludes 1,896 pairs
from diagnostic components and 383 additional matching frames, and deduplicates
2,857 remaining frame repetitions. Every frame was read back and hash checked.
The binary has 25,437,952 bytes; preparation used 0.57 GB peak host RSS. This corpus
has not trained a model and is not admitted as semantic source–summary supervision.

Next, qualify a fixed decoder-only adaptation on this broader application text
while preserving the codec encoder/codebooks and frozen Llama body. A matched
source-tree experiment still needs correctly typed, conditioned task pairs and
its optimizer/restart checks. The two ongoing model-training jobs remain unchanged.
Latest completed application NCM semantic DEV remains 0/128, and the final 320
sources remain sealed. No speedup or 95% model-quality claim follows from this work.

Artifacts are under `data/experiments/ncm_paper_regime/`:

- `20261027_recovered_label_review_{prepare,pilot,grade,audit}_v1`
- `20261027_recovered_label_diagnostics_v1`
- `20261027_recovered_label_codec_v1` and `20261027_recovered_label_codec_audit_v1`
- `20261027_recovered_released_codec_v1` and `20261027_recovered_released_codec_audit_v1`
- `20261027_recovered_codec_corpus_v1`

The first preparation launch stopped before artifacts or model calls because
the deterministic runtime environment variable was absent. Its failure is
preserved separately; unchanged preparation succeeded with the required setting.
