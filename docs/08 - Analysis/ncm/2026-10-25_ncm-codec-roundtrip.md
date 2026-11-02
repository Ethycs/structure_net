# The codec preserves most summary tokens, but decoder fidelity remains imperfect

Completed exploratory diagnostic `ncm-summary-codec-roundtrip-v1`: **99.29%
content-token agreement and 45/64 exact reconstructions** for the repaired
parent codec alone on reserved summaries. This is not 99.29% complete-summary
task accuracy. The latest decoder with source attention reconstructed only
3/64 summaries exactly despite being supplied correct target codes.

The [fixed plan](../07%20-%20Status%20Reports/2026-10-25_ncm-codec-roundtrip-plan.md)
compared five configurations on all 64 reserved Sol teacher/component summaries
plus the existing 16 engineering TRAIN controls. No examples were selected by
individual judgment outcome. Identity checks found no overlap with either the
732 engineering training examples or the earlier native training population.
All targets fit without truncation or unknown tokens. The reserved summaries
span 50–86 native lexical tokens; this does not establish behavior across the
entire application range or maximum output length.

Each summary was encoded directly, bypassing the source-to-code generator.
The resulting 641 discrete codes were identical across the three checkpoints.
The codec-only arms received no source features. The source-attention arms used
the existing correctly bound full FP32 Llama features and unchanged attention
adapters. The two comparison factors were decoder checkpoint and the presence
of source attention. Every model remained frozen in evaluation mode.

| Configuration | Matching content tokens | Content agreement | Exact native-rendered summaries | Digit sequence preserved |
| --- | ---: | ---: | ---: | ---: |
| Released codec alone | 4,316/4,378 | 98.58% | 31/64 (48.44%) | 17/18 |
| Repaired parent codec alone | 4,347/4,378 | **99.29%** | **45/64 (70.31%)** | 16/18 |
| Repaired parent with source attention | 4,341/4,378 | 99.15% | 44/64 (68.75%) | 16/18 |
| Latest adapted decoder alone | 4,315/4,378 | 98.56% | 34/64 (53.13%) | 17/18 |
| Latest adapted decoder with source attention | 4,056/4,378 | **92.65%** | **3/64 (4.69%)** | 14/18 |

Only 18 reserved references contain digits. Digit-sequence equality is a
mechanical check, not a test of numerical reasoning or units. A high average
token score can coexist with important factual errors: one reference's 2021
became 2019, and another reconstruction introduced a duplicate 70. With the
latest decoder and source attention, one percent sign became “cinnamon”.

Exact comparison uses the existing uncased tokenizer's reference rendering;
capitalization and formatting changes caused by tokenization are not counted
as codec errors. Exact reconstruction is a strict preservation diagnostic,
not the semantic summarization judge used for the main task. None of the arms
passed the descriptive reconstruction gate of at least 61/64 exact summaries
and 99% content-token agreement. No semantic accuracy estimate is inferred
from these exact-match fractions.

On the 16 TRAIN controls, both application checkpoints with source attention
reconstructed all 16 exactly, with 640/640 content tokens. Thus the latest
checkpoint retains the training control while generalizing poorly in this
reserved control. This does not extend the earlier 732/732 oracle result to
all training examples for the latest checkpoint; only these 16 were re-tested.

The decoder-only adaptation left the source adapters unchanged. Disabling
those adapters on the latest checkpoint raises reserved content agreement
from 92.65% to 98.56%, and exact reconstruction from 3/64 to 34/64. That is
direct evidence of harmful interaction on this oracle-code population. It
does not establish that removing source attention improves source-only
summarization, which was not measured in this diagnostic.

A separate, explicitly post-hoc numerical probe examined the observed
2021-to-2019 error. It kept the entire reference fixed except for that year,
and compared the 2019 and 2021 variants. **57/641 code positions differed**;
the reconstructed latent vectors also differed (L2 distance 46.96285). All
three codec-only decoders nevertheless emitted 2019 for both inputs. There
was no full-code or latent collision in this pair. The representation
distinguishes this input change; the existing decoders fail to recover the
distinction. One chosen error is not a general estimate of numerical capacity,
nor proof that every summary distinction survives quantization.

This supports keeping the encoder/codebook fixed for the next controlled
repair, retaining clean-code reconstruction as an anchor while training for
generated-code errors, and testing source-attention calibration separately.
It does not justify adding code capacity before checking those failures.
These are proposed next interventions, not completed improvements.

The comparison used the locked Pixi `ncm` environment, FP32 inference, TF32 off,
and physical GPU1 (RTX 2060). Peak GPU allocation was 1,292,789,248 bytes. Source
features streamed losslessly through a 256 MiB cache. Four focused tests passed;
the initial test invocation failed because the repository-wide conftest requires
SciPy, absent from this locked environment. The focused invocation used
`--confcutdir=tests/structure_net`; no environment was changed. The two-case CUDA
pilot repeated all ten outputs exactly, and its outputs matched the full run.
An independent auditor replayed all 400 case metrics, code identities, target
bindings, tokenizer renderings, aggregate denominators and artifact hashes.
No training, new judge requests, sealed final-test access, or speed test ran.

Readable [side-by-side reconstructions](../../data/experiments/ncm_paper_regime/20261025_codec_roundtrip_results_v1/review.html)
include every reference, original source, and all five outputs. The prior
[architecture radar](../../data/experiments/ncm_paper_regime/20261025_component_radar_v3/component_radar.png)
separates component diagnostics from complete-summary quality; this new codec
comparison is additional evidence, not a replacement of its historical scores.

Artifacts under `data/experiments/ncm_paper_regime/`:

- `20261025_codec_roundtrip_prepare_v1`: fixed cases, native frames, features and split proof.
- `20261025_codec_roundtrip_pilot_v1`: repeated two-case systems check.
- `20261025_codec_roundtrip_eval_v1`: all 400 outputs and actual successful process receipt.
- `20261025_codec_roundtrip_results_v1`: independent aggregate and readable viewer.
- `20261025_codec_year_discrimination_v1`: two-year probe, six decoded outputs and source snapshot.

Full-run outputs SHA-256:
`458540dec97dd0863595b33b9fa7f7a7ffcbb09e9234f480d38982e33c16e5f7`.
