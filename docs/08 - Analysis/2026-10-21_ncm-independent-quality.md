# Independent grading of the Llama-distilled NCM

The frozen NCM passed **0/200 human-dialogue summaries** and **0/40 constructed
stress cases** under the independent judge's strict rubric. It fails the 95%
operational target. The previous 100/100 measured a narrow synthetic
project-decision task; that competence does not transfer to these dialogue
summaries. However, the rubric accepted only **11/40 human reference summaries**,
so its percentages are not calibrated estimates of human-rated task accuracy.

Date: 2026-10-21. Hypothesis: `llama32-3b-ncm-independent-quality-v1`.
Final assessment: `ncm_transfer_failed_accuracy_metric_unqualified`.
Original operational classification: `independent_summary_quality_below_95`.
Conformance: **EXPLORATORY**, one frozen adaptation seed, with prospective
quality thresholds and disclosed evidence-format recovery after limited outcome
exposure. Main grading, raw-evidence reconstruction, input audit and supplementary
human-reference diagnostic completed. The reference diagnostic exposes an
unqualified coverage criterion; there is no defensible production-accuracy
percentage from this study.

Aggregate:
`data/experiments/ncm_quality/20261021_judge_v3/20261021_registered_v1/campaign_results.json`

SHA256: `abc58a6cf9786aa88b9f0de7a8541eded8b2090cbd882a4b716810ae1d67f0f0`.

The separate assessment preserves the raw grade and qualifies its interpretation:
`data/experiments/ncm_quality/20261021_judge_v3/20261021_assessment_v1/assessment.json`.
SHA256: `6eae3a3eb1d96e194bb3983489144b1131cbea05e6bcb3cce08eec4823332d7b`.
It records `judge_qualified_for_accuracy:false` and `production_accuracy:null`.

The [original plan](../07%20-%20Status%20Reports/2026-10-21_ncm-independent-quality-preregistration.md),
[judge amendment](../07%20-%20Status%20Reports/2026-10-21_ncm-independent-quality-amendment-v2.md),
and [citation recovery](../07%20-%20Status%20Reports/2026-10-21_ncm-independent-quality-amendment-v3.md)
preserve the original rules and failed attempts.

## Measured grade

These are **LLM-judged rubric pass rates**, not measured production accuracy or
human gold labels. In particular, the judge sometimes makes overly literal
decisions about reasonable Llama paraphrases. Its 18% strict pass rate must not
be presented as an objective estimate of Llama's useful-summary accuracy.

| Human dialogues, n=200 per model | Frozen NCM | Original Llama 3.2 3B |
| --- | ---: | ---: |
| Strict summary passes | 0/200, 0% | 36/200, 18% |
| Wilson 95% interval | 0–1.88% | 13.29–23.91% |
| Fully supported summaries | 0/200 | 117/200 |
| Supported material claims | 6/670, 0.90% | 886/979, 90.50% |
| Coverage failures or critical omissions | 200/200 | 146/200 |
| Coherence failures | 192/200 | 1/200 |
| Empty outputs | 0 | 0 |
| Outputs exceeding 128 stored tokens | 0 | 0 |
| Outputs hitting the native generation limit | 0 | 0 |

Llama minus NCM strict pass rate: **18 percentage points**, paired-bootstrap
95% interval **13–23.5 points**, 10,000 resamples with seed 20261021. There were
36 teacher-only passes and 164 cases both failed. Confidence intervals measure
case sampling under these labels; they do not include judge error, domain shift
or variation across independently trained models.

The point target required at least 190/200 NCM passes: failed. The stronger
Wilson-lower-bound target of 95%: failed. Retaining Llama's point pass rate:
failed. No speed measurement ran in this study. The earlier speed results do
not establish faster useful summarization on this population.

The separate stress cohort scored **NCM 0/40**, **Llama 13/40**. Fully supported
summaries were 0/40 and 26/40 respectively. All 40 stress sources are below the
application's 192-token summarization trigger, so this cohort is a semantic
diagnostic, not a measurement of the actual summarization branch. It is excluded
from the headline 200-case rate.

## What went wrong

The outputs show strong project-template bias, not merely missing details:

| Whole word, case insensitive | Human sources containing it | NCM outputs containing it |
| --- | ---: | ---: |
| project | 4/200 | 189/200 |
| launch | 0/200 | 157/200 |
| budget | 1/200 | 90/200 |
| deadline | 1/200 | 120/200 |

This post-outcome count is descriptive, not the primary scoring rule. Its receipt
is `20261021_judge_v3/20261021_template_diagnostic_v1/result.json` beneath the
study root. All **117** primary dialogues within the old training-source length
range of 256–480 GPT-2 tokens also failed; the problem is not confined to inputs
outside that previous length range.

For `test_346`, the source concerns Nathan's summer newspaper internship in
Chicago. NCM instead describes a project launch and invents a November 11
deadline. For `test_419`, the source concerns anniversary jewellery and a
marriage proposal; NCM emits a project owner, a $2,500 budget and a deadline.
The full source, every native output ID and each judge claim remain available
in the raw artifacts. These source mismatches support the transfer-failure
conclusion independently of marginal disagreements about summary coverage.

This is a failure of the selected task-distilled checkpoint. It does not
establish a limit of the NCM architecture or a result for a broadly trained NCM.

## Data, models, and execution integrity

The primary population is the public [DialogSum](https://github.com/cylnlp/DialogSum)
test split, pinned commit `848cc8c05e0a04def79326539de827f1b28786a0`, with
three human summaries per dialogue. Upstream attribution and CC BY-NC-SA 4.0
license information are retained in the downloaded README. Of 500 dialogues,
217 fit both the full 512-GPT-2-token source capacity and the application's
at-least-192-Llama-token trigger. A prospectively fixed ID hash selected 200;
276 short sources and seven over-capacity sources were excluded. Seventeen
eligible cases were not selected; two of them served as systems pilots.

All 240 evaluated sources were re-encoded and checked against their saved padded
prefixes, without truncation. The primary GPT-2 length range is 205–504.
There is no exact source overlap with the previous 1,188 distillation fixtures.
Pretraining contamination cannot be excluded. These are public research
dialogues, not the user's production traffic; the 2,048-token fragments and
6,300-token batching prompts are outside this evaluation.

The NCM is the unchanged approximately 149M-parameter generator plus codec.
Generator SHA256:
`ff7133b2de50bc1f66ef1e490fec4c723e0fc5a5c1957107209e923b198c3e3c`.
Codec SHA256:
`b87f7288edfc0eb1f5b9122d1139cc2e61aa2331be7d6fe53f0582993086ede2`.
FP32 weights use BF16 autocast and the existing native nine-level CUDA graph;
the output canvas is 48 GPT-2 positions. The failed factorized candidate was
not used. There was no training, checkpoint selection or output postediting.

The control is the complete original Llama-3.2-3B-Instruct at revision
`0cb88a4f764b7a12671c53f0838cd831a0843b95`, physically hash-checked, with FP16,
FlashAttention2 and greedy native generation. Both receive the same complete
source. Llama receives its normal instruction wrapper asking for at most 32
words, with a 96-native-token generation cap. Both are checked against the
128-Llama-token stored-summary cap; the vocabularies and native output caps
are different. Mean output lengths are 24.57 common tokens/15.63 words for
NCM and 33.13 common tokens/25.91 words for Llama on the primary cohort.

Physical GPU0, RTX3060 12GB, ran one local model worker at a time. Other GPUs
were not used. Torch 2.4.1+cu121, Transformers 4.53.3, FlashAttention 2.6.3,
two CPU threads, the existing MoBA environment and pinned local tokenizers
were recorded in the dependency manifest. Both excluded CUDA lifecycle pilots
passed under each producing version: eight pilot outputs total, excluded from
all quality rates. All 480 main model outputs completed without inference
retry. V3 reused exactly these frozen V2 outputs.

## Judge, controls, and limitations

Primary judge: gateway alias `claude_code/claude-sonnet-4-6`, requested and
returned alias checked and every raw reply retained. Each request contains two
anonymous candidates in a deterministic randomized order, the full source and
human references. Model names, checkpoint identities and speed results are
hidden. Each candidate is scored independently; both may pass or fail.

A pass requires nonempty text within 128 common tokens; faithfulness 4/4;
every material claim and overall verdict supported; coverage at least 3/4;
coherence at least 3/4; and no critical omission. Every claim is labeled
supported, contradicted or insufficient evidence. V3 cites losslessly identified
source turns; code saves the exact original text, avoiding reconstructed quotes.
The quality dimensions are motivated by [G-Eval](https://aclanthology.org/2023.emnlp-main.153/);
this is not a reproduction of that metric.

The V3 judge passed **48/48** known positive/corrupted-summary controls and
rejected all **21** factual negatives. The real-source run produced 240 valid
paired judgments: **251 requests**, including **11 inconsistent overall/claim
verdict replies** across **nine pairs** that required allowed identical-request
retries. No valid low quality score was retried, and no final pair was missing
or invalid. The aggregate was reconstructed from native replies, complete
candidate IDs, source/text hashes and the fixed pass function.

The independent GPT route could not complete calibration even with one worker:
three pairs/six summaries were valid and correct; the fourth pair exhausted
HTTP500 retries. Therefore the planned 50-primary/20-stress second-judge audit
did not run. **Inter-judge agreement is unavailable.** The gateway aliases do
not expose immutable weight revisions, and reported usage is not a billing
measurement.

The early direct Anthropic endpoint lacked provider credit. The first GPT
calibration stopped with 15 completed pairs, five valid and ten invalid, after
40 requests. The V2 free-quote primary stage stopped with eight pairs, seven
valid and one invalid, because quotes omitted intervening speaker labels.
Both failures remain separate from the recovered grade. The source-turn change
followed limited outcome exposure, so this is exploratory evidence rather than
a newly blind confirmatory campaign.

The judge can over-penalize reasonable paraphrases. For example, it marked
“Asha initially quotes an estimate” contradicted because Asha initially called
the amount an invoice, although another speaker establishes that it was an
estimate. That is an overly literal reading of an observer's summary. Similarly,
“agrees to load” was penalized where a speaker volunteered to load. These are
reasons to avoid interpreting the strict Llama rate as human accuracy. They do
not explain the NCM's unrelated projects, fabricated deadlines and incoherence.

The [human-reference diagnostic](../07%20-%20Status%20Reports/2026-10-21_ncm-quality-reference-diagnostic.md)
scored two references from each of 20 independently hash-selected primary
sources. All 20 pairs completed without retry and were reconstructed from their
native replies and exact source citations:

| Human reference diagnostic | Result |
| --- | ---: |
| Strict passes | 11/40, 27.5% |
| Fully supported summaries | 32/40, 80% |
| Supported material claims | 139/144, 96.53% |
| Coverage failures or critical omissions | 26/40 |
| Coherence failures | 0/40 |
| Over-128-token outputs | 0/40 |
| Mean common output tokens | 33.525 |

Human references have nearly the same mean common-token length as Llama's
outputs, so a simple mean-length difference does not explain this low strict
acceptance. The diagnostic exposes a coverage/omission criterion that is too
demanding or insufficiently calibrated for human-quality summarization. Passing
the simple 48-example controls did not establish real-dialogue judge validity.
**The strict pass fraction must not be called final task accuracy.** The
operational below-95 classification is retained without upgrading it to a
validated human-quality measurement. The NCM's severe unsupported content and
incoherence remain direct evidence against the claimed transfer of quality.

This diagnostic is post-outcome and does not change the endpoint. References
can contain errors and are visible as grading context, which can bias this
check upward. The 40 references share 20 dialogues and are not independent
replicates. This check cannot establish human-rater agreement or production
accuracy. The next quality evaluator needs realistic acceptable-paraphrase
controls and human-rated production examples before a 95% accuracy claim.

The complete diagnostic is in
`data/experiments/ncm_quality/20261021_judge_v3/20261021_reference_diagnostic_v1/`,
with source-bound scores, raw responses and a separate immutable result.
Its result SHA256 is
`7048dcde1ba32a1c14ae0a510f489647a19905509f84a681302ebd48f291f818`.

## Audit and reproduction

Main artifacts are under `data/experiments/ncm_quality/`:

- `20261021_data_v1/cases.json`: complete frozen sources, references and exclusions.
- `20261021_judge_v2/20261021_{ncm,llama}_generation_v1/`: frozen model outputs,
  physical input readback and source snapshots.
- `20261021_judge_v3/20261021_judge_primary_v1/`: anonymous requests, original
  replies, exact source evidence and all retry attempts.
- `20261021_judge_v3/20261021_registered_v1/case_scores.json`: per-model,
  per-source grade and reproducible code-computed pass.
- `20261021_judge_v3/20261021_input_audit_v1/result.json`: full prefix
  re-encoding, disjointness, termination, length and storage-cap checks.

Ten focused tests passed, along with the real CUDA pilots and full raw-result
audit. Source, plans, codebooks, model checkpoints, tokenizer files and producing
versions were hashed; completed failed roots were preserved. Registry status
remains `confirmed:false`, `underpowered:true`, production and mobile accuracy
untested. The searchable registry uses the qualified final-assessment
classification and links both immutable records; its exact/semantic/full-JSON
readback receipt is `20261021_assessment_v1/registry_readback.json`.

Reconstruct the aggregate from saved replies, without inference or network:

```bash
env CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=src:. \
  data/experiments/moba_paper/environment/bin/python \
  -m experiments.structure_net.ncm_quality_evidence aggregate \
  --output /tmp/ncm-quality-replay-20261021/campaign_results.json
```

The destination must be fresh. The existing raw roots are append-only. The
entry points are `ncm_quality.py` for frozen model generation,
`ncm_quality_evidence.py` for calibrated independent grading and reconstruction,
and `ncm_quality_reference_audit.py` for the separate reference diagnostic.
