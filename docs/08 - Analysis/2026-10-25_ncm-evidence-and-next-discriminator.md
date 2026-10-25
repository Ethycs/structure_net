# What the NCM failures establish

The current attached summarizer is unsuccessful. That result does not establish
fabrication or impossibility of the published method. Our experiments combine
changes to conditioning, domain, training supervision and output range; repeated
failures sharing those changes are not independent replications of the paper.

There is already an original-domain positive control, so do not describe the
release as wholly untested or rerun that control merely because an adaptation
failed. The untouched XSum checkpoint on the fixed 128 cases scored
ROUGE-1/2/L 34.7230/12.2160/27.8859. Its separate reference-code reconstruction
recovered 3,283/3,289 tokens (99.8176%). These are a small sanity check and a codec
diagnostic, not a full benchmark, training replication or 95% summary pass rate.
Raw result: `data/experiments/ncm_recipe_update/20261022_control_v2/result.json`.

The current full-732 parent passed 0/128 source-disjoint whole-summary judgments.
It had learned 94.59% teacher-forced code agreement but 66.13% free-running code
agreement and 36.88% active word agreement on TRAIN. Correct reference codes
previously reconstructed all 732 training targets. The latest decoder-only
adaptation raised TRAIN word agreement to 42.38%; its complete DEV grade is
pending. These controls make code prediction, error propagation and task transfer
plausible failure points. They do not prove that code capacity is sufficient for
unseen texts, that the source interface is correct, or that more epochs will help.

The first two fixed outputs in each DEV cohort remain visibly off-topic after
decoder adaptation. For example, the source about a 2 pm volleyball game produces
comparisons, unrelated quantities and training terminology. The four-case
inspection is descriptive, not a replacement for all 128 fixed semantic judgments.
Training agreement and partial fluency must not be reported as task accuracy.

## Data hypothesis

The current 732 pairs represent only eight source components. This is weak coverage
for a new conditional mapping evaluated across broader topics. A prior audit also
found that a 4,949-pair historical cache pool contains extracted atoms rather than
complete-source summaries, including targets such as `That` and `Here`. Hash
binding and source separation do not qualify these targets for the summary task.
The newly joined cached-label candidate set remains unadmitted; compare its
identities against the existing task-mismatch audit before any future use.

BabyLM is a reasonable controlled language-training candidate. Ten existing local
files contain 9,447,603 whitespace-counted words and match the old cache manifest;
exact release provenance and document-level exclusions have not been qualified.
The existing BPE cache cannot be reused as native NCM word IDs. This inventory is
at `data/experiments/ncm_paper_regime/20261025_babylm_readiness_v1/result.json`.
No BabyLM training has started. The official current alternative is
[BabyLM 2026 Strict-Small](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small).

The current NCM starts from pretrained LM1B components. Additional plain-language
pretraining therefore does not uniquely test whether a new Llama-feature
conditioning interface can learn summarization. A better discriminating study
would compare the native conditioning route and the attached route on identical
source/target examples, with the same codec, initialization, output frame and
declared training budget. Source-conditioned reconstruction or denoising on
BabyLM could test that interface before broader paired summary fine-tuning.
Success at that auxiliary task would remain separate from summary success.

Interpretation should follow the controls: native success with attachment failure
implicates the new interface; training success with unseen failure implicates
generalization; failure to fit a small clean training control calls for an
implementation/objective diagnosis before scaling. New data or extra updates
need a fixed comparison rather than selecting favorable examples afterward.

## Efficiency evidence has a separate scope

The earlier standalone 149M prototype did demonstrate 259.62 versus 77.18 tokens/s
and 0.713 versus 6.425 GB model storage on the narrow 512-input/48-output synthetic task
on RTX 3060. That result is not the speed of the current frozen-Llama-body model.
The verified aggregate hash is
`27a541e247d2da4e4fe52eba72dbf54cc08ade19d4a00638aff3f0811e381ec6` at
`data/experiments/ncm_host_io/20261021_registered_v1/campaign_results.json`.

The authors report 31× XSum speed versus AURORA-LM, 21× LM1B versus DUO and 10× OWT
versus their strongest diffusion comparator. Their timing setup uses H100, batch 16.
These are not 31× Llama claims. Their LM1B tokenizer plus generator consumed
13.36B token exposures including repeated passes. See
[the paper, Sections 4.3 and 4.6](https://arxiv.org/html/2610.14162v1#S4.SS6).
The unchanged 95% complete-summary goal and full-path speed target remain unmet.
