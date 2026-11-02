# What the untouched NCM weights can do

October 30, 2026. New native inference checks completed; no adaptation training
or stopped semantic grading was restarted. The released models have measurable
specialist capabilities, especially Sudoku. The LM1B checkpoint used for our
attachment does not supply a pretrained conversation-summarization interface,
and its native text samples already have serious language-quality limitations.

**Was our attached head pretrained on related subject matter?** Only loosely.
The original LM1B generator is an unconditional language model, with a single
CLS prefix and a 128-position output canvas. It was not pretrained to read
Llama hidden states or to produce faithful summaries from conversation sources.
The released XSum model is closer: it is a conditional news summarizer with a
512-position source and a 48-position output canvas. These are different
checkpoints, not capabilities inherited by the LM1B model.
We moved to LM1B to obtain the longer native output canvas. That improved
output-range coverage while giving up the XSum model's pretrained conditional
summarization mapping; it did not establish a stronger summarization base.

We did subsequently train the attached LM1B-based generator on summary pairs:
the latest stage used 43,894 main pairs plus 2,384 anchors, four epochs and
6,416 updates. Thus the answer is **task-mismatched pretraining, followed by
related task adaptation**. It was not an entirely untrained head. That
adaptation also had to learn the new conditioning interface; it did not retain
Llama's original language-generation behavior simply by reading its features.
See the [training/code review](2026-10-30_ncm-failure-literature-and-code-review.md)
and [released LM1B configuration](https://github.com/HazyResearch/ncm/blob/c70274273f489abc73832b2b98584a8483b6b203/config/eval/lm1b-small.yaml).

| Untouched checkpoint | Evidence | Practical interpretation |
| --- | --- | --- |
| Sudoku easy | **1,988/2,000 = 99.4%** exactly correct, also independently valid with original clues | Strong specialist solver on its released benchmark |
| Sudoku hard | **1,726/2,000 = 86.3%** exactly correct, also independently valid with original clues | Substantial capability on harder boards; still 274 failures |
| LM1B small | New fixed 16-sample native generation; substantial grammatical defects visible in raw text | Language-pattern prior and research baseline; unsuitable as a ready-made summarizer |
| XSum | Existing 128-case native test control: ROUGE-1/2/L **34.72/12.22/27.89** | Learns topical news-summary overlap; this is not a summary pass percentage |
| XSum, separate semantic probe | **3/128 = 2.34%** complete passes under Terra on public TRAIN-derived probes | Insufficient quality under our faithfulness/salience/readability criteria; not an official test-set accuracy estimate |
| Native text codecs | Existing XSum oracle reconstruction: **3,283/3,289 tokens**; framed LM1B: **63,872/65,063 tokens** | Can reconstruct supplied text with errors; reconstruction does not select the right source facts |

The XSum ROUGE and semantic rows use different populations. The semantic
probe's public TRAIN sources were excluded from adapter fitting but may have
appeared in the released model's training. These numbers cannot be combined
into an application accuracy estimate. The codec figures use correct target
codes and different datasets, not source-only generation. Existing reports:
[native XSum control](2026-10-25_ncm-evidence-and-next-discriminator.md),
[XSum semantic probe](2026-10-27_ncm-xsum-author-semantics.md),
[LM1B reconstruction](2026-10-24_ncm-lm1b-framed-codec.md).

**The new Sudoku result is a real positive control.** Each difficulty used its
own original generator and codec on all 2,000 released puzzles. The scorer
checks output length, separators, all 81 digits, all rows, columns, boxes and
every original clue. Invalid outputs stay in the denominator. A separate CPU
readback recomputed correctness from the emitted strings and matched every
case. Easy and hard reuse the same underlying solution boards, so their
denominators are not independent populations.

Each Sudoku model has 25,457,600 generator parameters and 14,896,512 codec
parameters: approximately **40.35 million total**, including reconstruction.
The measurements are close to the paper's 99.5%/86.1% easy/hard results and
match another user's 99.4%/86.3% released-checkpoint reproduction. This supports
the narrow claim that the released architecture can learn this structured
conditional task. It does not establish general reasoning or summarization
transfer. [Paper, Table 3](https://arxiv.org/html/2610.14162v1#S4.SS4),
[independent reproduction](https://github.com/HazyResearch/ncm/issues/9).

**Native LM1B text is already weak.** We used the authors' complete sampling
and word-decoding policy, including the root-code prior, temperature 1,
frequency penalty 11.5, deficit 2 and adjacency penalties. No Llama, adapters,
modified codec or trained checkpoint was involved. This model has 97,065,600
generator parameters and 111,404,672 codec parameters, approximately **208.47
million total**.

Short excerpts from the first four fixed samples, in generation order:

1. `it has respect with me, there ' s just so the different of people can give him out`
2. `these test are, simply a little player.`
3. `no obligation to have the date of these looking statements on some stars`
4. `mccain ' s members are now - to the north and what has a kind of work.`

These are qualitative observations, not a numerical language-accuracy grade.
The sample canvases contain multiple CLS-delimited fragments; the concern is
also broken grammar within fragments, not merely topic changes between them.
All 16 unselected full canvases are preserved in
[the sample file](../../../data/experiments/ncm_pretrained_capability/20261030_native_v1/readback/lm1b_samples.txt).

There is also an author CLI display issue: all 16 generated canvases begin with
CLS101, which the display helper also treats as EOS. Its first-EOS truncation
therefore makes all 16 `hyps` empty. All 16 raw canvases contain text. We
preserved both forms and inspected the full raw canvases. The author's FLM
perplexity evaluator reads the raw token IDs, so this display issue does not
mean its language benchmark scored empty strings. We did not run that
perplexity benchmark here, or assign a summary accuracy to unconditional text.
The application has already had separate CLS-framing corrections; this is not
evidence that all prior adaptation failures were a display bug.

The release also offers WMT14 German-to-English translation, TinyStories and
OpenWebText checkpoints. Their intended tasks are documented in the
[official repository](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203).
They were not evaluated in this check. We have no new claim about their local
performance or the larger LM1B checkpoints.

**Implication for our application:** the weights are useful as task-specific
research baselines and the Sudoku solver works well. We have not established
a pretrained text generator with the semantic quality needed for our memory
summaries. Native competence should be an admission criterion before another
attachment experiment. Choosing the closest conditional task and proving
faithfulness first is a better next discriminator than assuming an
unconditional generator will acquire both the interface and summary quality
through further updates. The pretraining mismatch is a supported contributor,
not proof that it is the sole cause or that a task-matched model will reach 95%.

**Reproducibility:** the [plan](../../07%20-%20Status%20Reports/ncm/2026-10-30_ncm-pretrained-capability-plan.md)
fixed the new tests before outputs. The
[runner](../../../experiments/structure_net/ncm_pretrained_capability.py) executes
the pinned authors' inference script with strict weight loading; it changes
only checkpoint deserialization safety, local BERT-tokenizer lookup and raw
code capture. Model revision `d342ce8aaea58830eef3b8ad473664673e41f692`;
data revision `6580ba41a4b0a976dde876fbe84d25327617b0e6`. Download sizes and
hashes were verified. Source snapshots and hashes accompany every run.

Execution used locked Pixi, physical RTX3060 GPU0, FP32 with TF32 off, Sudoku
batch32 and LM1B batch8. Two-board Sudoku pilots and the eight-sample LM1B pilot
matched their full-run outputs exactly. Eight scorer tests passed. All jobs
exited successfully. The
[CPU readback](../../../experiments/structure_net/ncm_pretrained_capability_readback.py)
verified source bindings, author outputs and Sudoku scores.
No comparative speed claim, application final-test measurement or 95%
application success is made. Adaptation training and the stopped grading stay
stopped. Artifacts and results:
[readback summary](../../../data/experiments/ncm_pretrained_capability/20261030_native_v1/readback/summary.json),
[tool exits](../../../data/experiments/ncm_pretrained_capability/20261030_native_v1/actual_tool_exits.json).
