# Full released-checkpoint XSum benchmark check

Date: 2026-10-22. EXPLORATORY `ncm-released-xsum-benchmark-v1`.
Question: does the untouched released XSum checkpoint, run through the unchanged
author evaluator, give ROUGE scores compatible with the paper? This is a
checkpoint/evaluation reproduction, not a from-scratch training replication or
an evaluation of the Llama adaptation. The known adaptation result is0/64
development summaries passing Terra; author/local inference parity is32/32.
Neither establishes the published full-test scores.

Use all11,334 public test inputs/references already downloaded and SHA-verified
in `20261022_released_xsum_data_v1`. Use the original pinned XSum checkpoint,
config, GPT-2 tokenizer and unchanged vendor `scripts/eval_shard.py` followed by
`scripts/eval_merge.py`. No Llama features, adapted codec, training, CFG, MBR,
coarse AR or hyperparameter search. Keep FP32, first512 source tokens, default
published generation penalties and batch32. Use one shard on physicalGPU0;
the released config suggests two shards. Record this execution difference and
the released nine-level schedule versus the paper's eight-NFE XSum row. Do not
silently modify either configuration to make the scores agree.

Primary endpoints: mean per-example stemmed ROUGE-1/2/L F1 times100, as in the
author's merge script. Predeclare checkpoint-score compatibility as all three
absolute differences from Table2's35.0/12.8/28.0 at most0.5points. This is a
practical compatibility margin, not a statistical replication guarantee.
Report raw scores and differences even when this gate fails. Capture the
unchanged merge script's GPT-2 generative perplexity and mean word count as
secondary measurements with no success threshold. ROUGE is not a measure of
complete-summary factual accuracy and cannot pass the application's95%gate.

Install only ROUGE0.1.2, NLTK3.9.2 and absl-py2.3.1 in a new isolated dependency
directory; use existing runtime dependencies for their requirements. Hash this
directory, all relevant sources, inputs, config/checkpoints and cached GPT-2
files. Preserve the environment and exact commands. No live producer imports
this directory. Original evaluator/model files remain unchanged.

Before the full benchmark, run a32-case real CUDA pilot at batch32 with both
unchanged scripts. Require complete unique index coverage, finite metrics,
nonempty outputs, exact first32 text agreement with the prior32-case control
(batch4), and independent ROUGE calculations agreeing with printed author
values within0.0051points. Label pilot systems-only; do not compare its scores
to the paper. The full benchmark requires the same producer/input hashes and
passed pilot, plus exact first32 output agreement with that pilot. Check all
full reference strings against downloaded references before scoring. Stop on
OOM, mismatch, nonfinite metrics or dependency/hash drift; preserve failures
without seed retries or configuration changes. Do not interrupt the existing
GPU2 residual experiment or teacher-label worker.

Fresh roots: `20261022_released_xsum_benchmark_{pilot,full}_v1`. Freeze the
producer before execution. Save author pickles/logs, structured full-precision
ROUGE per example, output seals, stage durations and strict result JSON. Stage
durations are execution telemetry on a shared3060, not a speedup claim. The
conversation application's final test stays sealed. Report/register completed
results conservatively, including any checkpoint-score mismatch.

Reproduction: `python -m experiments.structure_net.ncm_released_xsum_benchmark
{pilot|full}` using the existing moba CUDA environment, GPU0 selected before
Python startup, offline HF, empty LD_LIBRARY_PATH and OMP_NUM_THREADS=2.

Primary references: [paper Table2](https://arxiv.org/html/2610.14162v1),
[official evaluator](https://github.com/HazyResearch/ncm).
