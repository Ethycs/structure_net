# Word-final frozen-parent NCM: training in progress

Date:2026-10-23. Hypothesis `ncm-word-final-frozen-parent-v1`.
The goal remains at least95%complete-summary task accuracy on source-disjoint
evaluation. It is unmet. No new model accuracy, compression or speedup has been
established by the systems work below.

The complete nonfinal data set contains252TRAIN summaries,64short development
sources and64longer diagnostic sources. All380complete sources have frozen
Llama3.2-3B features, with the original28-layer body hash unchanged. The maximum
source frame is2,816GPT-2 positions. Full-size initialization reused every
released generator weight, repeated root queries, copied the word projection
and parallel text decoder, and zero initialized the parent-feature residual.

The first CUDA head pilot terminated with exit1before updates because the
released RoPE helper created CPU frequencies for CUDA positions. Its producer,
traceback and terminal receipt remain in the original root. A prospective
[device-only recovery](2026-10-23_ncm-word-final-cuda-recovery.md) left the
completed preparation sources unchanged. Exact CPU and real CUDA geometry
tests passed2/2. The recovered two-update full-size pilot passed finite
gradient, maximum-source, frozen-codec and model/optimizer/RNG/source-only
checkpoint reload checks. Peak allocated memory was4,848,129,024bytes.

The fixed512-update run started from fresh released/new-copy initialization
after the pilot's observed exit0. It is running on physicalGPU0 in root
`data/experiments/ncm_paper_regime/20261023_word_final_train_cuda_v2`, unified
exec session7330. At this snapshot at least128updates and the first recovery
checkpoint had completed. Only the final512checkpoint will be selected.
Original parent weights and MLPs remain frozen; new module copies train.

The evaluation source producer completed with exit0. Its file contains64short
and64longer sources and no target summaries or concept labels; both cohorts
are disjoint from training by case, session, group and source hash. Its SHA256
is07786d65a4d29201ba7be302d8c4fcc9af56a5f1bb48beec55a9cddcd756d3e1.
The evaluation module includes source-only inference/reload qualification,
retention of the first two outputs, all192raw word IDs and predicted concepts,
one global EOS, exact public-source payload proofs, and qualified Terra then
Sonnet grading. Each cohort separately requires61/64passes and64valid judgments
under both judges. No repair, oracle concepts, fallback or output selection.
The evaluation contract tests passed2/2. Both final cohorts remain closed.

All802canonical earlier first judgments and their raw attempts replayed
successfully without any new external requests. Evaluation reuses their first
decisions, including negatives. The new finalizer will replay the512update
sample/noise/LR sequence, words/texts and complete-summary judgments before
writing an aggregate and conservative registry record. Its CLI imports and
parses successfully; finalization has not run and is not quality evidence.

Next: observe training terminal status, require successful final checkpoint
and reload evidence, run the two-case source-only inference pilot, generate
both64-case cohorts, freeze exact public judge payloads, then grade and archive
the actual outcome. Training loss and oracle reconstruction are not task
accuracy. Preserve every earlier failed codec, native hybrid and systems run.

Completion update: training exited0after512updates, with exact reload, frozen
codec and5,142,067,200bytes peak allocation. Source-only inference/reload and
all128outputs completed. Qualified Terra found0/64complete-summary passes on
each cohort, all128judgments valid. Sonnet was not admitted after either failed
gate. The measured failure and TRAIN-only mechanism diagnostic are documented
in the [analysis report](../08%20-%20Analysis/2026-10-23_ncm-word-final-frozen-parent.md).
The systems checks passed; the summarization model failed. The95%goal remains
unmet, both final cohorts closed, and no speedup established.
