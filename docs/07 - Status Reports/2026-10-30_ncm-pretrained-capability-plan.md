# Untouched NCM checkpoint capability checks

Question: what can the released weights do on their intended tasks, independently
of our Llama summarization adaptation? Exploratory inference only, October 30,
2026. Adaptation training and previously stopped semantic grading remain stopped.

Before observing new outputs, fix these checks:

- Sudoku easy and hard: all 2,000 released puzzles per difficulty. Use the
  corresponding original generator and codec, native 91-token inputs and
  89-token outputs, FP32, TF32 off, greedy published decoding, batch 32.
  Check exact reference agreement and independently validate all rows, columns,
  boxes, original clues and output formatting. Malformed outputs remain failures.
  Difficulties share reference boards; do not treat them as independent samples.
- LM1B small: 16 unconditional samples, seed 42, batch 8, original 128-token
  output frame and all published sampling/word-decoding penalties. These are
  qualitative examples, not instruction following, summarization accuracy or a
  generative-perplexity replication. Show the first examples without selection.
- Reuse existing native XSum ROUGE and semantic controls with their different
  populations clearly identified; do not reopen application final-test cases.

Pin Hugging Face model revision d342ce8aaea58830eef3b8ad473664673e41f692
and data revision 6580ba41a4b0a976dde876fbe84d25327617b0e6. Verify download
sizes and LFS SHA256s (Git blob IDs for small configs), and preserve manifests.
Use the existing pinned author inference script via a root-executed wrapper;
do not rewrite generation or word decoding. Restrict checkpoint loading to
weights-only and route BERT tokenization to our existing local snapshot.
Record source hashes, checkpoint hashes, raw IDs, generated codes, all outputs,
runtime identity and actual tool exits in a new dated artifact root.

Run scorer tests on CPU and a small GPU pilot before the full run (two Sudoku
puzzles or eight LM1B samples). Physical GPU0 RTX3060 is currently free; select
it before Python, use logical cuda:0, locked Pixi ncm environment, no optimizer.
Check pilot/full output agreement and retain any differences. No comparative
speed claim: these checks prioritize native behavior and use FP32.

This is a direct exploratory capability audit, not a registered training study
or a claim that the application's 95% goal has been met.
