# Source guidance did not rescue the recovered TRAIN summaries

Exploratory study `ncm-recovered-source-guidance-TRAIN-v1` completed all sixteen
outputs and their exact code repetitions. The four preselected TRAIN summaries
remain repetitive, incomplete or unrelated in every arm on qualitative inspection.
No semantic judge scored this diagnostic, and it does not estimate task accuracy.

| Fixed inference setting | Reference-code matches / 2,564 | Agreement | Generator calls per prediction |
|---|---:|---:|---:|
| Baseline |156|6.08%|12|
| Increasing guidance, maximum 0.5 |141|5.50%|23|
| Constant guidance 0.5 |158|6.16%|24|
| Constant guidance 1.0 |103|4.02%|24|

All four root codes differ from their references in every arm. Constant 0.5
adds only two matching codes across all four complete hierarchies. This tiny
code-count change is not evidence of improved summary quality. The added model
calls are known algorithmic work; no latency or throughput comparison ran.

For example, the data-visualization source remains repeated fragments about
data and analytics. The Houston sports source remains generic football/club
text, with stronger guidance introducing unrelated legal and medical fragments.
The invoice-permissions source remains damaged code fragments or unrelated
health/flooding text. Sunscreen recommendations still become unrelated food,
fabric or herb lists. All original and guided texts are retained without repair
in `20261028_recovered_source_guidance_replay_v1/examples.txt`.

The model, complete source matrices, improved standalone decoder and renderer
were fixed across settings. Baseline codes and decoded words exactly matched
the previous saved control outputs. The increasing schedule preserved the root,
as required. All code outputs repeated exactly; model and codec hashes remained
unchanged; the source-context bypass was restored after every use. Peak GPU
allocation was 1,885,302,272 bytes on the RTX2060, below the 3.5 GB limit.

The unconditioned forward bypassed the added source adapters while retaining
the native prefix and the same generated history. This model was not trained
with a dedicated null-source condition. Consequently this experiment tests our
local guidance adaptation and does not invalidate the authors' published method.
The prospective plan fixes this limitation and all four settings before outputs.

Result root:
`data/experiments/ncm_paper_regime/20261028_recovered_source_guidance_TRAIN_v1`.
Result SHA256: `34dac902ac21bfcff16922c408a5dbc00854eaa5377a4446e48bf253dfc7b707`.
Actual session 27676 ended at chunk 56d2f5, exit 0; receipt SHA256
`ee5262d98eec2ad65245eba0cafc12298cddb1822afb4f4210ff996b647403b9`.
Producing code/tests/plan: commit c137b68. Three CPU contracts passed beforehand.

Independent replay recomputed all code counts, root matches, renderer outputs,
baseline code equality and generator-call counts from the saved artifacts.
It did not rerun the neural models. Replay actual session 46106, chunk f5579e,
exit 0; receipt SHA256
`608d18c730e72317057f40e1037e713bd48df68f7b98c3cea14b953aad8f7327`.

The run was paused by the user with SIGSTOP and resumed in the same process with
SIGCONT. No optimizer existed and no updates, sources or sampling states were
restarted. The historical before/after training evidence remains unchanged.
The code diagnostic and fixed-code reconstruction controls still point to weak
source-to-code learning and accumulated generation errors as unresolved issues.
This decoding-only follow-up did not provide a useful remedy.

Separately, the expanded reviewed-data candidate completed all 128 source-only
DEV outputs before the pause. Its semantic pilot had sent no requests and was
retired after the endpoint repair changed two provider files covered by its
provenance check. A fresh isolated service and unchanged judge qualification
are recorded under the repaired-route amendment. That infrastructure work does
not add an application accuracy measurement. Final320 remains sealed, the 95%
goal remains unmet, and no NCM inference speedup has been established.
