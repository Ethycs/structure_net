# Native parent still fails the summary competence gate

Date: 2026-10-23. `ncm-native-parent-competence-v1` is complete, **native_parent_task_gate_failed**.
The fixed native Llama configuration passes **49/64=76.5625%**
under qualified Terra, all64 judgments valid. Wilson95% interval:
64.87–85.25%. The61/64 gate fails;
independent Sonnet grading did not run. This is the original full28-layer
Llama-3.2-3B-Instruct, not a trained NCM model. NCM95% and speed remain unproved.

The stricter source-attribution/qualifier instruction and128-native-token
generation allowance were fixed before inference. The same64 source-disjoint
development passages, unchanged faithful/salient/readable rubric and128-cl100k
storage cap remain. No clipping, case exclusion or semantic retries. Failure
flags are `{'secondary': {'faithful': 15, 'salient': 1, 'readable': 1}}`. Exact request/parser/first-judgment origins were replayed.

The original generic parent passed50/64. This combined prompt/budget change
passes49/64 and does not establish improvement or isolate either
change. It also does not prove the frozen representations cannot support a
separately trained readout. Parent outputs are not qualified as95% teacher
labels. A stronger teacher-label intervention needs an independent audit.

BF16/SDPA, physicalGPU0, batch4, greedy, seed42. Original body hash and tied
native vocabulary stayed unchanged; real-memory pilot, repeated greedy output
and serialization passed. Main peak allocation:
7017425408bytes. The generator's
original execution handle was lost in a truncated tool response; a separate
receipt records observed process disappearance and completed result, without
inventing an exit code. The grading worker's exit0 was directly observed.

Final256 task cases remain sealed. No model training, NCM success or matched
speed claim is admitted. Aggregate `data/experiments/ncm_paper_regime/20261023_native_parent_results_v1/campaign_results.json`;
SHA256 `ff30ebaf10ede5cd1e02378c44d461e5ea3948665e9a1ea93c72dc58d5cb2286`. All raw outputs, first responses and source
snapshots remain append-only. Registry confirmed=false.
