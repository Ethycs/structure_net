# Public epoch2: unchanged TRAIN-control diagnostic

Exploratory hypothesis `ncm-public-epoch2-training-probe-v1`: additional public
training improves unaided code rollout on the 16 application TRAIN controls
fixed by the earlier BROAD experiment. Epoch1 and epoch2 both already scored
0/128 on semantic DEV. This diagnostic cannot revise either score or qualify
the final task. The epoch4 continuation remains unchanged on physical GPU1.

Compare the completed BROAD and public epoch2 checkpoints, using the original
ordered `diagnostic_ids` from BROAD preparation. Include all 16 controls,
including its longest source. Keep every source position and the original
641 oracle codes and 128 native word IDs. No DEV, reserved, or final case is
used. Reuse `ncm_sequential_engineering.diagnostics` for matched and cyclically
shuffled source features. Capture its teacher-forced argmax codes without
altering the returned logits, so code counts can be independently replayed.

Primary descriptive measurement: change in matched-source rollout code
agreement. Also report teacher-forced code agreement, their difference,
per-level agreement, native-word agreement, and shuffled-source results.
These are TRAIN interpolation diagnostics, not semantic summary accuracy.
Classify the paired rollout-count change as improved, unchanged, or decreased;
there is one trained trajectory and no independent-replicate inference.
Noise strength is a possible later experiment, not an intervention here.

Use locked Pixi `ncm`, physical GPU0 RTX3060 for read-only FP32 inference,
TF32 off, deterministic algorithms, CUBLAS `:4096:8`, seed42, and a 256MiB
feature cache. No optimizer or backward pass exists. Enforce peak allocated
memory <=4.9GB. The RTX2060 keeps training; leave GPU2's service intact.
Do not derive speed or GTX/RTX portability claims from this diagnostic.

First run focused CPU contracts for selection, output replay, capture cleanup,
and invalid inputs. The separate two-case pilot uses the first control and
the longest source. For BROAD, require exact code/word/count/text parity with
the archived matched outputs; teacher CE allows rtol1e-5 and atol1e-6 across
devices. For epoch2, repeat both predictions exactly. Verify complete tensor
hashes before and after each condition, no parameter gradients, and the memory
limit. This is the real architecture and intended two-GPU concurrency pilot.
No tolerance widening, fallback, or silent retry can rescue a failure.

After the pilot's actual successful receipt, run both matched and cyclically
shuffled conditions for both checkpoints on all 16 controls. Require BROAD
parity with all archived matched and shuffled outputs. Preserve raw argmax
codes, words, per-level diagnostics, source identities, checkpoint identities,
configuration, source/input hashes, runtime, and actual tool completion.
The epoch2 matched pilot cases must repeat in the full run. A CPU audit then
recomputes counts and aggregates from the saved outputs and unchanged labels.
Require its actual receipt before writing and registering the report.

Roots: `data/experiments/ncm_paper_regime/20261027_public_epoch2_TRAIN_probe_`
followed by `pilot_v1`, `run_v1`, `audit_v1`, or `results_v1`.
Entry point: `experiments.structure_net.ncm_public_training_probe` with stages
`pilot`, `run`, `audit`, and `report`. All roots are fresh and append-only.
Retain failures. Frozen prior producers, the training curriculum, semantic
rubric, judgment history, and model-selection gates are unchanged.
