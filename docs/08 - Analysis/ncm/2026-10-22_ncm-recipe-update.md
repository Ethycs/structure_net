# NCM recipe update

**Updated recipe training-summary passes: 29/32.**
These are training cases, not held-out accuracy. No 95% application-quality or
throughput claim follows from this bounded experiment.

Date: 2026-10-22. ID: `ncm-recipe-update-v1`. Conformance: EXPLORATORY, one seed.
Classification: `recipe_update_training_positive_control_failed`.
Aggregate: `data/experiments/ncm_recipe_update/20261022_training_v1/campaign_results.json`.
SHA256: `03b4bb30a584c1df7b35e2ac3210be9cec4d9787d6d1b52690f204e1a3d69903`.
[Plan](../07%20-%20Status%20Reports/2026-10-22_ncm-recipe-update-plan.md).

The update adds actual cumulative-latent cutoffs for decoder adaptation,
conditioning-code corruption with clean generator targets, equal per-level code
loss, independent saved augmentation RNG, and explicit generator-to-codec SHA256
pairing. Encoder/codebooks and Llama body remain frozen; both arms reuse the same
saved body features and initialized hybrid. This adapts pretrained weights; it
does not reproduce full codec/codebook training. No old producer was edited.

The original released NCM passes the fixed 128-case XSum control: ROUGE-L
27.89, reconstruction 3283/3289
= 99.82%. The unmodified released evaluation shard and
decoder settings produced these outputs. This is a small original-domain sanity
check, not the full 11,334-example paper evaluation or semantic task accuracy.
The local codec scorer initially rejected long references; the
[windowing correction](../07%20-%20Status%20Reports/2026-10-22_ncm-recipe-control-windowing-fix.md)
preserves that failure and reuses unchanged original outputs with SHA receipts.
All reference tokens are covered in 48-token windows without truncation.

The original saved-target audit accepted only 5/128 candidates, with one invalid
judgment. That failed audit is preserved and those snippets were not used for
the quality-training run. The
[teacher-target amendment](../07%20-%20Status%20Reports/2026-10-22_ncm-recipe-teacher-target-amendment.md)
then fixed the first 64 public training passages and generated one greedy local
Llama3B summary each, requesting approximately 24 words to fit the codec. No
quality-based generation retry or prompt tuning occurred. Of
64 native-compatible generated targets,
57 passed the unchanged source-grounded
judge, selecting the first 32 accepted in fixed source order. Only public source
passages and newly generated summaries were sent. Cached body features retain
their original approximately-32-word prompt; both arms share that condition.
The judge controls were previously qualified and are agent-authored, not
independent human gold labels. Selection grades are not held-out task accuracy.

Each matched arm trained 128 decoder updates and 512 generator/adapter updates,
batch eight, AdamW lr1e-4, warmup32/cosine floor .1, clipping1. Legacy uses full
latents and clean conditioning with equal position weights. Updated uses .5
nested dropout, uniform retained cutoff 1..K-1 when dropped, corruption decreasing
from .30 to zero by level, and alpha=1 equal-level loss. Those rates and weighting
are experimental choices, not known optimal XSum settings. Loss values between
these differently weighted objectives must not be directly ranked.

| Blinded candidate | Passing complete training summaries |
| --- | ---: |
| legacy | 30/32 |
| updated | 29/32 |
| target | 30/32 |

The primary judge is `claude_code/claude-sonnet-4-6`, using the unchanged source-grounded
faithfulness, salience, readability, nonempty and <=128-token rubric. Candidates
were anonymous and shuffled. Raw requests, replies, evidence citations and every
retry are retained. Valid failures were not retried. The train competence gate
requires at least 31/32 updated outputs and 31/32 reference controls to pass;
classification above applies that gate. There is no held-out endpoint in this
study and no model was selected using a held-out score.

Frozen encoder/codebook identities, unchanged codec during generator training,
actual removal of finer latent levels, nonzero adapter/coarse-query gradients,
matching batches and exact checkpoint reload were checked in a separate CUDA
pilot and the full run. The vendor cumulative-history helper aliases entries;
the new helper explicitly accumulates only retained levels and is tested against
the full vendor latent. Physical GPU0 only; FP32 parameters, BF16 compute, one
worker. Body feature caching is a training convenience, not an application reuse
assumption or a deployment speed measurement. The native output canvas remains
48 GPT2 tokens and source prefix512, below the application's full range.

Reproduce from the repository root with the moba environment Python,
`LD_LIBRARY_PATH='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=src:.`, selecting
physical GPU0 before CUDA commands. Add the memory dependencies and tokenizer
cache for the CPU/network grading stages:

```text
python -m experiments.structure_net.ncm_recipe_control pilot --output NEW_CONTROL_PILOT
python -m experiments.structure_net.ncm_recipe_control control --pilot NEW_CONTROL_PILOT --output NEW_CONTROL
python -m experiments.structure_net.run_ncm_recipe_update pilot --output NEW_TRAIN_PILOT
python -m experiments.structure_net.run_ncm_recipe_update train --pilot NEW_TRAIN_PILOT --output NEW_TRAIN
python -m experiments.neural_architecture_lab.finalize_ncm_recipe_update grade --root NEW_TRAIN --report NEW_REPORT
```

The direct scripts' pinned input roots are declared in source and fingerprinted.
Changing those roots or parameters defines a new producer. The registry must
retain `confirmed: false`, explicit training-only evidence and no speed claim.

## Separate fresh-body integration and lineage readback

A fresh frozen Llama forward, paired codec and saved updated generator completed
without cached body features on one public training example. This is a systems
check, not a new quality or speed measurement. Artifacts are under
`data/experiments/ncm_recipe_update/20261022_inference_v1/`. Fresh/cached text
match: `True`; all 48 output IDs match: `True`.
The output preserves the explicit codec SHA binding. The reusable entry point is
`experiments/structure_net/infer_ncm_recipe_update.py`.

`data_lineage.json` in the training root records that retained store/section and
summarizer-identity fields describe the original shared pair. Replacement target
text and its SHA come from the local frozen-Llama teacher producer, not that
historical summarizer field. Trace readback confirms 128 decoder and 512
generator updates per arm. Thirty-one focused CPU tests passed, along with the
separate CUDA training lifecycle pilot and fixed comparison.
