# Broader paired training data while the curriculum experiment runs

Date:2026-10-22. EXPLORATORY data-preparation study
`ncm-expanded-public-pairs-v1`. The earlier full-data recipe scored0/64under
both judges for both arms. The revised curriculum is running and has no new
held-out quality result. Its small training control proved learning capacity,
but generalization from1,999pairs remains unproven. This study prepares broader
aligned data for a separately specified future training study; it does not
modify the running experiment or reclassify any failure.

Use only the SHA-verified public LongMemEval source spans in the existing
`20261022_data_v1/native_train.jsonl` partition. There are13,721native training
groups; the current2,048source selection used1,900of them, leaving11,821fresh
groups before length filtering. Select8,212fresh groups deterministically by
SHA256("ncm-expanded-pairs-v1:"+id), one source/group/session/hash each. Exclude
all current training groups/sessions/source hashes and all original native-test
groups/sessions/source hashes. Check the original partition file digests.
Require both native-source and unchanged Llama-chat prompt to fit512tokens;
never truncate. First20sources are systems-pilot-only; remaining8,192are new
training sources. Strip all old summaries/targets before any new request.
Verify every outgoing exact span against the public file and its published SHA.

Teacher: the same qualified `claude_code/claude-sonnet-4-6`, temperature0,
4passages/request, max1,024output tokens,4concurrent workers,180second timeout,
up to3transport/schema attempts. Use the original24-word prompt with an added
instruction to preserve explicit qualifiers, modality and negation, and omit
unestablished roles/quantifiers. The added instruction is motivated by the two
preserved original audit failures; none of those sources is relabeled/rejudged.
Do not claim attribution between data volume and this prompt change.

Before full generation, run one serial4-source batch, then4concurrent batches
on the20separate pilot sources. Require all5valid responses and >=16labels
fitting47GPT2tokens+EOS. System outputs are not training or quality evidence.
Full generation uses8,192different sources; require all2,048batches valid and
>=7,000nonempty eligible labels. Exclude oversized labels without truncating or
regenerating them. Preserve every raw response, exclusion, timeout and retry.
No valid semantic failure is retried. Stop a full data campaign on pilot failure.

Audit128deterministically hash-selected eligible labels with the unchanged
qualified secondary judge, one candidate/request and4workers. A fresh4-request
systems pilot must pass before the audit. Keep first valid negative judgments;
report whole-summary pass numerator/denominator, Wilson interval and every error.
This audit estimates label noise; it is not a model-accuracy result or proof
that a student cannot outperform its teacher. Missing judgments are not passes.
No new conditional training is authorized by an eligibility count alone: a
future training protocol must explicitly use the observed audit and current
curriculum results before its own run.

Roots under `data/experiments/ncm_paper_regime`:
`20261022_expanded_data_v1`, `20261022_expanded_teacher_pilot_v1`,
`20261022_expanded_teacher_v1`, `20261022_expanded_audit_pilot_v1`,
`20261022_expanded_audit_v1`. Freeze all sources/config/inputs and public proofs.
Data preparation is explicitlyCPU-only; no GPU worker or resident service is
changed. Generation can overlap the ongoing GPU training after earlier quality
grading has completed. Development64and final256sources remain excluded from
training; the final test stays sealed. Record the dataset/audit separately.
