# Why the current frozen-body NCM summarizer is failing

Date: 2026-10-22. The final5,120-update NCM continuation still fails actual
complete-summary grading. Training reconstruction and code accuracy are not
summary accuracy. **The95% task goal and any speedup claim remain unestablished.**

| Model, same64 source-disjoint development passages | Complete summaries passing Terra | Wilson95% interval |
| --- | --- | --- |
| Final NCM continuation |0/64=0%|0–5.6624%|
| Original full Llama-3.2-3B-Instruct |50/64=78.125%|66.5672–86.4977%|

All128 judgments were valid, with zero retries. Both use the unchanged rubric:
every material claim supported, salient main information, readable, nonempty,
and at most128cl100ktokens. Neither uses exact-reference matching. NCM failed
all three semantic flags on all64 cases. Parent failures include13faithfulness,
one salience and one readability flag; these overlap. Examples include changing
an optional advertising suggestion into a definite plan, generalizing one
item's properties to every item in a list, and reversing story chronology.
This is model-based evaluation, not independently collected human gold.

Parent generation used the same source prompts and all28frozen layers, with
its original output head and96-Llama-token cap. NCM has a48-GPT2-token canvas.
Output budgets therefore differ; this is a task-quality reference, not a
matched speed benchmark. Parent performance is not a logical upper bound on a
student taught by a stronger teacher, but preserving this reference alone would
not satisfy95%. The original parent body and its fully connected layers have
not been retrained.

The intended primary Sonnet quality runs are **still pending**. A fresh combined
four-judge/four-teacher systems pilot returned three valid judgments and one
invalid after its schema/consistency retry cap; it did not pass. Primary quality
therefore waits for the existing broader-label job to finish. No joint-judge
accuracy or final-test result is reported here. The final256-case test is sealed.

## The model uses both source channels, but does not generalize

The completed follow-up isolates each input on128hash-selected training cases.
All conditions use the exact final checkpoint, frozen polished codec, BF16,
batch4, coarse AR and the same discrete generation procedure. Changed native
inputs carry their own padding masks; cyclic shuffles preserve channel marginals.
Zero and disabled-body conditions are out-of-distribution stress tests.

| Input condition | Correct coarse codes | Correct target text tokens | Exact training texts |
| --- | --- | --- | --- |
|matched|506/512=98.8281%|4393/4,477=98.1237%|94/128|
|shifted_body|154/512=30.0781%|929/4,477=20.7505%|0/128|
|zero_body|158/512=30.8594%|877/4,477=19.5890%|0/128|
|native_only|233/512=45.5078%|1536/4,477=34.3087%|0/128|
|shifted_native|48/512=9.3750%|270/4,477=6.0308%|0/128|
|both_shifted|3/512=0.5859%|129/4,477=2.8814%|0/128|

The matched128-case sample reconstructs98.1237% of training tokens while every
held-out complete summary fails. Shifting Llama features reduces reconstruction
to20.7505%, and shifting native text reduces it to6.0308%. This establishes
dependence on both channels for training predictions. It does not establish
semantic transfer from Llama or prove the one-vector interface is sufficient.
Strong memorization/task-transfer failure is supported; ignored Llama input
does not explain these results. More data alone has not been proved a cure.

The four-case lifecycle pilot and full run reproduced original training codes
and text tokens exactly and reloaded exactly. All codec, generator and adapter
weights remained fixed. All1,999training sources, source hashes and parent
prompt IDs match their original cached-feature records. The development feature
path uses the same parent input construction before right-packing its valid
hidden states; final-valid-state extraction handles both padding directions.

## The released evaluator is consistent with the local inference path

The untouched original XSum model and the untouched author's evaluator produce
the same32/32 output texts as our local FP32 path on the author's first32 test
inputs. All outputs are nonempty; exact reference matches are
0/32. Data are pinned to official revision
`6580ba41a4b0a976dde876fbe84d25327617b0e6` with published LFS SHA256 verification.
This control explicitly uses the author's first512tokens, with no task-transfer
finetuning or Llama bridge. Original models remain unchanged.

The first four inspected outputs still show factual and wording defects, such
as confusing which football club appointed a director and producing a malformed
description of a Chicago kidnapping case. These examples are qualitative,
not a measured XSum task-accuracy rate. **Exact evaluator parity is an
implementation control, not evidence of95% faithful summarization.** The
author's released evaluation reports ROUGE/perplexity rather than our full
faithfulness/readability/salience pass criterion.
[Official release and evaluation instructions](https://github.com/HazyResearch/ncm).

An initial author-subprocess pilot stopped before model loading because the
existing NCM dependency path was not forwarded. Its log/producer remain intact.
Fresh v2 pilot/control forward that existing path without modifying vendor
code, packages, checkpoints or data. Both passed their declared parity checks.

The recent codec polish reaches99.7603% training-token reconstruction with
known correct codes, and the residual-capacity oracle can repair over92% of
token losses after nearby coarse-code changes when it knows the target latent.
Neither gives the generator the right facts on a new passage. The next training
experiments need broader audited teacher supervision and explicit tests of
error-conditioned finer-code correction, with real development accuracy as
the deciding endpoint. These are proposed remedies, not established outcomes.

Aggregate: `data/experiments/ncm_paper_regime/20261022_generalization_diagnosis_v1/campaign_results.json`; SHA256
`6a97793c5d1bd034f34ef6929d5922f9fa5c5378ecf7f1a3d1311600578306fb`. Raw predictions, first-valid
judgments, producer/source copies, checkpoint seals, public-data receipts,
failed pilot and exact reload records remain in their immutable run roots.
Registry records for the source-channel diagnostic and released-path control
remain confirmed=false and underpowered=true; the two-judge task campaigns
remain unfinished. The prototype's512source/48target-token range still does
not cover the full application limits.
