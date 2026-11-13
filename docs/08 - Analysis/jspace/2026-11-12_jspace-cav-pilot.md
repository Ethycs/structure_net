# J-derived CAV pilot: targeted changes, no demonstrated compression

**Verdict:** the experiment did not clear its joint gates. The parent answered
20/48 held-out questions correctly, below the required 24. J-derived swaps
produced four intended counterfactuals among those twenty parent-correct cases;
both controls produced zero. The retained-only reconstruction failed. No model
compression, attention replacement, training result, or speedup was established.

Status: completed, exploratory, `confirmed=false`. Date: 2026-11-12
(America/Los_Angeles). Hypothesis: `JSPACE-CAV-PILOT-01`.

- [Original plan](../../07%20-%20Status%20Reports/jspace/2026-11-12_jspace-cav-pilot-plan.md)
- [Numerical/model amendment](../../07%20-%20Status%20Reports/jspace/2026-11-12_jspace-cav-pilot-amendment-01.md)
- Main evidence: `data/experiments/jspace_cav_pilot/20261113T0728Z_pilot02/campaign_results.json`
- Fingerprint: `8b9f965b9fd01bc71da37603cbd5280dff1849751ec9e94b52f2b6c49f5d3a5d`

## Task accuracy and causal intervention

The parent was frozen
[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, in native BF16. Its standard chat
template requested one missing word or number. Generation was greedy, capped at
eight tokens, with complete-word parsing and the fixed normalization rules.

Development accuracy was 9/12. Held-out accuracy fell to **20/48, or 41.7%**
(Wilson 95% interval 28.8–55.7%). The 48 cases were independent unordered
concept pairs, not 48 trained-model replicates. The result's classification is
`underpowered_parent_accuracy`.

| Intervention | Intended target matches, all 48 | Success among 20 parent-correct cases |
| --- | ---: | ---: |
| J-derived concept swap | 7/48 | 4/20 = 20.0% |
| Norm-matched unembedding control | 2/48 | 0/20 |
| Norm-matched random control | 2/48 | 0/20 |

The conditional J success interval is 8.1–41.6%; the controls' intervals are
0–16.1%. Against either control, the exploratory paired bootstrap gives a
5–40 percentage-point interval, but the exact paired test gives **p=0.125**.
This is a small, suggestive signal, not a confirmed causal advantage. It also
misses the prespecified 50% success threshold. Two of the seven unconditional
target matches already gave that target before intervention; raw matches alone
must not be treated as causal changes.

The four parent-correct changes were:

| Hidden intermediate swap | Original answer | Counterfactual answer |
| --- | --- | --- |
| Egypt → Russia, language of Cairo's country | Arabic | Russian |
| China → Brazil, ocean east of the Great Wall's country | Pacific | Atlantic |
| England → Germany, capital of the Thames's country | London | Berlin |
| Canada → China, currency of Toronto's country | Canadian | yuan |

The vocabulary candidates were selected from the task inventory before outcomes,
not learned from held-out answers. The six Jacobian lenses used 16 independent
WikiText2 training paragraphs, 64 tokens each with the specified position mask.
The selected rows were computed by VJPs equivalent to contraction of the full
published estimator. CPU comparisons against the pinned upstream implementation
and a tiny full-J reference passed.

## Retention is not accuracy

The selected source was block output 11 (zero-based), with block 12 providing
downstream concept scores. No rank passed both development sufficiency tests;
rank 64 was the prespecified fallback. The complete selected-bank span had rank
94 in a residual space of width 2,048. Its semantic factorization residual was
approximately 1.19e-6; rank-64 truncation left 17.7% relative bank reconstruction
error.

| Basis and intervention | Baseline answers retained | Actual correct answers | p95 next-block concept-score drift |
| --- | ---: | ---: | ---: |
| J rank 64, donor complement | 46/48 = 95.8% | 20/48 = 41.7% | 3.91% |
| PCA rank 64, donor complement | 48/48 = 100% | 20/48 = 41.7% | 7.50% |
| Random rank 64, donor complement | 46/48 = 95.8% | 20/48 = 41.7% | 7.73% |
| J full rank 94, donor complement | 48/48 = 100% | 20/48 = 41.7% | 3.57% |
| J rank 64, calibration-mean complement | 4/48 = 8.3% | 6/48 = 12.5% | 41.17% |
| J full rank 94, calibration-mean complement | 6/48 = 12.5% | 7/48 = 14.6% | 37.77% |

The **95.8% number is answer agreement**, including 26 preserved wrong answers.
It is not 95.8% accuracy. Donor replacement retains all twenty correct answers,
but the controls do too. J-specific answer retention was not established.

The donor arm meets the local numerical/retention thresholds at ranks 64 and 94.
The mean arm fails at every tested development rank and both held-out ranks.
Retained-coordinate errors for the J arms stay below 0.3%, within the 0.5%
tolerance. The large downstream drift is therefore not explained by a broken
projection or an incorrectly imposed coordinate constraint.

The quarter-strength donor sensitivity arm is retained in the raw results. It
does not change the conclusion. For example, full-rank J quarter-strength
replacement retains 45/48 outputs, compared with 48/48 at full strength. Small
changes in output choice are not monotone in perturbation strength.

## Why reconstruction failed: evidence and limits

A post-hoc CPU reconstruction audit, without further model inference or changed
scoring, found that rank-64 donor replacement changes the activation by a median
10.1% of its norm. The calibration-mean reconstruction changes it by **89.0%**.
Median absolute changes in residual norm are 0.57% and 23.45%, respectively.
The audit checks its reconstructed perturbations against the original recorded
sizes and is saved as `post-hoc-norm-audit.json`.

The affine mean comes from ordinary WikiText positions, whereas the intervention
acts at a final chat position. Of the 48 rank-64 mean reconstructions, **34 emit
EOS immediately**. The failed reconstruction can therefore reflect lost response
formatting or operating-state information as well as missing task semantics.
The audit does not establish that retaining norm, or changing the mean, will
repair it. In particular, within the J mean arm the norm-change/drift correlation
is negative; a simple norm-only explanation is not demonstrated.

The intact non-final positions and lower-layer caches can still supply original
information. Donor robustness is context-conditioned local invariance, not an
autonomous reduced transition. The complete 94-row bank is also only a small
selected dictionary, not the paper's full sparse nonnegative J-space. No
attention head's Q/K routing or output projection was replaced.

## Baseline error audit

All 48 baseline answers parsed as complete: 47 ended at EOS and one at an ordinary
word boundary. There were no cap-truncated answers. Most failures are substantive:
`four` instead of eight spider legs, `53` instead of calcium's atomic number 20,
and `Ontario` instead of Ottawa for the capital of Toronto's country.

Some outputs identify the intermediate but miss the requested property:
`Ruby` rather than its color red, or `May` rather than month number five. One
`lip`/`lips` mismatch is plausibly morphological. The butter-source question is
underspecified enough that `Yak` versus the gold `cow` deserves an annotation.
These are post-hoc observations; no gold labels or primary scores were changed.
They do not explain away the broad competence shortfall.

## Gate accounting

| Gate | Requirement | Measured verdict |
| --- | --- | --- |
| Development parent competence | At least 8/12 | Pass: 9/12 |
| Held-out parent competence | At least 24/48 | Fail: 20/48; underpowered |
| Targeted counterfactual success | At least 50% among parent-correct cases | Fail: 20% |
| Retained-only local sufficiency | Both donor and mean arms pass | Fail: mean reconstruction collapses |
| J-specific answer retention | Beat PCA and random with paired lower bounds above zero | Fail |
| Numerical coordinate preservation | At most 0.5% error | Pass for evaluated J arms |

The next useful experiment is a parent/task pair with substantially stronger
baseline competence, followed by an explicit test of a task-calibrated affine
origin and residual-norm observables. That would distinguish missing task
coordinates from lost chat-state information. It requires a new frozen protocol
and fresh evaluation evidence. Algebra compilation and speed benchmarking remain
premature.

## Execution, corrections, and reservations

The main campaign requested and completed one run, with no reused runs or
execution failures. It took **514.5 seconds** including loading, fitting,
development selection, and evaluation. Peak PyTorch allocation was 6,680 MiB;
peak allocator reservation was 6,832 MiB. The controller lease requested 10,240
MiB on the RTX 3060. These wall times are descriptive experiment costs, not a
throughput benchmark. The runtime was locked repository Pixi, Torch 2.5.1+cu121,
Transformers 4.53.3, eager attention, BF16 weights/forward, and FP32 analysis.

Earlier evidence is preserved separately in
`data/experiments/jspace_cav_pilot/20261113T0715Z_pilot01/`. Its 1.5B FP16 path
overflowed in block zero and incorrectly scored nonfinite logits as token zero.
`numerical-validity-correction.json` records that its 0/12 was invalid as a
knowledge measurement. Reserved FP32 diagnostics subsequently measured 3/12 and
5/12 for the 0.5B and 1.5B base parents. No such values were pooled into this
campaign. Finite-value checks now raise before argmax and scoring.

Four GPU jobs used explicit reservations: the original screen, the precision
diagnostic, the prompt-format diagnostic, and the amended campaign. Their leases
`824fc4d7249d`, `a936e261ae4f`, `b6a97a766455`, and `713bb96103db` were verified
released. Controller 1.5.1 returned to `normal` in enforce mode. The current
campaign's receipt is `reservation.log`; final verification is
`reservation-release-readback.json`. No reservation key was copied to disk,
TLS verification was retained, and no identity privileges were broadened.

The main implementation, numerical guards, intervention math, reservation
lifecycle, and registry checks have 82 passing focused CPU tests in total.
Registration uses one direct experiment and keeps `confirmed=false` for this
exploratory result. Its separate record is
`data/meta_hypotheses/JSPACE-CAV-PILOT-01_20261112_pilot02.json`, with required
Chroma readback recorded under `storage.readback`.

## Reproduction and raw artifacts

The completed root is append-only. To repeat, prepare a new root rather than
reusing its results:

```bash
CUDA_VISIBLE_DEVICES='' pixi run --locked python -m experiments.structure_net.jspace_pilot prepare \
  --output data/experiments/jspace_cav_pilot/NEW_RUN \
  --model-path /home/rabbit/models/hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --dtype bfloat16 --prompt-format instruction_chat \
  --parent-plan data/experiments/jspace_cav_pilot/20261113T0715Z_pilot01/plan.json

/usr/bin/python3 /usr/local/bin/inference-ctl lease run \
  --allocation 3060=10240 --minutes 60 --reason 'JSPACE-CAV-PILOT-01 replication' -- \
  pixi run --locked python -m experiments.structure_net.jspace_pilot run \
  --root data/experiments/jspace_cav_pilot/NEW_RUN
```

Within the completed root, `plan.json` contains all model/data/source hashes and
the frozen split. `producer/` retains producing code and protocol copies.
`runs/pilot/seed_7/` contains the direct `result.json`, per-case `events.jsonl`,
the locked development selection, competence screen, lens, semantic basis and
controls, and baseline residual tensors. The separate norm audit is explicitly
post-hoc and does not amend the primary scores or gates.
