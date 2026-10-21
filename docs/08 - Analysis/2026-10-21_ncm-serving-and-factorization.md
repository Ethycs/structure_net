# NCM serving: remove CPU validation overhead; factorization loses quality

Completed 2026-10-21. Conformance: EXPLORATORY, UNDERPOWERED, one frozen
Llama-distilled NCM and technical timing repetitions. All three completed
records retain `confirmed:false`.

| Hypothesis | Classification | Aggregate SHA-256 |
| --- | --- | --- |
| `llama32-3b-ncm-replicas-v1` | `replica_throughput_target_unmet` | `c52342213c791384df6632b74976ac8e1f5cf58a8b99570bb8f1e8c7d45dd9a3` |
| `llama32-3b-ncm-factorization-v1` | `factorized_development_quality_failed` | `e055140519ab06cf9864e070f9d741c858d5c7ce5300700566edbe1bf736e3e2` |
| `llama32-3b-ncm-host-io-v1` | `lossless_host_io_throughput_target_met` | `27a541e247d2da4e4fe52eba72dbf54cc08ade19d4a00638aff3f0811e381ec6` |

Canonical aggregates are respectively under
`data/experiments/ncm_replicas/20261021_registered_v1/`,
`data/experiments/ncm_factorization/20261021_registered_v1/`, and
`data/experiments/ncm_host_io/20261021_registered_v1/`, each named
`campaign_results.json`. The separate original batching campaign is
intentionally interrupted and superseded, rather than classified as passed.

The unchanged 149M-parameter NCM now serves **259.62 output tokens/s alone**
and **326.49 tokens/s in batches of nine**, versus a fresh original Llama3B
control at **77.18 tokens/s**. These are respectively **3.364x and4.230x**
Llama's aggregate token rate, with all1500 NCM outputs exactly matching the
parent's native output IDs and text. The runtime change caches an immutable
tokenizer vocabulary size. It adds no model training or content/KV reuse.

Eight literal model copies give only280.18 tokens/s, using almost the complete
Llama reserved-memory budget. Shared-weight batching gives higher throughput
with much less memory. The rank256 factorization experiment really shrinks
the model to534MB, but its repaired model passes only28/64 development source
reviews, so it is not promoted or installed as the default.

The prospective records are the
[replica plan](../07%20-%20Status%20Reports/2026-10-21_ncm-replicas-plan.md),
[factorization plan](../07%20-%20Status%20Reports/2026-10-21_ncm-factorization-plan.md),
[original batching plan](../07%20-%20Status%20Reports/2026-10-21_ncm-batching-plan.md),
and [host-I/O correction plan](../07%20-%20Status%20Reports/2026-10-21_ncm-host-io-plan.md).
The last plan records the known failed factorization field checks, completed
replica timings and diagnosed vocabulary lookup before corrected timings.

## What changed the speed conclusion

The [parent distillation report](2026-10-21_ncm-llama-distillation.md) remains
unchanged. Its81.89 tokens/s included the original Python output-validation
path. In `decode_ids`, every output ID was checked against `len(gpt)`.
On this host,100 calls to the fixed GPT2TokenizerFast vocabulary size took
1.1021s. The same size was therefore repeatedly looked up within one request.

`CachedVocabulary` obtains the exact vocabulary length50258 once after PAD
is added, delegates the unchanged tokenizer decode, and supplies the same
constant to every existing ID-bound check. EOS/PAD trimming, invalid-ID
handling and full-canvas behavior remain. The wrapper is used only after
vocabulary construction, with no later mutation. The default inference CLI
now uses it for both graph and eager execution.

Eight paired development profiles isolated the cost:

| Median paired component | Original validation | Cached vocabulary |
| --- | ---: | ---: |
| Full request, seconds | 0.360437 | 0.104556 |
| CUDA-event graph execution, seconds | 0.103812 | 0.103810 |
| CPU validation and decode, seconds | 0.256156 | 0.000189 |

These are systems diagnostics, not the full-cohort speed estimator. All eight
paired native48-ID/text outputs matched, and a separate batch9 A/reversed/A
check passed all27 comparisons. GPU generation was unchanged. The old flat
replica curve therefore did not establish GPU saturation as the cause of its
low end-to-end rate; a CPU bottleneck masked much of the available speed.

## Corrected full-cohort comparison

Each condition uses the same100 complete parent runtime-validation sources,
three repetitions, and a fresh isolated GPU process. Sources contain256–493
GPT-2 tokens, padded to512, with a48-position output canvas and the original
nine generation levels. Literal copies have independent complete weights,
codec, graph and serving stream. Batches share one model and graph. Final
partial static batches still compute padding requests; their wall time stays
in the denominator, while their outputs never count toward throughput.

Timing includes CPU source IDs, input transfer, all generation, output transfer,
every ID validation and decoded CPU text. Loading, graph construction, output
token counting, evidence writes and warmups are excluded. NCM has one excluded
development warmup wave per condition; Llama has an excluded warmup per source.
These warm steady-state measurements are not cold-start or machine-exclusive
latency measurements. GPU0 alone was used, with no overlapping GPU producer.

Throughput is total actual output tokens divided by total timed wave wall
seconds. Token counts use the common Llama tokenizer, not latent codes,
canvas positions, padding or invented N-times-single-copy estimates. Llama's
fresh control uses the identical total-token/total-time estimator and unchanged
graph/FlashAttention/current-prompt lookup8 runtime, through EOS.

| Configuration | Aggregate tokens/s | Tokens/s in three repetitions | Mean wave, s | Model state, GB | PyTorch GPU reserved, GB |
| --- | ---: | --- | ---: | ---: | ---: |
| Original Llama3B, fresh | 77.18 | 77.140 /77.198 /77.199 | 0.348281 per request | 6.425 | 6.795 |
| NCM, one independent copy | 259.62 | 259.670 /259.595 /259.586 | 0.104423 | 0.713 | 0.883 |
| NCM, shared batch1 control | 259.41 | 259.655 /259.210 /259.368 | 0.104506 | 0.713 | 0.883 |
| NCM, eight independent copies | 280.18 | 280.344 /279.935 /280.267 | 0.744297 | 5.706 | 6.786 |
| NCM, shared batch9 | 326.49 | 326.979 /326.308 /326.169 | 0.691966 | 0.713 | 1.336 |
| NCM, nine independent copies | 277.74 | 278.091 /277.392 /277.725 | 0.813423 | 6.420 | 7.625 |

GB means decimal bytes. Eight copies improve aggregate throughput7.92% over
one; shared batch9 improves25.86% over its batch1 control. The eight-copy
effective rate is35.02 tokens/s per copy, not280.18 per copy. Batch9 is an
aggregate-throughput improvement, with a longer response wave than batch1.
The tested maximum is not an optimum over every possible batch size.

Nine NCM model states total6419648592 bytes, within Llama's6425499904-byte
weight/buffer budget. Nine copies exceed the full reserved-memory budget;
eight fit at6786383872 versus6794772480 bytes. Allocated counters were reset
after graph construction and do not represent every transient capture
allocation; the reserved budget conservatively retains graph working pools.
Driver/context memory is separate. This is not a physical-VRAM equivalence
claim based solely on parameter count.

Both token and request throughput gates pass. Single-copy request-rate ratio
against fresh Llama is3.335x, and batch9's is4.194x. Batch9 uses only1335885824
reserved bytes. Its24 timed dummy outputs do not contribute to the300 actual
request cells. Original Llama's own batching was not tested or optimized;
this comparison does not establish optimal equal-memory serving superiority.

All1500 corrected NCM native48-ID/text outputs equal the saved parent outputs.
All300 fresh Llama native/text outputs equal its parent as well. Their100/100
synthetic source-fidelity results are inherited by this exact equality, not
new independent model-quality replications. Seven original Llama cache files
were physically hash-read back after timing, matching the pinned parent.

## The completed original replica experiment

Before the vocabulary correction, the fixed1/9/2/8/4/7-copy order completed all
1800 timed request cells with native-output parity and no failed/retried cells.
A separate nine-copy systems pilot passed27 concurrent/single-stream checks.

| Literal copies | Original aggregate tokens/s |
| --- | ---: |
| 1 | 76.52 |
| 2 | 77.95 |
| 4 | 78.84 |
| 7 | 77.97 |
| 8 | 78.85 |
| 9 | 78.28 |

Its2x reused-Llama target failed. Preserve this result as the performance of
that implementation. The corrected experiment has separate source snapshots,
configuration, artifacts, fresh controls and classification.

## Factorization and local repair

Apply the existing Llama input-channel-RMS-scaled randomized SVD to all24
generator MLP c_fc/c_proj matrices. Rank256 is one-third of NCM's hidden width,
analogous to1024/3072 in the Llama pass. Execute factors directly and preserve
every original bias. Attention, norms, heads, queries and the entire codec
remain unchanged. This is low-rank matrix factorization, without a dense
correction operator. FP16 deployment factors and FP32 remaining state run
under the original BF16 autocast. The size change includes both factorization
and lower factor-storage precision, rather than isolating those effects.

Calibration uses the first64 balanced training sources; eight development
sources select local checkpoints. Original Llama summary labels are encoded
with the unchanged codec to teacher-force200 code positions. Each of six
independent two-block neighborhoods receives original dense NCM boundaries.
Only factors train:96 AdamW updates/group, lr2e-5, zero weight decay, norm clip1,
seed51021, FP32 masters and BF16 autocast. Add1% factor-RMS Gaussian weight
perturbations once and5% channel-RMS input noise, recomputing original dense
targets on those same perturbed inputs. There is no random inference noise.

All six dense cache replay errors were exactly zero. The separate CUDA pilot
passed real updates, finite nonzero factor gradients, exact checkpoint reload,
unchanged teacher/codec digests and graph/eager native-ID parity. All576 full
updates completed; each local neighborhood selected update96. Local development
NMSE ranges fell from0.00178–0.00906 to0.000793–0.00255, but this did not preserve
free-running generation quality.

| Candidate | Runtime model bytes | Development field matches | Development source reviews |
| --- | ---: | ---: | ---: |
| Plain rank256 initialization | 533987792 | 7/64 | 2/64 |
| Noise plus local neighborhood repair | 533987792 | 39/64 | 28/64 |

The runtime state shrinks25.14%. Codex reviewed all128 complete development
texts against final source quotations; no independent judge model was used.
Field substrings alone accepted corrupted digits, inconsistent predicates,
unrelated suffixes and degenerate repetition. The source reviews reject those
cases. Minor repeated function words were allowed when the complete final
facts remained coherent without added claims. Both models fail even a95%
threshold, as well as the planned64/64 no-degradation promotion requirement.

No final100-source quality stage or new ranks/global training were run after
this failure. The planned bounded8-source/three-repeat diagnostic completed
48 requests. Repaired-model request latency worsened: median-time sum3.6663s
versus dense2.9034s. Its nominal84.28 versus74.74 tokens/s reflects longer,
often invalid text and is not a usable quality-preserving speed gain. The
vocabulary correction was not silently substituted into this earlier trial.

Checkpoints remain under
`data/experiments/ncm_factorization/20261021_training_v1/`:
`initial.pt`, SHA-256
`32cd1b3c638ec7ae94d17ef18329a05afe9d7e2828202455b334eb96a4fa85c5`,
and `repaired.pt`, SHA-256
`e2bb44e023dc4af53a14d2aed3a05f44708be12054f792698c2bb93427fc2334`.
They are research checkpoints, not the default summarizer.

## Integrity, interruptions and reproduction

Nine focused CPU tests passed, covering partial waves without inflated counts,
parity and quality gates, bias preservation, factor/teacher gradients,
checkpoint identity, review/source binding, static batch tails, and cached
vocabulary semantics. Python compilation, CUDA pilots and the updated default CLI also passed.
CLI demonstration artifacts are under `ncm_host_io/20261021_cli_v1`, excluded
from timing. Its request took0.1134s and retained the correct final facts.

There are1800 completed original-replica timing cells,48 factorization
diagnostic cells and1800 corrected-campaign cells, with zero failed/retried
timing cells in those completed campaigns. Their176 excluded request warmups
also include the interrupted legacy batch1 warmup; graph-construction warmups,
systems pilots, training and128 development outputs are separate.

The original shared-batching pilot passed27 checks. Its first batch1 arm
preserved all300 timed request rows, but its process was intentionally
interrupted during post-run verification after the CPU bottleneck was found;
no terminal result or completed900-cell aggregate was produced. Its batch4/9
primary arms were not run. Producer/rows/logs, `failure.json`, and the explicit
`ncm_batching/20261021_campaign_v1/interruption.json` remain. The unfinished
CPU replica audit was likewise interrupted while inside tokenizer length
lookup, then completed using identical validation through the cached wrapper.
Its separate interruption receipt is retained. Neither interruption is a
scientific threshold retry, and neither erases completed legacy evidence.

Physical GPU0 was the RTX3060 12GB; other reserved GPUs and unrelated user
worktree changes remain untouched. Software is the existing Torch2.4.1+cu121,
Transformers4.53.3 and FlashAttention2.6.3 environment. No libraries or weights
were upgraded. The frozen original generator SHA is
`ff7133b2de50bc1f66ef1e490fec4c723e0fc5a5c1957107209e923b198c3e3c`;
codec SHA is
`b87f7288edfc0eb1f5b9122d1139cc2e61aa2331be7d6fe53f0582993086ede2`.
All new corrected-condition fingerprints bind these same checkpoints.

The default single-source command now includes the verified host correction:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
HF_HUB_OFFLINE=1 LD_LIBRARY_PATH='' \
data/experiments/moba_paper/environment/bin/python \
  -m experiments.structure_net.infer_ncm_llama --input /path/to/source.txt
```

For experiments, use `run_ncm_replicas`, `run_ncm_factorization`, and
`run_ncm_host_io`; their corresponding campaign coordinators preserve serial
GPU execution. `finalize_ncm_optimizations` audits/registers the original
replica and failed factorization records; `finalize_ncm_host_io` audits the
corrected comparison and fresh teacher. Raw producer commands, source copies,
environment, per-request outputs, per-wave times, hashes, complete aggregates
and conservative registry readbacks are retained. Reproduce in fresh roots.

The supported conclusion is higher warm throughput for this unchanged small
model on the bounded synthetic summary task, with output parity across tested
serving configurations. It is not production95% accuracy, general Llama
capability transfer, lossless compression of Llama's weights, an optimum over
all serving choices, mobile performance, or support for the application's full
6300-input/128-output-token range. The model still accepts at most512 GPT-2
input tokens and generates a48-token canvas.
