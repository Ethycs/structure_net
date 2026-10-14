# Llama 3B tensorized initialization and neighbourhood distillation

Completed EXPLORATORY / UNDERPOWERED experiment, one trained student.
Hypothesis `llama32-3b-neighbourhood-tensorization-v1`.
Classification: **`tensorized_student_development_quality_failed`**.

Rank-1024 MLP factorization reduced FP16 parameter storage by **35.6%**, and
independent neighbourhood training reduced local development error. The
assembled student passed **0/8 complete-summary development requests**. The
original FP16 and existing Q4 controls each passed 8/8. This candidate does
not meet the quality or deployment speed target.

Primary aggregate:
`data/experiments/llama_neighbourhood_tensorization/20261014_registered/campaign_results.json`

SHA-256:
`1a235e05f7cb5f312c3568dd59567168d10f8be298561f9ddcb2d0fcb65bfd54`.

The judge was **assistant source-grounded review with structural/token checks**;
there was no separate or independent judge model. The rubric required every
fragment's current owner, approved budget and deadline, no invented or
superseded claims, complete fragment coverage, and at most 128 tokens per
stored summary. All 32 arm/request outputs have preserved review dispositions
and source quotations. Correct control outputs used at most 34 tokens per
labeled summary. The two eight-fragment control outputs put labels on separate
heading lines. The review accepted these as eight labeled summaries, with a
clarification saved before student outputs were reviewed. Literal one-line
format compliance is separately reported: 6/8 for each control, 0/8 for the
initialization, and 3/8 for the distilled student. Cosmetic formatting cannot
rescue incorrect or missing facts.

Prospective success required all eight development requests to pass before
opening the 20-request sealed test set. Final proxy quality would require at
least 95% complete requests and no fewer passes than either control. The speed
target was at least 2x current native Q4-plus-lookup throughput and lower mean
latency on matched complete-summary requests. Those final gates were not
reached. The sealed test set, five-repeat final latency campaign, native factor
kernel integration and factor quantization remained unexecuted after the
development stop rule. These synthetic requests do not measure production
accuracy or establish a general verdict on tensorization.

| Arm | Weight bytes, decimal GB | Complete requests | Fixed 32-token diagnostic, tokens/s |
| --- | ---: | ---: | ---: |
| Original FP16, PyTorch | 6.4255 | 8/8 | 11.10 |
| Initial MLP factors, PyTorch | 4.1354 | 0/8 | 10.88 |
| Distilled MLP factors, PyTorch | 4.1354 | 0/8 | 11.20 |
| Existing Q4_K_M + lookup 8, native | 2.0115 | 8/8 | 42.78 |

The fixed-length diagnostic used the same four development prompts and three
repetitions, included prefill and 32 generated token steps, and ignored EOS to
keep the work count equal. It counts special-token steps and is systems
evidence, not summary-quality evidence. Models were warm; loading,
tokenization and detokenization were excluded. The PyTorch prompt tensor and
empty cache were created before its timer; the native timer included cache
clearing and forward input transfer. Every request had fresh logical KV state.
All model execution used physical GPU 0, an RTX 3060, serially, with two CPU
threads. GPUs 1 and 2 were not used.

The distilled prototype was only 1.0085x the same-backend dense prototype and
0.2617x the native Q4-plus-lookup control. Three repetitions do not establish a
real sub-1% gain. The native control uses quantized weights, CUDA graphs and
prompt lookup; the PyTorch prototype has none of those runtime optimizations.
Consequently the native comparison is an implementation comparison, not an
isolated architectural or optimal-kernel comparison. Replacing 84 dense MLP
projections with two factors introduces 84 additional matrix multiplications.
Fewer parameters did not provide a meaningful measured speed gain here.

Natural-request mean latencies were 6.337 s for dense FP16, 53.154 s for the
initialization, 20.939 s for the distilled student, and 1.043 s for native Q4.
These averages include substantially different output lengths: the failed
initialization reached every 512/768-token cap, while the distilled student
reached three caps. They are not a matched useful-output throughput estimate.

All arms used the full 28-layer Llama architecture. The original cached
checkpoint was pinned by the cotenancy manifest at HF revision
`0cb88a4f764b7a12671c53f0838cd831a0843b95`. The native Q4 artifact retained
SHA-256 `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`.
Attention, embedding/output weights, norms and ordinary residual connections
were preserved. Each gate/up/down projection became two factors, retaining
the 3,072-dimensional layer interface and 8,192-dimensional nonlinear MLP
activation. There is no dense exact-correction branch or reconstructed dense
MLP operator during inference.

Initialization used original FP16 matrices, train-input channel RMS scaling,
randomized CUDA SVD with rank 1,024, 16 oversamples and two iterations, and
balanced square-root singular values. This is diagonal activation scaling,
not SVD-LLM's complete covariance whitening algorithm. Two-layer groups were
trained independently on original-teacher input/output activations, following
the local-subnetwork idea of [Neighbourhood Distillation](https://arxiv.org/abs/2010.01189).
The factorized initialization and recovery are also motivated by
[SVD-LLM](https://arxiv.org/abs/2403.07378). This is an exploratory adaptation,
not a reproduction of either paper's full experimental protocol.

The teacher stayed frozen. Only 968,884,224 factor parameters trained, using
FP32 master weights, FP16 forward operations, AdamW, learning rate 0.00002,
zero weight decay, clipping at norm 1, and 96 updates per group. Development
selection compared the actual FP16 deployment representation at updates
0/32/64/96; all groups selected update 96. Fourteen groups completed 1,344
updates in 306.35 seconds of group training/decomposition/evaluation time,
excluding cache preparation, validation and teacher loading. All factor
gradients were finite and nonzero. Dense two-layer extraction exactly matched
the cached teacher for all 14 groups, and every saved factor checkpoint
reloaded with exactly matching operational output. Peak training allocation
was 2,881,176,576 bytes, separate from driver/context memory.

The original learning-rate pilot at 0.0002 increased longest-request NMSE from
0.76694 to 2.79003 after one update. A prospective stability amendment reduced
the learning rate tenfold before primary training. Its eight-update pilot
reduced NMSE to 0.04228. Both pilots and their producing snapshots remain
preserved. Across the actual primary groups, local development error fell
67.7–99.9%, with a median reduction of 75.2%. These are relative reductions in
normalized squared hidden-state error, not task accuracy.

There were 64 training requests, eight development requests and 20 unopened
final-test requests with distinct seeds and prompt hashes. Sources contained
synthetic final decisions and superseded proposals. The inherited template
is repetitive and favors copying; no prior study's held-out requests were
used in training. Prompt families were approximately 500/2,000/6,300 raw
tokens and 1,000 hierarchy tokens. Actual development prompts spanned
493–6,227 tokens; every source fragment fit 2,048 tokens and every prompt fit
6,300. Output caps remained raw 768 and hierarchy 512. Teacher caches included
reference response tokens as well as prompts. The loss averaged all token
positions and did not give answer positions additional weight. This data and
rubric do not cover general summarization, the exact extractor, or all inputs
near the 192-token bypass threshold.

Observed failures were substantive. In one request, the source approved
**Nora Bell, $6,900, November 23**; the distilled student emitted
**“Owner, $6,800, November 15.”** Other outputs repeated generic phrases,
omitted required facts, produced blank-line loops, or copied the first
project's facts into different projects. A few correct fields and improved
termination do not make a complete request pass.

A separate post-decision diagnostic traced teacher-forced errors on the first
short request and first eight-fragment request. It is exploratory and cannot
rescue the failed quality gate:

| Prompt | Final group local NMSE with teacher inputs | Final assembled NMSE | Answer-position NMSE | Teacher/student next-token agreement with reference history |
| --- | ---: | ---: | ---: | ---: |
| 493 tokens | 0.02938 | 0.26734 | 0.39739 | 97.1% (33/34 positions) |
| 6,227 tokens | 0.02172 | 0.17654 | 0.64870 | 86.8% (224/258 positions) |

The first group's local and assembled errors matched exactly. By the final
group, assembled error was about 9.1x/8.1x the same-group local error. This
directly measures the mismatch between isolated teacher-input fitting and
assembled student-prefix behavior on these two trajectories. Teacher forcing
supplies correct reference history; its next-token agreement does not measure
free-generation task accuracy. The 97.1% agreement alongside a failed free
summary is a concrete reason not to use token agreement as the user's final
accuracy target. An answer-focused loss and training on student-prefix inputs
are plausible next experiments; these measurements do not establish that
either will solve quality or speed.

The diagnostic is preserved at
`data/experiments/llama_neighbourhood_tensorization/20261014_accumulation_v1/`.
Its result hash and producer readback are included in the completion receipts.
The primary producer roots are `20261013_cache_v1`,
`20261013_training_v4`, and `20261013_development_v4` under the same study
directory. Cache v1, lifecycle v2, optimizer v3 and generation-finiteness v4
source states have distinct preserved fingerprints. The reused native control
producer retains its earlier throughput-study identifier; the new campaign
explicitly maps its fresh input/output roots into the new comparison.

Reproduction uses the pinned model manifest and
`data/experiments/moba_paper/environment/bin/python`. Use new output roots;
completed artifacts reject replacement. The entry points are
`experiments.structure_net.llama_neighbourhood_tensorization`
(`fixtures`, `tiny-pilot`, `cache`, `train`, `evaluate`),
`experiments.structure_net.campaign_llama_neighbourhood_tensorization`,
`experiments.structure_net.llama_neighbourhood_accumulation`, and
`experiments.neural_architecture_lab.finalize_llama_neighbourhood_tensorization`.
The saved producer and execution records contain the exact commands, source
snapshots, input hashes and configuration. For CUDA execution set
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2`
before starting Python. The conservative registry record remains
`confirmed: false`, with production accuracy unmeasured.
