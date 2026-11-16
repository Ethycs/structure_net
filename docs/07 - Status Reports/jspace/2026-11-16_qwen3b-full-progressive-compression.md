# Complete progressive compression of Qwen2.5-3B

Study `JSPACE-QWEN3B-FULL-PROGRESSIVE-01`, exploratory, prospectively specified
before capturing the expanded banks or fitting layer34. The user's objective
is all36 transformer blocks compressed while retaining as much task quality as
possible. The allowed task-accuracy loss is5 percentage points. Completion of
compression and qualification of its quality are separate outcomes.

## Starting evidence and correction

The earlier narrow progressive run reached16/36 blocks. The new broad-corpus
rebuild reached only1/36: layer35. Its packed development MC score was198/256
(77.34375%) versus195/256 (76.171875%) for the original teacher. During its
recovery, response NLL fell from3.45472 to2.95058, still above the native2.17769.
Its controller stopped on a1.10times-native NLL gate and consequently never
measured generation. That stopping rule did not establish task-accuracy loss
and prevented the requested full compression. Preserve the original failed
gate, model outputs, producer files and reports; this study changes advancement
prospectively rather than rewriting those outcomes.

Begin from the completed layer35 `update_0046.pt`, SHA256
`d2d1b8613474d4b5891d004e11c87c5610b9215e67d34b95465e8bb70f15a486`,
in `jspace_3b_progressive_response_rebuild/20261116_v2/blocks_01/recovery`.
It tied the other saved candidates on MC and had the lowest response NLL.
Authenticate its completed source manifest, fit receipt, packed export and
model/corpus identities. Admit it explicitly as an exploratory seed whose old
quality gate failed. A separate task probe of the fixed seed may run in parallel
with CPU preparation; it cannot retroactively alter the original classification.

## Data and original teacher

Use the original frozen Qwen2.5-3B-Instruct, revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, with all36 native layers.
Reuse the broader complete-response corpus and original response teacher cache:
4441TRAIN and128DEV examples across eight task families, prompt cap288,
response cap128, total cap416. Preserve the v2 deterministic128 calibration,
32 disjoint TRAIN diagnostic and256 local alignment selections and the same
736-example joint order (25172 supervised response/EOS tokens per stage).
Do not use DEV to derive bases, fit relations or perform gradient updates.

Capture native boundaries0..36 in one forward pass per example, and native
boundaries0..35 for the existing256 MC DEV items. Check original teacher hidden
states, boundary20 and the old v2 boundary35 bitwise, plus exact corpus, model,
record order and MC identities. The new banks require approximately112.8GB;
check free disk before admission. Stream memory-mapped banks and evict completed
pages; do not load all banks into CPU RAM. Hash completed banks once, record
immutable file identities, and verify those identities for each subsequent
consumer. The sealed final benchmark is not an input. The previously disclosed
final-data exclusion incident remains disclosed; this work does not restore an
untouched final test claim.

## Stage schedule and recovery

Append layers34,33,...,0, giving2,3,...,36 compressed blocks. No stage-count
pilot cutoff and no NLL-based advancement cutoff are permitted. Use the existing
functional/Jacobian bases, rank1024 residual space, head dimension64 with original
attention scale, and the existing compiled MLP quadratic/residual recipe. Keep
the native frozen teacher as the target; do not replace it with the compressed
parent. Fit bases and relations in the established order before distillation.

For each new block, run a discarded real-CUDA systems pilot, then initialize
scientific training afresh. First align only the newly compressed block to the
original projected teacher boundaries using the fixed local TRAIN pass. Then
jointly recover all compressed blocks and final norm using complete-response
distillation, including EOS, for the same25172 supervised-token budget. Preserve
the fixed geometry, native prefix and readout. Record raw prefit, post-local,
approximately5k,10k and25k-token checkpoints. Prioritize early high-accuracy
checkpoints: rank by packed MC accuracy descending, then response NLL ascending,
then earlier update. Continue from the best observed candidate, never blindly
from the latest update. Do not label this exposed DEV selection a fresh test.

Always produce an authenticated continuation checkpoint for a scientifically
completed stage. NLL, generation and semantic quality failures remain explicit
diagnostics; they do not terminate the all36 compression schedule. An exploratory
parent is not a qualified parent. Systems failures, corrupted artifacts,
nonfinite training, invalid provenance or failed save/reload checks require
diagnosis and repair before continuing; never silently mark them completed.

## Packed evaluation, quality and resource use

Evaluate every candidate after packing, saving and fresh structural reload.
Measure the same128 response DEV losses and256 MC DEV items, retaining packed
state digests and reload evidence. On each stage's chosen candidate, always run
the same six canaries and16 response DEV generation cases, without an NLL
short-circuit, through the real native prefix plus packed suffix. Keep the
original generation limits and EOS tokens. GPU generation must finish and
release its lease before remote semantic grading. Queue CPU/network grading
serially while the GPU proceeds to the next stage; its verdict does not mutate
the already chosen parent. An unavailable judge remains unavailable, never zero.

Report paired native/student MC, generation, numerical/code task success,
semantic success and support. Preserve the existing native-relative5pp screens,
including the native's own low strict-canary baseline; do not invent an absolute
accuracy floor. Missing required evidence prevents a qualification claim but
does not prevent exploratory compression. All36 completion requires authenticated
36-block artifacts and the final evaluations; quality success additionally
requires the stated task screens. Tiny exposed samples cannot establish broad
accuracy,95% quality or deployment readiness. NLL stays visible as a distribution
diagnostic. Report actual bytes and measured speed separately; no1.4x/2x speed
or40% whole-model size claim is assumed.

Reserve the3060 with `inference-ctl lease run`,11264MiB allocation and11008MiB
allocator cap, separately for capture, fit, pilot and recovery/generation.
Use locked/offline Pixi and respect all other leases. Transient reservation or
authentication failure may be retried only before the scientific phase starts,
with separate append-only attempt logs. Never overwrite or rerun a partially
started scientific phase as though it had not happened.

For the new study only, candidate exports are declared temporary scratch: retain
their metrics, export/reload receipts and digests, but retain only the selected
packed payload at each stage. Retain the best three unpacked stage checkpoints
by the declared ordering. Garbage-collect only this study's explicitly declared
scratch/checkpoint candidates after the necessary evidence and selected parent
are secured. Do not delete old studies. Check disk admission before each stage;
reserve space for transient candidate exports and the final36-block export.

Write append-only artifacts under
`data/experiments/jspace_3b_full_progressive/20261116_v1`. Freeze producer source,
tests and this protocol before their use. Keep `confirmed:false`. Preserve the
completed all-at-once36 run and its failed quality measurements without restarting
it or using it as this study's parent.
