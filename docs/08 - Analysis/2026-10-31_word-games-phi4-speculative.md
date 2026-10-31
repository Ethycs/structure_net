# WORD_GAMES: smaller Phi draft was slower in the pilot

October 31, 2026. EXPLORATORY, systems pilot only.
Hypothesis `word-games-phi4-mini-speculation-v1`; classification
`pilot_output_mismatch`. The planned full 36-request comparison did not run.

**Phi-4-mini speculative decoding did not accelerate these three WORD_GAMES
pilot requests.** Target-only throughput was **14.63 output tokens/s**; draft
lengths 2, 4 and 8 gave **13.88, 13.11 and 12.77 tokens/s**. Each speculative
condition also changed the metaphor selector's output, so none passed the
exact-output gate. The two other requests matched exactly and were still slower.
The original serving endpoint remains unchanged; speculative decoding is
available in the experimental runner but is not enabled by default.

Primary artifact:
[pilot v2 aggregate](../../data/experiments/word_games/20261031_speculative_pilot_v2/run/campaign_results.json),
SHA-256 `67956e642b5cd3814690b459a927a59168064d14f7e81219cecdf9f3fcef42af`.
Its scientific fingerprint is
`87085f0ac339abb6ff0e5f686f539006dc6b731d1d436e8ac3cc10d1c80de7fb`.

## Configuration and implementation

The target was the existing Phi-4 14B Q4_K_M checkpoint on the RTX 3060, with
Q8 KV, 8,192-token context, JSON grammar and greedy decoding. The draft was
Phi-4-mini-instruct 3.8B Q4_K_M on the RTX 2060 SUPER, also with Q8 KV and an
8,192-token context. This setup consumes an additional GPU. It does not test
equal-resource efficiency.

The draft was downloaded from
[Bartowski's conversion](https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/tree/7ff82c2aaa4dde30121698a973765f39be5288c0),
revision `7ff82c2aaa4dde30121698a973765f39be5288c0`.
Its 2,491,874,688-byte GGUF hash is
`01999f17c39cc3074afae5e9c539bc82d45f2dd7faa3917c66cbef76fce8c0c2`.
Both checkpoint hashes were verified before launching. The separate temporary
container reused the existing llama-cpp-python 0.3.23 serving image and its
Pixi environment. The host controller ran through locked Pixi from the
repository root. Executable code stays under `experiments/`; `data/` holds
input records, copies of producing sources and results.

The target and draft have different vocabularies (100,352 versus 200,064) and
chat formats, verified from the loaded models. Direct draft-ID reuse would be
incorrect. The adapter renders the draft's native chat template, re-encodes
the confirmed continuation, generates draft tokens, and converts their new
bytes into target tokens. It never rewrites an accepted target prefix.
The target runtime samples and checks every candidate with its own greedy
sampler and grammar. This follows the cross-tokenizer candidate-conversion
principle described in
[Hugging Face's assisted-decoding documentation](https://huggingface.co/docs/transformers/assisted_decoding#universal-assisted-decoding),
but is a local adapter for the existing GGUF runtime, not Transformers' UAD
implementation or its exact alignment algorithm.

All 36 prior requests were frozen. The pilot used the metaphor selector, the
final normal-reserve sender turn and the cancellation receiver. Each was run
without a draft and with lengths 2, 4 and 8: **12 measured calls**, with
condition order reversed on the middle request. Both models reset their caches
between requests. Within-request draft prefix reuse is allowed. No answers
were repaired and no model training occurred.

## Measured results

| Draft length | Request seconds, summed | Output tokens | Output tokens/s | Latency change | Accepted / proposed candidates | Exact output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 18.66 | 273 | **14.63** | baseline | — | reference |
| 2 | 20.02 | 278 | 13.88 | +7.3% | 163/225 (72.4%) | 2/3 |
| 4 | 21.21 | 278 | 13.11 | +13.6% | 195/321 (60.7%) | 2/3 |
| 8 | 21.38 | 273 | 12.77 | +14.6% | 214/456 (46.9%) | 2/3 |

Throughput includes prompt processing, adapter setup, draft generation,
token conversion, target verification and Python overhead. It excludes model
loading, warmup and artifact I/O. Unaccepted draft tokens are not output.
These are raw pilot timings, not an established quality-preserving speed
comparison: the selector outputs differ. Both exact-matching requests were
slower in every speculative condition, so the negative timing observation
does not depend only on the selector's changed output length.

Draft generation plus conversion consumed 4.70, 5.54 and 6.52 seconds across
the three requests. Target `eval` calls fell from 276 to 118, 86 and 62, but
the total time increased. These are Python `eval` calls, not GPU kernel counts.
Their wall-clock subtimings may exclude asynchronous device work and must not
be interpreted as isolated GPU compute times. The whole-request timer is the
primary measure. Conversion cost was not isolated from draft inference.

The two smaller draft settings changed `shelve` / `return` to
`place-on-shelf` / `return-to-shelf`; length 8 changed `shelf-position` to
`shelf-number`. These are not evidence of worse task quality by themselves.
They are failures of the declared exact-output check. Batch-dependent
quantized numerics are a possible explanation; no numerical causality test was
performed, so this report does not attribute the differences to that cause
or claim that the adapter is universally lossless. The cancellation receiver's
original erroneous occupied-slot answer remained unchanged, not corrected.

## Integrity and limitations

The negative verification control deliberately proposed an unrelated token:
**0/20 proposals were accepted**, and target output matched the no-draft
control. Candidate conversion tests cover distinct token-ID spaces and partial
Unicode bytes. The focused project suite passes **45 tests**. Arithmetic,
output-parity flags, frozen inputs, source snapshots and aggregate hashes were
independently read back.

Pilot v1 was interrupted after four measured selector calls by a score-buffer
allocation defect in the new adapter. Switching logits collection on after
construction had retained only `n_batch` rows. Pilot v2 constructs the full
context score buffer and disables score collection for baseline calls. The
failed run and its four measurements remain separate. The
[plan amendment](../07%20-%20Status%20Reports/2026-10-31_word-games-phi4-speculative-plan.md)
records the defect and the added speculative warmup. Pilot v2 recorded 161.52
seconds of warmup, excluded from the timing table. Both models fit: total card
usage after loading was 9,822 MiB on the 3060 and 6,114 MiB on the 2060 SUPER,
including their pre-existing allocations. Both returned to their pre-test
memory levels when the temporary worker exited; unrelated services stayed up.

No setting passed all three exact-output checks. Following the declared gate,
no full 36-request timing run or semantic quality claim was made. This is one
small pilot and one hardware/runtime combination. It does not establish that
all speculative decoding is slower, that a same-card layout is slower, or that
a much smaller target-specific draft could not help. For these short JSON
requests, the measured extra work outweighed the saved target evaluations.

Reproduce the experimental setup with fresh output directories:

```sh
pixi run --locked -e word-games python -m experiments.word_games.prepare_speculative --output data/experiments/word_games/NEW_INPUTS
pixi run --locked -e word-games python -m experiments.word_games.run_speculative --inputs data/experiments/word_games/NEW_INPUTS --output data/experiments/word_games/NEW_PILOT --stage pilot
```

Source: [drafter](../../experiments/word_games/speculative_draft.py),
[worker](../../experiments/word_games/speculative_worker.py),
[launcher](../../experiments/word_games/run_speculative.py).
Raw [pilot v1](../../data/experiments/word_games/20261031_speculative_pilot_v1/),
[pilot v2](../../data/experiments/word_games/20261031_speculative_pilot_v2/), and
[download/input provenance](../../data/experiments/word_games/20261031_speculative_inputs_v1/)
are append-only. Registry status: tested pilot, **not confirmed**.
