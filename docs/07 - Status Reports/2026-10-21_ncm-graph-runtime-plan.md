# Frozen NCM GPU graph follow-up

Exploratory hypothesis `llama32-3b-ncm-graph-runtime-v1`, separate from the
unchanged `llama32-3b-ncm-distillation-v1` model study. Known at registration:
the adapted generator completed1024 updates and passed64/64 development field
checks; its frozen SHA-256 is
`ff7133b2de50bc1f66ef1e490fec4c723e0fc5a5c1957107209e923b198c3e3c`.
The selected decoder/codec SHA is
`b87f7288edfc0eb1f5b9122d1139cc2e61aa2331be7d6fe53f0582993086ede2`.
The parent's final comparison is still running. Early selected-arm cells take
about0.76s/request; no completed speed comparison is available yet. One fresh
output was inspected and all tested development outputs retained final facts.
This engineering follow-up must not replace or retroactively pass the parent's
eager-runtime gate.

Capture the complete original nine-level NCM generation, native top_k1
sampling, text decoder, token argmax and finite flag in one CUDA graph. Preserve
weights, input encoding, precision, sampling code, canvas and EOS/PAD trimming.
Do not introduce a new prediction rule, cache content between requests, train
weights or select a different model. Replay receives fresh512-position padded
source IDs. Include CPU input copy, all graph work, CPU output copy and text
decoding in request time; exclude graph construction and warmups.

First run a separate systems pilot on eight development sources. Require
graph/eager equality of all48 native output IDs, finite logits, A/B/A fresh
request reset, repeated same-input parity, and peak allocated/reserved memory
below the device's capacity. CPU tests cover shape/input/output contracts.
Physical GPU0 only, one worker. Do not overlap this worker with the parent's
isolated GPU timing blocks. Preserve any failed root; engineering corrections
need a new root/version. Do not tune from quality or timing threshold misses.

If the pilot passes, run all100 of the parent's fixed held-out sources with
one excluded per-case warmup and three timed repeats. These sources are used
runtime-validation inputs, not a new independent model-quality holdout. Require
all300 graph outputs to equal the saved eager selected outputs, including native
IDs, then inherit the exact source-review decisions by text/source hash.
Compare with the parent's completed eager selected, original teacher and
factorized FP16 controls. Reuse is explicit: verify exact prompt/source/model,
software/producer fingerprints and raw rows. These nearby sequential blocks
are not randomized or simultaneous controls; time drift is a limitation.

The model-quality gate stays >=95/100 source passes and no fewer than the
original teacher, with a >=95/100 teacher positive control. Primary runtime
success requires both complete-request latency and common-Llama-output-token
rate >=2x versus original teacher. Report the same ratios versus factorized
FP16 and eager NCM, plus graph setup and memory. A quality/parity failure cannot
be rescued by speed. Classifications: systems failed; output parity failed;
inherited model quality failed; speed target unmet; frozen NCM graph prototype
meets speed and quality. One frozen model, technical repeats, `confirmed:false`.
Retain the parent's independent classification. No mobile, production95%,
general-Llama-capability or full6300/128-token application claim.

Roots `data/experiments/ncm_llama_graph/20261021_*`. Freeze/hash the graph
producer and references, audit, report both stages, register/read back and
commit only this follow-up with the parent model study.
