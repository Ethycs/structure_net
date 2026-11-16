# Phi-4 pilot: data and execution readiness

Date: 2026-11-15. This is a preparation report, not a compression result.

The new Phi data helper is committed as `f3bce41`. Its 16 focused CPU tests
passed in 5.79 seconds, including actual local tokenizer behavior, source and
option-label identity, response-prefix/EOS matching, balanced pilot selection,
saved-artifact readback, tamper rejection and refusal to decode sealed files.
An independent source review found no blocking issue.

A separate read-only pass retokenized all 4,864 existing broad-recovery cases:
4,608 TRAIN and 256 DEV. All fit the declared limits without rejection or
truncation. The pilot retains the first 32 TRAIN and 16 DEV cases per family
in source order: 128 TRAIN and 64 DEV across ARC-Challenge, BoolQ, HellaSwag
and WinoGrande. The pilot has 24,935 prompt tokens, with a maximum of 331
against the 384-token prompt cap. The complete-sequence cap is 416.

Raw questions and existing option permutations were preserved. Phi answer
tokens are A/B/C/D = 32/33/34/35; the actual number of choices controls scoring,
including the original three-choice ARC case. Family balancing does not imply
that the small pilot's answer labels are perfectly balanced; those counts are
retained in the prepared plan.

`pilot_items.json` contains prompt tokens, four content/answer diagnostic
positions and contiguous offsets. Its matching `pilot_responses.jsonl` contains
the gold answer letter and actual Phi chat end token 100265 in exactly the same
case order. The independently rendered complete chat must extend the exact
generation prefix. These two-token gold responses are compatible with complete
response masking, but are not explanations, teacher-generated conversations or
evidence of general language-model recovery.

The available checkpoint is `/data/models/Phi-4-GPTQ-int4`. Its quantized-source
identity remains explicit. The bounded decoder and compact block use native
Phi fused QKV/SwiGLU geometry and a separate, untied output head. BF16 execution
of decoded int4-source weights must not be called the original unquantized
model, nor a storage improvement over the existing int4 files.

The controller readback during preparation showed the RTX3060 and RTX2060
SUPER free of experiments, while the separate Qwen all-at-once experiment
retained its ready RTX2060 lease. This is an observation, not a reservation.
The proposed native launch reserves only the RTX3060 with 11,264 MiB and caps
its worker allocator at 11,008 MiB. A thin launcher records one attempt without
automatic retries; completed scientific artifacts require the native
producer's separate validation.

No Phi GPU launch, native accuracy, recovered checkpoint, held-out assessment
or speed result is reported here. The preregistered first experiment compresses
only block 39 of 40. Native prompt/reference capture, calibration, recovery and
fresh selected-checkpoint verification remain required before describing that
pilot as successful.
