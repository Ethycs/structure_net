# TinyLLM native-2048 H2 crossover result

**Status:** COMPLETED AT PREREGISTERED EARLY STOP
**Hypothesis:** `tinyllm-native-2048-h2-crossover-v1`
**Classification:** `native_long_context_model_competence_failed`
**Campaign artifact:** `data/experiments/tinyllm_native_2048_h2_crossover/20260907_registered/campaign_results.json`
**Campaign SHA-256:** `1a11581da02f819e0a842092fe205ff94c56d15beff1b552a72001d302a448c4`

## Outcome

The native 2048-context TinyLLM completed all 12,000 optimizer updates and the
exact token-matched budget of 73,728,000 target tokens. It used one physical
2048-token sequence accumulated three times per update, matching the original
checkpoint's `24 * 256 = 6,144` tokens per update. The model was initialized
from scratch with a native learned 2048-position table and did not reuse or
extend the original checkpoint.

The resulting checkpoint is:

```text
data/experiments/tinyllm_native_2048_pretrain/20260907_seed7/checkpoint_step12000.pt
SHA-256 b8cd8a75ae4daac7555dc87d7c8b8e07e41362bba9ef826dc8b8c9ed730edcd6
```

Training took approximately 3,074 seconds on the RTX 2060 SUPER. The final
native-2048 validation loss was `5.2507434`. Training remained finite and did
not collapse.

## Model-competence gate

Both checkpoints were evaluated on the exact same 384 frozen length-256
validation windows. The native checkpoint was additionally evaluated on 48
frozen length-2048 windows with the same total validation-token count.

| Checkpoint and window | Token-mean loss |
| --- | ---: |
| original native-256 model at 256 | `4.8932135` |
| new native-2048 model at 256 | `5.3189554` |
| new native-2048 model at 2048 | `5.2507434` |

The matched short-context relative degradation was

```text
(5.3189554 - 4.8932135) / 4.8932135 = 0.0870066
```

or 8.70%, exceeding the preregistered maximum of 5%. The competence gate
therefore failed.

## Consequence for the H2 question

The preregistered decision order places model competence before H2 evaluation.
Accordingly, no H2 cell was executed and no claim is made about representation
or cost crossover for this checkpoint. Running the decomposition after this
failure could still produce an exploratory diagnostic, but it could not rescue
or alter the registered classification: unusually compressible attention might
reflect the checkpoint's short-context quality deficit.

This outcome does not refute A7. It establishes that matching architecture,
optimizer-update count, and total token count while increasing native context
from 256 to 2048 did not reproduce the original checkpoint's short-context
competence within the chosen 5% engineering margin.

## Systems incident and recovery

The final checkpoint was written after update 12,000. The original process then
ran out of GPU memory while beginning the post-training competence evaluation,
because an unrelated resident process reduced available memory. A fresh process
reloaded the unchanged final checkpoint and completed the identical locked
evaluation. No optimizer update was repeated and no weight was modified. The
recovered native-2048 loss exactly reproduced the final scheduled validation
loss from the training process.

## Next defensible options

The registered campaign is closed. A new study would be needed for either:

1. a native-2048 training protocol with a larger optimization/token budget or
   a context-length curriculum, followed by the same competence gate; or
2. an explicitly exploratory H2 decomposition of this failed-gate checkpoint,
   labeled as non-rescuing diagnostic evidence.

One trained checkpoint also leaves trained-model variability unmeasured.
