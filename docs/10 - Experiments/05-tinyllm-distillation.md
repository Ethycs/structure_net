# Experiment 05 — TinyLLM Distillation

**Verdict:** COMPLETE; LABEL-ONLY ARM SELECTED  
**Date:** 2026-08-16

All six NAL runs completed on the same 49,259 semantic groups. The best
label-only run was seed 17 at 55.27% balanced accuracy (58.33% accuracy). The
best Qwen-assisted balanced accuracy was 53.24%, so teacher assistance was
rejected for the frozen A interface. This confirms empirically that the
imperfect hard teacher did not improve either local head under the frozen
auxiliary-loss rule.
