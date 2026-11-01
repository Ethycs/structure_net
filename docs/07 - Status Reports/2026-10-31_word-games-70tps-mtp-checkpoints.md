# WORD_GAMES: remove unused prompt checkpoints from native MTP

Exploratory follow-up, registered before executing these arms. The expanded
GPU pilots fit all weights and show actual draft acceptance (105/118 with one
draft token, 146/190 with two). Decode reaches 94.6–95.8 tok/s but full-request
rates of 53.5–56.5 trail non-speculative pilots at 62.6–63.3.

Source inspection of the frozen b11960 native server identifies independent
state mechanisms. `server-context.cpp` around lines 4391–4587 gates prompt
context checkpoints on `n_ctx_checkpoints`; the draft/verification paths around
3918–3960 and 4945–5007 use `slot.spec_ckpt`, sequence removal and `n_rs_seq`
independently. Setting `--ctx-checkpoints 0` removes unused prompt snapshots
without disabling native speculative rollback or target verification.

After the expanded-GPU stage closes and restores services, amend the controller
to permit this combination. Preserve previous source snapshots and raw data.
Use a new root `20261031_70tps_mtp_nocp_v1`; hash the preceding campaign, this
plan, runtime and all source files. Keep both resident models gracefully
unloaded only during the measured lease, and restore both afterward.

Fixed pilot conditions (three frozen requests):

- `control`: expanded 8/14/18 whole-layer placement, microbatch 1024, batch 2048,
  native pipeline, no speculation, no prompt checkpoints.
- `mtp1_nocp512`: same placement, one draft token, microbatch 512/batch 1024.
- `mtp2_nocp512`: two draft tokens, microbatch 512/batch 1024.
- `mtp1_nocp1024`: one draft token, microbatch 1024/batch 2048.
- `mtp2_nocp1024`: two draft tokens, microbatch 1024/batch 2048.

All MTP head weights remain on 3060; keep original Q4_K_M weights, Q8 caches,
Flash Attention, native rollback states, context 8192, greedy JSON sampling,
seed 73, no cross-request caching and the 384 MiB memory guard. Native source
is not patched. Require actual drafting, configured `n_rs_seq` and zero created
prompt context checkpoints. No synthetic acceptance or altered verification.

Advance the fastest two candidates at least 98% of the matched control pilot
rate; two full 36-request passes, control/candidates then reversed. If none
qualify, run only the control twice and retain the rejected pilots. Report full
request, byte and decode rates, acceptance, memory, CPU, state checks and manual
output review. This remains exploratory, with no independent semantic judge
or population accuracy claim. Select a repeated, fitting configuration for
actual Triton load/infer/unload/reload validation afterward.
