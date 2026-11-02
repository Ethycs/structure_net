# MoE placement follow-up: experts only on the second GPU

Date: 2026-10-31. Amendment to `word-games-moe-placement-v1`, written after
the first full `gpu8` pass, before any experts-only outcome. The first full
whole-block pass took 192.04 seconds versus controls 185.60/185.39; decode was
47.56 versus 41.78 tok/s, but prefill grew to 86.96 from about 68.94 seconds.
The second whole-block pass and cache comparison remain scheduled and visible.

Test whether keeping attention, recurrent state, shared/dense work and the
remaining model on RTX 3060 avoids this prefill penalty while routed expert
banks for blocks 0-7 reside on RTX 2060 SUPER. Expert weights for blocks 8-25
remain on CPU. Keep all other controls identical: Q4_K_M, native b11960,
Flash Attention, Q8 KV, 12 unpinned CPU threads, batch/ubatch 512, context 8192,
temperature 0, no thinking/speculation/prompt reuse.

Use both devices but assign all layers to CUDA0 with layer split 1,0; explicitly
override only the three routed-expert projections for blocks 0-7 to CUDA1.
Verify all 42 generic layer assignments on CUDA0, exact GPU expert overrides
0-7, exact host expert overrides 8-25, and actual CUDA1 model allocation.
This is a different placement arm (`experts8`), not a correction to `gpu8`.

After the ongoing full campaign completes, freeze a new producer with this
arm and both protocol documents. Run a new three-request pilot, with the
same memory/service/JSON gates. Advance to two full 36-request passes only if
the pilot completion time is at least 5% lower than the mean completion time
for those same three requests extracted from the two initial full controls.
This selection remains exploratory, especially if output lengths differ.
Otherwise report the negative pilot and do not run a full arm. Retain all
prior outcomes. Run the ending full control after this follow-up. The
original exact parsed-output parity and speed gate remain unchanged.

Fresh roots `20261031_moe_placement_experts_pilot_v1` and, conditionally,
`20261031_moe_placement_experts_full_v1`. CPU affinity stays at its original
unpinned setting: full pinned passes were about 6% slower. No model training,
precision change, selector repair or service unloading is permitted by this
amendment.
