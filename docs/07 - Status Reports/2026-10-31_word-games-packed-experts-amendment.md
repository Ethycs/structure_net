# Use remaining VRAM: 38 of 40 expert banks on GPU

Exploratory follow-up to `word-games-three-gpu-experts-v1`, responding to
the user's 50 tok/s question. Written while the initial full 18-bank
offload replay finishes. Known: its pilot was 36.18 tok/s versus 30.01;
the full eight-bank baseline was 31.71 tok/s. Observed candidate usage is
about 8.7 GiB on RTX 3060, 4.9 GiB on RTX 2060 and 6.8 GiB on SUPER.

The native loader reports banks 18-22 occupy 2,292 MiB and bank 23 occupies
432 MiB. Move 18-22 onto the 3060 and 23 onto SUPER. Keep 0-7 on SUPER,
8-17 on RTX 2060, 24-25 on CPU and 26-39 on the 3060. Thus 38 of 40
serving banks reside on GPUs. Keep attention on 3060, unchanged precision,
context, batching, runtime and all top-8 expert contributions.

Wait for the previous campaign and Qwen3-8B restoration to finish. Freeze
a fresh producer, temporarily unload only Qwen3-8B again, and require the
two-sample release check. Run a fresh three-request 18-bank reference and
38-bank candidate. Advance the candidate to one full 36-request replay only
if valid JSON and at least 5% lower total pilot completion time. After the
full pass run an ending three-request 18-bank reference to diagnose drift.
Compare full speed to the immediately preceding full 18-bank pass and
report the temporal comparison explicitly; do not treat it as independent
replication. If pilot reference elapsed time drifts by more than 5%, do not
claim a precise speed ratio without another matched full control.

Retain 384 MiB minimum free memory on every GPU and preserve OCR. Any
allocation/headroom failure ends this follow-up without reducing precision
or context. Restore Qwen3-8B in finally and verify readiness/configuration.
Keep all earlier negative and positive outcomes, and preserve artifacts in
`20261031_packed_experts_v1`. Quality remains a separate frozen-task semantic
diagnostic, not a claim of population accuracy or lossless token outputs.
