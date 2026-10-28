# Reuse measured hashes during future source-only generation

The expanded inference pilot completed successfully at actual tool session15278,
chunk4677b7, exit0, generating both fixed probes with exact repeatability. Its
validation process had read150GB by the last observation before completion.
Repeated hashing of shared ancestors dominates this small inference stage.

For future, not-yet-started task DEV generation stages, apply the same qualified
process-local SHA cache already used in training and semantic evaluation. Every
new process reads each file in full on first access; only files at least1MiB,
older than2seconds, with unchanged device/inode/size/mtime/ctime may reuse that
locally measured digest. Never preload a receipt's claimed digest into the cache.
The original hash implementation detects mutation during a read. Reuse changes
verification work only: source-only computation, checkpoints, exact pilot replay,
data, split guard, output identities and evaluation gates remain unchanged.

The entry point `experiments.structure_net.ncm_verified_task_generation` accepts
`inference_pilot|outputs --arm existing|expanded`. Completed/live roots remain
immutable; original stage code still requires a fresh root. Bind the cache policy,
wrapper, plan, tests and CPU evidence into each new producer. Run the original
three cache contracts plus the new generation-binding contract before launch.
GPU selection and memory limits are enforced by the unchanged generation code.

This is a systems amendment, not a new model experiment or NCM speed claim.
Do not modify any current producer or reinterpret prior outcomes. Final320
remains sealed; semantic first judgments, judge routes and ordering are unchanged.
