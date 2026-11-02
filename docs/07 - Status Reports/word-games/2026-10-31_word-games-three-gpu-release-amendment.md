# GPU release lifecycle amendment

The first three-GPU attempt (`20261031_three_gpu_experts_v1`) unloaded
Qwen3-8B successfully, but its first telemetry sample still saw the old CUDA
allocation. The guard stopped the owned runtime before any request timing.
The allocation subsequently disappeared (5,507 MiB free), and Qwen3-8B was
successfully reloaded with identical configuration and readiness 200.

For `20261031_three_gpu_experts_v2`, wait for two consecutive one-second
samples showing at least 5,200 MiB free on RTX 2060 after unload, with a
90-second timeout. Record all samples. Keep every original experiment
parameter, speed gate, headroom threshold and restoration requirement.
This fixes startup sequencing; it does not relax a scientific gate.
