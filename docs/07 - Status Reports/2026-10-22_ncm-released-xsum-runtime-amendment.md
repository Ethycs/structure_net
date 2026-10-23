# Forward the existing NCM dependency path to the author subprocess

Date: 2026-10-22. The first released-XSum pilot stopped in the unchanged author
script at `from omegaconf import OmegaConf`, before model loading or generation.
The local loader normally inserts `ncm_llama/dependencies` into its own Python
search path; that insertion was not inherited by the separate author process.
Preserve the failed v1 producer, source copies, command, log and failure receipt.

The fresh v2 pilot/control explicitly forwards that existing dependency path
via subprocess PYTHONPATH. No package installation, vendor-source change,
precision change, data change, checkpoint change or endpoint change. Keep the
verified v1 download immutable; the four-case pilot and32-case control receive
fresh v2 roots. Full control still requires exact author/local parity and a
matching passed v2 pilot. This is infrastructure recovery, not a semantic retry.
