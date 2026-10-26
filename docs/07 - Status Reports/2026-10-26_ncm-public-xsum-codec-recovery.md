# Public XSum codec loader correction

Execution amendment to `ncm-public-xsum-codec-v1`, before any new reconstruction
outputs. The first pilot exited 1 while checking state keys, before its output
root was created and before an encoder or decoder forward. The implementation
looked under `codec.`; the complete hybrid checkpoint stores all 136 codec
tensors under `concepts.codec.`. A CPU extraction at the correct path produced
the originally specified codec hash
`eb962706f10c693871f9c6df007e9fd3e23004c91c6258f435cdb09abf7fb366`.

Preserve the failed terminal receipt and both CPU diagnostics under
`20261026_public_xsum_codec_pilot_launch_failure_v1`. Keep the original producing
source and completed preparation unchanged. A recovery entry point extracts
the exact prefix, requires the same complete key set and tensor hash, and runs
fresh `pilot_v2`, `eval_v2`, and `results_v2` roots. The original preparation,
selected cases, controls, checkpoints, metrics, gates, precision, and device
remain unchanged. The finalizer independently extracts and hashes the same
checkpoint tensor dictionary. This is an implementation correction, not a
change to the hypothesis or a retry of an unfavorable scientific result.
