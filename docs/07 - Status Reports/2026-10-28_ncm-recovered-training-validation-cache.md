# Reuse measured file hashes within future recovered-training stages

The full40,890-source disk replay has completed successfully. The original
complete-data preparation is currently running and remains required. Its v1
files, source, configuration and receipts are unchanged. No new recovered model
pilot, training or semantic outcome has been observed.

The existing verifier hashes each large dependency, then discards its guarded
in-process cache before checking the same dependency map again. The packed
source payload alone is80.45GB. Future stages will keep the existing hash cache
within their own process. Every first access still computes SHA256 from the
complete bytes. Cache hits require unchanged device, inode, size, modification
and change timestamps; files smaller than1MiB or changed within2seconds are
always rehashed. Hashing retains its open-descriptor and replacement checks.
No saved or declared digest is inserted into the cache as a measured digest.
Each new process, including each independent audit, starts with an empty cache.
Per-record compressed-payload and reconstructed-FP32 hashes are unchanged.

This removes redundant verification reads; it does not defer the first complete
corpus hash, omit a data join, alter any model/optimizer computation, or establish
an inference-speed improvement. Test actual read counts and mandatory first
reads in independent verifier instances, same-size mutation with restored mtime,
inode replacement, and small/recent-file rehashing before use.

Keep the original preparation root. Use fresh v2 pilot/pilot_audit/train/audit
roots and binary storage `training_v2`, with the validation policy, new source,
tests and this amendment bound into every producer fingerprint. Reuse all13
original training-contract tests and their existing CPU receipt; additionally
require the3 cache-contract tests. Preserve the unchanged4 epochs,6,416 updates,
initial checkpoint, data, seeds, precision, memory bounds and semantic gates.
The full-population CUDA lifecycle and its independent audit still precede
training. No v1 fitting attempt has run or is replaced by this amendment.
