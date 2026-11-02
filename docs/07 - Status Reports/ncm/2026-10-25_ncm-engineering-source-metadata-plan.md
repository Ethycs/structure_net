# Source-only preparation for engineering training and held-out inputs

After actual completion of the engineering v2 reservation, prepare all 732 TRAIN
and 156 engineering-test sources. Whitelist only source identity/text/provenance
fields before tokenizer access. No target text, word IDs or code IDs may reach
the source preparation interface. Use the existing frozen parent prompt, pinned
training date 23 Oct 2026, exact native Llama chat template, full GPT2 source
tokenization and character-overlap alignment. Keep every source position.

Verify first/longest original TRAIN geometry parity with the unchanged source
preparation functions before processing new inputs. Freeze source files, tokenizer
assets, candidate hashes/partitions and the source-only policy into a new root.
Save each complete prepared source plus a hash index, token counts and feature
storage estimate. CPU only: no parent weights loaded, features extracted, model
updates or judge calls. The engineering test sources never enter gradients.

This advances the real data path for a separately frozen attention-repair
training experiment; it does not change sealed final or existing DEV cohorts,
and does not replace the 95% complete-summary semantic accuracy objective.
