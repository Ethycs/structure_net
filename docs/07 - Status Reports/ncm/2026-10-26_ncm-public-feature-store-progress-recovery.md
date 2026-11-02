# Preserve failed feature pilot and correct append-only progress logging

The first feature-store pilot exited 1 because its second progress write used
the same `progress.json` path with the shared exclusive-create JSON writer.
The failure occurred after one reported stored case and eight repeated GPU
extractions; no quality or full-pilot result was produced. Preserve the whole
v1 evidence root, frozen producing sources, partial feature shards and actual
tool receipt (session6476, chunk67f7e1, exit1).

Change only progress filenames to include the completed case count, and use
fresh v2 evidence and storage roots. Retain the original 16 selected probes,
original model/precision, four encoding threads, eight pending matrices,
8.6 GB GPU bound, all byte equality gates and full-population plan. The v1
source snapshot remains in its failed root; v2 freezes the corrected producer.
The focused regression checks progress path uniqueness under the actual
exclusive-create writer.
