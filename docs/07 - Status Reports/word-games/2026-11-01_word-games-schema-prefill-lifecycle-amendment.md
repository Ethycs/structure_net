# Public prefix lifecycle correction

Before any full speed replay, pilot v1 stopped before changing deployment:
an absolute source path caused a snapshot SameFileError. Pilot v2 loaded the
candidate, then rejected native `n_predict=0` because the sampled-token counter
was one. The original serving repository was restored and loaded successfully.

Pinned runtime source checks its generation budget after the first sampled
token. The native log reports prompt/cache length 466 and one sampled token,
with zero decode time. The sampled token has not been fed back into sequence
state. The next pilot uses the same native request and verifies BOTH the
completion's cached-token count and saved sequence length equal the exact
public token list. The sampled response is recorded but never fed back or
returned to an application. No generated token may enter a saved snapshot.
This refines the original plan's zero-generation assertion to the relevant
state invariant; no speed or quality outcome has been inspected.

Retry lifecycle in a fresh `20261101_schema_prefill_pilot_v3` artifact root.
The fixed full replay, quality gates and baseline restoration remain unchanged.

Pilot v3 successfully built the two snapshots, then stopped before inference
because the host evidence collector could not traverse their container-owned
0700 directory. Use 0755 for this public-only directory and retry lifecycle in
`20261101_schema_prefill_pilot_v4`. Neither failure supplied speed/quality data.
