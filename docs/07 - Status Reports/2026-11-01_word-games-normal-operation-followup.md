# Ordinary operation coverage after the first live workflow

Known outcome before this follow-up: the four independent cases passed. The
persistent workflow passed four events, then the receiver returned an empty
occupied-slot list after Clock was cancelled, omitting Heron still in N1.
Sender operations and actual backend state were correct for all nine executed
events. The two remaining dependent events were not run and stay unexecuted.

Without changing model, prompts, policy or the original result, test three
independent ordinary fixtures under `word-games-normal-operation-followup-v1`:

1. Reserve an object by its current title only: rename the existing permitted
   Clock record to Copper Timekeeper, then ask to reserve Copper Timekeeper in
   North N2 through 2027-01-10. No source ID is supplied in the request. Record
   whether search/read retrieval actually occurs and whether identity binds.
2. Move an already placed Heron from North N1 to N2, keeping its date.
3. Return the only placed Heron from North N2 to home custody.

All fixtures and expected operations are deterministic, valid under the same
policy, and host-only. Model-selected metaphor is freshly obtained from the
unchanged public task prompt. Stop on invalid selector or transport failure;
retain ordinary model failures without retries. Reuse the same separate sender,
backend, receiver and disclosure grading. These additional probes establish
coverage and do not rescue or alter the failed persistent-workflow result.

Use a new append-only root:
`data/experiments/word_games/20261101_normal_operation_followup_v1`.
Freeze source/configuration and retain every live retrieval and state mutation.
No serving configuration changes or new speed claims.
