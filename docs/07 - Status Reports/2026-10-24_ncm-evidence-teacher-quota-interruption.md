# Bulk teacher interruption after confirmed reviewer quota exhaustion

2026-10-24,22:13UTC. The immutable bulk teacher producer had completed roughly
1,000new cases when inspection found repeated Sonnet responses containing a
weekly-limit notice instead of JSON. An actual200response identifies the
unchanged `claude_code/claude-sonnet-4-6` model and states resetOct26at4pm
America/Los_Angeles. Earlier requests also hit repeated180second read timeouts.
The model-list endpoint's200response did not establish inference availability.

This is an EXTERNALresource interruption, not a semantic quality failure and
not grounds to change the frozen recipe, discard invalid cases or retry valid
failures. Preserve ALLoriginal first labels, schemas, exhausted failures and
raw draft/review attempts, including qualified64and8pilotfirsts. Preserve the
full reserved10000source population and all unattempted sources. No new teacher
model, prompt, repair, semantic reroll or target clipping is authorized by this
interruption record.

Prevent thousands of predictable quota-message failures: sendSIGINT ONLYto the
verified bulk generate child while leaving its supervisor and other jobs alone.
The ThreadPoolExecutor context waits for existing candidate tasks to finish;
raw attempt writes remain intact, although completion of the parent wait loop
may be interrupted before its per-case JSON save. Preserve those raw requests
and recover their first-valid candidate deterministically under a NEW recovery
protocol; never regenerate an observed valid response. Record exactPID/command,
state snapshot, source hashes and actual child/whole-tool exits. No silent
restart or overwrite of the old producer.

Any later recovery must first audit all saved case files and pending-task raw
attempts, preserve full prior invalid/first chronology and use fresh roots.
Unattempted sources may resume only after the SAMEqualified reviewer is
available or a separately qualified prospective teacher protocol exists. An
exhausted original stage is not automatically granted three more attempts.
All attempted/invalid sources remain in the10000label denominator. This record
neither changes canonical3222semantic first judgments nor opens final320.

This is not a pause of the user goal. Original-body feature extraction onGPU0
and the new source-only FP16candidate onGPU2 continue independently. Student
quality still needs real held-out semantic evaluation; no95%claim follows.
