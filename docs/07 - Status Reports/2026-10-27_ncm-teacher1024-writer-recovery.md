# Correct the exclusive producer writer before continuation

The v3 inventory command exited1 before raw replay or any new model request.
Its wrapper called the original freeze function, which wrote producer.json,
then attempted a second exclusive-create write to append configuration fields.
The repository writer correctly rejected that overwrite with FileExistsError.
The original producer, source snapshots, failure and actual tool receipt remain
unchanged. This is a local bookkeeping failure, not a teacher or model result.

The v4 wrapper constructs the complete configuration and fingerprint before
writing producer.json exactly once. A filesystem test uses the real exclusive
writer and verifies the resulting fingerprint and source snapshot. Re-entering
the same root still fails. All scientific controls and the prospective v3
timeout-continuation plan remain unchanged: replay438 original firsts, retain
the eight exhausted negatives, and submit only586 untouched sources after
an eight-case continuation pilot and independent replay. New roots and durable
units use v4. No existing request, summary, source or stage is replaced.
