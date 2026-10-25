# Correct generic-turn overexclusion in the engineering reservation

The completed v1 inventory retained 223 TRAIN and 156 test candidates but
excluded the full build-session component because five duplicate records contain
the exact two-token turn `try again`. No source-family, informative source text,
or open-heldout overlap was found for that component. Preserve v1 unchanged.

In a new v2 producer/root, count shared full turns as corpus-overlap evidence
only when their pinned public metadata has at least 32 cl100k tokens. Remove
known public turns below that threshold from inherited turn sets too; unknown
turn lengths remain conservative matches. This follows the distinction between
informative content and generic bypass text already used for component joins.
Family, source hash, normalized text and 32/80 five-gram checks remain intact.
Recompute the whole shard, all exclusions, deduplication and hash-based component
split with the same seed. No outputs or quality scores informed this correction.

For the text-only open-cohort comparison, verify the pinned source-case file and
its original successful worker receipt. Do not reload/hash unrelated GPU feature
producers: this inventory never uses those features. Include the v1 result and
actual completion receipt, this amendment, the new source/tests and all metadata
and shard hashes in the new producer. All original limits and target bytes remain
unchanged. No model training, semantic requests, or sealed final text access.
