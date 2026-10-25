# Engineering data: 732 training candidates and 156 held-out candidates

All 12 manifest-listed handoff payloads (38,349,085 bytes) were downloaded from
immutable lakeFS commit `ded03c25a12e56d620710d2bff334ba5beed923584783c460084eb5922a8270a`
and independently matched their SHA256 and size receipts. The Git data pointer
`1fbb00c3602c2d7fb22baf363030481df23b2a7e` was also fetched into an isolated checkout.
The complete large archive was not downloaded.

The engineering shard has 4,542 records: 3,315 raw-summary candidates, 601
hierarchy-merge candidates and 626 structural candidates. Repeated runs reduce
the raw records to 1,488 exact inputs across ten source components. The final
reservation retains 732 TRAIN pairs across eight components and 156 test pairs
across two components; 600 distinct inputs follow the below-192-token application
bypass, and 1,827 repeated raw inputs are excluded. Every retained target fits
the unchanged 128-token application limit and complete native BERT frame.

The first inventory incorrectly excluded the full build-session component
because a generic two-token turn, `try again`, also appears in the old corpus.
That inventory remains immutable. A prospective v2 amendment counts informative
public full turns of at least 32 tokens and retains conservative matching for
unknown turn lengths. Exact source/family, normalized/near-text and open cohort
checks remain unchanged. The same hash-based split seed was used. No model
outputs or quality scores informed the correction. Four v1 and five v2 focused
tests passed; inventory sessions 5519 and 95974 actually exited 0.

The checks exclude all 16,414 known core source families, 152,681 informative or
otherwise protected turn hashes and all reserved metadata; they also compare
against 192 open DEV/component sources. No protected overlap was found after
the generic-turn correction. This does not certify arbitrary paraphrase
independence. All sealed final source files remain unopened and unhashed.

Target and span hashes/character bounds were checked locally. Source and summary
database hashes bind to the frozen artifact index; their full bytes have not yet
been independently restored here. Labels remain existing, unreviewed stored
summaries, not individually accepted teacher targets or recovered exact prompts.
No new training, semantic grading or speed test has run. Historical checkpoint
LongMemEval exposure remains recorded; it is not erased by these new exclusions.

The larger index contains 96 unclassified source stores for further inspection.
The overwhelming majority of atomic-cache files are in LongMemEval-named bundles,
so the archive's raw file count must not be presented as new independent training
data. The next training plan will bind these candidates, frozen parent features,
target codes, fixed optimization budget and source-only semantic evaluation.

Evidence: `data/experiments/ncm_paper_regime/20261025_engineering_summary_inventory_v2/`.
Candidate file SHA256: `9aa656375afb489c9027aafc4804237460160fb2cc86a526ada6eb5936a5346e`.
Complete-summary semantic 95% accuracy and a speedup remain unestablished.
