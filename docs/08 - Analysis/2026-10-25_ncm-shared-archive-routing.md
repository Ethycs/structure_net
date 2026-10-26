# Most cached summary files are in LongMemEval-named roots

The metadata-only inventory completed with 422,402 indexed files. It found
244,688 of the 253,366 atomic-cache files in LongMemEval-named roots. The archive
therefore does not establish a quarter-million additional independent,
non-LongMemEval training pairs.

| Path routing hint | Atomic-cache files | Merge-cache files | Source databases |
| --- | ---: | ---: | ---: |
| LongMemEval-named | 244,688 | 12,735 | 37 |
| Engineering-named | 1,417 | 287 | 25 |
| Unclassified | 7,261 | 821 | 96 |
| Total | 253,366 | 13,843 | 158 |

These are physical file counts. Across the whole index, atomic caches have
229,673 unique file hashes and merge caches have 13,326. Neither count measures
independent sources, complete-summary pairs, or semantic label quality. Path
hints also cannot prove that an unclassified source is outside LongMemEval.

An independent metadata check matched all 25 engineering source-store paths to
the existing engineering export. That shard already contains 4,542 candidates:
3,315 raw-summary, 601 hierarchy-merge and 626 structural candidates. It is not
an undiscovered additional engineering corpus. Source independence and complete
task alignment still require the existing guards and label checks.

The next useful archive action would require source-provenance metadata for the
96 unclassified databases before admitting or opening their source payloads.
The current model's 1,376-pair training set remains fixed. No new payload was
downloaded, no source or summary text was opened, no teacher/judge request was
made, and no additional training pair was admitted. The sealed final sources
remain closed.

The first metadata pass incorrectly expected `source_root` directly on each
index row. It exited before creating the result. The corrected pass joins each
bundle to the verified archive catalog; the failed producer and receipt remain
preserved. The completed pass exited 0, and a separate process verified every
input hash, category totals, routing totals, and exact engineering-store coverage.

Evidence: `data/experiments/ncm_paper_regime/20261025_handoff_routing_inventory_v1/`.
Preserved failure: `20261025_handoff_routing_inventory_CPU_checks_v1/`.
Producer: `experiments/structure_net/ncm_handoff_routing_inventory.py`.
This is data-feasibility evidence, not a model-accuracy or speed result.
