# Combined prepared NCM data

The prepared catalogs combine into **208,631 distinct source-summary pairs from
202,340 distinct source texts**. Classification: `prepared_catalog_union_indexed`.
This is a data inventory; training on the combined mixture has not started.

Result: `data/experiments/ncm_paper_regime/20261028_combined_prepared_pair_inventory_v1/result.json`.
SHA256: `a10aacce5d41a80ecc9fa1119b3623d527a43d8e01e99d2de9467517356f35cd`.

| Prepared population | Pairs | Distinct source texts |
|---|---:|---:|
| XSum public TRAIN | 162,353 | 162,353 |
| Recovered conversations | 43,894 | 38,484 |
| Reviewed anchors | 2,384 | 2,384 |
| Combined content union | **208,631** | **202,340** |

There are zero duplicate exact source-text/target-text pairs across the union.
The conversation and anchor sets share 881 exact source texts with different
targets; their original records and source groups are retained. There is zero
exact source-text overlap between XSum and either application population.
These are exact-content counts, not independent-conversation counts or proof
of absence of near duplicates.

Every union row points back to its immutable original record. Recovered
role/date-conditioned source pointers remain attached. No source or target was
truncated, no new split was assigned, and no model or judge was called. An
independent pass read all 208,631 index rows and reproduced the pair, source,
domain, overlap and original TRAIN-split counts. Three contract tests passed.
The execution completed with actual terminal chunk `b75834`, exit zero; the
independent readback completed with `cab730`, exit zero.

The largest prepared paired population is XSum: 162,353 eligible pairs from
204,045 downloaded training rows. The latest model's public epoch2 initializer
already received XSum pretraining. Its latest 6,416-update run mixed the 43,894
recovered targets with 2,384 anchors. A future mixed-domain run would retain both
news and conversational supervision during continued training; it would not
constitute first exposure to all 208,631 pairs.

The shared archive has a larger candidate supply: 1,069,574 ordinary pairs,
including 259,636 that fit the application's content-length limits. That total
contains the recovered subset and cannot be added to it. The archive candidates
still require the applicable provenance, source-group, native-codec and label
checks. The 99,367-frame decoder corpus is target-only reconstruction data and
does not add 99,367 source-summary pairs.

The combined index is not yet admitted for a new training experiment. Its
union-wide near-duplicate split audit and training mixture remain to be defined
and executed; individual catalog admissions remain unchanged. Complete current
candidate evaluations before selecting another training intervention. Final320
remains sealed, and no accuracy or speed improvement is established here.

Index: `data/experiments/ncm_paper_regime/20261028_combined_prepared_pair_inventory_v1/pairs_index.jsonl`.
Its SHA256 is `85bb472e5ea553d11f91709a04623ed1baae4f08b2e11b365663450f4d942533`.
The separate replay is under
`20261028_combined_prepared_pair_inventory_readback_v1`.
