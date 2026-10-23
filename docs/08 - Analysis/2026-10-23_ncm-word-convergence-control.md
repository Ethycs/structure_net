# Word convergence diagnosis: four passes were not qualified

Date: 2026-10-23. Hypothesis `ncm-word-convergence-control-v1`.
Classification `causal_fixed_point_reached_quality_unmeasured`. Aggregate `data/experiments/ncm_paper_regime/20261023_native_word_convergence_results_v1/campaign_results.json`,
SHA256 `d138500ba1d045e7eda4be24f8034a5c35212fa7937991051a29d48ce9f74217`.

The original failed 2,048-update model was restored unchanged. On the first
four fixed training summaries, source-only predicted coarse plans remained
fixed while the word stage ran 128 parallel updates. A 129th pass verified
exact full-frame fixed points on every case. No oracle plan or previous gold
words entered generation. This is a decoding diagnosis, not a new summary
accuracy or throughput result. The original nine-pass model remains 0/64 on
source-disjoint development summaries.

| Word updates | Training token recovery, not summary accuracy |
| --- | --- |
| 4 | 32/346 (9.25%) |
| 8 | 44/346 (12.72%) |
| 16 | 64/346 (18.50%) |
| 32 | 86/346 (24.86%) |
| 64 | 82/346 (23.70%) |
| 128 | 81/346 (23.41%) |

On the first fixed case, four updates produced:

> A suggested conference packing list clothing clothing-–-–––22--,2222,,,-,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, and and and and and and and and,, and and and and and and and and airline airline airline airline airline

At 128 updates, the same unchanged model produced:

> A suggested conference packing list covers clothing (2–3 tops, 2–3 bottoms, 1–2 jackets or blazers, walking shoes, and dress shoes for evening events), toiletries, electronics including a laptop and phone, conference materials such as business cards and a notebook, and miscellaneous items including snacks, a water bottle, a first-aid kit, and travel documents. Checking the conference website, weather forecast, and airline baggage restrictions is advised.

These texts have no new semantic judgments. Convergence and training-target
matches do not establish faithful, salient summaries or generalization.
Increasing the iteration count can remove a parallel compute advantage; it
does not demonstrate acceleration. Raw states, configured stop IDs, exact
four-pass replay, token matches and prefix traces were retained and replayed.
Original body, books, new weights and selected cache remained unchanged.

The reference JacobiForcing implementation uses iterative block decoding and
trains clean/noisy trajectories. Our fixed whole-frame four-update shortcut
was never qualified to converge. The neighboring paired-loss control improved
training recovery but failed its 90% oracle capacity gate; it was not scaled.
Before more large training, distinguish decoder convergence from the quality
of the final fixed point and from a learned codec's ability to skip sequential
dependencies. Any adaptive/block decoding would be a separately registered
hybrid study, with its full compute costs and complete unseen-summary grading.
It cannot retroactively rescue the failed fixed-nine-pass NCM endpoint.
