# Qwen2.5-3B: hidden-feature drift after the thirteenth block

The completed diagnostic shows that entrance feature alignment reduced downstream representation drift, but **did not recover task accuracy**: raw, aligned and selected candidates each scored **46/64**, versus the accepted twelve-block parent’s **50/64**. This is a measurement on the existing development cases, not evidence that a final-feature loss will repair the model. No weights were trained or selected in this diagnostic; `confirmed=false`.

Root: `data/experiments/jspace_3b_hidden_feature/20261115_v1`. Fingerprint: `51225002a2e9993131b90d9dae0e525a271ccfef04ff1188e9c8dbc434786512`. The parent is accepted cut 24; students are cut 23’s raw initializer, entrance-aligned state and selected epoch 0. All inherited coordinates use their actual stored bases and bridges.

The packed parent’s inference and input-autograd paths matched **exactly** at all twelve inherited layers, final output and choice logits. The unpacked parent differed from packed arithmetic: final answer-coordinate relative RMSE was **0.961%**, maximum choice-logit difference 0.25, and one formerly correct decision changed, yielding 49/64. This separates the measured arithmetic discrepancy from the larger student drift.

Answer-position coordinate relative RMSE against the packed parent:

| Inherited layer | Unpacked parent | Raw student | Aligned/selected student |
|---|---:|---:|---:|
| 24 | 0.438% | 33.773% | 12.179% |
| 29 | 1.304% | 32.009% | 21.011% |
| 35, final compact coordinates | 0.961% | 13.208% | 8.900% |

Metrics first average each case’s squared relative error, then take the square root. They cover 64 cases and 8,985 valid tokens; answer metrics use the last valid token. Final-coordinate error across all valid tokens fell from 7.945% to 5.191%. Both raw and selected students gained two answers and lost six against the parent; selected predictions changed on eight cases. Smaller feature error therefore did not imply a higher correct count here.

The original-native target is substantially different from the accepted parent: parent coordinates differ from projected native final states by **3.0466 answer relative RMSE**, normalized by each case's projected-native target energy. This feature discrepancy is not a task-error rate, and the two targets are not interchangeable. Stored BF16 projection/reconstruction also has a nonzero roundtrip floor; actual parent exit-coordinate roundtrip error was 0.247%. Native projection uses the model’s existing coordinate map, not a fitted inverse.

The selected checkpoint is the aligned epoch 0, so their traces coincide. Joint epoch 5’s weights were not retained: this diagnostic cannot reconstruct a joint-training trajectory or identify why its task count differed. The next controlled comparison can distinguish adding accepted-parent final-feature matching from changing to original-native feature/KL targets; neither benefit is established.

Separately captured training targets contain exactly **512 training-only final-answer vectors**, shape `[512,1024]`, stored as BF16 bits. No development rows enter that bank or its energy statistics. Held-out evaluation and throughput were not run.

Audit verified all 11 manifest artifact hashes, 95 live/archive source pairs, 514 input identities, row alignment and aggregates for 77 feature comparisons, and disjoint training/development identities. Evidence hashes:

- `run/result.json`: `d203042c5c7b0101a821ba6d38dadb63fe492eae3186d4d37ba4f9f21a1b904f`.
- `run/manifest.json`: `e7781dad96944f93640270bae25fa37000114838f9a7e46f9caf7e1e21cee135`.
- `run/parent_training_final_answers.npy`: `72cfe1da22f6dc39d2981e4b7f1dc79cb7fcf6bd47d022f8623e123374988876`.
