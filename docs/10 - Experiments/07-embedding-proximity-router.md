# Experiment 07 — Embedding-Proximity Router

**Verdict:** COMPLETE; ACCURACY FLOOR MET, LIMITED SAVINGS  
**Date:** 2026-08-16

The selected partition-2 configuration was unweighted 15-NN with `τ=0.80`.
It achieved 78.14% realized accuracy versus 79.09% for always-C, meeting the
locked one-percentage-point floor. It escalated 99.45% of examples, had mean
relative cost 150.13 versus 156.86 for C, ECE 0.049, and routing regret 115.33.

Thus embedding proximity was valid but weakly economical for these distilled
heads; the result is not presented as a large routing win.
