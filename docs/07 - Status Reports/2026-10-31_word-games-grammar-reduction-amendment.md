# Cheaper greedy preview follow-up

Exploratory implementation amendment to `word-games-grammar-on-demand-v1`,
written after the first full pair in `20261031_grammar_replay_v1` completed.
That pair has exact output equality and lower raw-vocabulary transfers, but
candidate throughput is 47.446 versus 50.035 tok/s. The second pair is still
running and will be retained regardless of its result. Do not run competing
CPU builds or GPU tests during those timings.

The next implementation changes two plausible overhead sources together:

- Find two winners per 128-token group and then two global winners, using
  existing backend operations. Every global top-two candidate is represented
  in the intermediate set. Retain the tie/non-finite/full-logit fallback gates.
  Use the original global operation for unsupported sizes; FP32 index
  arithmetic is restricted to exact integers at or below 2^24.
- Let the context own the fixed, stateless preview sampler so a new request's
  CPU grammar does not force rebuilding an identical backend sampling graph.
  Ordinary samplers retain their existing ownership and lifecycle.

This jointly tests the revised implementation, not the individual causal
contribution of either change. Preserve the first implementation and evidence.
Build into a fresh root using verified preceding source/build inputs, original
patch-1 control binaries and unchanged CPU/CUDA backends. Add lifecycle tests
for replacing/freeing request samplers while a context remains live, plus
ties across groups. Repeat the original native correctness gates, pilot and
two reversed-order full pairs under a new artifact root. No live endpoint
promotion follows from the experiment.
