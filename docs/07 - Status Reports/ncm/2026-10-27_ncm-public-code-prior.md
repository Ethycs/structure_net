# Interpret public code agreement against a source-blind baseline

Exploratory hypothesis `ncm-public-code-prior-v1`. The completed public TRAIN
probe found epoch2 generated-code agreement of 32.85% with matched sources and
37.25% with shuffled sources. This follow-up is specified after seeing those
results. It cannot qualify semantic accuracy or retroactively alter that probe.

Fit one empirical majority code per position from all 162,353 admitted public
TRAIN hierarchies except the 32 fixed diagnostic examples. Resolve ties by the
lowest code ID. This baseline never reads a source or runs a model. Evaluate
its exact code agreement on the same 32 examples and compare the saved BROAD
and epoch2 matched/shuffled predictions. The descriptive classification is
whether the baseline equals or exceeds epoch2 matched code agreement.

Report positional code entropy, occupied vocabulary, mode concentration and
per-level agreement. Also report model agreement only on reference positions
whose code differs from the fitted positional mode. Fix this definition before
computing the baseline; do not select a favorable subset after inspection.
Reference length statistics are descriptive. Do not equate frequent codes with
padding or entropy with semantic information without additional evidence.

Use original immutable code labels and TRAIN index bindings. Exclude all 32
diagnostic identities from fitting, check every public record and bind raw
counts, mode hierarchy and comparisons to the completed prior probe. No raw
source text, feature matrices, DEV or final-test payloads are needed. Stream
code records on CPU through the locked Pixi ncm environment, retaining counts
and indexes only. No model training, gradients, GPU use, new judgments or speed
measurement occurs. Neither current training producer changes.

First run focused CPU tests and a 256-record lifecycle pilot. Full execution
requires its successful actual receipt and exact reproduction of its counts.
Independently replay all full-corpus counts using Python Counters rather than
the producer's NumPy indexed accumulation, then replay mode selection and all
reported comparisons. Preserve append-only artifacts under
`20261027_public_code_prior_{pilot,run,audit,results}_v1`. Report and register the
result conservatively. Any error stops dependent stages and remains recorded.

Even a successful majority-code baseline is not a summarizer: independently
chosen modal codes need not form a coherent text hierarchy. It only tests
whether exact code agreement is informative about source-grounded generation.
Application semantic DEV remains 0/128; the 95% task goal is unchanged.
