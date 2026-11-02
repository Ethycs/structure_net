# Information theory options for the NCM summarizer

The present evidence favors improving reliable prediction and recovery of facts
before enlarging the code hierarchy. This is an exploratory interpretation of the
failed full732 checkpoint, not a successful quality experiment. The generated-code
decoder study is a separate fixed run and remains unchanged by these observations.

## Measured code usage

The hierarchy has 641 indices, each with 8,192 choices: 8,333 fixed-width bits.
Its 128 native word IDs need at most 1,920 fixed-width bits. These are nominal
index budgets, not empirical entropy, parameter memory, compute or evidence of
semantic sufficiency. A large hierarchy can still allocate information poorly.

Across the failed parent's 128 public DEV cases, the root chooses code266 in
112 cases (87.5%), compared with387/732 TRAIN predictions (52.9%). DEV uses eight
distinct root codes, and root empirical entropy is0.784bits. TRAIN uses171 root
codes with3.949bits. All128 DEV complete hierarchies remain distinct. Twelve
positions in the24-position level are constant across DEV. Different sample
counts, domain composition and padding prevent interpreting these marginal
statistics as mutual information or proof of codebook collapse.

The independent number audit found278/941 matching digit-bearing TRAIN tokens
with generated codes, versus941/941 with correct codes. Other word tokens in
the same303 cases matched29.8%, close to29.5% for digit-bearing tokens. This is
consistent with broader prediction/reconstruction failure, rather than a measured
number-specific representation ceiling. Unseen-number reconstruction is untested.

Raw evidence: `data/experiments/ncm_paper_regime/20261025_ncm_information_diagnostic_v1/result.json`.
Replay: `pixi run --locked -e ncm python -m experiments.structure_net.ncm_information_diagnostic --replay data/experiments/ncm_paper_regime/20261025_ncm_information_diagnostic_v1/result.json`.

## Priorities

1. Train the reconstruction path on the actual prediction-error distribution.
   The active decoder experiment does this with fixed generated codes. Random
   code corruption does not necessarily approximate these structured errors.
2. Add an exact source-span channel for quantities, names and units, conditioned
   on their role in the summary. Copying a source number alone does not prove it
   is attached to the right fact. This changes the available information at the
   output rather than relying solely on coarse codes.
3. Optimize a distortion measure that reflects factual damage. Quantity/unit
   binding, important omissions and contradictions should matter more than
   innocuous paraphrase. Train with differentiable supervision where possible;
   the independent complete-summary semantic judge remains the quality endpoint.
4. Diagnose useful code information with matched source/code interventions and
   reconstruction controls. Greater marginal entropy can be useless noise.
   Consider codebook changes only after showing an unseen reconstruction or
   utilization bottleneck. Increasing entropy blindly could worsen predictions.
5. Evaluate uncertainty-preserving decoding separately: a small set of candidate
   codes or a trained soft representation may reduce early commitment errors,
   but mixtures can blur discrete numerical facts and extra candidates cost work.

The data-processing inequality limits any correction layer receiving only a
fixed code: it cannot add source information absent from that code. It can still
learn a better decoder. Our existing word decoder also attends to source features,
so the relevant representation includes that side information; exact source-token
access would add a different route for literal details.

Residual VQ and straight-through word-gradient losses have already appeared in
earlier local experiments. They are not untried solutions. Error-correcting codes
protect against specified corruption channels; a model systematically selecting
the wrong fact is not automatically such a channel. Entropy coding could reduce
stored code bytes but does not itself improve summary quality or token throughput.

## Literature and scope

- [Lossy Compression for Lossless Prediction](https://arxiv.org/abs/2106.10800)
  motivates compression that preserves task-relevant information. Its image
  results do not establish a gain for this NCM or its summarization task.
- [Pointer-generator summarization](https://aclanthology.org/P17-1099/) provides
  a source-copy mechanism alongside generation.
- [SQ-VAE](https://arxiv.org/abs/2205.07547) and
  [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505) offer alternatives
  when quantizer learning/utilization is a demonstrated limitation. Adopting
  either would require a new codec and generator compatibility study.
- [Gumbel-Softmax](https://arxiv.org/abs/1611.01144) supplies a differentiable
  categorical relaxation; that estimator alone does not solve factual grounding.
- [Information theory lecture notes](https://web.stanford.edu/class/stats311/lecture-notes.pdf)
  state the data-processing constraint underlying the decoder argument.

Quality remains the first constraint. Rate, model size and measured compute can
be optimized after establishing the original95% complete-summary task criterion.

## Conditional coding and useful refinement

Let X be source text, S the source information accessible to the decoder, Z the
predicted codes, and Y the desired facts in the summary. The useful uncertainty
to reduce is H(Y | Z,S), not simply the marginal entropy H(Z). Our decoder already
has source-feature attention. This means a code can identify which fact to express
while a separate source-span route supplies its exact spelling, number and unit.
An incorrect span or role still produces an incorrect fact; copying needs its own
supervision and end-to-end evaluation.

[Wyner and Ziv](https://web.mit.edu/6.962/www/www_fall_2001/kusuma/wynerziv.pdf)
formalize source coding with decoder side information. This is a design analogy,
not a claim that their asymptotic rate bound applies to our finite neural system:
our predictor also sees the source, and runtime compute is a separate constraint.
For this application, an appropriate engineering objective is factual distortion
plus weighted code rate and measured compute. Entropy coding lowers storage rate;
it does not remove a forward pass or reduce a fixed number of predicted positions.

Another candidate is to train later code levels to repair specific remaining
factual errors and allow early exit when a separately calibrated check succeeds.
[Successive refinement](https://isl.stanford.edu/~cover/papers/paper94.pdf)
explains why a hierarchy is not automatically an efficient progressive code:
some source/distortion combinations are not successively refinable. The existing
residual VQ structure alone therefore does not justify a refinement or speed claim.
Any early exit needs source-disjoint quality/latency measurements, including tail
latency and missed-error rates. The current fixed run does not change its levels.

Before replacing the quantizer, compare reconstruction with correct versus
predicted codes, and hold codes fixed while changing source access. Those controls
distinguish decoder robustness, useful side information and generator failure.
Neither higher code entropy nor a lower reconstruction loss substitutes for the
complete-summary semantic endpoint.
