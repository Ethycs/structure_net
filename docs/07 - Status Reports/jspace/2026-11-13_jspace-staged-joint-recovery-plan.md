# Staged Qwen compression, joint recovery, and local distillation controls

Status: next-stage design, not a launched study. Date: 2026-11-13.

The user supplied the successful small-image training account, then explicitly
asked to retain the new distillation angle. Preserve both: staged output-level
distillation and joint recovery form the baseline; local Euclidean and
task-Jacobian supervision are additional matched experimental arms.

## Evidence and scope

The [supplied account](</home/rabbit/.codex/attachments/dabdb122-3aa4-4259-abcf-1d46b7d56a3d/Pasted text.txt>) reports
compression of the final block against the original teacher, followed by
compression of its predecessor and joint recovery of both blocks against the
previously recovered compact teacher. Its reduced classifier and embeddings
were trainable in the second stage. It used KL+CE at temperature two and no
separate hidden-state or attention-map loss. The reported 97.44% test result is
user-supplied evidence, not a measured Qwen result or independently inspected
checkpoint in this workspace.

The ongoing `JSPACE-QWEN-LAYERKD-01` tests local losses on a single already
compiled final block. It cannot establish the joint-recovery behavior. The
two-block compiler now exists in `jspace_qwen_affine_stack.py`; its CPU tests
establish algebra, gradient, cache and reload behavior only.

## Required second-stage sequence

1. Select one recovered compact final-block checkpoint as the common teacher
   before the next stage. All three arms inherit exactly that checkpoint.
2. Run the task benchmark through that teacher. Derive the preceding block's
   functional basis and shared Gröbner relations before eliminating any part.
3. Compile block 34 and retain block 35's recovered compact weights. Connect
   them through a fixed affine bridge with no full-width intermediate residual
   in the deployed student path.
4. Recapture endpoint teacher logits from native block 34 followed by the
   recovered compact teacher block 35. The original teacher cache has different
   targets and must not be silently reused.
5. Train both compressed blocks jointly. Recompute both student blocks on every
   minibatch. A cache before block 34 is valid only while blocks 0–33 and the
   embeddings remain frozen. Do not cache the trainable block-34 output.
6. Compare output-only recovery, output plus Euclidean supervision, and output
   plus J-space supervision with identical initialization, trainable modules,
   data, optimizer budget and selection criterion. Pack, reload and measure.

For bases `U1,U2` and anchors `a1,a2`, the bridge is
`z2 = U2.T @ U1 @ z1 + U2.T @ (a1-a2)`. The `a2` term vanishes in exact
orthogonal geometry; retain the general expression in implementation. Bridge,
bases and geometric buffers remain fixed. Both sets of folded weights adapt.

## Retain the new local-distillation angle

Local supervision is a new variable, not part of the reported successful
baseline. Keep the endpoint KL+CE signal in every arm.

For the second compact block, run its frozen teacher on the actual current
student input each step. Since both versions share a coordinate system, using
the same detached bridge coordinates avoids an unnecessary reconstruct/round/
encode mismatch. Distinguish this same-input local target from the frozen
teacher's original complete-trajectory endpoint target.

For block 34, downstream sensitivity includes block 35's attention. A tokenwise
pooled metric drops cross-token effects. Investigate an exact teacher-block
JVP followed by the final RMSNorm/readout quadratic, with explicit-Jacobian
value and gradient tests on a tiny causal model. Fresh online full-sequence
VJPs are a fallback if JVP training is unsupported or prohibitively costly.
Do not optimize repeatedly against a small fixed probe bank and then describe
its training loss as an unbiased full-metric measurement.

## Qwen-specific readout and prefix constraints

Qwen ties the 151,936-by-2,048 output matrix to its embeddings. Updating that
matrix changes the supposedly frozen prefix and invalidates its cache. The
single-block comparison therefore freezes it and trains the final norm gain.
For a closer joint-recovery counterpart with a trainable classifier, the
proposed second-stage adaptation is a shared identity-initialized 2,048-square
linear map after the final norm and before the frozen output matrix. Train it
in all three arms, together with both compact blocks and the norm gain.

This gives a trainable readout within the original vocabulary matrix's column
space, preserves the input embedding and frozen-prefix cache, and adds 8 MiB
of BF16 deployed weights. Count its actual storage and runtime. It is a
declared constrained readout adaptation, not a fully trainable untied vocabulary
matrix or a literal reproduction of the small-image classifier.

## Checks and decisions before launch

- Bridge execution must match explicit reconstruct/project in FP64, with
  nonzero anchors, compact heads and polynomial channels. Packed BF16 has a
  different rounding path and requires measured quality.
- Gradients must reach both blocks and the shared readout adapter; teacher,
  frozen prefix, bases, anchors and bridge must stay fixed.
- Changing the first compressed block must immediately change the second
  block's live input. Same-input teacher targets must refresh with it.
- Freeze task identities, learning rate, batch/update budget, validation
  selection rule and auxiliary weights before any new test results. Record
  differences from the source's 12 epochs, batch 256, learning rate 4e-4 and
  validation-CE selection rather than calling changed settings a replication.
- Compare actual end-task accuracy and deployed bytes/timing, including the
  bridge and adapter. No automatic whole-model speed claim follows from
  smaller dense projections or successful recovery.
- Use a fresh source snapshot, systems pilot, locked Pixi and an experiment
  reservation. No producing files from the active one-block campaign change.

## Readiness after the fused execution experiment

The [measured performance follow-up](../../08%20-%20Analysis/jspace/2026-11-13_jspace-compact-performance.md)
now supplies a concrete faster one-block starting point: the recovered Gröbner
block with BF16 geometry, aligned output packing and Triton fusion. Against the
native block with both implementations using CUDA graphs, it measured 1.297×
prefill and 1.275× decode speed. Fresh accuracy was 391/512, versus 393/512
native and 390/512 for the previous compact checkpoint. Its exact exported
weights passed fresh-process reload. These remain one-block measurements;
two-block calibration, joint recovery and GPU stack timing have not run.

Before stage two, explicitly pin the accepted teacher's exported weights and
geometry. If that teacher is the faster version, generate its targets through
native block 34 and that exact compact block 35. Do not substitute the original
native teacher's cached logits. Initialize trainable block 35 from the selected
unpacked checkpoint, then verify its packed structure against the accepted
teacher; a final packed checkpoint has merged polynomial columns and removed
offset parameters, so loading it directly into an unpacked block is invalid.

Block 34's MLP Jacobian must include **all context-token rows**: downstream
block-35 attention mixes them. The four-output-position slice used for the
final block is insufficient here. The new `StreamedJacobianRows` adapter feeds
a read-only NumPy bank in bounded chunks to the existing unchanged anchor
selector and Gröbner fitter. Seven CPU tests verify fit agreement and four
complete streaming passes for the existing one-power-iteration SVD. It avoids
allocating a full resident tensor for the roughly 13 GB derivative bank;
operating-system file caching is separate. It has not yet been exercised in
the real stage-two GPU calibration. Use blocking transfers
initially so staging-buffer reuse cannot corrupt derivatives.
