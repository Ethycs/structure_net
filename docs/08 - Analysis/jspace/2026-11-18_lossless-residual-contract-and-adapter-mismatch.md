# Critical correction: lossless residual compression is not compact projection

Date: 2026-11-18 UTC. Status: source-backed architectural analysis; no model correction or new GPU experiment performed.

## Finding and governing requirement

The user's current requirement is **lossless compression with lossless inter-block adapters, apart from measured floating-point error**. The governing architecture is a cheaper basis-covered contribution **plus the retained remainder**, not a smaller substitute trained to compensate for discarded information.

**The inspected Qwen compact-suffix implementation does not satisfy that requirement.** It carries only reduced activation coordinates and a fixed background anchor. It does not carry each input's complementary activation component. It also substitutes fitted polynomial predictions without carrying their prediction defect. Its existing recovery measurements concern that approximate architecture, not a verified lossless implementation of the user's method.

This is an implementation/experiment-contract mismatch, not evidence that the intended residual-plus-compressed research direction fails. Conversely, the user's intended losslessness is not proof that the current code is lossless. Those two statements must remain separate.

The earlier [residual-compression direction](../../07%20-%20Status%20Reports/jspace/2026-11-13_jspace-residual-compression-direction.md) already required preserving the uncovered contribution, corrections and interactions. It explicitly rejected requiring the compressed branch to replace the whole function autonomously. The [implemented design](../../01%20-%20Design/jspace/residual-compression-and-recovery.md), lines 69–99 and 119–132, already documented both the discarded activation component and the omitted polynomial defect. **The failure was not merely missing documentation: these distinctions were not enforced when interpreting experiments or answering the user.**

## 1. What we are trying to build

The intended sequence is:

1. Use task benchmarks and task-Jacobian probes to identify functionally relevant candidate coordinates; validate any claimed CAV interpretation separately.
2. Find relations and a basis that identify computation which can be rewritten or factored.
3. Compile the covered computation into cheaper operations.
4. Retain the uncovered computation, relation defects where nonzero, and cross-component interactions.
5. Transport the complete represented state between blocks without losing information.
6. Measure the cost of the combined implementation against the same native reference.

The compressed component need not be autonomous. It may read context supplied by the remainder. The objective is to eliminate redundant **work**, not to delete every direction outside a selected subspace.

A useful accounting identity is

\[
F(x)=C_B(x)+R_B(x).
\]

Here the functional remainder includes whatever is needed for equality. It is **not necessarily orthogonal to the basis**: a covered-branch error can require correction inside the retained span. An orthogonal activation complement is one component of state bookkeeping, not a complete definition of the functional remainder.

J-space sensitivity supplies a search criterion, not permission to discard a direction. Low local sensitivity does not establish that finite removal has zero effect. Similarly, exact polynomial algebra certifies its stated relations/domain, not automatically the neural activation distribution.

## 2. Four different meanings of “residual”

| Term | What it preserves | What it does not establish |
| --- | --- | --- |
| Transformer skip connection | Adds a sublayer update to the state actually carried | If that state is only reduced coordinates, missing original coordinates are not restored |
| Per-input activation complement | The component outside the retained activation basis | Passing it through unchanged does not reproduce the native sublayer's dependence on it or its updates to it |
| Fixed affine anchor | A calibration-derived background component | It is not the current input's missing activation component |
| Functional/relation remainder | The difference between the original contribution and its compiled covered contribution | It is not automatically cheap, orthogonal, or supplied by other retained channels |

Previous explanations blurred these meanings. In particular, “we still have residual connections” and “we retain most MLP channels” are not evidence of lossless correction.

## 3. The inter-block invariant

For an orthonormal basis \(U_A\), anchor \(a_A\), and native state \(h\), one explicit lossless representation is

\[
z_A=U_A^\top(h-a_A),\qquad
r_A=h-a_A-U_Az_A,\qquad
h=a_A+U_Az_A+r_A.
\]

For another basis and anchor, a lossless transition is

\[
z_B=U_B^\top(a_A-a_B+U_Az_A+r_A),
\]

\[
r_B=a_A-a_B+U_Az_A+r_A-U_Bz_B.
\]

Therefore \(a_B+U_Bz_B+r_B=h\). The adapter must redistribute information between the covered and remaining components; it must not replace the remaining component with zero or a mean.

The general contract, independent of this particular storage representation, is

\[
D_B\bigl(T_{A\to B}(E_A(h))\bigr)=h
\]

in exact arithmetic, with separately measured numerical error in deployment. It also has to compose across a chain of different bases. Equal ranks, exact checkpoint reload and matching decode/re-encode of a projected state are insufficient.

This construction is a **correctness reference**, not a demand to store a redundant full-width residual plus reduced coordinates in the final executor. Native-width carry, complementary coordinates, structured residual factors or a proven codec are possible representations. Compression must account for the state and work of whichever remainder is actually used. Keeping a lossless activation representation alone does not prove smaller parameter storage or faster computation.

With quantized geometry, do not silently assume \(\widehat U^\top\widehat U=I\). A reconstruction residual defined by subtraction can still preserve the state, but formulas that drop cross terms using orthogonality require their own justification and numerical qualification.

## 4. What the executable path actually does

Inspected sources:

- [CompactQwenStack and _Bridge](../../../experiments/structure_net/jspace_qwen_compact_stack.py), lines 21–36 and 145–180.
- [AffineQwenBlock](../../../experiments/structure_net/jspace_qwen_affine_block.py), lines 69–109, 202–278 and 309–316.
- [Reduced execution](../../../experiments/structure_net/jspace_qwen_affine_fast.py), lines 161–207.

The stack starts with `z = U_first.T @ h`, carries `z`, applies the fixed bridge

\[
z_B=U_B^\top U_Az_A+U_B^\top(a_A-a_B),
\]

and reconstructs only \(a_{last}+U_{last}z\) at its exit. There is no input-dependent `r` in this interface. The omitted term relative to the lossless transition is \(U_B^\top r_A\), and the next residual is not retained either.

The anchor is constructed as \(a=(I-UU^\top)\mu\), so its orthogonality explains why the current entry encoder does not explicitly subtract it in exact arithmetic. The problem is **not** merely a missing anchor subtraction. It is replacement of the input-dependent complement by a fixed anchor.

The reduced executor's `z + update` preserves its reduced state, not the discarded native component. Its normalization uses reduced-coordinate energy plus fixed anchor energy, consistent with the represented affine state but not with arbitrary original input states.

### Executed counterexample

A CPU-only command through locked, offline Pixi imported the actual `_Bridge` and used FP32 exact unit bases:

- Native width 3; retained width 2; zero anchors.
- \(U_A=[e_1,e_2]\), \(U_B=[e_2,e_3]\).
- Input \(h=(1,2,3)\).

Observed output:

```text
original: [1.0, 2.0, 3.0]
current_compact_entry_reconstruction: [1.0, 2.0, 0.0]
current_after_bridge_reconstruction: [0.0, 2.0, 0.0]
with_residual_next_coordinates: [2.0, 3.0]
with_residual_next_remainder: [1.0, 0.0, 0.0]
with_residual_reconstruction: [1.0, 2.0, 3.0]
```

Assertions confirmed exact reconstruction for the residual-carry arithmetic in this example and failure for the current compact path. This is a counterexample to universal lossless transport, not a measurement of the trained Qwen bases' average error or a benchmark of a corrected Qwen implementation. No GPU reservation was needed; CUDA was hidden. No production source was modified.

## 5. Preserving the boundary is necessary, not sufficient

For a linear operation,

\[
W(a+Uz+r)+b=(WU)z+(Wa+b)+Wr.
\]

Dropping `Wr` is a separate approximation even if the original `r` is carried unchanged around the sublayer. For RMSNorm, the denominator must depend on the complete represented state:

\[
\sqrt{\|a+Uz+r\|^2/d+\epsilon},
\]

using the original width \(d\). With exact orthogonality, \(\|a+Uz+r\|^2=\|z\|^2+\|a+r\|^2\) when both \(a\) and \(r\) are orthogonal to \(U\); even then the anchor/residual cross term remains inside \(\|a+r\|^2\).

Attention must preserve the complete score and value contributions, not merely keep the unused input for a later skip. For example, with a common orthogonal split, \(q^\top k=q_c^\top k_c+q_r^\top k_r\); removing the second term changes softmax unless it is otherwise accounted for. SwiGLU likewise requires its coupled gate/up computation, not separate linear processing followed by an assumed additive nonlinear identity.

The historical [ResidualCarryBlock](../../../experiments/structure_net/jspace_residual_carry.py) already returns `h + U(z_out-z_in)` and uses dynamic complement energy. Its own module documentation states that incoming QKV/SwiGLU numerators still approximate the complement using fixed shifts. It is useful prior work, **not an already-proven lossless native block**. Restoring that wrapper alone would not complete the requested architecture.

## 6. Gröbner reduction must retain the appropriate remainder

For a polynomial expression and basis \(G=\{g_i\}\), division gives

\[
f=\operatorname{NF}_G(f)+\sum_i q_i g_i.
\]

The normal form is the retained reduced expression. The second term vanishes on a domain where the relations hold exactly. If fitted neural relations have nonzero defects, their contribution cannot simply be declared zero. Normal-form remainder, relation defect, functional remainder and activation complement are different objects.

The current MLP substitution has the accounting identity

\[
W a=W_Ka_K+W_Rp(z)+W_R\bigl(a_R-p(z)\bigr).
\]

The implementation computes the first two terms and removes the selected original gate/up rows; it does not compute the last defect term. Keeping `a_K` does not supply that correction. Source: `compile_polynomial` and `_mlp_update` in `jspace_qwen_affine_block.py`.

A lossless variant must either establish the required identity on its declared operating domain or retain the nonzero correction through an executable representation. Merely evaluating all deleted native work again to calculate the correction is a useful oracle, not evidence of a speedup. The research question is which algebraic structure lets the **combined** covered and remaining computation cost less.

The [algebra classification study](2026-11-13_jspace-groebner-classification.md) contains exact constructed controls and separately scoped neural results. It must not be cited as certifying exact neural relations that it did not establish.

## 7. Reconcile the supplied example rather than rewriting its meaning

The user-provided attachment `/home/rabbit/.codex/attachments/3a9e1366-cc2e-4086-92d0-193f8890637b/Pasted text.txt` explicitly says:

- Lines 40–50: `h ≈ a + Uz`; the anchor retains the average of discarded directions.
- Line 99: folding is exact **for the chosen affine representation**; discarded directions and projected updates introduce approximation.
- Lines 198–200: adaptation compensates for discarded information and need not preserve an exact folding of the original weights.

Thus that particular small-model recipe is an approximate compact-model example. Its reported recovery and speed do not establish a lossless original-model transformation. Earlier permission for bounded task-accuracy loss also does not make a lossy adapter mathematically lossless.

These historical materials explain the origin of the compact path, but do not excuse conflating it with the user's residual-plus-compressed architecture. The latest clarification requires lossless transport and compression; existing approximate runs must retain their historical labels and protocols rather than be retroactively presented as satisfying that contract.

## 8. Errors in the assistant's interpretation

1. **Changed the object being preserved.** Exact execution of a projected model was discussed as if it meant preserving the original model.
2. **Confused residual meanings.** Reduced skip connections, fixed anchors and retained MLP channels were treated as if they supplied the missing dynamic remainder.
3. **Investigated recovery before establishing architecture fidelity.** Training could improve an approximate student, but cannot certify that the requested decomposition was implemented.
4. **Overinterpreted seam checks.** Small recomposition KL measures numerical/folding agreement for the already-projected computation. It neither proves lossless native-state transport nor bounds the value of a corrective adapter.
5. **Accepted the latest correction too broadly.** The current compression cannot be called lossless merely because losslessness is the intended contract. The explicitly lossy supplied example and the existing code have to be reconciled openly.
6. **Failed to act on caveats already in the design documents.** Better wording alone is insufficient; future admission gates must check the invariant before launching a recovery campaign.

The proposed learned affine bridge correction is an approximate recovery experiment, not the missing lossless adapter. It cannot uniquely recover information from genuinely indistinguishable encoded states. Correlations may make some missing effects predictable, but that is a different claim. A learned adapter is compatible with losslessness only if its complete representation and inverse/correction preserve the required information.

## 9. Correct experiment order and acceptance gates

These are prospective requirements, not completed implementation claims.

1. **Freeze the reference and scope.** Identify native checkpoint, dtype, operations, masks, positions and cache semantics. Separate algebraic equality, floating-point equivalence and empirical task quality. A five-point task gate is not a numerical-equivalence gate.
2. **Specify complete state and operation accounting.** Name the covered contribution, remaining state, remaining computation, relation defect and all interactions. For every removed operator, identify its replacement or the identity making it unnecessary.
3. **Prove and test the adapter before training.** Include arbitrary unequal subspaces, unequal ranks, nonzero anchors, pure-complement inputs, sequential transitions and native islands. Compare decoded states to original states, not only to projected controls. Include a negative control which drops the residual.
4. **Qualify one complete block without recovery.** Compare attention inputs/scores, normalization, MLP intermediates and final output with the native reference. Exact floating-point bit equality may not survive reassociation, so predeclare precision-specific absolute/relative error checks, reference rounding controls and conditioning analysis. Do not explain structured missing terms as rounding.
5. **Qualify composition and deployment.** Repeat with actual fitted geometry across multiple blocks, prefill and cached decode, then folded/packed kernels. Source-state round trips and operator equivalence are separate gates. Preserve checksums and unchanged native controls.
6. **Measure the full cost only after correctness.** Include residual weights/state, decoder/adapter work, caches, temporary buffers and all correction paths. Use an equally optimized native baseline. Exact decomposition alone guarantees neither parameter savings nor speed.
7. **Keep approximation/recovery as a separately named branch.** If a component fails exactness and is deliberately approximated, identify that change explicitly; do not use post-distillation accuracy to certify a lossless compiler. Distillation is not the acceptance mechanism for an algebraically exact rewrite.

Do not require standalone closure of the compressed component: the retained computation may provide its context. Do not force every component into a fixed removal budget if the available relations do not justify it.

## 10. Interpretation and operational scope

- Existing quality and timing observations remain observations of their actual architectures. They are not falsified by this audit, but they do not prove the requested lossless design.
- This audit establishes structural information loss in the inspected compact path. It does not quantify how much of the real Qwen recovery failure comes from each omission, nor show that a corrected implementation meets a particular speed/size target.
- No active training source, pinned standard, checkpoint or historical result was changed. No job was stopped, restarted or relabeled during this analysis. A corrected experiment needs a distinct frozen source/artifact identity and the reservation system for GPU execution.
- No claim is made here that an efficient exact correction already exists for every Qwen operation. That is the research obligation, not a reason to silently substitute projection and recovery.

**Working understanding going forward:** preserve the full function through a cheaper covered contribution and an explicitly represented remainder; use lossless coordinate transport between those representations; prove the invariants before attributing success or failure to recovery, the basis method or Gröbner compression.
