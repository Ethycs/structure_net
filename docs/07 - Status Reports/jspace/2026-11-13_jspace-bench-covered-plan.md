# Strict cross-task coverage and factored Gröbner tensor follow-up

Date: 2026-11-13. Parent: JSPACE-BENCH-TENSOR-01, revision 2.
Artifact root: `data/experiments/jspace_bench_tensor/20261113_covered_v1`.

Known before this follow-up: the common formulas qualify on all four development
tasks for 20, 34 and 19 channels in blocks 8, 17 and 26, respectively. The larger
parent budgets missed the development KL gate; 128 channels preserved all 64
development labels but had KL 0.021506 versus the fixed 0.02 limit. No held-out
accuracy result was inspected when this follow-up was specified. Parent gates
and outputs are unchanged. Reusing the same evaluation pool does not create an
independent confirmation.

Select exactly the nonanchor channels whose common degree-two relation has
development NRMSE <=0.5 on every task. Recompute eligibility from the frozen
captures and fit using the original CUDA arithmetic. Do not tune the threshold,
add channels for a nominal budget, or select using held-out accuracy. Preserve
all other channels as the nonlinear residual. This tests reliable coverage
directly; the parent also explored task-sensitivity-weighted channel budgets.

With these small covered sets, expanding the entire output-by-quadratic tensor
would cost more than retaining its algebraically derived factors. The primary
implementation therefore keeps `T = W_elim diag(scale) C` factored, applies
polynomial features to C, and then applies the output factor. The factors follow
from Gröbner substitution, not weight SVD. Removed gate/up rows are not evaluated.
Compare to the explicitly expanded tensor and to zeroing the identical channels.

Vectorize the quadratic feature products with fixed triangular index buffers;
avoid one Python-dispatched kernel per monomial. CPU value/gradient tests must
show equivalence to the parent's feature implementation for both contraction
orders. All arms keep the same eager attention, precision, token limits and data.

Evaluate original and primary model on development and all 256 held-out questions.
Apply the original development gate and held-out label/accuracy rules. Evaluate
the expanded tensor and zeroed-channel controls on the identical held-out pool.
Time original before and after, factored primary, and expanded tensor using the
same eight development prompts, 32-token generation, warmup and wall-time rules.
Positive tensor-byte reduction and >=1.02x point-estimate throughput constitute
the efficiency gate. Report failed gates and both controls without substitution.

Freeze this extension, tests and parent identities before launch. Reuse captured
data and the common basis; no new basis training or data-driven threshold choice.
Use a new 10,240 MiB RTX 3060 lease through inference-ctl and root locked ncm Pixi.
Record this as an exploratory follow-up, `confirmed=false`, alongside the original
campaign result. The algebra certifies the polynomial graph representation; neural
relation defects remain measured approximations in the combined architecture.
