# Exact speed iteration: shared physical weights for paired controls

Date: 2026-10-10, before primary outcomes. Pilot v2 again stopped at packed
prefill 512: the reference and packed parameter buffers plus CUDA graph pools
exhausted GPU memory. Its `mm` cells remained exact, packed prefill 128 again
failed exactness, and the tiny scheduling candidates passed. Preserve v2.

For paired tests, let the reference's individual projection tensors become
views into the packed tensor after verifying every source parameter bit. Give
each view the original tensor shape and strides, including the size-one batch
stride. Only the storage address changes. The candidate and reference now
share one physical weight set, as the non-packed arms already did. They remain
alternative execution paths, never two independent models.

Before any packing, compute and save all six full-model reference controls
(two inputs for each workload, including complete generation traces). Every
subsequent reference eager result and graph result must match these immutable
pre-packing anchors. This prevents a changed control implementation from
silently moving the exactness target. Retain and hash those anchors.

No candidate, threshold, timing sample, shape, model arithmetic or final
decision changes. New pilot root `20261010_pilot_v3`, primary `20261010_v3`.
All previous failed pilots are excluded from primary timing. Packing and
capture overhead remain reported. Test preservation of reference strides and
physical storage sharing on CPU before the new pilot.
