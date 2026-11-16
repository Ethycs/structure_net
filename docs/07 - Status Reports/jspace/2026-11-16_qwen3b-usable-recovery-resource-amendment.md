# Usable recovery: concurrent reservation adjustment

The initial generation diagnostic launch requested 3,840 MiB on the plain
RTX 2060 alongside ABLATION32's 2,048 MiB QR allocation. The controller rejected
it before starting the worker: the card has 5,705 MiB bookable, so those requests
would exceed available reservations. No diagnostic GPU work or training ran.

Use a fresh diagnostic preparation with 3,584 MiB reserved and a 3,328 MiB
allocator cap. The full-corpus worker uses that same amended profile. This
changes only the resource ceiling; precision, inputs, arithmetic, checkpoints,
training data, objectives and scoring remain as declared. A real memory pilot
must fit before training. Existing progressive and ABLATION32 leases are
preserved.

The original prepared diagnostic `diagnostics/update_0046` remains intact.
The replacement preparation is `diagnostics/update_0046_v2`, under
`/data/structure_net_experiments/jspace_3b_usable_recovery/20261116_v1`.
Keep the full-corpus data preparation and its original protocol immutable;
this separately logged resource amendment does not alter their science.
