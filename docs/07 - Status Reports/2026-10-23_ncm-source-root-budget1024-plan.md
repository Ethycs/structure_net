# Fixed1024-update follow-up for isolated source-root learning

Date:2026-10-23. Exploratory hypothesis
`ncm-source-root-capacity-budget1024-v1`. The prior256-update capacity run
completed with exact reload and unchanged frozen words/other heads/codec.
Active-root recovery improved6/176to121/176; shuffled-source recovery stayed
6/176. All-root recovery was200/256. Its95%capacity gate failed. The complete
word-final summarizer remains0/64short and0/64long, not95%task accuracy.

The source path is learning, but the fixed256-update schedule did not fit even
the16TRAINcases. Test a separately declared longer training budget. Preserve
and register the256-update failure before starting. This is a fresh experiment,
not a continuation of its optimizer or a retrospective extension of its gate.

Start again from the same original word-final final512checkpoint, SHA256
fff3e05b6715754d200da761bdd0481d54be6ccd7d988e7899ada34c85eed2b8.
Discard all256-update/pilot weights. Keep the16TRAINcases, selected order,
source features, labels, active/padded loss, dropout, optimizer, gradient scope,
precision, memory limit, source-shuffle control and95%capacity gates unchanged.
Original Llama weights/MLPs, codec, copied word decoder/projection and nonroot
heads remain frozen. The original parent is not loaded.

Fix1024updates, batch1, same model/dropout seed13143and sampler seed14143.
Keep16warmup updates to3e-4, then cosine to3e-5at1024. The longer cosine horizon
changes the learning-rate schedule after warmup. Report this as a budget-and-
schedule follow-up, not an isolated effect of update count. No intermediate
selection or further extension; only final1024can be selected. The unchanged
implementation saves128as a recovery checkpoint only.

Reuse the completed actual two-update source-root CUDA pilot: operation shapes,
loss, initialization, first two warmup rates, precision, optimizer state and
maximum2,816-position source are identical. Its source-only/model/optimizer/
RNG reload and3,762,940,416-byte peak checks passed. Verify the first16warmup
rates match exactly before launch. The wrapper changes only declared config,
budget, schedule and output identity, and binds the preserved256-update result
and registry receipt. New model/input operations would require a new pilot.

Use physicalGPU0and fresh root
`20261023_word_final_root_capacity_budget1024_train_v1`, followed by a new
results root. Observe actual terminal status, replay all1024sample/frame/LR
entries and raw root counts, archive/register the actual outcome conservatively.
A TRAINcapacity pass may support a later full-hierarchy/source-only protocol;
it cannot satisfy the95%complete-summary task goal. The930canonical first task
judgments, including every failure, remain unchanged. Both final cohorts stay
closed. No speed or compression claim.
