# MoBA scope history and superseded branches

The current experiment is public-Qwen attention feasibility, documented in
[the Qwen protocol](../07%20-%20Status%20Reports/2026-09-09_qwen8b-moba-feasibility.md)
and its [shared-GPU amendment](../07%20-%20Status%20Reports/2026-09-09_qwen8b-moba-shared-gpu-amendment.md).
The user explicitly authorized a representative public model and generic
feasibility experiments; exact application code is not required for that scope.
Application accuracy remains distinct from representation-preservation proxies.

Earlier branches are retained as history, not evidence for this Qwen claim:

| Branch | Execution state | Interpretation |
| --- | --- | --- |
| Original TinyLLM MoBA distillation | Completed and registered in commit `8e321e7` | `distillation_not_confirmed`; the original negative result is unchanged. |
| Routing-95 optimization draft | Withdrawn before implementation or execution | The user meant final accuracy, not routing recall. |
| All-weight TinyLLM BLiMP search | First candidate completed its 1,600-update budget; second stopped after its 300-update check | Out of scope after the user clarified that only attention should be modified. No final test was evaluated. Validation scores do not establish Qwen or application accuracy. |
| Attention-only TinyLLM accuracy draft | Protocol only; never launched | Superseded by the user's attention-readout application and public-Qwen feasibility scope. |
| Exact-application protocol | Intermediate scope document | Its request for exact app code was superseded by authorization to use representative experiments. |
| Custom Triton sparse kernel | Auxiliary numerical and systems tests | Not the implementation used for the Qwen findings; those use official MoBA. |

The all-weight search root is
`data/experiments/tinyllm_moba_accuracy95/20260909_search_v1/`.
Its `scope_correction_stop.json` records the stop and hashes the available
validation evidence. Producing source
`experiments/structure_net/tinyllm_moba_accuracy95.py` has SHA-256
`4adb6beff68f9af100699f062fb25bd9d139536eff30573de38f34fe3e866d3d`.
It is preserved to interpret those historical artifacts. Unexecuted draft
registration/finalization code for that abandoned branch was removed; it never
wrote a registry entry.

The Qwen experiment has zero training updates. All loaded weights, including
MLPs, are frozen. It executes five full transformer blocks and then the sixth
attention module. It is not an attention-only network consisting of six
attention modules with all preceding MLPs removed.
