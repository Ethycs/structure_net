# Qwen MoBA feasibility: durable provenance

The completed exploratory study passed 10/10 held-out joint gates on a shared GPU. The registry retains `confirmed: false` for uncontended performance; application accuracy was not measured.

```json
{
  "evidence": {
    "campaign_path": "data/experiments/qwen_moba_prefix/20260909_registered_shared_v1/campaign_results.json",
    "campaign_sha256": "1ac332c137d7d8d596053d23ca538573b83c05d2e99beb8d23fe955e371509a6",
    "classification": "shared_gpu_prefix_feasibility_supported",
    "completed_cells": 130,
    "final_cells": 10,
    "final_joint_passes": 10,
    "report_path": "docs/08 - Analysis/2026-09-09_qwen8b-moba-feasibility.md",
    "report_sha256": "8924ca9f9ce9a893a92b2d97fd72d526cfb2c474781958fc5dc58d0d3e30dd79"
  },
  "hypothesis_id": "qwen3-8b-prefix6-moba-feasibility-v1",
  "limits": [
    "no_application_accuracy_measurement",
    "shared_GPU_runtime_only",
    "generic_text_inputs",
    "six_layer_prefix_not_full_8B_forward"
  ],
  "registry_readback": {
    "confirmation_status": "shared_gpu_prefix_feasibility_supported",
    "confirmed": false,
    "direct_experiment_count": 130,
    "evidence_count": 130,
    "exact_readback_verified": true,
    "hypothesis_id": "qwen3-8b-prefix6-moba-feasibility-v1",
    "registry": "data/hypothesis_registry/qwen3-8b-prefix6-moba-feasibility-v1.json",
    "semantic_readback_verified": true,
    "tested": true
  },
  "schema": "nal.qwen-moba-durable-provenance.v1",
  "source_sha256": {
    "experiments/neural_architecture_lab/audit_qwen_moba_feasibility.py": "6fbf327e509bf29cfea55db0f3719466bbaf33136f70369752ee52471ac2e5f7",
    "experiments/neural_architecture_lab/finalize_qwen_moba_feasibility.py": "543cea0906b5bb0b4fbff3358e862ab4e7029184f45fc0dedf3e84b88cfb63d8",
    "experiments/structure_net/moba_head_benchmark.py": "2a80ab15c855a445751d2df56aba4d639227c5ce575226d93fb7ce75a9a3d8d2",
    "experiments/structure_net/moba_paper_latency_diagnostic.py": "3cd94019fc7a50d58c963420593230a20ff61a95060b59a184340a79c0284fe9",
    "experiments/structure_net/qwen_moba_prefix.py": "f5a298d2750f2dbb4cd2026c84036da84f481dd23e74c4d4fd2a9328f84762a3",
    "experiments/structure_net/qwen_moba_shared_assessment.py": "01b6be186ed866eef19e1bd59a19c68f9ae3c53a6e52feaef106258e0ebe4dd1",
    "experiments/structure_net/run_qwen_moba_feasibility.py": "8213278196a3229db5b73fa9e74091e8760e72813449624e67398ef09e32d2e4",
    "src/neural_architecture_lab/qwen_moba_feasibility_hypothesis.py": "d1c1b94121fdfc57259bbc0867c466901e7bd6d6d33cfc93a57125fa85de2dfa"
  },
  "validation": {
    "command": "env LD_LIBRARY_PATH= PYTHONPATH=/home/rabbit/structure_net:/home/rabbit/structure_net/src data/experiments/moba_paper/environment/bin/python -m pytest -q tests/neural_architecture_lab/test_qwen_moba_audit.py tests/neural_architecture_lab/test_qwen_moba_feasibility_hypothesis.py tests/structure_net/test_qwen_moba_shared_assessment.py tests/structure_net/test_qwen_moba_prefix.py",
    "figure_visually_inspected": true,
    "final_gate_passes": "10/10",
    "focused_cpu_tests_passed": 16,
    "frozen_weights_verified": true,
    "pinned_weight_download_verifier": "passed; existing files unchanged",
    "real_prefix_graph_parity": true,
    "stage_audits_verified": true
  }
}
```
