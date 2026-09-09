# MoBA distillation provenance receipt

The measured report, artifacts, implementation and registry read-back were verified before committing this receipt.

```json
{
  "classification": "distillation_not_confirmed",
  "confirmed": false,
  "focused_tests": {
    "failed": 0,
    "files": [
      "tests/structure_net/test_tinyllm_moba_distillation.py",
      "tests/neural_architecture_lab/test_moba_distillation_hypothesis.py"
    ],
    "passed": 17
  },
  "hashes": {
    "data/experiments/tinyllm_moba_distillation/20260909_registered/campaign_results.json": "abcb16c37bc37ffc62afe8f1060075944f86225ae9af38fc82acd8c47f86f6d6",
    "data/hypothesis_registry/tinyllm-moba-distillation-v1.json": "03aed9fee5cf71c93f8d1052f7cef2b79a5231dc069b54fdbd198f01e7213f12",
    "docs/07 - Status Reports/2026-09-09_tinyllm-moba-distillation-preregistration.md": "4911a9b03eca50ea3fcd96aaf5ee1d8406d9eb508518d07280ab318f6045746e",
    "docs/08 - Analysis/2026-09-09_tinyllm-moba-distillation.md": "f693fcfb3d419eed33812a834516312982ef1dde5ba46dd53f37b97a8cccacc2",
    "docs/08 - Analysis/figures/2026-09-09_tinyllm-moba-distillation.pdf": "831d1d50f96f8f8fed569a4766134728cf7aff8ab52b93acac5711849d16f1e8",
    "docs/08 - Analysis/figures/2026-09-09_tinyllm-moba-distillation.png": "69662766a3f31bd07bbc0ec881173a55d984fec81bd0695f5b406d48d80c157a",
    "experiments/neural_architecture_lab/finalize_moba_distillation.py": "0d907a240d69dc81dcf99ae2873b6a14b6485e02db9e88d7d79fef0b356e7fd1",
    "experiments/structure_net/tinyllm_moba_distillation.py": "3c863504bc4898863ca2a8e8ff1b1e16230a59dd3061f495c19a53366c91f792",
    "src/neural_architecture_lab/moba_distillation_hypothesis.py": "f0b9116157b14cbbdbe6837ed5799e9d5972f290f0d367205aed15ba26f07d55"
  },
  "hypothesis_id": "tinyllm-moba-distillation-v1",
  "producer_fingerprint": "ca7ff90dc823dbfcad703320dbd9b15d2dde6990df46bda445ae610db1f57b2e",
  "registry_compatibility": "Chroma 0.4 uses the removed NumPy 2 NaN alias; np.NaN=np.nan is applied only in registry processes. Metadata and semantic query both verified. Unrelated legacy duplicate-ID and telemetry warnings persisted.",
  "registry_readback": {
    "confirmation_status": "distillation_not_confirmed",
    "confirmed": false,
    "direct_experiment_count": 40,
    "evidence_count": 40,
    "hypothesis_id": "tinyllm-moba-distillation-v1",
    "tested": true,
    "verified": true
  },
  "schema": "nal.moba-distillation-provenance.v1",
  "semantic_query_readback": {
    "classification": "distillation_not_confirmed",
    "evidence_count": 40,
    "filter": {
      "id": "tinyllm-moba-distillation-v1"
    },
    "ids": [
      [
        "tinyllm-moba-distillation-v1"
      ]
    ],
    "query": "TinyLLM teacher-distilled MoBA conversion",
    "verified": true
  },
  "summary": {
    "completed": 40,
    "excluded": 0,
    "failed": 0,
    "joint_passes": 0,
    "requested": 40,
    "retried": 0,
    "reused": 0,
    "unexecuted_by_gate": 0
  }
}
```
