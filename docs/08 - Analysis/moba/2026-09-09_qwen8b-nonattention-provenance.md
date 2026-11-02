# Qwen nonattention acceleration: finalization provenance

Date: 2026-09-09. All raw producing roots remain immutable.

```json
{
  "application_accuracy_tested": false,
  "artifacts": {
    "pilot": {
      "path": "data/experiments/qwen_moba_prefix/20260909_nonattention_pilot_v1/campaign_results.json",
      "producer_fingerprint": "4da1ba5235a0963bef1ec9abe5b6f7b36780d14dae9f8e2d4eb0bdba6c8527f2",
      "sha256": "261c2ed6832e65bf3b7ea544b2c30593b2438570d795ff85913dd639ed83388d"
    },
    "prefixpilot": {
      "path": "data/experiments/qwen_moba_prefix/20260909_nonattention_prefixpilot_v1/campaign_results.json",
      "producer_fingerprint": "a5db11c22195157f8e97afb60379ad60effcdb5ffcfd895ee69033e530697d45",
      "sha256": "f855779946620eda0d7aa737c630a9e050a5014eee133edba23659add1d44565"
    },
    "regression": {
      "path": "data/experiments/qwen_moba_prefix/20260909_nonattention_regression_v1/campaign_results.json",
      "producer_fingerprint": "31310b67384f33e75d1c9327ddd3b06f9c9ce2f30ecbbb05fea7086b92df5f58",
      "sha256": "c71ca35429b9c5ad887d88d606864b188afc537de96fc5427afe8fef166e7b8c"
    },
    "screen": {
      "path": "data/experiments/qwen_moba_prefix/20260909_nonattention_screen_v1/campaign_results.json",
      "producer_fingerprint": "27e1a460f1a4d54a4289555957c15ef303b90955bfebd42cc415b284f7cf7160",
      "sha256": "ebd4196d5989407f627c0211dc42a89c302c4f28d28b686316857b103a2a1eab"
    }
  },
  "audit_tamper_controls_rejected": [
    "producer fingerprint mismatch",
    "cell mismatch"
  ],
  "classification": "execution_regression_supported",
  "confirmed_application_accuracy": false,
  "exact_readback_verified": true,
  "focused_tests_passed": 7,
  "hypothesis_id": "qwen3-8b-prefix6-nonattention-acceleration-v1",
  "registry_path": "data/hypothesis_registry/qwen3-8b-prefix6-nonattention-acceleration-v1.json",
  "registry_sha256": "dbbd74e27e59f86bfcde92bf240a7cef79aab20eed8b73c54d55a4bf2f5260bc",
  "report_path": "docs/08 - Analysis/2026-09-09_qwen8b-nonattention-acceleration.md",
  "report_sha256": "a4e148f858995dd2d056b513fa7c758ff70bc8eb47a3e8269cdc7a283ad1f7e7",
  "semantic_readback_verified": true,
  "trained_parameters": 0
}
```
