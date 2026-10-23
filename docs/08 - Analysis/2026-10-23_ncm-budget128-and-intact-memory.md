# Corrected budget qualified; intact-memory lifecycle passes

Date: 2026-10-23. Exploratory, systems/evaluation evidence only. Classification
`budget128_judges_qualified_and_memory_lifecycle_passed_only`. The95%summary-task objective remains unmet.

The inherited judge rubric targeted approximately32words, while the application's
stored summary limit is128tokens. One of10teacher-pilot judgments explicitly
penalized a supported summary for length inside that storage cap. The new profile
changes only this budget instruction; source-grounded faithfulness, main-point
coverage, readability and the code-enforced128-token cap remain. Every old score
and first negative judgment remains unchanged. The old pilot scored8/10; one
other failure was a real recipe-order contradiction, not a budget artifact.

Both Terra and Sonnet passed **50/50** authored/source-reviewed calibration
controls, with13/13valid requests per judge. Each set contains the48original
controls plus a long supported coffee summary and a terminal-number contradiction.
All raw responses and decisions were replayed. This is judge qualification,
not model accuracy, independent human gold or an NCM result. Fixed model outputs
have not yet been re-evaluated under this new profile; old50/64,49/64and0/64
counts must not be relabeled as new-profile measurements.

The original zero-output memory correction passed CPU tests but failed its GPU
initial exact-identity check before any updates. The v2 inference branch returns
the exact original hidden tensor while its output matrix is zero and autograd is
disabled. This retains storage/strides; training still runs correction arithmetic
so its initial output projection receives gradients. The original numerical
cause was not isolated. Original producer and failure stay append-only.

The v2 GPU lifecycle passed initial exact native logits and32-token greedy IDs,
two real updates, no native gradients, frozen full-model hash, immutable source
features, correction bound and exact new-module reload. Native16-block1B weights
were FP16, the new7,340,032parameters FP32, batch2, physicalGPU2. Peak allocation
was3,226,645,504bytes. Four accepted training cases and target-independent
synthetic concept vectors were systems probes only. This is not an NCM model;
no new held-out summary quality or speed was measured. Its no-op initialization
does not guarantee quality after training.

Aggregate `data/experiments/ncm_paper_regime/20261023_budget128_and_memory_results_v1/campaign_results.json`, SHA256 `4cd0a14a49f2194efad6adc8798299e178af15199a891f84db843a1170612eae`. Registered records retain
confirmed=false. Follow-up label production uses a separately fixed draft /
critique / one-revision algorithm and a prospective128-case audit; neither these
systems passes nor old training memorization admit an NCM success claim.
