# Native655 source feature pipeline diagnostic

Status **six_probe_pipeline_parity_passed**. Read-only systems check after the recorded0/128DEVfailure.
Artifact `data/experiments/ncm_paper_regime/20261024_native655_feature_pipeline_check_v1/audit/result.json`, SHA256 `105ef2d3d7b789687cb6705cbfa78c04a146515eec48c8b38ed8ba8c3f7336aa`.

All783source records (655TRAIN,64shortDEV,64longDEV) were independently rebuilt: 0 prompt/tokenization/date/alignment differences. Every cached feature had its source/prompt/alignment binding, fullFP32shape and file hash checked.

An unchanged28layer frozen Llama body recomputed six fixed first/longest sources. CPUreplay found 6/6 byte-exactFP32tensors. Original BF16body and FP32alignment were retained; peak GPUallocation 6,606,946,816bytes. No training or new semantic judging occurred.

Feature-scale means (TRAIN/shortDEV/longDEV) were1.588/1.592/1.599RMS and1.359/1.357/1.411centeredRMS. These descriptive values and the six probes provide no evidence of the suspected feature mismatch; they do not rule out every possible integration error.

The adapted checkpoint still failed all128complete-summary judgments. Fluent TRAINoutputs alongside unreadable DEVoutputs remain consistent with memorization and poor conditional generalization. The LM1Brelease is an unconditional language generator; the new frozen-body source-conditioning path was trained on only655pairs. It is not a matched replication of the paper's conditional summarization training. Larger, higher-quality paired training and a corrected level-dependent conditioning-corruption experiment are next hypotheses, not demonstrated fixes. Final320sources remain closed and unhashed;95%quality and a speedup are unestablished.
