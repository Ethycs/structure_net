# Token-dictionary inversion does not repair the NCM text codec

Date: 2026-10-22. `ncm-codec-inverse-control-v1` is complete, classified
**dictionary_inverse_failed**. Neither released nor adapted codec meets
the target-blind replacement gate. No new summary accuracy or speed is measured;
the current attachment remains at its earlier0/64 Terra development result.

Entries are correct non-PAD target tokens, with exact target texts in parentheses.

| Codec | Cohort | Neural raw | Blind inverse,4 steps | Quantized, oracle statistics | Unquantized, oracle statistics |
| --- | --- | --- | --- | --- | --- |
|released|seen|96.2798% (4/20)|25.8929% (0/20)|75.4464% (0/20)|100.0000% (20/20)|
|released|novel|96.9317% (8/20)|37.5174% (0/20)|74.7559% (0/20)|100.0000% (20/20)|
|released|raw_validation|96.4181% (124/256)|34.8464% (0/256)|76.2390% (0/256)|100.0000% (256/256)|
|adapted|seen|99.8512% (19/20)|44.9405% (0/20)|80.5060% (0/20)|100.0000% (20/20)|
|adapted|novel|92.6081% (4/20)|43.9331% (0/20)|78.9400% (0/20)|100.0000% (20/20)|
|adapted|raw_validation|95.3373% (99/256)|39.8641% (0/256)|77.8138% (0/256)|100.0000% (256/256)|

The encoder is a lookup of token plus positional embeddings, followed by
8-group normalization. The candidate estimates normalization from neural token
predictions, reverses it, subtracts positional embeddings, then uses nearest
token-embedding distance. Four fixed alternating steps use no target IDs or
target statistics. They perform substantially worse than the original neural
decoder. No iteration/metric/decoder search or outcome-based rerun occurred.

Using true normalization statistics with the quantized latent also performs
worse than the neural decoder. That observation does **not** prove irreversible
information loss: the neural decoder demonstrably recovers more tokens than
this distance rule, and information can be distributed across positions. This
rejects the proposed geometric inverse as a decoder remedy. It cannot establish
an optimal decoding bound or excuse source-to-code prediction failure.

With the original unquantized normalized latent and true statistics, the same
inverse recovers every token ID exactly across all592 codec/case combinations,
including EOS and PAD. This supports implementation correctness for the stated
normalization inversion, not applicability of nearest-neighbor geometry after
learned residual quantization. Oracle variants know reference-derived
statistics and are not inference candidates. Actual generated-summary codes
were not tested here.

The20 seen/20 novel labels and raw256 source-disjoint validation windows are
unchanged. Every readout shares the same target codes. FP32, batch4, GPU1;
frozen states, finite values, code/prediction reload and the lifecycle pilot's
readout-order parity passed. Own peak allocation was
494,885,376bytes. The novel targets and raw validation
are development diagnostics; final task-test data remain sealed.

Aggregate: `data/experiments/ncm_paper_regime/20261022_codec_inverse_results_v1/campaign_results.json`; SHA256 `146675db0f9f2c4d84f64e788b08c9dd3c2e8bef380b7d7f01f3b44c9e494db7`. Full readout IDs,
intermediate inverse iterations, target codes, original source snapshots and
input seals remain in immutable roots. Registry confirmed=false,
task_accuracy_measured=false, irreversible_information_loss_established=false.
