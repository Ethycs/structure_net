# Later residual codes can partly correct generated-prefix errors

Completed exploratory diagnostic `ncm-residual-recoding-probe-v1`. On the same
16 fixed TRAIN summaries, keeping the first nine generated levels and
recomputing the final three residual levels raised source-bound reconstruction
agreement from **62.84% to 81.20%**. All 16 cases improved on that comparison.
This measures correction capacity with access to reference summaries. It is
not a trained improvement or source-only summarization accuracy; the latest
[semantic DEV result remains 0/128](2026-10-25_ncm-broad-generator.md).

Aggregate: `data/experiments/ncm_paper_regime/20261025_residual_recoding_probe_results_v1/campaign_results.json`.
SHA-256: `10fef7e4c2185765ef6bcd0a34ba4e0b337920779246f4f6e3ce92dfbd5d0644`.
Classification: `oracle_residual_recoding_capacity_measured`.

The [prospective plan](../07%20-%20Status%20Reports/2026-10-25_ncm-residual-recoding-probe-plan.md)
compared the original clean suffix codes with a suffix re-quantized against the
remaining reference latent after subtracting an actual generated prefix.
All model tensors stayed fixed. Both arms received the same prefix; both used
the reference to supply their suffix. The codebook, decoder and source-attention
weights were unchanged.

| Generated prefix levels retained | Original suffix: token agreement | Recomputed suffix: token agreement | Exact frames, original / recomputed |
| --- | ---: | ---: | ---: |
| 0 | 98.32% | 98.32% | 6/16 / 6/16 |
| 2 | 98.32% | 98.32% | 6/16 / 6/16 |
| 3 | 98.41% | 98.85% | 6/16 / 7/16 |
| 4 | 98.41% | 98.59% | 5/16 / 7/16 |
| 6 | 95.85% | 97.88% | 2/16 / 5/16 |
| 9 | 62.84% | 81.20% | 0/16 / 0/16 |
| 11 | 42.28% | 45.01% | 0/16 / 0/16 |
| 12 | 19.15% | 19.15% | 0/16 / 0/16 |

These are active native-token counts over 1,133 positions, including boundary
tokens and excluding padding. Exact frames are reconstruction diagnostics, not
semantic summary judgments. With all twelve levels generated, there is no
remaining suffix to correct. That endpoint exactly reproduces the saved model
readouts; zero generated levels reproduces the clean oracle hierarchy.

After nine generated levels, correct active tokens increased from 712 to 920.
After eleven, they increased from 479 to 510, but three cases worsened. At four
levels, two cases worsened despite a small positive aggregate. Correction is
neither universally beneficial nor sufficient for full reconstruction after a
long erroneous prefix. The codec-only branch showed the same broad pattern:
64.78% to 82.88% after nine levels, and 44.75% to 46.78% after eleven.

The evidence supports testing correction earlier and repeatedly during
generation. Infinity's bitwise self-correction revises later target residuals
after perturbing previous codes; it supplies a related visual-model precedent,
not proof of transfer to this text model.
[Infinity, section 3.4](https://arxiv.org/html/2412.04431v2).
Our existing generator corruption keeps clean targets unchanged. A new matched
training comparison should distinguish exposure to actual generated histories
from the additional effect of recomputing residual targets, while retaining
clean-history training and the frozen reconstruction path.

Nine focused tests passed. The first pilot failed before diagnostic outputs
because it assumed a single-path feature manifest; its failure is retained.
The corrected version validates separate metadata, header and compressed
payload files. A two-case CUDA pilot repeated all outputs exactly, and all pilot
outputs matched the full run. Every forced clean prefix re-encoded to the native
gold hierarchy. The original source-only output matched the full-prefix control.

The full diagnostic retained 256 arm/case/prefix comparisons and 512 decoded
readouts. An independent CPU finalizer replayed every target/source binding,
prefix and original-suffix identity, raw token count, rendering, aggregate and
paired change. Latent MSE values were aggregated from producer measurements;
the independent finalizer did not rerun the encoder.

Locked Pixi ncm ran from repository root on physical GPU1 RTX2060, in FP32 with
TF32 disabled. Peak GPU allocation was 1,230,972,416 bytes; full FP32 source
features used at most 175,448,064 bytes in the 256 MiB disk-streaming cache.
No examples were clipped, no optimizer update or new judge request ran, and
DEV and final-test sources were not used by this diagnostic.

Raw outputs:
`data/experiments/ncm_paper_regime/20261025_residual_recoding_probe_full_v2/outputs.json`.
SHA-256: `03bba3314c21d7e580ce4777449ed1754562b40836cc93027f8fa210e781c75e`.
Run entry point: `experiments.structure_net.ncm_residual_recoding_probe`.
Independent finalizer: `experiments.neural_architecture_lab.finalize_ncm_residual_recoding_probe`.
Completed roots are append-only evidence. No inference speedup is claimed.
