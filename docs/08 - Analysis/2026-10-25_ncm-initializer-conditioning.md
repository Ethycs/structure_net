# The current hybrid learns a different source-conditioning interface

The current NCM hybrid starts from the released LM1B model and learns added
attention over frozen Llama source features. The released XSum summarizer
instead conditions its generator on a raw text prefix. Configuration readback
verified these differences against the saved release and positive-control hashes.

| Configuration | Released XSum | Released LM1B-small underlying our hybrid |
| --- | --- | --- |
| Generator layers / width / heads | 12 / 768 / 12 | 12 / 768 / 12 |
| Native text prefix | 512 tokens | 1 CLS token |
| Word tokenizer | GPT-2 | BERT uncased |
| Native output positions | 48 | 128 |
| Hierarchy | 9 levels, 4,096 entries each | 12 levels, 8,192 entries each |
| Output heads | Tied to codebooks | Shared head with per-level bias |
| Separate context encoder | Disabled | Disabled |

The current hybrid adds source attention to the LM1B generator and word decoder;
the active experiment trains only the generator and its source-attention modules.
Matching transformer widths do not make the complete released checkpoints
interchangeable. Their word vocabularies, quantized-code namespaces, hierarchy
lengths and head conventions differ; positional settings also differ.

This makes pretrained source-conditioning transfer a possible separate
experiment. It would require explicit treatment of these interfaces and a
measured quality comparison. Partial weight transfer has not been tested here,
and no change was made to the running 4,096-update study.

The earlier 128-case original-domain XSum control remains valid evidence:
ROUGE-1/2/L was 34.723/12.216/27.886 and codec reconstruction was 3,283/3,289
tokens. Those measurements are not complete-summary semantic accuracy on our
application or a full replication of the paper.

Evidence: `data/experiments/ncm_paper_regime/20261025_initializer_geometry_comparison_v1/result.json`.
Original-domain control: `data/experiments/ncm_recipe_update/20261022_control_v2/result.json`.
No new model inference, training, teacher calls or held-out source access occurred
in this configuration comparison.
