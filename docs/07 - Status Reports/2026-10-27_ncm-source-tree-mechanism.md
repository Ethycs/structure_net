# Source hierarchy accessible to every NCM layer

Exploratory, unregistered mechanism check `ncm-source-tree-mechanism-v1`.
This follows the user's clarification that the tree organizes source information,
not connections between transformer layers. This is a CPU-only synthetic
contract check; no trained-quality or inference-speed hypothesis is tested.

Build an eight-way tree over the shared projected source K/V: tokens, contiguous
eight-token groups, 64-token groups, 512-token groups, and a document root for a
2,048-token source. Padding is excluded and parents are weighted by valid leaf
counts, avoiding mean-of-means bias for incomplete groups. Compute the tree once
per source context and share it across all twelve existing NCM blocks. Each block
has its own query and learned attention over hierarchy levels. Preserve the full
leaf attention path so exact facts remain accessible. Pooled ancestors are lossy;
the retained leaves do not imply lossless generated summaries.

Start from a loaded model and use the existing source-coverage expansion for the
nine blocks that lack direct source attention. Retain original parameter objects;
reuse each leaf adapter's query projection. Add zero-initialized hierarchy output
projections and zero level-routing weights, without consuming RNG. In deterministic
evaluation, initial outputs must exactly equal the original model. Training dropout
draw order is not claimed equal after adding the branches. The frozen source
features remain detached. Generator depth and the word codec are unchanged.

The fixed CPU checks cover masked weighted pooling, root correctness with an
incomplete group, initial exact evaluation parity, original tensor preservation,
RNG preservation, strict state reload, finite nonzero gradients to all twelve new
output projections, subsequent gradients to level routing after activating the
outputs, detached source features, invalid geometry, repeated installation, and
hook cleanup. Any failure blocks a real-model pilot; preserve failed receipts.
Artifacts use the fresh root `20261027_source_tree_CPU_qualification_v1`.

Systems amendment before mechanism outcomes: the first test invocation stopped
in repository-wide `tests/conftest.py` because the locked NCM environment lacks
SciPy. It executed no mechanism tests. Preserve that collection failure in v1;
rerun the same focused checks with `--noconftest` in fresh root
`20261027_source_tree_CPU_qualification_v2`. These tests use no repository-wide
fixtures and do not require SciPy.

This dense hierarchy adds work. For 2,048 leaves it stores 256 + 32 + 4 + 1 = 293
ancestor positions in addition to the leaves. A subsequent sparse-routing design
must measure actual latency and numerical-fact retention rather than claiming a
speedup from asymptotic attention counts. No optimizer run, source split, semantic
judge call, or sealed-final access is part of this check. Current training workers
and the ongoing pair census remain unchanged.

A later matched quality comparison should use the same recovered-data split and
training exposure for: current three-source-layer baseline, dense source attention
at all twelve layers, and the twelve-layer hierarchy. This separates increased
source coverage from hierarchy effects. Freeze its checkpoint/data identities,
training budget and semantic endpoint before launching it. The primary evidence
must remain complete-summary faithfulness and salience, with numerical errors
reported separately. That comparison has not run.

Related primary literature: [Hierarchical Attention Transformers](https://arxiv.org/abs/2104.07545)
use hierarchical source representations for generation; [Treeformer](https://arxiv.org/abs/2208.09015)
uses learned tree navigation for sparse attention. This prototype is a dense,
function-preserving multiscale adapter, not a replication of either paper.
