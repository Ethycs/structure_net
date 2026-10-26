# Full source-only tokenization for bounded public-data feature streaming

Prepare every admitted pair in `ncm-public-training-catalog-v1` for the same
frozen original Llama3B body used in previous NCM experiments. This phase reads
only an explicit source-field projection before calling the original source
tokenization/alignment function. Reference text, target IDs and VQ codes never
enter that function. No model weights, features or gradients are computed.

Keep the original system prompt, date23 Oct2026, full passage, native prompt
tokenizer, GPT2 source positions and exact character-overlap alignment. Reuse
`ncm_additional_source_metadata.prepare_one` and its first/longest old TRAIN
geometry controls. Preserve full source strings and all original positions;
the8192 source frame maximum is a hard error, never a truncation rule.

First prepare a CPU pilot: the first admitted row, plus distinct rows nearest
192,256,384,512,768,1024,1536,2048 cl100k source tokens (ID tie break). Compare
the two existing TRAIN controls exactly. After actual pilot exit0, prepare
the entire admitted catalog in original order, requiring every pilot row to
reproduce exactly. Write immutable indexed JSONL metadata and per-record
hashes. A separate index contains lengths and offsets, not article strings or
feature tensors. A PyTorch Dataset reader must preserve requested batch order,
including duplicate draws, and reject changed bytes or identities.

Record the total and maximum actual source/native prompt positions. These
measurements inform the later bounded feature pipeline; do not allocate an
eager whole-corpus FP32 feature cache. Body feature extraction will use the
existing original BF16 body and exact FP32 alignment/lossless feature storage,
after a separate CUDA extraction pilot. A larger training launch still needs
fixed optimizer/sampling budgets and a complete batch4 optimizer lifecycle.

Fresh roots under `data/experiments/ncm_paper_regime`:
`20261026_public_source_metadata_pilot_v1` and
`20261026_public_source_metadata_full_v1`. Use locked Pixi `ncm` and CPU only:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  pixi run --locked -e ncm python -m experiments.structure_net.ncm_public_source_metadata pilot
```

After recording its actual terminal receipt, run the same module with `full`.
Full output requires an independent binding/tokenization audit before feature
or training use. The application95% complete-summary goal and sealed final
protocol remain unchanged. These are data preparation results, not quality.
