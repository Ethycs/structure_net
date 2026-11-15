# Raw-text inference from the compressed Qwen2.5-3B bundle

This entry point uses the custom 36-block `CompactQwenStack` loader. It is for the eventual exported Qwen2.5-3B bundle; it does not create a bundle or establish accuracy, generation quality, or speed. A generic Hugging Face `AutoModelForCausalLM.from_pretrained` call cannot load this custom architecture.

Use the repository revision and locked Pixi environment retained with the export. The CLI checks the bundle's runtime source hashes and exact PyTorch/Transformers versions, then the existing loader checks all bundle file hashes and tensor metadata. Tokenizer, chat template, generation configuration, embedding and weights come only from the local bundle. Dense source weights, fitted bases and training checkpoints are unnecessary at inference.

From the repository root, replace `/absolute/path/to/compact_export/model` with the completed final-gate export directory:

```bash
CUDA_VISIBLE_DEVICES='' pixi run --locked --offline -e ncm python \
  -m experiments.structure_net.jspace_3b_infer \
  --bundle /absolute/path/to/compact_export/model \
  --prompt 'Explain why a queue preserves insertion order.' \
  --max-new-tokens 32 --device cpu
```

The default applies the exported chat template with a user message and assistant-generation prefix. Add `--plain` to use raw completion text without that template. Generation is greedy; pinned EOS/padding tokens, repetition penalty and other generation processors remain in effect. `--max-new-tokens` explicitly sets the output budget. Add `--json` to see generated token IDs, input/output counts, source fingerprint and runtime versions alongside the decoded completion.

**The supported capacity is at most 416 positions total**, or the smaller limit declared by a particular bundle. The input count includes the chat template and special tokens. Input tokens plus the entire requested output budget must fit before generation begins, even when EOS might stop earlier. Oversized requests fail; the CLI does not truncate them. Native long-context settings in `config.json` do not expand this calibrated limit.

For CUDA on this machine, acquire a reservation around a fresh process:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 /usr/local/bin/inference-ctl lease run \
  --allocation 3060=11264 --minutes 10 \
  --reason 'Compressed Qwen3B raw-text inference' -- \
  pixi run --locked --offline -e ncm python \
  -m experiments.structure_net.jspace_3b_infer \
  --bundle /absolute/path/to/compact_export/model \
  --prompt 'Explain why a queue preserves insertion order.' \
  --max-new-tokens 32 --device cuda:0 --json
```

The reservation wrapper supplies the visible physical GPU UUID. The CLI verifies the active lease before its first CUDA query, caps PyTorch's native allocator at the leased bytes minus 256 MiB, and rechecks the same lease before loading, on every forward pass, and after generation. Missing, changed or undersized reservations fail. The cap bounds allocation; it cannot make an oversized live working set fit. This command is an inference entry point, not a throughput benchmark.

CPU smoke tests use a tiny 36-block bundle and remove its native source files before text tokenization, greedy generation and decoding. They also test capacity, runtime-source/version checks and reservation rejection without starting a GPU:

```bash
CUDA_VISIBLE_DEVICES='' pixi run --locked --offline -e ncm python -m pytest \
  --confcutdir=tests/structure_net tests/structure_net/test_jspace_3b_infer.py -q
```

The frozen runtime archive and Pixi lock remain the reproducibility boundary. A companion receipt can pin this CLI and runbook after the real final bundle exists; no portable installation format or real-model smoke result is claimed here.
