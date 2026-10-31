# Bonsai 2 27B for WORD_GAMES

Research assessment, 2026-10-31. **Worth testing as the target on the RTX 3060.**
No Bonsai inference or local accuracy/speed measurement has been performed.
Qwen2.5 remains the active experimental pair pending a direct comparison.

The requested release is PrismML's Ternary Bonsai 2 27B, derived from
Qwen3.8-27B. Its advertised 98.2% is the ratio of aggregate benchmark scores
(83.9 versus 85.4 in the announcement), not 98.2% task accuracy or exact
preservation. Their advertised 143 tok/s is on an RTX 5090, not our hardware.
[Announcement](https://prismml.com/news/bonsai-2-27b).

The model combines ternary weights, group scales and a fixed Hadamard rotation;
the runtime applies the corresponding activation transform. The paper's
quality evaluation uses thinking mode and xhigh reasoning, with output budgets
of 32,768–81,920 tokens. That is far from our 128-token stored summaries and
bounded JSON workflow. Evaluate concise/non-thinking operation directly, and
count all generated reasoning tokens in latency if thinking is enabled.
[Whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf).

| Language-model packing | Download size | Runtime |
| --- | ---: | --- |
| PTQ1_0 | 5.947 GB | PrismML fork |
| PQ2_0 | 7.206 GB | PrismML fork |
| Q2_0 | 7.626 GB | Upstream llama.cpp b11556+ |

These are weight-file sizes, not total VRAM. All are smaller than the 8.988 GB
Qwen14B checkpoint we downloaded. I expect an 8K text-only configuration to be
a plausible fit on the 12 GB 3060, subject to a real memory pilot. Our pinned
b11960 meets the model card's Q2_0 minimum version. The optional vision tower
is unnecessary for WORD_GAMES. PQ2_0 and PTQ1_0 trade unpacking cost against
memory traffic; neither is universally faster.
[Model card and files](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).

The maintainers distinguish low-bit format support from the required model
graph, sign flips and activation transforms. A file loading successfully in
an older engine does not establish correctness. The fork supports the compact
packings on CUDA; hardware-specific speed still needs measurement.
[Backend support](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/BACKEND-SUPPORT.md).

## Locally checked tokenizer compatibility

I fetched only the first 16 MiB of the official 27B Q2_0 file, at repository
revision `94dd51cdbbec5309ed8b76a12f3268bc4fdceb93`, and parsed its GGUF metadata.
Its vocabulary has 248,320 entries versus Qwen2.5's 152,064. Of their shared
index range, 151,783 token spellings differ; EOS, pre-tokenizer and merges also
differ. **Our Qwen2.5-3B cannot be used as a direct shared-ID draft for Bonsai.**
A cross-tokenizer adapter or a matching-family drafter would be required.
The current Qwen IPC reference intentionally rejects this mismatch.

Evidence: `data/experiments/word_games/20261031_bonsai_research_v1/27b_tokenizer_comparison.json`.
Header SHA256: `37e69113a76ffa3e2e12d3750fe884083c8b11577beb2d907a6f41276bdd6d72`.
No full checkpoint was downloaded. An earlier 2B header inspection remains
separately recorded; the conclusion above uses the requested 27B itself.

The next useful experiment is **Bonsai 27B Q2_0 target-only**, with the same
WORD_GAMES structural and semantic checks and a bounded generation policy,
before adding speculation. That isolates whether it improves the target's
currently poor 5/12 task success. If it passes, compare the compact fork
packings on the same card and inputs. Neither published retention nor nominal
parameter count predicts that result.
