# Quantized deployment follow-ups to neighbourhood distillation

Research date: 2026-10-14. This adds deployment research while the fixed
clean/noise continuation is running. It changes neither training nor gates.
The original [research memo](2026-10-14_embedded-neighbourhood-distillation-research.md)
remains the producer's frozen motivation.

| Primary source | Connection to our work | Deployment boundary |
| --- | --- | --- |
| [EfficientQAT, ACL 2025](https://aclanthology.org/2025.acl-long.498.pdf), [author implementation](https://github.com/OpenGVLab/EfficientQAT) | Trains weights and quantizer parameters block by block, then freezes integer weights and trains step sizes across the assembled model. This is a particularly close analogue for local fitting followed by memory-efficient global recovery. | Uses standard uniform quantization with existing low-bit backends. Its 2-bit models still lose benchmark accuracy, and experiments use 4,096 calibration examples and A100 hardware. Neither our 64 training requests nor factorized operators have been validated by that work. |
| [MobileQuant](https://arxiv.org/pdf/2408.13933) | Learns weight equalization, clipping and static activation ranges globally, addressing saturation of blockwise fitting. Relevant to balancing the intermediate activation between two tensor factors before quantization. | Tests TinyLlama 1.1B, StableLM 1.6B and Gemma 2B on Galaxy S24/Snapdragon 8 Gen 3 HTP. Reports 20–50% latency/energy reductions relative to its on-device baselines. Uses W8A8/W4A8 with exceptions for nonlinear operations; it is not a demonstrated 2x improvement over our native Q4+lookup baseline. Calibration length is 2,048, below our 6,300-token batch prompt. |
| [Microsoft T-MAC author repository](https://github.com/microsoft/T-MAC) | Supplies low-bit lookup-table operators and llama.cpp integration; explicitly documents conversion of EfficientQAT GPTQ models. This is a plausible embedded runtime path for trained quantized weights. | Documents ARM kernels and Android cross-compilation. Its README warns that old x86 systems may not speed up. Support for ordinary Llama projections does not automatically support our two-factor projections: conversion, operator shapes and kernels need validation. This table lookup concerns matrix multiplication, distinct from prompt lookup decoding. |

The engineering implication is to treat quality recovery and deployable
compression as successive measured stages. If the fixed continuation recovers
quality, test low-bit factors and the remaining attention/output weights with
the same summarization gates, then implement or select a native backend that
executes that representation efficiently. EfficientQAT suggests a cheaper
global scale-only recovery after local quantization; MobileQuant suggests
matching activation granularity to mobile hardware rather than copying a
server quantizer. These are proposed adaptations, not results from our model.

For our linear factorization, a diagonal rescaling can move range between
factors: `(L D) (D^-1 R) = L R` in real arithmetic. Such a transform can be an
initialization for quantizer calibration; finite-precision rounding and the
quantized result still need quality tests. It does not allow arbitrary scaling
through SiLU, RMSNorm or softmax, and it does not make approximation lossless.

For speed, benchmark both prefill and complete bounded-summary latency at
roughly 500, 2,000 and 6,300 prompt tokens on the intended device. Keep the
128-token storage rule and actual generation caps. Report energy and resident
memory if phone hardware is available. Our current RTX 3060 PyTorch experiment
cannot establish embedded-device performance.

Assuming independent full weight copies, two same-precision models in one
original model's weight space require more compression than the current
rank-1024 student. Derived from the parent's
measured parameter counts, retaining the same base weights and reducing every
factor rank to 512 would yield 1,583,262,720 parameters per student: two FP16
students use 98.56% of one original model's FP16 weight bytes. A conventional
FFN with width 1,920 instead would yield 1,594,272,768 parameters per student,
or 99.25% for two. The latter retains the ordinary three-projection MLP shape,
which is a reason to investigate it for native deployment. Both are planning
arithmetic, not trained quality or runtime measurements; metadata, two KV
caches and other runtime buffers are additional. Nonuniform sensitivity and
measured operator latency should decide the eventual widths/ranks.

These two same-parent variants also retain identical frozen base weights.
Sharing those immutable weights would give 6,073,178,112 unique FP16 weight
bytes for the two rank-1024 students together, before buffers and metadata.
That arithmetic describes weight deduplication, separate from cross-request
prefix/KV reuse. We have not measured such a combined deployment or its speed.
