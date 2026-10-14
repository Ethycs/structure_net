# Embedded and blockwise distillation research for the Llama continuation

Research date: 2026-10-14. Primary papers, proceedings and author repositories
only. This memo informs the new continuation; it does not change the failed
parent result or claim measured phone performance.

| Work | Relevant evidence | Application and boundary |
| --- | --- | --- |
| [Neighbourhood Distillation, 2020](https://arxiv.org/pdf/2010.01189) | Independently trains subnetworks, assembles them, then fine-tunes the complete student. Its Gaussian-noise experiments feed noise into neighbourhood inputs. | Supports noisy-activation experiments and an assembled recovery stage. The reported experiments are vision models; Gaussian inputs are not randomly initialized weights and do not prove Llama summarization quality. |
| [Puzzle, ICML 2025](https://proceedings.mlr.press/v267/bercovich25a.html), [paper](https://arxiv.org/pdf/2411.19146) | Local distillation builds alternative LLM blocks; hardware-aware search chooses the architecture, and global KD repairs compatibility. Authors report 2.17x throughput and 98.4% relative benchmark retention for their 49B/51B models. | Closest LLM analogue. It explicitly warns that parent-input local training cannot adapt later blocks to earlier student errors. Results use H100/FP8 and billions of training tokens, not phones or our small dataset. |
| [LANA, ECCV 2022](https://arxiv.org/abs/2107.10624) | Distills candidate layer replacements and selects them under a measured latency budget. | Useful precedent for selecting ranks or block types by actual runtime and quality. Vision experiments do not establish autoregressive Llama speed. |
| [Pipe-BD, DATE 2023](https://arxiv.org/abs/2301.12443), [code](https://github.com/hongsunjang/pipe-bd) | Pipelines teacher/student blocks to reduce repeated teacher execution and improve training-device utilization. | Relevant to training efficiency, not a deployed-model compression method. We have only GPU 0, so cached teacher states provide the practical reuse of teacher computation here. |
| [MobileLLM, ICML 2024](https://arxiv.org/abs/2402.14905), [author code](https://github.com/facebookresearch/MobileLLM) | Builds small decoder models using deep/thin layouts, tied embeddings, GQA and optional immediate block sharing. | Relevant mobile architecture reference. It is not neighbourhood distillation, and it does not supply a lossless retrofit for Llama 3B. Native operator support and model design matter beyond parameter count. |
| [MobileBERT, ACL 2020](https://aclanthology.org/2020.acl-main.195/), [author code](https://github.com/google-research/google-research/tree/master/mobilebert) | Uses bottleneck architecture and layer-wise transfer for resource-limited devices. | Useful for local representation matching and compact encoders. It is a bidirectional encoder for understanding tasks, not an autoregressive summarizer. |
| [SVD-LLM V2, 2025](https://arxiv.org/abs/2503.12340) | Assigns compression according to layer-dependent truncation sensitivity. | Supports investigating nonuniform ranks instead of rank 1024 everywhere. It is an initialization/compression complement, not evidence that more independent local training alone repairs assembly. |
| [CALDERA, NeurIPS 2024](https://arxiv.org/abs/2405.18886), [author code](https://github.com/pilancilab/caldera) | Represents weights with a low-precision matrix plus low-rank, low-precision factors. | Relevant to getting below the existing Q4 footprint. Its compressed residual matrix differs from our previous dense exact correction; quantization quality and kernels still need direct tests. It is not neighbourhood distillation. |

The common practical sequence is original-model initialization, local fitting,
assembled task/output recovery, and runtime measurement on the intended device.
Our parent stopped after local fitting and showed substantial assembled error.
The next experiment therefore continues the tensor factors, compares clean
training with Gaussian perturbations, emphasizes response positions, and adds
global output-distribution supervision. This is an engineering inference from
the papers and our measured failures, not a proven recipe for 95% application
accuracy.

Noise adds training diversity; it does not reduce inference work by itself.
Our two-factor MLPs remain 4.135 GB in FP16, versus 2.012 GB of native Q4
weights. Achieving the speed goal would also require a quality-preserving
representation and supported efficient kernels. A smaller model's parameter
count, a vision training speedup, and an H100 throughput result cannot be
substituted for measured summarization latency on a phone.

The new prospective design is
`docs/07 - Status Reports/2026-10-14_llama32-noise-tensor-continuation-plan.md`.
Current judging remains source-grounded assistant review with explicit fact
and length checks. Independent judging and representative application data
remain separate validation requirements for production quality.
