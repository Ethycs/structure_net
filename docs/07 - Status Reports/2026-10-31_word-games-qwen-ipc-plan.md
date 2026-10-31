# WORD_GAMES Qwen worker IPC

Prospective systems check, 2026-10-31. User explicitly selected the RTX 2060
SUPER for the 3B draft and requested preserving the Qwen3-8B service on the
6 GB RTX 2060. Keep the 14B target on the RTX 3060.

Use two isolated native b11960 worker processes, one GPU each, connected by
persistent local Unix-domain HTTP sockets in an owner-only directory. No TCP
listener or external network is needed. Exchange shared token IDs and bind
draft responses to request identity, tokenizer identity and confirmed prefix.
Measure health-request IPC latency separately from real draft generation.

The first implementation is a transport and correctness reference: the target
generates the verification IDs autoregressively. It does not yet replace
llama.cpp's internal batched speculative verifier with an external drafter,
and it cannot establish speculative speedup. Explicitly keep that limitation
in results. Compare candidate IDs against target-generated IDs; include a
deliberately wrong candidate as a rejection control and unit tests for stale
responses, mismatched tokenizers and changed prefixes.

Use the pinned Qwen pair and frozen three-request pilot inputs. Hash producer,
weights, binary and tokenizer evidence; retain command, memory, IPC payload
sizes, 100 warmed health round trips and per-request reference acceptance.
Stop on identity mismatch, missing GPU offload, OOM, transport error or invalid
token response. Preserve other resident services and close only owned workers.
Use the locked Pixi controller and keep compilation caches outside data/.

No new model-quality claim, training, production deployment or CUDA peer-memory
sharing is implied. The firewall still owns the receiver's authorized view.
