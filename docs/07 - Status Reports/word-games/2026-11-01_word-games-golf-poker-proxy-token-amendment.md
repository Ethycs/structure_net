# Proxy token instrumentation amendment

The `codex_sdk/gpt-5.6-sol` route returned a valid model-generated mapping but
reported zero prompt/completion/total tokens. Attempt v2 stopped at the usage
validation before gameplay. Preserve it. No player outcomes have been observed.

Continue in fresh root `20261101_golf_poker_proxy_v3`, importing the exact v2
mapping request/response and its measured elapsed time; do not regenerate the
mapping. The saved HTTP call took about 10 seconds. All playing calls are fresh.

Count visible prompt and response text using the already running Qwen GGUF
tokenizer through its read-only `/tokenize` API, without special tokens. Report
this explicitly as Qwen-token-equivalent throughput, not the provider's native
token rate. Prompt counts exclude chat framing and SDK-injected context. Neither
hidden reasoning nor provider context is observable; reported zero API usage
does not mean no tokens or cost. Save raw usage alongside reference counts.
Use reference output tokens for the approximately 10K stop rule. Tokenizer calls
are outside the timed inference round trip. Keep the game, seed, prompts and
mechanical scoring unchanged. Source and tokenizer manifest identities are frozen.
