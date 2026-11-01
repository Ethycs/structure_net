# Golf/poker proxy route amendment

The first attempt (`20261101_golf_poker_proxy_v1`) ended before any model
generation: the advertised `claude-sonnet-5` route returned HTTP 400 with an
Anthropic insufficient-credit error. No mapping or task outcomes were observed.
Preserve that result. Start a new root, `20261101_golf_poker_proxy_v2`, using the
advertised `codex_sdk/gpt-5.6-sol` alias. Keep producer, courses, prompts, scoring,
randomness and stop rules unchanged. This is a transport/model-route substitution,
not an answer retry; report actual returned identity and usage. Bind this amendment
by SHA256 in a separate receipt before generation.
