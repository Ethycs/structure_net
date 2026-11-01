# Golf through poker on the proxy

Exploratory diagnostic `word-games-golf-poker-proxy-v1`, requested by the user.
Use advertised `claude-sonnet-5` on the existing authenticated central LiteLLM
proxy. This measures that route, not the local three-GPU WORD_GAMES server.
No deployment changes, training or communication with other people.

A new finite stochastic golf fixture has seven positions, seven named actions,
explicit transition probabilities, one stroke per action and specified additional
penalties. This is a simplified game, not regulation golf or a physical simulator.
A fresh model call chooses an injective poker vocabulary for all positions and
actions. The host validates the mapping and translates exact transition tables;
it supplies no optimal action labels. Fresh poker calls receive only the translated
game and current authorized synthetic state. Fresh direct-golf calls are controls.
No conversation history, private notes, expected answers or source/poker dictionary
are supplied to the poker player. This is a new small V3-style operational mapping
fixture; it does not claim integration of the V3 placement codec or V4 capsules.

Use paired courses and common random numbers by hole and move, alternating arm
order each wave. Start with nine holes per arm, then add three-hole paired blocks
until generated token count reaches at least 9,000 (approximately the requested
10,000). Finish the active paired block. Report input and output tokens separately;
do not pad outputs to hit a count. At most 27 holes per arm, 12 moves per hole,
70 requests and 2,400 generated tokens per playing request. Mapping cap 1,200.
Temperature zero; use no answer retries or repairs. Illegal moves fail that hole;
malformed batches, truncation or transport errors stop the experiment and preserve
partial evidence. Do not replace failed outputs with oracle actions.

Compute the expected-stroke optimum with value iteration before inference.
Report legal actions, completed holes, exact ledger copying, optimal decisions
(both overall and excluding positions with only one legal move), expected excess
strokes versus the optimum, and realized strokes versus par. Stochastic realized
scores are not an accuracy percentage. Do not grade by the model's self-assessment.
Mechanical codec checks do not establish broad semantic firewall security.

Primary speed is proxy-reported completion tokens divided by full HTTP round-trip
seconds, including proxy overhead, prompt processing and generation. Nonstreaming
requests do not measure time to first token or isolated decode speed. Save every
request, raw response, provider usage, response model identity, timings, actions,
state transitions and frozen producer under a new append-only artifact root.
Run from repository root with Pixi. Validate transitions, oracle residuals,
mapping invertibility and common-random-number behavior before generation.
