# WORD_GAMES Bridges

Last verified: 2026-11-11. The endpoint is active as a managed **rabbit user
service**, with lingering enabled for logout/boot persistence. Dedicated-account
infrastructure adoption is prepared but has not been performed.

The CPU bridge exposes the Neighborhood V3/V4 reservation workflow over the existing
`qwen3.8-35b-a3b` LiteLLM → WORD_GAMES router → Triton deployment. It invents an
operational metaphor, validates a reversible schema map, compiles the exact action
contract through that map, translates scoped state
and requests, lets a controller propose actions, and lets the doer commit through
the original invariant-checking Engine. Sessions persist in SQLite. No Claude path
or additional GPU process is used.

This first release supports the supplied museum reservation grammar and fixture.
It is **not yet a general messaging proxy or arbitrary tool executor**. The engine
records are a local simulation; committing a reservation does not book a real
external resource. Source authority and the Python runtime remain fixed. V4
recursion translates the deployment and generates a fresh mapping; the host codec
transcodes the checkpoint exactly. This differs from asking a model to reproduce
the entire checkpoint in one response. Unlike the original bundle, the exact
manual and action schema are compiled from the fixed Engine grammar; the model
generates the metaphor, vocabulary, task language and descendant deployment prose.
Invalid table generations are retained, with at most one new generation per
table using validation feedback; there is no silent repair or model fallback.
The controller similarly receives at most one validation-feedback retry before
the host rejects a proposal. Both original and replacement outputs are retained.
Structural checks do not prove semantic
equivalence of a generated metaphor.

## Endpoint and credentials

Preferred URL: **`http://blackbox.zt:18092`**, bound only to
`10.158.218.141:18092` (ZeroTier). HTTP traffic is carried inside ZeroTier; this
listener does not terminate TLS. Outbound LiteLLM HTTPS verifies the infrastructure
CA installed at `~/.local/lib/word-games-bridges/ca.crt` on blackbox.

Every application request supplies `Authorization: Bearer <LITELLM_KEY>`. Use an
existing inference key limited to `qwen3.8-35b-a3b`. The bridge authenticates it at
LiteLLM's `/v1/models` and requires that alias in the returned catalog. The existing
inference key cannot call `/key/info`; it returns 403. Invalid keys return 401 from
the catalog. No new service key is required. Caller keys exist in request memory
and are forwarded to LiteLLM for that request; they are never stored or logged.
The service runs without core dumps as the unprivileged `rabbit` user. Its API
separates agent capabilities; agents with filesystem access as that same Linux
user are outside this isolation boundary.

Session creation returns separate random capabilities for the two roles. Subsequent
requests also supply `X-Bridge-Capability`. Give each agent only its own role file.
A controller capability cannot read source state, change mappings, issue authority
or commit actions. Doer authority includes source views and committing proposals.
The session creator receives both capabilities and is therefore trusted.

Only `GET /health` is unauthenticated. It checks that the bridge listener is running;
it deliberately does not load Qwen or claim Qwen is resident. All model calls use
`num_retries=0`, `fallbacks=[]` and verify the precise LiteLLM deployment ID. A parked
or unavailable model returns HTTP 503 without committing session state. Startup
does not call inference or load a model. The existing WORD_GAMES router retains
responsibility for its normal request-time model selection.

## Use from StructureNet

Export your existing inference key as `LITELLM_KEY`, then from the repository root:

```sh
pixi run -e word-games python -m experiments.word_games.bridges.cli create --output /tmp/my-bridge-session
pixi run -e word-games python -m experiments.word_games.bridges.cli view --session /tmp/my-bridge-session/doer.json
```

Create a `prepare.json` containing:

```json
{
  "request_id": "place-1",
  "version": 0,
  "goal": "Reserve loan:bronze in gal:north slot N1, due 2027-01-10.",
  "context": {
    "scope": ["gal:north"],
    "loan_ids": ["loan:bronze"],
    "allowed_ops": ["reserve"],
    "event_id": "place-1"
  }
}
```

```sh
pixi run -e word-games python -m experiments.word_games.bridges.cli prepare --session /tmp/my-bridge-session/doer.json --body prepare.json
pixi run -e word-games python -m experiments.word_games.bridges.cli view --session /tmp/my-bridge-session/controller.json
```

The controller can submit its own target-language action JSON, or use Qwen via
`solve` with `{"request_id":"solve-1","version":1}`. The doer then uses `apply`
with `{"request_id":"apply-1","version":2}`. Pass each JSON file with `--body`.
Every successful mutation increments `version`. Read the current role view after
an uncertain response; replaying the exact same `request_id` and body returns the
original receipt without repeating the action. Reusing an ID with a different
body or supplying a stale version returns 409.

## HTTP contract

| Method and path | Role | Result |
|---|---|---|
| `POST /v1/sessions`, `{"profile":"reservations-v4"}` | authenticated creator | Fresh model-selected mapping and separate capabilities |
| `GET /v1/sessions/{id}/controller` | controller | Target manual, translated task/view, proposal and minimal result |
| `GET /v1/sessions/{id}/doer` | doer | Source state and pending task |
| `POST /v1/sessions/{id}/doer/prepare` | doer | Issue scoped context and translate goal; see example |
| `POST /v1/sessions/{id}/controller/solve` | controller | Qwen proposes target actions from the translated view only |
| `POST /v1/sessions/{id}/controller/submit` | controller | Same, using a caller-supplied `proposal` object |
| `POST /v1/sessions/{id}/doer/apply` | doer | Decode and atomically commit or clarify |
| `POST /v1/sessions/{id}/doer/retranslate` | doer | New metaphor/deployment, exact checkpoint preservation; no pending task; maximum generation 3 |

All POSTs after creation require `request_id` and `version`. Requests are bounded
to 128 KiB. No generic shell, file-read, Python REPL or external action endpoint
exists. The bridge preserves engine state, context, receipts, aliases and mapping
across restarts; the Qwen controller sees the current scoped state each turn, not
an unbounded model conversation history.

Protected titles, notes, names, actors and source identifiers become stable opaque
handles before crossing. Quoted exact payload writes round-trip through the host.
This preserves equality/identity and exact writes; semantic searches within
arbitrary note prose need a later workflow extension. It prevents direct source
retrieval, but is not a proof against all semantic inference or model mistranslation.

## Lifecycle and persistence

```sh
curl -fsS http://blackbox.zt:18092/health
systemctl --user start word-games-bridges
systemctl --user stop word-games-bridges
systemctl --user restart word-games-bridges
systemctl --user status word-games-bridges
journalctl --user -u word-games-bridges -n 50
```

Run these commands as `rabbit` on blackbox. Application:
`~/.local/lib/word-games-bridges`, a CPU-only locked Pixi Python 3.11.13 runtime.
State: `~/.local/share/word-games-bridges/bridges.sqlite` and its SQLite WAL files,
accessible only to `rabbit`/root. The private database contains source messages,
model requests/responses, mappings and capability hashes. **Not yet in a backup
set.** Stop the service before copying the database, or use SQLite's backup API;
copying only the main file while it runs can omit committed WAL data.

The current deployment uses `deploy_user.py` with no sudo:

```sh
WG_RELEASE=/path/to/unpacked/release pixi exec --spec pyinfra==3.11.0 pyinfra @local deployments/word_games_bridges/deploy_user.py -y
```

The separate `deploy.py` is prepared for infrastructure's
`pipelines/deploy_word_games_bridges.py` and a dedicated service account. It uses
`/opt/word-games-bridges`, `/etc/word-games-bridges/ca.crt` and
`/var/lib/word-games-bridges/bridges.sqlite`. Do not start it alongside the current
user unit on the same port. Adoption requires stopping the user unit, migrating
the database with its permissions, and then enabling the dedicated-account unit.
Set `WG_RELEASE` to the unpacked release directory for either pipeline. The archive
contains application code, the Pixi lock, unit and public CA, never credentials or
session state. It restarts only this CPU service, drains requests on stop, and
never restarts Triton or changes its model files. A redeploy restarts the bridge.

HTTP 403: check key access to Qwen and the role-specific capability. HTTP 409: read
the current version or replay the original request. HTTP 422: invalid input,
mapping or proposal; the session state has not changed. HTTP 502: malformed,
truncated or wrong-route model output. HTTP 503: bridge capacity or model route
unavailable. Start with `/health`, then the WORD_GAMES router/Triton readiness and
LiteLLM catalog. Errors deliberately omit private source details.

Validation command (creates a new immutable attempt directory):

```sh
pixi run -e word-games python -m experiments.word_games.bridges.smoke --output data/experiments/word_games/NEW_ATTEMPT
```

The smoke test checks a small reservation trajectory and recursive checkpoint
resumption. It does not establish production task accuracy or parity without the proxy.
