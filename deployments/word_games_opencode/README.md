# Native Qwen actor in OpenCode

Last verified: 2026-11-11.

Use the installed **OpenCode 1.19.5** harness with **Qwen3.8-35B-A3B** as the
native task actor. OpenCode owns the model/tool loop, file operations, session
history, permissions and session resumption. The Python launcher only configures
and invokes the existing CLI and retains its events; it does not implement an
agent loop. The attempted custom actor loop was removed before deployment.

The model route is OpenCode → LiteLLM `https://central-dev.zt:4000/v1` → the
WORD_GAMES Triton router → the existing Qwen worker. Native function calls were
checked through that route: Qwen called an `add` function with the requested two
integers. No checkpoint or inference configuration changed.

The configuration uses the **1.19.5 schema**, verified against
[its provider configuration source](https://github.com/anomalyco/opencode/blob/v1.19.5/packages/core/src/v1/config/provider.ts).
Current [OpenCode documentation](https://opencode.ai/docs/providers/) describes
the newer schema, so do not substitute its field names without upgrading and
revalidating the harness. The launcher checks the installed version.

## Running a task

From the StructureNet repository root, with `LITELLM_KEY` set:

```sh
pixi run -e word-games python -m experiments.word_games.opencode_harness \
  --workspace /absolute/path/to/task-workspace \
  --prompt 'Read the task files, make the requested changes, and verify the result.' \
  --output data/experiments/word_games/NEW_NATIVE_ACTOR_RUN
```

`--key-env-file .env` can read the existing inference key instead of requiring an
export. Its contents are not passed as a prompt or written into the provider config.
The provider config contains only `{env:LITELLM_KEY}`. The infrastructure CA is
supplied to OpenCode using `NODE_EXTRA_CA_CERTS`.

Use a dedicated task workspace. This profile enables OpenCode's native file read,
edit, glob and grep tools within that workspace; shell, network tools, delegation
and external-directory access are disabled for this first task demonstration.
These are ordinary OpenCode permissions, not a replacement tool interpreter.
The configured context is the actual 8,192 tokens, with 1,024 output tokens and
12 maximum agent steps. Automatic compaction is disabled in this initial measured
profile; long-running task memory remains to be evaluated.

To continue the **same OpenCode session**, use the session ID recorded in
`result.json`:

```sh
pixi run -e word-games python -m experiments.word_games.opencode_harness \
  --workspace /absolute/path/to/task-workspace \
  --session SESSION_ID \
  --prompt 'Continue the task using the preceding conversation.' \
  --output data/experiments/word_games/NEW_CONTINUATION
```

OpenCode state is isolated under `.pixi/word-games-opencode-runtime/`; existing
personal OpenCode settings and sessions are not changed. The launcher uses `--pure`
and disables project config loading and automatic updates. Raw tool events,
per-step token usage and process duration are retained under the requested output
directory. A zero exit code is not treated as task success.

## What was actually verified

The file-backed reservation task asked Qwen to inspect a ledger and reserve Bronze
Heron in the first free North slot while preserving other records and rules. N1
was occupied, so N2 was the required choice. The harness was not given scripted
tool steps; Qwen selected its `read` and `edit` calls.

- The first integration attempt hit a permission-pattern error and changed no task
  state. The configuration was corrected; the failed attempt remains recorded.
- On the next fresh run, Qwen used `read → edit → edit → read` and selected N2.
  **The strict task check failed:** it also changed a gallery deadline rule.
- A follow-up identifying that unintended change resumed the **same session**.
  Qwen used `read → edit → read`, restored the rule and retained the correct
  reservation. The independently checked final file then passed exactly.
- Durations were 20.76 seconds for the initial task and 14.94 seconds for the
  corrective continuation, including OpenCode process/transport overhead. These
  are not model decoding-speed measurements or a general accuracy estimate.

Evidence:
`data/experiments/word_games/20261111_opencode_native_v1/`,
`20261111_opencode_native_v2/`, and `20261111_opencode_native_resume_v1/`.

This establishes an existing native-actor harness for the bridge work. It does
**not** turn the previously deployed reservation proxy into a complete double-pass
OpenCode integration. The current bridge service and its earlier systems checks
remain separate from these actor results; the same OpenCode actor can now be used
for a direct baseline and for the subsequent bridge comparison.
