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

The launcher now runs OpenCode inside **bubblewrap 0.6.1** on blackbox. It fails
if bubblewrap or the installed OpenCode binary is unavailable; there is no
unsandboxed fallback. It uses the installed Snap's ELF directly, since the Snap
launcher expects host services that are outside the sandbox.

The selected `--workspace` is writable. Configuration, OpenCode's binary, system
files and the infrastructure CA are read-only. Other home directories and projects,
`.claude`, `.codex`, `.ssh`, `/data`, `/opt` and the host process tree are hidden.
The sandbox has private mount, user, PID and IPC namespaces and private `/tmp`
and `/dev`. **Networking is shared** for the existing LiteLLM connection; this
is filesystem/process isolation, not a network firewall. All files inside the
selected workspace, including dotfiles, are deliberately exposed to the actor.
Use a dedicated task workspace when the full repository contains private data.

Only the required `LITELLM_KEY` is forwarded, with a small explicit environment.
Subscription credentials, SSH agent sockets and unrelated environment variables
are not inherited. The launcher reads `--key-env-file` outside the sandbox;
that file is not separately mounted. A model credential must remain available
to the trusted OpenCode process for inference.

OpenCode history, state and caches persist under
`.pixi/word-games-opencode-runtime/sandboxes/<workspace-path-hash>/`. Only the
current workspace's store is mounted. Other histories are masked even when the
whole StructureNet repository is the selected workspace. Existing personal
settings and the earlier unsandboxed test sessions are unchanged; those legacy
sessions are not automatically imported into the new stores. Resume a sandboxed
session with the **same workspace path** and its recorded session ID.

The launcher uses `--pure` and disables project config loading and automatic updates. Raw tool events,
per-step token usage and process duration are retained under the requested output
directory, along with the sandbox command (no key), startup result and policy
source hash. A zero exit code is not treated as task success.

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

## Bubblewrap verification

The original infrastructure handoff described sandboxing the **bridge HTTP
service**, but no live unit had been changed. The user clarified that the intended
target was **OpenCode sessions**. This change wraps the actor launcher; the
`word-games-bridges` service on port 18092 remains unchanged and is not bubblewrapped.

The real OS-boundary checks run with:

```sh
pixi run -e word-games python -m pytest --confcutdir=tests/word_games -q tests/word_games/test_opencode_sandbox.py
```

Five tests passed, covering allowed writes, hidden host files and processes,
symlink escapes, read-only configuration including a second path inside the
workspace, environment filtering, separate workspace histories, persistence,
and actual OpenCode startup/configuration loading. These are checks of the stated
policy, not an exhaustive sandbox security assessment.

The first two sandbox attempts stopped before inference because OpenCode tried
to initialize its read-only configuration directory and `.gitignore`. The
launcher now precreates those files without making the configuration writable.
Both failed attempts remain recorded. The first attempt also had a missing test
fixture, caught before any actor ran.

The working live attempt used Qwen's native `read → edit → read` sequence and
passed full-ledger equality on the initial reservation request in **15.04 s**.
Live process observations confirmed separate mount, PID and user namespaces;
the network namespace was shared as intended. A fresh sandbox invocation resumed
the same OpenCode session, cancelled only that reservation, and passed exact
equality with the original ledger in **15.74 s**. The evidence is in
`data/experiments/word_games/20261111_opencode_bubblewrap_v3/`; earlier failures
are `20261111_opencode_bubblewrap_v1/` and `20261111_opencode_bubblewrap_v2/`.
The continuation is `20261111_opencode_bubblewrap_resume_v1/`; see
`bubblewrap-validation.json` for the verification receipt. Timings include
process and transport overhead and are not a controlled speed comparison.

This establishes an existing native-actor harness for the bridge work. It does
**not** turn the previously deployed reservation proxy into a complete double-pass
OpenCode integration. The current bridge service and its earlier systems checks
remain separate from these actor results; the same OpenCode actor can now be used
for a direct baseline and for the subsequent bridge comparison.
