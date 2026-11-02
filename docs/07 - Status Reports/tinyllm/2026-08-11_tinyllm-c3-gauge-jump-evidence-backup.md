# TinyLLM C3 gauge-jump evidence backup receipt

**Status:** VERIFIED FINAL META-INCLUSIVE BACKUP

**Date:** 2026-08-11

## DVC

The final repository data tree is tracked by:

```text
root:   7c2789905560eaf86d36abbcc0dfb032.dir
size:   54,464,901,441 logical bytes
files:  4,143
```

The last DVC push uploaded three objects. An immediate replay reported
`Everything is up to date`.

The final meta-hypothesis record is:

```text
data/meta_hypotheses/tinyllm-c3-gauge-jump-joint-typed-score-v1.json
SHA-256: 72ad47130714768dfa6e2cb94073cb39fa09fdf9b64e58e26f08e0f1c02b1691
```

## lakeFS

The final DVC object graph is sealed on `lakefs://artifacts/main`:

```text
commit:  4ba92780a3b51b56d8591a997f25a107a7184c3136974bf9b1d44fc7f755acd6
parent:  1d399a19bb4032b0926f356a44f98b37859c6177def8bf835ff19c2e3fcc00b8
message: Seal final TinyLLM C3 gauge-jump meta evidence
```

Post-commit `lakectl diff --prefix structure-net lakefs://artifacts/main`
returned no objects. Direct object inspection recovered checksum
`7c2789905560eaf86d36abbcc0dfb032` for the `666,418`-byte DVC directory
manifest.

The parent commit is the intermediate report-before-meta checkpoint. Its own
parent is the preceding program head
`d96febecfa829fc860965429d33173c98cbfa6d97bc160ba079a3614175e14bc`.

