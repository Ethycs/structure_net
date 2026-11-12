# Coding-recovery input presentation amendment

2026-11-11, after attempt `20261111_coding_recovery_v1`, before any paired task
or recovery result. The preserved first native actor made no implementation:
OpenCode's read tool truncated the single-line canonical task JSON to 2,000
characters. Since canonical sorting put example_input first, the task goal was
not visible. Repeated reads/grep exhausted the 8,192-token model context, ending
with ContextOverflowError (8,603 requested tokens). The campaign stopped with
0/8 complete pairs, not a measured zero accuracy for eight tasks.

For v2, serialize actor task.json using indented JSON in insertion order, placing
the goal first. The task payload, prompt, two arms, hidden tests, model, budgets,
bridge mapping and recovery attacks are unchanged. The full original failure and
producer snapshot remain immutable. Run all eight pairs in fresh workspaces and
sessions; do not pool incomplete v1 execution with v2. The one trivial startup
check succeeded but did not exercise long JSON lines. No held-out answer or
grading feedback is used to repair a model answer.
