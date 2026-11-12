# Coding-recovery resource-failure accounting

2026-11-11. Preserved attempt v2 completed native c01 (4/4 hidden tests), then
stopped on translated c01 ContextOverflowError (8,244 requested versus 8,192
available tokens). The translated actor had already written code, but its session
did not complete. No recovery attacks or complete task pair were recorded.

This is a measured resource limitation of the existing configuration, not a
service outage. For a fresh v3 run, count context overflow and the fixed actor
timeout as failures of the corresponding arm and continue independent tasks.
Other transport errors still stop the campaign. Keep tasks, payloads, prompts,
serialization, model, 8K context, disabled compaction, time/step limits, mapping,
and exact hidden-test requirements unchanged. Do not repair answers or retry a
failed arm. Preserve v1/v2 and do not pool their partial executions into v3.

Primary end-to-end task/test success requires completed actor delivery AND correct
execution. If code exists after an actor resource failure, evaluate it once and
record artifact-only test correctness as a secondary diagnostic. It cannot turn
a failed delivered task into a pass. This amendment is made before any full pair,
information-recovery outcome or other task has been inspected.
