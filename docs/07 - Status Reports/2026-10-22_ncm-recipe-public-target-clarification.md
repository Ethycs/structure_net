# Public target eligibility clarification

Date: 2026-10-22, before target-quality grades. In addition to the recipe plan's
20..40-word filter, require every candidate saved target to be an exact substring
of its publicly verified source passage. This makes both the passage and the
candidate judgment payload demonstrably public. Retain substring offsets and
exact public LongMemEval source-span verification. No private targets are sent.
There are 503 eligible pairs in the fixed training selection before judging;
take 128 by the declared fixed ID-hash order, then the first 32 passing targets.
This explicitly restricts the bounded diagnostic to extractive target pairs and
does not represent the entire application distribution. No outcome has been
used to choose this filter or change the quality rubric.
