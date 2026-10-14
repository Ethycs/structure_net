# Native warmup implementation amendment

Written before development runtime outputs or new final outputs were inspected.
The existing native runner has an internal 64-token warmup, rather than a full
longest-request warmup. Preserve that producer and add a first fixture named
`excluded_warmup_longest` using the longest final prompt. The native five-repeat
loop executes it five times; exclude these five cells prospectively from all
quality and timing endpoints. The remaining twenty cases remain unchanged.
This replaces the originally planned single native longest-request warmup with
five. Student arms retain one longest-request warmup each. Report this difference;
no outcome-based exclusion or changed speed/quality threshold is permitted.
