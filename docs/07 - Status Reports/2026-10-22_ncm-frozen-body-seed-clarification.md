# Frozen-body NCM seed clarification

2026-10-22, during frozen feature extraction for the main run, before main-run
optimizer updates or held-out NCM quality outcomes were inspected.

The plan called the model/dropout seed 42. The frozen implementation uses 42
for adapter initialization, 2042 for minibatch sampling, and **52 (42 + 10)**
for dropout in both arms. Resetting dropout after adapter initialization keeps
the arms' dropout streams aligned despite the hybrid's extra initialization.
The producing source already encoded this before the successful CUDA pilot.

This corrects the description of the seed streams. No producing code, data,
configuration, checkpoint, threshold, requested update count, or endpoint is
changed. The original plan, successful pilot, main producer and raw artifacts
remain preserved. The study is exploratory and underpowered in all cases.
