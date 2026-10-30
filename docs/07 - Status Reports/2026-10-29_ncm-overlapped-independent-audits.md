# Overlap independent audit admission with generation

The corrected two-case inference pilot completed at tool chunk34bbc6/session83542
and exactly reproduced the earlier saved prediction. The CPU semantic worker's
own admission completed at chunk5a0357/session73015. The full128 generation is
running; no latest-model semantic judge call or grade has occurred. The two
pilot outputs have been inspected and remain repetitive and unsupported.

Admission rereads about170GB of historical weights and features on rotational
storage. Start each readout's independent audit in its own fresh CPU-only Pixi
process while full generation runs. Each starts with an empty measured hash map,
reads the complete unchanged admission dependencies itself, and retains only its
own live measurements with the original inode/device/size/mtime/ctime and recent
write guards. No cache is received from the semantic worker, the other auditor,
disk or another process. This changes scheduling, not validation or scoring.

After its admission completes, each auditor waits for its own actual successful
grade-process receipt binding the exact result bytes. A quality-failed grade is
auditable; a failed process or altered result is not. Upstream failures stop the
waiting auditors. Then run the unchanged recovery CLI audit in that same fresh
independent process, reusing only that process's guarded measured hashes. Bind
the start/warmup records, adapter, tests, CPU qualification and this amendment
into the audit producer. Actual terminal tool receipts remain necessary before
the second readout or paired comparison can consume an audit. No model or judge
call is added. The first readout still closes before the second readout begins.

Pass five scheduling/receipt/cache contracts and the three existing hash-cache
contracts before launch. Preserve every previous producer and receipt. The
semantic CPU worker remains prohibited from running audits. No metric, candidate
order, source, prediction, judge, first-decision cache, threshold or sealed final
test boundary changes. This remains exploratory DEV evaluation, not a claim of
95% final accuracy or model-speed improvement.
