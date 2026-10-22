"""Audit and record the exploratory shared-summary training run."""
import argparse
import json
import math
from pathlib import Path
import shutil

from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_llama import decode_ids, tokenizers
from experiments.structure_net.train_ncm_memory import Config, rows, verify, word_unigram_f1


def read(path):
    return json.loads(Path(path).read_text())


def audit(root):
    producer = read(root / "producer.json")
    fingerprint = text_hash(json.dumps({k: v for k, v in producer.items() if k != "fingerprint"}, sort_keys=True))
    if fingerprint != producer["fingerprint"]:
        raise ValueError("producer fingerprint mismatch")
    verify(producer)
    snapshots = {file_hash(p) for p in (root / "producer_source").iterdir()}
    if not set(producer["source_hashes"].values()) <= snapshots:
        raise ValueError("producing source snapshot incomplete")
    result = read(root / "result.json")
    if result["producer_fingerprint"] != fingerprint:
        raise ValueError("result fingerprint mismatch")
    return producer, result


def finalize(args):
    if (args.root / "campaign_results.json").exists() or args.report.exists():
        raise ValueError("fresh aggregate/report required")
    producer, result = audit(args.root)
    pilot_producer, pilot = audit(args.pilot)
    if result["status"] != "completed" or pilot["status"] != "passed":
        raise ValueError("completed run and successful pilot required")
    for key in ("input_hashes", "source_hashes", "config"):
        if producer[key] != pilot_producer[key]:
            raise ValueError("pilot producer differs")
    data = read(args.data / "manifest.json")
    for name, item in data["files"].items():
        if file_hash(args.data / name) != item["sha256"]:
            raise ValueError("split hash mismatch")
    cfg = Config()
    for kind, steps in [("codec", cfg.codec_steps), ("generator", cfg.generator_steps)]:
        stage = result["stages"][kind]
        trace = read(args.root / (kind+"_trace.json"))
        if stage["updates"] != steps or [r["step"] for r in trace] != list(range(1, steps+1)):
            raise ValueError("training update accounting differs")
        if any(not math.isfinite(r[k]) or r[k] < 0 for r in trace for k in ("loss", "gradient_norm")):
            raise ValueError("invalid training trace")
        if file_hash(stage["checkpoint"]) != stage["sha256"]:
            raise ValueError("final checkpoint changed")
    test = rows(args.data / "native_test.jsonl")
    selected = sorted(test, key=lambda r: text_hash("evaluation:"+r["id"]))[:cfg.eval_limit]
    if [r["id"] for r in selected] != read(args.root / "test_selection.json")["ids"]:
        raise ValueError("test selection changed")
    gpt, _ = tokenizers()
    for label in ("original", "trained"):
        scored = read(args.root / ("test_"+label+".json"))
        outputs = scored["outputs"]
        if [r["id"] for r in outputs] != [r["id"] for r in selected]:
            raise ValueError("missing or duplicate evaluation cases")
        for output, case in zip(outputs, selected):
            if output["source"] != case["source"] or output["target"] != case["target"]:
                raise ValueError("evaluation source/target changed")
            if decode_ids(output["native_output_ids"], gpt) != (output["text"], output["termination"]):
                raise ValueError("native decode differs")
            if output["word_unigram_f1"] != word_unigram_f1(output["text"], case["target"]):
                raise ValueError("word metric differs")
            if output["exact_reference_match"] != (output["text"].strip() == case["target"].strip()):
                raise ValueError("reference match differs")
        if scored["mean_word_unigram_f1"] != sum(r["word_unigram_f1"] for r in outputs)/len(outputs):
            raise ValueError("aggregate metric differs")
        if scored["exact_reference_matches"] != sum(r["exact_reference_match"] for r in outputs):
            raise ValueError("reference-match count differs")
        if {k: v for k, v in scored.items() if k != "outputs"} != result["test_"+label]:
            raise ValueError("terminal metrics differ")
    artifacts = [args.data / "manifest.json", args.pilot / "result.json",
                 args.pilot.parent / "20261022_pilot_v1/error.json"]
    artifacts += [p for p in args.root.iterdir() if p.is_file() and p.suffix in (".json", ".pt")]
    campaign = {"schema": "nal.ncm-memory-campaign.v1", "status": "completed_exploratory",
                **{k: v for k, v in result.items() if k != "status"},
                "data_splits": data["splits"], "data_counts": data["counts"],
                "source_session_overlap": data["source_session_overlap"],
                "eligible_input_overlap": data["eligible_input_overlap"],
                "artifacts": {str(p): file_hash(p) for p in artifacts}}
    copy = args.root / "analysis_source"
    copy.mkdir()
    shutil.copyfile(__file__, copy / Path(__file__).name)
    campaign["analysis_source_sha256"] = file_hash(__file__)
    write_json(args.root / "campaign_results.json", campaign)
    a, b = result["test_original"], result["test_trained"]
    train, test_split = data["splits"]["train"], data["splits"]["test"]
    report = f"""# NCM training on the shared memory summaries

The existing NCM completed an initial adaptation using a dataset of
**{train['native_rows']:,} original-text / saved-summary pairs**, with
**{test_split['native_rows']:,} compatible pairs held out by source conversation.
The fixed training budget samples from that dataset; it is not a full epoch.
This continues the existing model weights. The final checkpoints are saved.

Date: 2026-10-22. Hypothesis: `ncm-memory-summaries-v1`.
Classification: `{result['classification']}`.
One training seed; underpowered for an architecture comparison.
Aggregate: `{args.root / 'campaign_results.json'}`.
SHA256: `{file_hash(args.root / 'campaign_results.json')}`.

## Data

Restored all 13 shared bundles and verified all 1,500 member hashes. Input
repository: `Ethycs/memory_condense`, Git commit
`{data['git_commit']}`, lakeFS commit `{data['lakefs_commit']}`.

| Split | All unique atomic pairs | Source sessions | Native-compatible pairs |
| --- | ---: | ---: | ---: |
| Train | {train['rows']:,} | {train['source_sessions']:,} | {train['native_rows']:,} |
| Test | {test_split['rows']:,} | {test_split['source_sessions']:,} | {test_split['native_rows']:,} |

The deterministic 90/10 split assigns source-session groups, not individual
rows. Repeated non-bypass input text is grouped together. Source-session overlap
and eligible exact-input overlap are both zero. This partition uses LongMemEval
as training data and is **not the untouched LongMemEval benchmark**. Saved
summaries are training labels, not independently human-verified references.

Targets are unchanged stored atomic summaries. Every raw span was recovered
from the original turn and checked against its saved SHA256. All atomic pairs
are exported to `train.jsonl` and `test.jsonl`; compatible training inputs are
`native_train.jsonl` and `native_test.jsonl` under `{args.data}`.

This run retains the model's 512-GPT-2-token source and 48-position output
canvas (47 text tokens plus EOS). It excludes application bypass inputs below
192 cl100k_base tokens, over-capacity sources/targets and targets over the
128-cl100k_base-token stored limit. Nothing is truncated. Exclusion counts,
which can overlap, are in the dataset manifest. Hierarchical/parent sections
remain in the restored stores and were not training targets in this run.

## Training and held-out diagnostics

The encoder and multiscale codebooks stayed frozen. The text decoder completed
512 AdamW updates, batch 16, LR 2e-4. The generator completed 1,024 updates,
batch 8, peak LR 1e-4 with warmup/cosine decay. We used BF16 computation and
FP32 weights/code assignment on physical GPU0, an RTX3060. Final checkpoints
were fixed in advance; no held-out checkpoint selection or retuning occurred.
Full optimizer/RNG continuation checkpoints and loss traces are retained.

The decoder sampled {result['stages']['codec']['unique_training_pairs_sampled']:,}
unique training pairs; the generator sampled
{result['stages']['generator']['unique_training_pairs_sampled']:,}. The first
systems check stopped before any optimizer update because a JSONL reader split
on Unicode paragraph separators inside strings. A physical-line reader and
regression test fixed it; the second CUDA pilot passed. Both roots are retained.

Both original and trained models were evaluated on the same {result['test_pairs_evaluated']}
test examples selected by a fixed ID hash after optimization finished.

| Diagnostic | Original checkpoint | Trained checkpoint |
| --- | ---: | ---: |
| Code cross entropy | {a['code_cross_entropy']:.4f} | {b['code_cross_entropy']:.4f} |
| Codec text cross entropy | {a['codec_text_cross_entropy']:.4f} | {b['codec_text_cross_entropy']:.4f} |
| Codec token accuracy (excludes PAD) | {a['codec_token_accuracy']:.2%} | {b['codec_token_accuracy']:.2%} |
| Mean word-unigram F1 | {a['mean_word_unigram_f1']:.4f} | {b['mean_word_unigram_f1']:.4f} |
| Exact saved-summary matches | {a['exact_reference_matches']} | {b['exact_reference_matches']} |
| Empty generated summaries | {a['empty_outputs']} | {b['empty_outputs']} |

Word-unigram F1 uses lowercased regex word counts, with both missing and extra
words penalized; it does not measure factual support or semantic equivalence.
Codec token accuracy measures reconstruction from the target's own codes,
not generation from the source. These diagnostics are not calibrated final task
accuracy. The 95% quality target and a speed improvement remain unestablished;
no speed comparison was run here. Per-case outputs remain in the raw artifacts.

## Artifacts and reproduction

- Codec: `{result['stages']['codec']['checkpoint']}`,
  SHA256 `{result['stages']['codec']['sha256']}`.
- Generator: `{result['stages']['generator']['checkpoint']}`,
  SHA256 `{result['stages']['generator']['sha256']}`.
- Data and training entry points: `experiments/structure_net/ncm_memory_data.py`
  and `experiments/structure_net/train_ncm_memory.py`.
- The dated training plan records exact inputs, budgets and boundaries.

Run the data entry point's `restore` and `prepare` commands first, then the
training entry point's `pilot` and `train` commands against fresh output roots.
Use the existing MoBA Python environment, `LD_LIBRARY_PATH=''`,
`HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=2`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`CUDA_VISIBLE_DEVICES=0` and `PYTHONPATH=src:.`. Preparation additionally uses
the separately installed tiktoken 0.11.0 dependency. Exact producing source,
configuration and data/checkpoint hashes are saved in `producer.json`.
"""
    args.report.write_text(report)
    print(json.dumps({"status": campaign["status"], "report": str(args.report),
                      "test_original": a, "test_trained": b}), flush=True)


def register(args):
    campaign_path = args.root / "campaign_results.json"
    c = read(campaign_path)
    for path, expected in c["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("completed evidence changed")
    hid = c["hypothesis_id"]
    record = {"id": hid, "name": "NCM continuation on shared conversational summaries",
              "description": f"Continued the existing NCM with a dataset of {c['training_pairs']} saved atomic summary pairs, "
              f"with source-session-disjoint train/test partitions and {c['test_pairs_evaluated']} held-out "
              "reference-fit comparisons. Native 512-input/48-output capacity only; no calibrated human "
              "task accuracy or speed comparison. LongMemEval was used as training data.",
              "category": "architecture", "tested": True, "confirmed": False,
              "status": "completed_exploratory", "underpowered": True,
              "confirmation_status": c["classification"], "independent_adaptation_seeds": 1,
              "direct_experiment_count": 2*c["test_pairs_evaluated"],
              "training_pairs": c["training_pairs"], "application_accuracy_tested": False,
              "speed_tested": False, "campaign": str(campaign_path),
              "campaign_sha256": file_hash(campaign_path), "report": str(args.report),
              "report_sha256": file_hash(args.report)}
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(
        project_name="structure_net_completed_hypotheses", queue_dir="data/experiment_queue",
        sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    exact = logger.hypotheses_collection.get(ids=[hid], include=["metadatas"])
    if exact["ids"] != [hid] or any(exact["metadatas"][0][k] != record[k]
                                    for k in ("tested", "confirmed", "confirmation_status")):
        raise ValueError("registry exact readback failed")
    semantic = logger.hypotheses_collection.query(query_texts=[record["description"]], n_results=10)
    if hid not in semantic["ids"][0]:
        raise ValueError("registry semantic readback failed")
    receipt = {"status": "registered_and_read_back", "record": record,
               "exact_readback": exact["metadatas"][0], "semantic_ids": semantic["ids"][0]}
    for path in [args.root / "registry_readback.json", Path("data/hypothesis_registry") / (hid+".json")]:
        if path.exists():
            raise ValueError("fresh registry receipt required")
        write_json(path, receipt)
        if read(path)["record"] != record:
            raise ValueError("JSON readback failed")
    print(json.dumps({"status": receipt["status"], "hypothesis_id": hid}), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
