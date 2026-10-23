"""Read-only exposure audit; no model outputs or sealed final sources are read."""
import json
from pathlib import Path
import statistics

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_native_label_quality import dependencies
from experiments.structure_net.ncm_paper_data import ROOT, PREP, freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-input-range-audit-plan.md")
DEST = ROOT / "20261023_application_input_range_audit_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-input-range-audit.md")


def lengths(values):
    values = list(values)
    if not values:
        raise ValueError("nonempty fixed cohort required")
    return {"count": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def main():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh append-only audit required")
    dependencies()
    import tiktoken
    from transformers import AutoTokenizer
    from experiments.structure_net.ncm_frozen_body import MODEL_PATH
    from experiments.structure_net.ncm_native_parent_control import native_inputs
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    data = ROOT / "20261023_native_query_neighborhood_data_v1"
    old = Path("data/experiments/ncm_recipe_update/20261022_quality_v2/selected.json")
    draft = ROOT / "20261023_revised_labels_main_draft_v1/labels.json"
    candidates = ROOT / "20261022_expanded_data_v2/train_sources.json"
    inputs = [data / n for n in ("producer.json", "result.json", "cases.json")]
    inputs += [old, draft, candidates, PREP / "development_sources.json", Path(__file__), PLAN,
               Path("experiments/structure_net/ncm_native_parent_control.py")]
    inputs += [MODEL_PATH / n for n in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")]
    p = freeze(DEST, "CPU_input_exposure_only_not_quality", inputs,
        {"application_source_tokens": "cl100k_base", "bypass_below": 192, "fragment_max": 2048,
         "raw_prompt_max": 6300, "raw_fragments_max": 8, "stored_summary_max": 128,
         "sealed_final_sources_read": False, "model_execution": False})
    cohorts = q.read(data / "cases.json")
    source_rows = q.read(old)["train"] + q.read(draft)
    by_id = {row["id"]: row for row in source_rows}
    summaries = cohorts["summary_train"]
    sources = []
    for row in summaries:
        source = by_id[row["id"]]
        if (any(row[k] != source[k] for k in ("source_id", "group_id", "source_sha256"))
                or text_hash(source["source"]) != row["source_sha256"]):
            raise ValueError("fixed summary source identity differs")
        sources.append(source)
    _, prompt_ids = native_inputs(sources, tokenizer, "cpu")
    if prompt_ids != [row["context_ids"] for row in summaries]:
        raise ValueError("actual cached prompt IDs differ from recovered sources")
    development = q.read(PREP / "development_sources.json")
    _, development_ids = native_inputs(development, tokenizer, "cpu")
    source_tokens = lambda rows: [len(encoding.encode(row["source"], disallowed_special=())) for row in rows]
    summary_lengths, development_lengths = source_tokens(sources), source_tokens(development)
    candidate_lengths = source_tokens(q.read(candidates))
    result = {"status": "input_exposure_audited_not_task_quality", "model_execution": False,
        "actual_native_context_lengths": {name: lengths(len(r["context_ids"]) for r in rows) for name, rows in cohorts.items()},
        "native_target_lengths": {name: lengths(len(r["target_ids"]) for r in rows) for name, rows in cohorts.items()},
        "summary_source_cl100k": lengths(summary_lengths), "development_source_cl100k": lengths(development_lengths),
        "candidate_source_cl100k": lengths(candidate_lengths),
        "summary_target_cl100k": lengths(len(encoding.encode(r["target_text"], disallowed_special=())) for r in summaries),
        "summary_train_bypass_cases": sum(n < 192 for n in summary_lengths),
        "development_bypass_cases": sum(n < 192 for n in development_lengths),
        "summary_train_sources_over512_cl100k": sum(n > 512 for n in summary_lengths),
        "development_sources_over512_cl100k": sum(n > 512 for n in development_lengths),
        "candidate_sources_over512_cl100k": sum(n > 512 for n in candidate_lengths),
        "full_range_quality_established": False, "NCM_success_established": False,
        "task_accuracy": None, "speed_tested": False, "final_test_opened": False,
        "producer_fingerprint": p["fingerprint"]}
    verify(p)
    write_json(DEST / "result.json", result)
    REPORT.write_text(f'''# Current NCM data covers short sources only

Date: 2026-10-23. Read-only CPU input-exposure audit; no model quality measured.
Actual native context lengths: `{result['actual_native_context_lengths']}`.
These include native chat instructions for summary inputs. The language arms
see their stored token windows, not their original complete raw sources.

Summary training source cl100k lengths: `{result['summary_source_cl100k']}`.
Development source cl100k lengths: `{result['development_source_cl100k']}`.
Approved candidate-source cl100k lengths: `{result['candidate_source_cl100k']}`.
Summary target cl100k lengths: `{result['summary_target_cl100k']}`.
Training/development bypass cases below 192 source tokens:
{result['summary_train_bypass_cases']}/{result['development_bypass_cases']}.
Sources over 512 cl100k tokens in summary training/development/candidates:
{result['summary_train_sources_over512_cl100k']}/{result['development_sources_over512_cl100k']}/{result['candidate_sources_over512_cl100k']}.

Every summary source ID/session/group/hash and actual native prompt was matched
to its fixed label record. Application units differ: a raw fragment may contain
2,048 cl100k tokens; batches allow eight fragments within a 6,300-token prompt.
Stored summaries allow 128 tokens, while the extractor's 32-word target is a
separate task. No raw batch or hierarchy-merge quality is measured here.

The current run remains unchanged. Passing its short-source 64-case development
gate cannot establish those longer operating ranges. Final source-disjoint,
representative task evaluation remains required; the original 256 sealed sources
were not read. Neither exposure counts nor baseline scores establish 95% student
accuracy, compression or speedup.

Result `{DEST / 'result.json'}`, SHA256 `{file_hash(DEST / 'result.json')}`.
Producer hashes and strict JSON remain append-only.
''')
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
