"""CPU-only recovery corpus admission; never reads benchmark TEST answers.

prepare --root DIR --spec FILE inventories cached sources, authenticates TRAIN
payloads and writes immutable ResponseRecord JSONL, separate evaluation contexts,
and calibration-only exact TRAIN prefixes. run/check authenticate this artifact;
no model execution is performed. Confirmation is encoded as ResponseRecord.test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from statistics import NormalDist
import subprocess

from . import jspace_3b_response_data as old

data = old.data
SCHEMA = "jspace.recovery-reset.corpus.v1"
APPS_REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
APPS_SHA256 = "45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae"
FAMILIES = ("math", "code", "instruction", "summary")
TRAIN_QUOTAS = dict(math=275000, code=275000, instruction=275000, summary=275000)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def sample_size(*, discordance=1., margin=.05, power=.8, alpha=.05):
    """Normal paired-difference planning approximation, not a realized CI."""
    if not 0 < discordance <= 1 or not 0 < margin < 1 or not .5 < power < 1 or not 0 < alpha < .5:
        raise ValueError("Invalid prespecified paired-power assumptions")
    normal = NormalDist()
    return math.ceil(discordance * (normal.inv_cdf(1-alpha/2) + normal.inv_cdf(power))**2 / margin**2)


def parquet_rows(path):
    # Separate locked environment supplies pyarrow; records, not whole tables,
    # cross the pipe. Neither environment is modified and CUDA remains hidden.
    script = "import json,sys\nimport pyarrow.parquet as pq\nfor b in pq.ParquetFile(sys.argv[1]).iter_batches(batch_size=64):\n for r in b.to_pylist(): print(json.dumps(r),flush=True)\n"
    process = subprocess.Popen(["/home/rabbit/.pixi/bin/pixi", "run", "--locked", "--offline", "-e", "default",
        "python", "-c", script, str(path)], cwd=old.ROOT, stdout=subprocess.PIPE, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"))
    try:
        for line in process.stdout:
            yield json.loads(line)
        if process.wait() != 0:
            raise RuntimeError("Locked CPU Parquet reader failed")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()


def source_rows(source):
    family, path = source["family"], source["path"]
    rows = parquet_rows(path) if str(path).endswith(".parquet") else jsonl(path)
    for index, original in enumerate(rows):
        extra = {}
        if family == "dolly":
            instruction, context, response = (original[k] for k in ("instruction", "context", "response"))
            task = "instruction:" + original["category"]
            group = "dolly:" + data.digest(old.normalize(context or instruction))
            example = f"dolly:{source['revision']}:{index}"
        elif family == "xsum":
            p = original["public_provenance"]
            if original.get("split") != "TRAIN" or original.get("source_truncated") is not False or p["split"] != "train" or p["revision"] != source["revision"]:
                raise ValueError("Non-TRAIN/truncated/unpinned XSum source")
            instruction, context, response = old.SUMMARY_INSTRUCTION, original["source"], original["target_text"]
            task, group, example = "summarization:xsum", original["source_component"], "xsum:" + original["id"]
            extra = dict(article_id=p["article_id"], original_row_number=p["original_row_number"])
        elif family == "gsm8k":
            instruction = "Solve this problem. Show the reasoning and give the final answer."
            context, response, task = original["question"], original["answer"], "benchmark:gsm8k"
            group = "gsm8k:" + data.digest(old.normalize(context))
            example = f"gsm8k:{source['revision']}:train:{index}"
        elif family == "mbpp":
            instruction = "Write Python code for the following task. Return only the complete code.\n\n" + original["text"]
            context, response, task = "Required behavior examples:\n" + "\n".join(original["test_list"]), original["code"], "benchmark:mbpp"
            group = "mbpp:" + data.digest(old.normalize(context or instruction))
            example = f"mbpp:{source['revision']}:train:{index}"
            extra = {k: original.get(k, "" if k == "test_setup_code" else []) for k in ("test_list", "test_setup_code", "challenge_test_list")}
        elif family == "apps":
            solutions = json.loads(original["solutions"]) if isinstance(original["solutions"], str) else original["solutions"]
            tests_error = None
            try:
                tests = json.loads(original["input_output"] or "{}") if isinstance(original["input_output"], str) else original["input_output"]
            except (ValueError, TypeError) as error:
                # Preserve invalid/unrepresentable public test metadata as
                # unavailable, without changing integer semantics or weakening
                # Python's bounded parser. The complete reference is independent.
                tests, tests_error = None, type(error).__name__
            instruction = "Write a complete Python solution to this programming problem. Return only the code."
            context = original["question"]
            # First published solution is fixed before tokenization/outcomes; no
            # choosing whichever reference happens to fit/pass a test.
            response = solutions[0] if solutions else ""
            task, group = "benchmark:apps", "apps:" + str(original["id"])
            example = f"apps:{source['revision']}:train:{original['id']}"
            extra = dict(apps_tests=tests, apps_tests_error=tests_error,
                apps_tests_raw_sha256=data.digest(original["input_output"]),
                difficulty=original.get("difficulty"), starter_code=original.get("starter_code", ""))
        else:
            raise ValueError("Unsupported source adapter")
        yield dict(example_id=example, family=family, category=source["category"], instruction=instruction,
            context=context, response=response, task=task, source_uri=source["uri"], source_revision=source["revision"],
            source_row=index, source_group=group, source_split="train",
            reference_quality="complete_public_TRAIN_reference_not_independently_factual_gold", **extra)


def identity_keys(row):
    # Exact normalized source/prompt and declared source group only. This is
    # deliberately NOT advertised as a near-duplicate or paraphrase audit.
    return ("group:" + row["source_group"],
            "source:" + data.digest(old.normalize(row["context"] or row["instruction"])),
            "prompt:" + data.digest(old.normalize(row["instruction"] + "\n\n" + row["context"])))


class LineageIndex:
    """Disk-backed exact identity guard, retaining prior TRAIN/DEV ancestry."""
    def __init__(self, path, seed):
        self.db = sqlite3.connect(path)
        self.seed = seed
        self.db.executescript("CREATE TABLE identities (key TEXT PRIMARY KEY, split TEXT, group_id TEXT, historical INTEGER);"
                              "CREATE TABLE selected (prompt TEXT PRIMARY KEY, content TEXT UNIQUE, example TEXT UNIQUE);")

    def historical(self, row):
        for key in identity_keys(row) + ("oldgroup:" + row["group_id"],):
            previous = self.db.execute("SELECT split FROM identities WHERE key=?", (key,)).fetchone()
            if previous and previous[0] != row["split"]:
                raise ValueError("Conflicting historical TRAIN/DEV source identity")
            self.db.execute("INSERT OR REPLACE INTO identities VALUES (?,?,?,1)", (key, row["split"], row["group_id"]))

    def assign(self, row):
        keys = identity_keys(row)
        found = [value for key in keys if (value := self.db.execute(
            "SELECT split,group_id,historical FROM identities WHERE key=?", (key,)).fetchone())]
        if len({v[0] for v in found}) > 1:
            raise ValueError("cross_split_exact_identity")
        historical = any(v[2] for v in found)
        if found:
            split, group = found[0][:2]
        else:
            group = "recovery-group:" + data.digest(keys[1])
            bucket = int(data.digest([self.seed, group])[:16], 16) % 100
            split = "train" if bucket < 45 else "development" if bucket < 55 else "test"
        for key in keys:
            self.db.execute("INSERT OR IGNORE INTO identities VALUES (?,?,?,?)", (key, split, group, int(historical)))
        return split, group, historical

    def unique(self, record):
        try:
            self.db.execute("INSERT INTO selected VALUES (?,?,?)", (data.digest(record.prompt_ids), data.digest(record.input_ids), record.example_id))
            return True
        except sqlite3.IntegrityError:
            return False

    def close(self):
        self.db.commit()
        self.db.close()


def inventory(spec):
    plans, corpora = [], set()
    for key in ("historical_data_root", "full_response_data_root"):
        path = Path(spec[key]) / "plan.json"
        value = json.loads(path.read_text())
        plans.append(dict(path=str(path.resolve()), sha256=sha(path), corpus=value["corpus"],
                          calibration_indices=value.get("calibration_indices"),
                          training_examples=value.get("training_examples"),
                          scope="All original corpus TRAIN conservatively treated as inherited recovery/calibration exposure"))
        corpora.add(Path(value["corpus"]))
    manifests = [(root, json.loads((root/"manifest.json").read_text())) for root in sorted(corpora)]
    root, manifest = manifests[0]
    sources = []
    for family, category, path, revision, uri, expected in (
        ("gsm8k", "math", root/"sources/gsm8k_train.parquet", old.GSM_REVISION, "openai/gsm8k", manifest["artifacts"]["sources/gsm8k_train.parquet"]),
        ("mbpp", "code", root/"sources/mbpp_train.parquet", old.MBPP_REVISION, "google-research-datasets/mbpp", manifest["artifacts"]["sources/mbpp_train.parquet"]),
        ("apps", "code", Path(spec["artifact_root"])/"public_sources/apps_21e74ddf_train.jsonl", APPS_REVISION, "codeparrot/apps", APPS_SHA256),
        ("dolly", "instruction", root/"sources/dolly.jsonl", old.DOLLY_REVISION, "databricks/databricks-dolly-15k", manifest["artifacts"]["sources/dolly.jsonl"]),
        ("xsum", "summary", Path(manifest["source_receipts"]["xsum"]["path"]), old.XSUM_REVISION, "EdinburghNLP/xsum", old.XSUM_SHA256),
    ):
        actual = sha(path)
        if actual != expected:
            raise ValueError("Pinned TRAIN source changed: " + str(path))
        sources.append(dict(family=family, category=category, path=str(path.resolve()), sha256=actual,
            bytes=path.stat().st_size, revision=revision, uri="https://huggingface.co/datasets/"+uri, split="train"))
    return plans, manifests, sources


def prepare(root, spec_path):
    from transformers import AutoTokenizer
    root, spec_path = Path(root).resolve(), Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    if spec["schema"] != "jspace.recovery-reset.spec.v1" or spec["target_context"] != 2048:
        raise ValueError("Frozen 2048-total study specification required")
    root.mkdir(parents=True, exist_ok=False)
    save(root/"spec.json", spec)
    save(root/"plan.json", dict(schema="jspace.recovery-reset.corpus-plan.v1",
        spec_path=str(spec_path), spec_sha256=sha(spec_path),
        implementation_sha256={str(Path(p).resolve()): sha(p) for p in (__file__, old.__file__, data.__file__)}))
    try:
        return _prepare(root, spec_path, spec, AutoTokenizer)
    except Exception as error:
        save(root/"failure.json", dict(status="failed", error_type=type(error).__name__, reason=str(error)))
        raise


def _prepare(root, spec_path, spec, tokenizer_class):
    plans, historical_manifests, sources = inventory(spec)
    save(root/"source_inventory.json", dict(plans=plans, sources=sources,
        historical=[dict(path=str(p/"manifest.json"), sha256=sha(p/"manifest.json"), source_inventory=m["source_inventory"],
                         exposure_incident=m.get("prior_preparation_access_incident")) for p, m in historical_manifests]))
    model = Path(spec["native_model_path"])
    tokenizer_files = {name: sha(model/name) for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json")}
    tokenizer = tokenizer_class.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    tokenization = dict(tokenizer_sha256=data.digest(tokenizer_files), chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())
    index = LineageIndex(root/"lineage.sqlite", spec["seed"])
    old_receipts = []
    for corpus, manifest in historical_manifests:
        for name in ("train.jsonl", "development.jsonl", "text.jsonl"):
            if sha(corpus/name) != manifest["artifacts"][name]:
                raise ValueError("Historical corpus artifact changed")
            old_receipts.append(dict(path=str(corpus/name), sha256=sha(corpus/name)))
        for row in jsonl(corpus/"text.jsonl"):
            index.historical(row)
    exact, windows, exclusion_receipts = old.exposed_questions(old.EXPOSED)
    n = sample_size()
    confirmation_per_family = math.ceil(n/len(FAMILIES))
    counts = {split: {family: Counter() for family in FAMILIES} for split in ("train", "development", "test")}
    exclusions, scanned, calibration_counts = Counter(), Counter(), Counter()
    paths, groups, guard = {}, defaultdict(set), data.SplitGuard()
    with ExitStack() as stack:
        streams, contexts = {}, {}
        for split, label in (("train", "train"), ("development", "development"), ("test", "confirmation")):
            paths[label+"_records"] = str(root/(label+".jsonl"))
            paths[label+"_contexts"] = str(root/(label+"_contexts.jsonl"))
            streams[split] = stack.enter_context(Path(paths[label+"_records"]).open("x"))
            contexts[split] = stack.enter_context(Path(paths[label+"_contexts"]).open("x"))
        paths["calibration_records"] = str(root/"calibration.jsonl")
        calibration = stack.enter_context(Path(paths["calibration_records"]).open("x"))
        for source in sources:
            for row in source_rows(source):
                family = row["category"]
                scanned[source["family"]] += 1
                # Historical question exposure is excluded, even if that costs
                # examples. No old TEST labels are opened for this comparison.
                texts = (row["instruction"], row["context"] or row["instruction"])
                if any(old.normalize(t) in exact or set(old.windows(t)) & windows for t in texts):
                    exclusions[family+":historical_benchmark_exposure"] += 1
                    continue
                try:
                    split, group, historical = index.assign(row)
                except ValueError as error:
                    exclusions[family+":"+str(error)] += 1
                    continue
                c = counts[split][family]
                wanted = c["supervised_tokens"] < TRAIN_QUOTAS[family] if split == "train" else c["records"] < (128 if split == "development" else confirmation_per_family)
                want_calibration = split == "train" and calibration_counts[family] < 8
                if not wanted and not want_calibration:
                    continue
                # Tokenize the complete source first; no completion clipping.
                try:
                    record = old.tokenize_reference(tokenizer, row, split=split, group_id=group,
                        tokenization=tokenization, max_prompt=1000000, max_response=1000000, max_total=1000000)
                except ValueError as error:
                    if str(error) not in {"embedded_chat_special_token", "empty_response", "prompt_over_limit", "response_over_limit", "total_over_limit",
                                         "assistant_prefix_mismatch", "terminal_eos_mismatch", "unexpected_template_footer"}:
                        raise
                    exclusions[family+":"+str(error)] += 1
                    continue
                lineage = dict(historical_exposure=historical, inherited_group=group if historical else None,
                               original_source_group=row["source_group"], original_source_split="train")
                provenance = dict(record.provenance, category=family, historical_lineage=lineage,
                                  historical_train=bool(historical and split == "train"),
                                  source_payload_sha256=source["sha256"])
                record = data.ResponseRecord(**dict(asdict(record), provenance=provenance))
                if want_calibration and len(record.input_ids) >= spec["target_context"]:
                    span = dict(example_id=record.example_id, group_id=group, split="train", input_ids=record.input_ids[:2048],
                        provenance=dict(provenance, source_complete_tokens_sha256=data.digest(record.input_ids),
                            source_complete_token_count=len(record.input_ids), derivation="exact_prefix_crop_for_geometry_only"))
                    calibration.write(json.dumps(span, sort_keys=True)+"\n")
                    calibration_counts[family] += 1
                if not wanted:
                    continue
                if len(record.input_ids) > 2048:
                    exclusions[family+":total_over_limit_complete_reference_rejected"] += 1
                    continue
                if split != "train" and group in groups[(split, "all")]:
                    exclusions[family+":repeated_heldout_source_group"] += 1
                    continue
                if not index.unique(record):
                    exclusions[family+":exact_duplicate_prompt_or_content"] += 1
                    continue
                guard.add(record)
                groups[(split, "all")].add(group)
                streams[split].write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False)+"\n")
                contexts[split].write(json.dumps(dict(row, split=split, group_id=group, historical_lineage=lineage,
                    prompt_text=old.messages_for(row)[-1]["content"]), sort_keys=True, ensure_ascii=False)+"\n")
                c.update(records=1, input_tokens=len(record.input_ids), prompt_tokens=len(record.prompt_ids),
                         supervised_tokens=len(record.response_ids), historical_examples=int(historical),
                         numeric_reference=int(bool(re.search(r"\d", row["response"]))),
                         entity_reference=int(bool(re.search(r"\b[A-Z][a-z]+\b", row["response"]))))
                c["max_total"] = max(c["max_total"], len(record.input_ids))
                c["length_"+("short" if len(record.input_ids) <= 416 else "medium" if len(record.input_ids) <= 1024 else "long")] += 1
                groups[(split, family)].add(group)
            index.db.commit()
    index.close()
    for split in counts:
        for family, c in counts[split].items():
            c["unique_groups"] = len(groups[(split, family)])
    total = sum(c["supervised_tokens"] for c in counts["train"].values())
    admitted = total >= 1000000 and all(counts["train"][f]["supervised_tokens"] >= TRAIN_QUOTAS[f] for f in FAMILIES)
    manifest = dict(schema=SCHEMA, status="admitted" if admitted else "insufficient_train_corpus", training_admitted=admitted,
        spec_path=str(spec_path), spec_sha256=sha(spec_path), study_id=spec["study_id"], **paths,
        sha256={path: sha(path) for path in paths.values()}, tokenization=tokenization, tokenizer_files=tokenizer_files,
        tokenizer_revision=spec["native_model_revision"], max_length=2048, counts=counts,
        unique_train_supervised_tokens=total, minimum_unique_train_supervised_tokens=1000000,
        train_token_targets=TRAIN_QUOTAS, source_inventory=sources, source_rows_scanned=dict(scanned),
        exclusions=dict(exclusions), historical_overlap_evidence=dict(plans=plans, artifacts=old_receipts,
            exposed_question_receipts=exclusion_receipts, inherited_all_original_train=True,
            prior_access_incidents=[m.get("prior_preparation_access_incident") for _, m in historical_manifests]),
        calibration=dict(records=sum(calibration_counts.values()), by_family=dict(calibration_counts), length=2048,
            supervision=False, derivation="Unmodified exact prefixes of complete original TRAIN sources; no fabricated EOS"),
        confirmation=dict(sealed=True, record_split="test", family_weights={f:.25 for f in FAMILIES},
            target_per_family=confirmation_per_family, required_pairs=n, margin=.05, power=.8, alpha=.05,
            assumed_paired_discordance=1.,
            sample_size_method="Normal paired-difference screening approximation only; not 80% power under the reported Hoeffding interval",
            power_guaranteed_for_reported_interval=False,
            hoeffding_power_bound_pairs=math.ceil(2*(math.sqrt(math.log(2/.05))+math.sqrt(math.log(1/.2)))**2/.05**2),
            power_note="The 3140-pair approximation targets normal-theory power; the wider finite-sample Hoeffding interval has lower power. Count readiness is not a power certificate; the actual interval controls noninferiority.",
            sample_size_ready=all(counts["test"][f]["unique_groups"] >= confirmation_per_family for f in FAMILIES),
            selection_must_be_frozen_before_unseal=True),
        policy=dict(training="Only train_records; confirmation/test never opened by gradient or calibration loaders",
            reference="Complete untruncated dataset response with EOS observed in pinned template, not appended after a cap",
            rollout="Separate runtime student_rollout partial histories may end at length without fabricated EOS",
            mixture_shift="Equal minimum supervised token budgets across math/code/instruction/summary; APPS TRAIN added to historical pool; both arms share these files",
            lineage="Inherited TRAIN remains TRAIN; inherited DEV remains exposed DEV; unexposed groups assigned 45/10/45 before tokenization",
            near_duplicate="Exact normalized source/prompt plus declared source ancestry and historical exposed 20-word windows; SplitGuard is exact-only, not a near-duplicate guarantee",
            uncertainty="Semantic paraphrases, related stories, dataset benchmark overlap and native pretraining contamination remain unresolved; no contamination-free claim"),
        producer_sha256=sha(__file__))
    manifest["fingerprint"] = data.digest(manifest)
    save(root/"manifest.json", manifest)
    if not admitted:
        raise ValueError("Fewer than required unique complete TRAIN tokens/family quotas; not admitted for training")
    return manifest


def check(root):
    plan = json.loads((Path(root)/"plan.json").read_text())
    for path, expected in dict(plan["implementation_sha256"], **{plan["spec_path"]: plan["spec_sha256"]}).items():
        if sha(path) != expected:
            raise ValueError("Corpus preparation dependency changed: " + path)
    manifest = json.loads((Path(root)/"manifest.json").read_text())
    fingerprint = manifest.pop("fingerprint")
    if fingerprint != data.digest(manifest) or manifest["schema"] != SCHEMA or not manifest["training_admitted"]:
        raise ValueError("Unauthenticated/unadmitted corpus")
    spec = json.loads(Path(plan["spec_path"]).read_text())
    for name, expected in manifest["tokenizer_files"].items():
        if sha(Path(spec["native_model_path"])/name) != expected:
            raise ValueError("Pinned tokenizer payload changed: " + name)
    evidence = manifest["source_inventory"] + manifest["historical_overlap_evidence"]["plans"] + manifest["historical_overlap_evidence"]["artifacts"] + manifest["historical_overlap_evidence"]["exposed_question_receipts"]
    for item in evidence:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Consumed corpus source changed: " + item["path"])
    for path, expected in manifest["sha256"].items():
        if sha(path) != expected:
            raise ValueError("Corpus payload changed: " + path)
    guard, total = data.SplitGuard(), 0
    # Sealed confirmation is authenticated above by bytes, never decoded here.
    for key, split in (("train_records", "train"), ("development_records", "development")):
        for record in data.stream_records(manifest[key], allowed_splits=(split,)):
            guard.add(record)
            if len(record.input_ids) > 2048 or record.finish_reason != "eos":
                raise ValueError("Incomplete/oversize reference")
            if split == "train":
                total += len(record.response_ids)
    if total != manifest["unique_train_supervised_tokens"] or total < 1000000:
        raise ValueError("TRAIN token budget mismatch")
    return dict(status="authenticated", unique_train_supervised_tokens=total, fingerprint=fingerprint)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.spec is None:
            parser.error("prepare requires --spec")
        value = prepare(args.root, args.spec)
        result = {k: value[k] for k in ("status", "fingerprint", "unique_train_supervised_tokens", "counts", "confirmation")}
    else:
        result = check(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
