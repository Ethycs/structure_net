"""Admit real complete-response Qwen recovery examples under frozen length limits.

Only named public TRAIN sources and exposed task files are read. No final-test
discovery, response generation, truncation, model loading, or GPU allocation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from experiments.neural_architecture_lab.distillation_recovery import module

data = module("distillation_data")
ROOT = Path(__file__).resolve().parents[2]
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL = Path("/home/rabbit/models/hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots") / MODEL_REVISION
XSUM_REVISION = "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
XSUM = ROOT / "data/experiments/ncm_paper_regime/20261026_public_training_catalog_v1/training.jsonl"
XSUM_SHA256 = "5e8edfdf652ef37514d1935f458ed6fc0a38143eedd74b4c6303c44540acd8d2"
SYSTEM = "You are a helpful assistant."
SUMMARY_INSTRUCTION = "Summarize the following text accurately and concisely. Preserve important names, numbers, and dates."
EXPOSED = (
    ROOT / "data/experiments/jspace_3b_broad_recovery_data/20261115_v1/items.json",
    ROOT / "data/experiments/jspace_3b_sanity_data/20261115_v1/tasks.json",
)
SEALED_TASKS = ROOT / "data/experiments/jspace_accuracy_recovery_data/20261114_v1/tasks.json"
BENCH_ROOT = ROOT / "data/benchmarks/jspace_four_tasks_20261113"
GSM_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MULTI_QUOTAS = dict(dolly=20000, xsum=20000, arc=3000, boolq=4000,
                    hellaswag=15000, winogrande=15000, gsm8k=20000, mbpp=8000)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def windows(text, width=20):
    words = normalize(text).split()
    if len(words) < width:
        return ()
    return tuple(" ".join(words[i:i+width]) for i in range(0, len(words)-width+1, 10))


def messages_for(row):
    content = row["instruction"]
    if row["context"]:
        content += "\n\n" + row["context"]
    return [dict(role="system", content=SYSTEM), dict(role="user", content=content)]


def tokenize_reference(tokenizer, row, *, split, group_id, tokenization,
                       max_prompt=288, max_response=128, max_total=416):
    """Take EOS from the complete pinned template; remove its formatting newline.

    A corpus reference is complete text, not a token-capped generation. The
    template contributes its actual terminal EOS. The only discarded tokens are
    the exact, verified newline after that EOS, never response content.
    """
    if any(x in field for field in (row["instruction"], row["context"], row["response"])
           for x in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
        raise ValueError("embedded_chat_special_token")
    messages = messages_for(row)
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if len(prefix) > max_prompt:
        raise ValueError("prompt_over_limit")
    full = tokenizer.apply_chat_template(messages + [dict(role="assistant", content=row["response"])],
                                         tokenize=True, add_generation_prompt=False)
    eos = tokenizer.eos_token_id
    if full[:len(prefix)] != prefix:
        raise ValueError("assistant_prefix_mismatch")
    tail = full[len(prefix):]
    if tail.count(eos) != 1:
        raise ValueError("terminal_eos_mismatch")
    eos_index = tail.index(eos)
    if tail[eos_index+1:] != tokenizer.encode("\n", add_special_tokens=False):
        raise ValueError("unexpected_template_footer")
    response = tail[:eos_index+1]
    if not row["response"].strip() or len(response) <= 1:
        raise ValueError("empty_response")
    if len(prefix) > max_prompt:
        raise ValueError("prompt_over_limit")
    if len(response) > max_response:
        raise ValueError("response_over_limit")
    if len(prefix) + len(response) > max_total:
        raise ValueError("total_over_limit")
    provenance = dict(tokenization, source_uri=row["source_uri"], source_revision=row["source_revision"],
        source_row=row["source_row"], source_group=row["source_group"], source_family=row["family"],
        prompt_provenance=("original_human_instruction_and_context" if row["family"] == "dolly"
                           else "declared_task_instruction_plus_complete_original_source"),
        response_provenance="complete_dataset_reference", reference_quality=row["reference_quality"],
        source_text_sha256=data.digest(row["context"]), instruction_sha256=data.digest(row["instruction"]),
        reference_sha256=data.digest(row["response"]), template_terminal_eos=True,
        template_post_eos_newline_removed=True, historical_train=bool(row.get("historical_train")),
        original_source_split=row.get("source_split", "train"))
    return data.record_from_tokenized(full_ids=prefix+response, assistant_boundary=len(prefix),
        expected_prompt_ids=prefix, provenance=provenance, example_id=row["example_id"],
        group_id=group_id, split=split, task=row["task"], finish_reason="eos", eos_token_id=eos)


class Groups:
    def __init__(self, count):
        self.parent = list(range(count))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def grouped_ids(rows):
    """Conservative exact/context/shared-window components before splitting.

    Shared 20-word source windows deliberately group some nonduplicates. This
    is a stated near-duplicate heuristic, not a semantic-paraphrase guarantee.
    """
    groups, seen = Groups(len(rows)), {}
    for i, row in enumerate(rows):
        source = row["context"] or row["instruction"]
        keys = [("group", row["source_group"]), ("source", normalize(source)),
                ("prompt", normalize(row["instruction"] + " " + row["context"]))]
        keys += [("window", x) for x in windows(source)]
        for key in keys:
            key = data.digest(key)
            if key in seen:
                groups.join(i, seen[key])
            else:
                seen[key] = i
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[groups.find(i)].append(row["example_id"])
    labels = {root: "response-group:" + data.digest(sorted(ids)) for root, ids in members.items()}
    return [labels[groups.find(i)] for i in range(len(rows))]


def split_for(group_id, seed):
    return "development" if int(data.digest([seed, group_id])[:16], 16) % 10 == 0 else "train"


def exposed_questions(paths):
    exact, chunks, receipts = set(), set(), []
    def visit(value):
        if isinstance(value, dict):
            for key in ("question", "prompt"):
                text = value.get(key)
                if isinstance(text, str):
                    text = re.split(r"\n\nA[.)] ", text, maxsplit=1)[0]
                    normalized = normalize(text)
                    if normalized:
                        exact.add(normalized)
                        chunks.update(windows(text))
            for item in value.values():
                if isinstance(item, (dict, list)):
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    for path in paths:
        path = Path(path)
        if path.resolve() == SEALED_TASKS.resolve() or path.name == "audit_tasks.json":
            raise ValueError("Sealed task contents cannot enter response-data preparation")
        if not path.is_file():
            raise FileNotFoundError(path)
        visit(json.loads(path.read_text()))
        receipts.append(dict(path=str(path.resolve()), sha256=sha(path)))
    return exact, chunks, receipts


def source_rows(dolly_path, xsum_path, *, xsum_pool=30000):
    rows, counts = [], Counter()
    with Path(dolly_path).open() as stream:
        for index, line in enumerate(stream):
            r = json.loads(line)
            counts["dolly_source_rows"] += 1
            fields = [r.get(k) for k in ("instruction", "context", "response", "category")]
            if any(not isinstance(x, str) for x in fields):
                raise ValueError("Dolly schema changed")
            instruction, context, response, category = fields
            content = normalize(context or instruction)
            rows.append(dict(example_id=f"dolly:{DOLLY_REVISION}:{index}", family="dolly",
                instruction=instruction, context=context, response=response, task="instruction:"+category,
                source_uri="https://huggingface.co/datasets/databricks/databricks-dolly-15k",
                source_revision=DOLLY_REVISION, source_row=index, source_group="dolly:"+data.digest(content),
                reference_quality="human_authored_dataset_reference_not_independently_verified"))
    # Stable, bounded sample over the entire admitted public TRAIN catalog.
    pool = []
    with Path(xsum_path).open() as stream:
        for index, line in enumerate(stream):
            r = json.loads(line)
            counts["xsum_source_rows"] += 1
            if r.get("split") != "TRAIN" or r.get("source_truncated") is not False:
                raise ValueError("Only complete original TRAIN articles admitted")
            provenance = r["public_provenance"]
            if provenance["revision"] != XSUM_REVISION or provenance["split"] != "train":
                raise ValueError("XSum source revision/split changed")
            priority = int(data.digest(["xsum-pool-v1", r["id"]]), 16)
            if len(pool) < xsum_pool or priority < -pool[0][0]:
                row = dict(example_id="xsum:"+r["id"], family="xsum", instruction=SUMMARY_INSTRUCTION,
                    context=r["source"], response=r["target_text"], task="summarization:xsum",
                    source_uri="https://huggingface.co/datasets/EdinburghNLP/xsum",
                    source_revision=XSUM_REVISION, source_row=provenance["original_row_number"],
                    source_catalog_row=index, source_group=r["source_component"],
                    article_id=provenance["article_id"], reference_quality=r["reference_kind"])
                item = (-priority, r["id"], row)
                if len(pool) < xsum_pool:
                    heapq.heappush(pool, item)
                else:
                    heapq.heapreplace(pool, item)
    rows += [x[2] for x in pool]
    rows.sort(key=lambda r: r["example_id"])
    counts["xsum_bounded_pool"] = len(pool)
    return rows, dict(counts)


def benchmark_rows(*, gsm_path, mbpp_path, arc_dev_path):
    """Read public TRAIN plus ARC validation; only identity metadata for reserved data."""
    def read_parquet(path):
        # PyArrow is pinned in the default Pixi environment, while Torch and
        # transformers are pinned in ncm. Use that locked CPU reader directly.
        script = "import json,sys; import pyarrow.parquet as pq; print(json.dumps(pq.read_table(sys.argv[1]).to_pylist()))"
        raw = subprocess.check_output(["/home/rabbit/.pixi/bin/pixi", "run", "--locked", "--offline", "-e", "default",
            "python", "-c", script, str(path)], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        return json.loads(raw)
    from .jspace_bench_data import normalize as bench_normalize
    from .jspace_recovery_data import identity
    manifest_path = BENCH_ROOT/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    known = json.loads(EXPOSED[0].read_text())["items"]
    known_train = {v for r in known if r["split"] == "train" for k in ("source_identity", "identity", "semantic_identity") if (v := r.get(k))}
    all_history, receipts = set(), [dict(path=str(manifest_path), sha256=sha(manifest_path))]
    # Existing frozen exclusion inventories contain only identity hashes, not test records.
    identity_paths = [ROOT/"data/experiments/jspace_3b_broad_recovery_data/20261115_v1/exclusions.json",
        ROOT/"data/experiments/jspace_3b_broad_recovery_data/20261115_v1/audit_identities.json",
        ROOT/"data/experiments/jspace_recovery_expansion/20261114_v1/confirmation_identities.json"]
    hash_pattern = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
    for path in identity_paths:
        all_history.update(x.decode() for x in hash_pattern.findall(path.read_bytes()))
        receipts.append(dict(path=str(path), sha256=sha(path)))
    reserved = set(json.loads(identity_paths[-1].read_text())["identities"])
    if reserved & known_train:
        raise ValueError("Historical TRAIN overlaps reserved identity manifest")
    all_history.update(v for r in known for k in ("source_identity", "identity", "semantic_identity") if (v := r.get(k)))
    rows, counts = [], Counter()
    for family, entry in manifest.items():
        paths = [("train", BENCH_ROOT/entry["files"]["train"]["local"])]
        if family == "arc":
            paths.append(("validation", Path(arc_dev_path)))
        for source_split, path in paths:
            actual = sha(path)
            if source_split == "train" and actual != entry["files"]["train"]["sha256"]:
                raise ValueError("Pinned benchmark TRAIN payload changed")
            receipts.append(dict(path=str(path.resolve()), sha256=actual, family=family, split=source_split,
                source_uri=f"https://huggingface.co/datasets/{entry['repository']}", revision=entry["revision"]))
            originals = read_parquet(path)
            counts[family+":"+source_split+":source_rows"] = len(originals)
            selected = []
            for index, original in enumerate(originals):
                try:
                    question, choices, answer = bench_normalize(family, original)
                except (ValueError, IndexError):
                    counts[family+":unsupported"] += 1
                    continue
                key, semantic = identity(question, choices), identity(question, sorted(choices))
                historic = key in all_history or semantic in all_history
                trained = key in known_train or semantic in known_train
                if historic and not (source_split == "train" and trained):
                    counts[family+":historical_nontrain_excluded"] += 1
                    continue
                if family == "arc":
                    instruction = "Answer the science question using the full answer text, not an option letter."
                    context = original["question"] + "\n\nPossible answers:\n" + "\n".join("- "+c for c in choices)
                    response = choices[answer]
                elif family == "boolq":
                    instruction = "Read the passage and answer the question with Yes or No.\n\n" + original["question"]
                    context, response = original["passage"], choices[answer] + "."
                elif family == "hellaswag":
                    instruction = "Continue the following situation with a plausible next event."
                    context, response = original["ctx"], choices[answer]
                else:
                    instruction = "Fill the blank and return the complete sentence."
                    context, response = original["sentence"], original["sentence"].replace("_", choices[answer])
                selected.append(dict(example_id=f"{family}:{entry['revision']}:{source_split}:{index}",
                    family=family, instruction=instruction, context=context, response=response,
                    task="benchmark:"+family, source_uri=f"https://huggingface.co/datasets/{entry['repository']}",
                    source_revision=entry["revision"], source_row=index, source_group=family+":"+semantic,
                    source_split=source_split, historical_train=trained, benchmark_identity=key,
                    semantic_identity=semantic, forced_split="development" if source_split == "validation" else "train" if trained or family == "arc" else None,
                    reference_quality="original_benchmark_correct_answer_reformatted_as_complete_text"))
            if family != "arc":
                selected.sort(key=lambda r: data.digest(["bench-pool-v1", r["example_id"]]))
                selected = selected[:5000]
            rows += selected
    for family, path, repo, revision in (("gsm8k", Path(gsm_path), "openai/gsm8k", GSM_REVISION),
                                       ("mbpp", Path(mbpp_path), "google-research-datasets/mbpp", MBPP_REVISION)):
        receipts.append(dict(path=str(path.resolve()), sha256=sha(path), family=family, split="train",
                             source_uri="https://huggingface.co/datasets/"+repo, revision=revision))
        originals = read_parquet(path)
        counts[family+":train:source_rows"] = len(originals)
        for index, original in enumerate(originals):
            if family == "gsm8k":
                instruction = "Solve this problem. Show the reasoning and give the final answer."
                context, response = original["question"], original["answer"]
                extra = {}
            else:
                instruction = "Write Python code for the following task. Return only the complete code.\n\n"+original["text"]
                context = "Required behavior examples:\n"+"\n".join(original["test_list"])
                response = original["code"]
                extra = dict(test_list=original["test_list"], task_id=original["task_id"],
                    test_setup_code=original.get("test_setup_code", ""), challenge_test_list=original.get("challenge_test_list", []))
            rows.append(dict(example_id=f"{family}:{revision}:train:{index}", family=family,
                instruction=instruction, context=context, response=response, task="benchmark:"+family,
                source_uri="https://huggingface.co/datasets/"+repo, source_revision=revision,
                source_row=index, source_group=family+":"+data.digest(normalize(context or instruction)),
                source_split="train", reference_quality="original_benchmark_training_reference", **extra))
    return rows, dict(counts), receipts


def prepare(output, *, dolly_path, xsum_path=XSUM, model_path=MODEL, seed=582470,
            train_tokens_per_family=70000, dev_per_family=64, xsum_pool=30000,
            exposed_paths=EXPOSED, multi_bench=False, gsm_path=None, mbpp_path=None, arc_dev_path=None):
    from transformers import AutoTokenizer
    output, model_path = Path(output), Path(model_path)
    if output.exists():
        raise FileExistsError(output)
    source_paths = {"dolly": Path(dolly_path), "xsum": Path(xsum_path)}
    source_receipts = {key: dict(path=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size)
                       for key, path in source_paths.items()}
    if Path(xsum_path).resolve() == XSUM.resolve() and source_receipts["xsum"]["sha256"] != XSUM_SHA256:
        raise ValueError("Historical admitted XSum payload changed")
    tok_files = {name: sha(model_path/name) for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json")}
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenization = dict(tokenizer_sha256=data.digest(tok_files),
                        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())
    exact_excluded, window_excluded, exclusions = exposed_questions(exposed_paths)
    rows, inventory = source_rows(dolly_path, xsum_path, xsum_pool=xsum_pool)
    bench_receipts = []
    if multi_bench:
        # Stable bounded Dolly pool, independent of outcomes.
        dolly = sorted((r for r in rows if r["family"] == "dolly"), key=lambda r: data.digest(["dolly-pool-v2", r["example_id"]]))[:6000]
        rows = dolly + [r for r in rows if r["family"] == "xsum"]
        extra, extra_counts, bench_receipts = benchmark_rows(gsm_path=gsm_path, mbpp_path=mbpp_path, arc_dev_path=arc_dev_path)
        rows += extra
        inventory.update(extra_counts)
        rows.sort(key=lambda r: r["example_id"])
    groups = grouped_ids(rows)
    forced_groups = defaultdict(set)
    for row, group in zip(rows, groups):
        if row.get("forced_split"):
            forced_groups[group].add(row["forced_split"])
    eligible, rejected, seen_pairs = defaultdict(list), Counter(), set()
    for row, group in zip(rows, groups):
        source = row["context"] or row["instruction"]
        if not row.get("historical_train") and (normalize(row["instruction"]) in exact_excluded or normalize(source) in exact_excluded
                or set(windows(source)) & window_excluded or set(windows(row["instruction"])) & window_excluded):
            rejected[row["family"]+":exposed_benchmark_overlap"] += 1
            continue
        pair = data.digest([normalize(row["instruction"]), normalize(row["context"]), normalize(row["response"])])
        if pair in seen_pairs:
            rejected[row["family"]+":normalized_exact_duplicate"] += 1
            continue
        seen_pairs.add(pair)
        if len(forced_groups[group]) > 1:
            rejected[row["family"]+":forced_split_group_conflict"] += 1
            continue
        split = next(iter(forced_groups[group])) if forced_groups[group] else split_for(group, seed)
        try:
            record = tokenize_reference(tokenizer, row, split=split, group_id=group, tokenization=tokenization)
        except ValueError as error:
            if str(error) not in {"prompt_over_limit", "response_over_limit", "total_over_limit", "empty_response", "embedded_chat_special_token"}:
                raise
            rejected[row["family"]+":"+str(error)] += 1
            continue
        eligible[(row["family"], split)].append((record, row))
    selected = {"train": [], "development": []}
    coverage = {}
    families = tuple(MULTI_QUOTAS) if multi_bench else ("dolly", "xsum")
    for family in families:
        for split in ("train", "development"):
            candidates = sorted(eligible[family, split], key=lambda x: data.digest([seed, x[0].example_id]))
            picked, tokens = [], 0
            for record, row in candidates:
                quota = MULTI_QUOTAS[family] if multi_bench else train_tokens_per_family
                if ((split == "train" and tokens >= quota)
                        or (split == "development" and len(picked) >= dev_per_family)):
                    break
                picked.append((record, row)); tokens += len(record.response_ids)
            if (split == "train" and (len(picked) < 32 if multi_bench else tokens < train_tokens_per_family)) or (split == "development" and len(picked) < dev_per_family):
                raise ValueError(f"Insufficient admitted {family}/{split}: {len(picked)} records, {tokens} response tokens")
            selected[split] += picked
            coverage[family+":"+split] = dict(eligible=len(candidates), selected=len(picked), response_tokens=tokens,
                eligible_response_tokens=sum(len(x[0].response_ids) for x in candidates),
                target_response_tokens=quota if split == "train" else None)
    if multi_bench and sum(len(r.response_ids) for r, _ in selected["train"]) < 100000:
        raise ValueError("Multi-benchmark corpus falls below 100k distinct complete-response tokens")
    guard = data.SplitGuard()
    for pairs in selected.values():
        for record, _ in pairs:
            guard.add(record)
    output.mkdir(parents=True)
    (output/"sources").mkdir()
    # Preserve the small newly fetched public input; large historical input stays bound in place.
    shutil.copyfile(dolly_path, output/"sources/dolly.jsonl")
    if multi_bench:
        for name, path in (("gsm8k_train.parquet", gsm_path), ("mbpp_train.parquet", mbpp_path), ("arc_validation.parquet", arc_dev_path)):
            shutil.copyfile(path, output/"sources"/name)
    artifacts, split_stats = {}, {}
    texts = []
    for split, pairs in selected.items():
        pairs.sort(key=lambda x: data.digest([seed, "order", x[0].example_id]))
        path = output/(split+".jsonl")
        with path.open("x") as stream:
            for record, row in pairs:
                stream.write(json.dumps(asdict(record), sort_keys=True, ensure_ascii=False)+"\n")
                texts.append(dict(row, split=split, group_id=record.group_id,
                                  prompt_text=messages_for(row)[-1]["content"]))
        artifacts[path.name] = sha(path)
        split_stats[split] = dict(records=len(pairs), input_tokens=sum(len(x[0].input_ids) for x in pairs),
            prompt_tokens=sum(len(x[0].prompt_ids) for x in pairs), response_tokens=sum(len(x[0].response_ids) for x in pairs),
            supervised_tokens=sum(len(x[0].prediction_positions) for x in pairs),
            max_prompt=max(len(x[0].prompt_ids) for x in pairs), max_response=max(len(x[0].response_ids) for x in pairs),
            max_total=max(len(x[0].input_ids) for x in pairs), groups=len({x[0].group_id for x in pairs}),
            historical_train_examples=sum(bool(x[1].get("historical_train")) for x in pairs),
            with_numeric_reference=sum(bool(re.search(r"\d", x[1]["response"])) for x in pairs),
            with_temporal_reference=sum(bool(re.search(r"\b(?:19|20)\d\d\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", x[1]["response"])) for x in pairs),
            by_task=dict(Counter(x[0].task for x in pairs)))
    with (output/"text.jsonl").open("x") as stream:
        for row in texts:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False)+"\n")
    artifacts["text.jsonl"] = sha(output/"text.jsonl")
    artifacts["sources/dolly.jsonl"] = sha(output/"sources/dolly.jsonl")
    if multi_bench:
        artifacts.update({"sources/"+name: sha(output/"sources"/name) for name in ("gsm8k_train.parquet", "mbpp_train.parquet", "arc_validation.parquet")})
    manifest = dict(schema="jspace.qwen-complete-response-data.v1", status="prepared", model_id="Qwen/Qwen2.5-3B-Instruct",
        model_revision=MODEL_REVISION, model_path=str(model_path), tokenization=tokenization, tokenizer_files=tok_files,
        seed=seed, limits=dict(prompt=288, response_including_eos=128, total=416, oversize="reject_without_truncation"),
        source_receipts=source_receipts, benchmark_receipts=bench_receipts, source_inventory=inventory, pool_rows=len(rows), coverage=coverage,
        rejected=dict(rejected), splits=split_stats, artifacts=artifacts, exposed_exclusion_receipts=exclusions,
        benchmark_policy="Multi-benchmark mode uses original sourceTRAIN, with priorTRAIN permitted only asTRAIN; ARC sourcevalidation supplies DEV. Historical nontrain and reserved identities excluded. Named exposed questions/window matches excluded except authenticated priorTRAIN. Semantic or pretraining contamination not certified." if multi_bench else "No benchmark family used as corpus; named exposed question/window matches excluded.",
        split_policy="Hash source/normalized-context/exact-prompt/shared20-word-window connected components before eligibility; 90/10 TRAIN/DEV; no TEST material.",
        near_duplicate_policy="Conservative shared20-word windows at stride10 plus normalized exact source/prompt. Related stories and semantic paraphrases beyond this heuristic remain possible.",
        label_policy="Complete original human/reference text; not new teacher generations or factual gold. Frozen native teacher must score every response token and terminal EOS.",
        training_admitted=True, requires_study_manifest_binding=True, test_opened_by_this_producer=False,
        prior_preparation_access_incident="v1 mistakenly decoded reserved512 tasks only for question exclusion, including JSON labels in memory; no model scoring or labels used. v1/access_correction.json records it. A new untouched confirmation is required.",
        producer_sha256=sha(Path(__file__)))
    manifest["fingerprint"] = data.digest(manifest)
    data._json(output/"manifest.json", manifest)
    return manifest


def load_records(root, split):
    """Read only a declared TRAIN/DEV artifact after checking its frozen hash."""
    if split not in {"train", "development"}:
        raise ValueError("Only TRAIN/DEV admitted")
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    fingerprint = manifest.pop("fingerprint")
    if data.digest(manifest) != fingerprint:
        raise ValueError("Corpus manifest fingerprint changed")
    path = root/(split+".jsonl")
    if sha(path) != manifest["artifacts"][path.name]:
        raise ValueError("Corpus payload changed")
    return list(data.stream_records(path, allowed_splits=(split,)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dolly", type=Path, required=True)
    parser.add_argument("--xsum", type=Path, default=XSUM)
    parser.add_argument("--train-tokens-per-family", type=int, default=70000)
    parser.add_argument("--dev-per-family", type=int, default=64)
    parser.add_argument("--xsum-pool", type=int, default=30000)
    parser.add_argument("--multi-bench", action="store_true")
    parser.add_argument("--gsm8k", type=Path)
    parser.add_argument("--mbpp", type=Path)
    parser.add_argument("--arc-validation", type=Path)
    args = parser.parse_args()
    result = prepare(args.output, dolly_path=args.dolly, xsum_path=args.xsum,
        train_tokens_per_family=args.train_tokens_per_family, dev_per_family=args.dev_per_family, xsum_pool=args.xsum_pool,
        multi_bench=args.multi_bench, gsm_path=args.gsm8k, mbpp_path=args.mbpp, arc_dev_path=args.arc_validation)
    print(json.dumps({k: result[k] for k in ("fingerprint", "splits", "coverage", "rejected")}, indent=2))
