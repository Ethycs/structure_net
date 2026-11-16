"""Shared broadened recovery data; option balancing and sealed audit are explicit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil

from . import jspace_3b_sanity_data as sanity
from . import jspace_recovery_expansion as expansion
from .jspace_bench_data import DATASETS, normalize
from .jspace_recovery_data import SYSTEM, digest, identity, read, sha, write

ROOT = sanity.ROOT
CONFIG = dict(seed=582464, audit_seed=582465, train_count=4608, development_per_family=64,
    audit_per_family=32, max_tokens=384, permutation_seed_offset=100, family_shuffle_seed_offset=1000)
PROTOCOL = ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-broad-recovery.md"
POSITION_REFERENCE = re.compile(r"\b(?:all|none|both)\s+(?:of\s+)?(?:the\s+)?(?:above|following|previous)\b|"
    r"\b(?:option|choice|answer)s?\s+[A-D]\b|\b[A-D]\s+(?:and|or)\s+[A-D]\b", re.I)
QUESTION_REFERENCE = re.compile(r"\b(?:option|choice|answer)s?\s+[A-D]\b", re.I)


def question_from_case(case):
    suffix = "\n\n"+"\n".join(f"{chr(65+i)}. {c}" for i,c in enumerate(case["choices"]))
    if not case["prompt"].endswith(suffix):
        raise ValueError("Original question/option formatting cannot be recovered exactly")
    question = case["prompt"][:-len(suffix)]
    if identity(question, case["choices"]) != case["identity"]:
        raise ValueError("Original semantic source identity changed")
    return question


def semantic_identity(question, choices):
    return identity(question, sorted(choices))


class ContentLayout:
    """Locate user content from the actual pinned template and fast-tokenizer offsets."""
    def __init__(self, tokenizer, label_ids):
        self.tokenizer, self.label_ids = tokenizer, label_ids
        marker = "JSPACE_UNIQUE_USER_CONTENT_MARKER_582464"
        rendered = tokenizer.apply_chat_template([dict(role="system", content=SYSTEM),
            dict(role="user", content=marker)], tokenize=False, add_generation_prompt=True)
        if rendered.count(marker) != 1:
            raise ValueError("Cannot identify exact user-content boundaries")
        self.prefix, self.suffix = rendered.split(marker)

    def encode(self, question, choices, answer):
        prompt = question+"\n\n"+"\n".join(f"{chr(65+i)}. {c}" for i,c in enumerate(choices))
        messages = [dict(role="system", content=SYSTEM), dict(role="user", content=prompt)]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered != self.prefix+prompt+self.suffix:
            raise ValueError("User-content rendering differs from pinned template")
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = list(encoded["input_ids"]), encoded["offset_mapping"]
        if ids != self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True):
            raise ValueError("Offset encoding differs from original chat-tokenization path")
        begin, end = len(self.prefix), len(self.prefix)+len(prompt)
        user_tokens = [i for i,(a,b) in enumerate(offsets) if begin <= a < b <= end]
        if not user_tokens or user_tokens != list(range(user_tokens[0],user_tokens[-1]+1)):
            raise ValueError("User content must form one contiguous token span")
        option_labels, cursor = [], begin+len(question)+2
        for i, choice in enumerate(choices):
            option_labels.append((cursor,cursor+2))
            cursor += len(f"{chr(65+i)}. {choice}")+1
        eligible = []
        for target in user_tokens[1:]:
            a,b = offsets[target]
            text = rendered[a:b]
            if (target-1 in user_tokens and any(c.isalnum() for c in text)
                    and "\n" not in text and "\r" not in text
                    and not any(a < right and b > left for left,right in option_labels)):
                eligible.append(target-1)
        if len(eligible) < 3:
            raise ValueError("At least three meaningful user-content targets required")
        content = [eligible[0], eligible[(len(eligible)-1)//2], eligible[-1], len(ids)-1]
        legacy = [0,(len(ids)-2)//2,len(ids)-2,len(ids)-1]
        targets = lambda positions: [ids[p+1] for p in positions[:3]]+[self.label_ids[answer]]
        return dict(prompt=prompt,input_ids=ids,user_content_char_span=[begin,end],
            user_content_token_span=[user_tokens[0],user_tokens[-1]+1],
            content_loss_positions=content,content_targets=targets(content),
            content_eligible_count=len(eligible),legacy_loss_positions=legacy,legacy_targets=targets(legacy))


def permute_training(original, layout, *, seed=582464, max_tokens=384):
    pools, summaries, source_seen = defaultdict(list), {}, set()
    for index,case in enumerate(original):
        if case["split"] != "train":
            continue
        question = question_from_case(case)
        if case["identity"] in source_seen or case["source_split"] != "train":
            raise ValueError("Every original training case must occur once")
        source_seen.add(case["identity"])
        reason = "position-dependent option/reference" if (QUESTION_REFERENCE.search(question)
            or any(POSITION_REFERENCE.search(c) for c in case["choices"])) else None
        pools[case["family"]].append((index,case,question,reason))
    transformed = {}
    for fi,family in enumerate(DATASETS):
        rows = pools[family]
        rng = random.Random(seed+CONFIG["permutation_seed_offset"]+fi)
        rng.shuffle(rows)
        # Fixed-order cases contribute to balance before freely permutable cases.
        rows.sort(key=lambda row:row[3] is None)
        counts, by_choices, output, flags = Counter(), defaultdict(Counter), [], []
        for source_index,case,question,reason in rows:
            n = len(case["choices"])
            original_encoded = layout.encode(question,case["choices"],case["answer"])
            if original_encoded["input_ids"] != case["input_ids"]:
                raise ValueError("Pinned tokenizer no longer reproduces original training input")
            permutations = [tuple(range(n))] if reason else list(itertools.permutations(range(n)))
            rng.shuffle(permutations)
            permutations.sort(key=lambda p:by_choices[n][p.index(case["answer"])])
            for permutation in permutations:
                answer = permutation.index(case["answer"])
                choices = [case["choices"][i] for i in permutation]
                encoded = original_encoded if permutation == tuple(range(n)) else layout.encode(question,choices,answer)
                if len(encoded["input_ids"]) <= max_tokens:
                    break
            else:
                raise ValueError("Original case cannot be preserved under the token bound")
            counts[answer] += 1;by_choices[n][answer] += 1
            output.append(dict(encoded,identity=identity(question,choices),source_identity=case["identity"],
                semantic_identity=semantic_identity(question,case["choices"]),family=family,split="train",
                choices=choices,answer=answer,source_answer=case["answer"],source_row=case["source_row"],
                source_split=case["source_split"],source_case_index=source_index,permutation=list(permutation),
                permutation_preserved_reason=reason))
            if reason:
                flags.append(dict(source_identity=case["identity"],reason=reason))
        random.Random(seed+CONFIG["family_shuffle_seed_offset"]+fi).shuffle(output)
        transformed[family] = output
        summaries[family] = dict(count=len(output),answer_counts={str(i):counts[i] for i in range(max(len(c["choices"]) for c in output))},
            answer_counts_by_choice_count={str(n):{str(i):counter[i] for i in range(n)} for n,counter in by_choices.items()},
            choice_counts=dict(Counter(len(c["choices"]) for c in output)),preserved_flags=flags,
            changed_order=sum(c["permutation"] != list(range(len(c["choices"]))) for c in output))
    # Each source occurs exactly once. Late batches contain only non-exhausted families.
    interleaved = [transformed[f][i] for i in range(max(map(len,transformed.values())))
                   for f in DATASETS if i < len(transformed[f])]
    if len(interleaved) != len(source_seen) or {c["source_identity"] for c in interleaved} != source_seen:
        raise ValueError("Interleaving dropped or duplicated training cases")
    return interleaved,summaries


def select_fresh(family, rows, excluded, semantic_excluded, tokenizer, *, seed, count):
    # No actual answer column is present in rows. Semantic rejection precedes tokenization.
    order = list(range(len(rows)));random.Random(seed).shuffle(order)
    selected, seen, semantic_seen, counts = [],set(excluded),set(semantic_excluded),Counter(raw_rows=len(rows))
    for index in order:
        try:
            question,choices = sanity.prompt_parts(family,rows[index])
        except (ValueError,IndexError):
            counts["unsupported"] += 1;continue
        key,semantic = identity(question,choices),semantic_identity(question,choices)
        if key in seen or semantic in semantic_seen:
            counts["excluded_or_duplicate"] += 1;continue
        seen.add(key);semantic_seen.add(semantic)
        prompt = question+"\n\n"+"\n".join(f"{chr(65+i)}. {c}" for i,c in enumerate(choices))
        ids = tokenizer.apply_chat_template([dict(role="system",content=SYSTEM),dict(role="user",content=prompt)],
            tokenize=True,add_generation_prompt=True)
        if not 3 <= len(ids) <= CONFIG["max_tokens"]:
            counts["outside_token_limit"] += 1;continue
        selected.append(dict(identity=key,semantic_identity=semantic,family=family,source_row=index,
            question=question,choices=choices,input_ids=list(ids)))
        if len(selected) == count:
            return selected,dict(counts)
    raise ValueError(f"Insufficient fresh {family}: {len(selected)} eligible; need {count}; no counts changed")


def validate_items(items, train_sources, excluded, audit_ids, audit_semantics):
    train = [c for c in items if c["split"] == "train"]
    dev = [c for c in items if c["split"] == "development"]
    if (len(train) != 4608 or len(dev) != 256 or len(items) != 4864
            or len({c["source_identity"] for c in train}) != 4608
            or {c["source_identity"] for c in train} != set(train_sources)
            or Counter(c["family"] for c in dev) != {f:64 for f in DATASETS}):
        raise ValueError("Broad recovery population or original-case preservation failed")
    train_ids = {c["source_identity"] for c in train}|{c["identity"] for c in train}
    dev_ids = {c["source_identity"] for c in dev}
    train_sem = {c["semantic_identity"] for c in train};dev_sem={c["semantic_identity"] for c in dev}
    if (len(dev_ids) != 256 or dev_ids & (set(excluded)|train_ids|set(audit_ids))
            or set(audit_ids) & set(excluded) or train_ids & set(audit_ids) or train_sem & dev_sem
            or (train_sem|dev_sem) & set(audit_semantics)):
        raise ValueError("Semantic/source identity leakage between broad splits")
    for c in items:
        n=len(c["input_ids"]);start,end=c["user_content_token_span"]
        if (not 3 <= n <= 384 or len(c["content_loss_positions"]) != 4
                or len(set(c["content_loss_positions"])) != 4 or c["content_loss_positions"][-1] != n-1
                or any(not start <= p < p+1 < end for p in c["content_loss_positions"][:3])
                or c["permutation"][c["answer"]] != c["source_answer"]):
            raise ValueError("Content positions, permutation answer or token limit failed")


def freeze(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Broad data freezing requires CUDA hidden")
    root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    expansion.verify(sanity.EXPANSION)
    reserved,confirmation,inputs=sanity.reserved_identities()
    for name in ("items.json","tasks.json","new_training.json"):
        inputs[str((sanity.EXPANSION/name).resolve())]=sha(sanity.EXPANSION/name)
    original=read(sanity.EXPANSION/"items.json")["items"]
    model=confirmation["tokenizer_model"]
    for name in ("tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","config.json"):
        path=Path(model["path"])/name
        if sha(path)!=model["files"][name]:raise ValueError("Changed pinned tokenizer")
        inputs[str(path)]=model["files"][name]
    tokenizer=AutoTokenizer.from_pretrained(model["path"],local_files_only=True,trust_remote_code=False)
    labels=[tokenizer.encode(chr(65+i),add_special_tokens=False) for i in range(4)]
    if any(len(ids)!=1 for ids in labels):raise ValueError("Single-token answer labels required")
    labels=[ids[0] for ids in labels];layout=ContentLayout(tokenizer,labels)
    train,balance=permute_training(original,layout,seed=CONFIG["seed"])
    if len(train)!=CONFIG["train_count"]:raise ValueError("All 4608 existing TRAIN cases required")
    excluded=set(reserved);historical=set()
    for pattern in ("jspace*/**/tasks.json","jspace*/**/items.json","jspace*/**/new_training.json","jspace*/**/exclusions.json"):
        historical.update(p.resolve() for p in (ROOT/"data/experiments").glob(pattern))
    historical.discard((sanity.CONFIRMATION/"tasks.json").resolve())
    for path in sorted(historical):
        excluded.update(sanity.identity_candidates(path));inputs[str(path)]=sha(path)
    excluded.update(c["source_identity"] for c in train);excluded.update(c["identity"] for c in train)
    semantic_excluded={c["semantic_identity"] for c in train}
    manifest=read(sanity.DATA_ROOT/"manifest.json");inputs[str(sanity.DATA_ROOT/"manifest.json")]=sha(sanity.DATA_ROOT/"manifest.json")
    development,audit,inventories=[],[],{f:{} for f in DATASETS}
    # All 256 development cases have priority; only then select the sealed audit.
    for role,count,seed,destination in (("development",64,CONFIG["seed"],development),("audit",32,CONFIG["audit_seed"],audit)):
        for fi,family in enumerate(DATASETS):
            entry=manifest[family];split=entry["evaluation_source_split"];file=entry["files"][split]
            path=sanity.DATA_ROOT/file["local"]
            if sha(path)!=file["sha256"]:raise ValueError("Changed pinned evaluation data")
            inputs[str(path)]=file["sha256"]
            rows=pq.read_table(path,columns=sanity.PROMPT_COLUMNS[family]).to_pylist()
            selected,details=select_fresh(family,rows,excluded,semantic_excluded,tokenizer,seed=seed+fi,count=count)
            answer_rows=pq.read_table(path,columns=[sanity.LABEL_COLUMNS[family]]).take([c["source_row"] for c in selected]).to_pylist()
            for case,answer_row in zip(selected,answer_rows):
                question,choices,answer=normalize(family,{**rows[case["source_row"]],**answer_row})
                encoded=layout.encode(question,choices,answer)
                if encoded["input_ids"]!=case["input_ids"]:raise ValueError("Fresh source tokenization changed")
                destination.append(dict(encoded,identity=case["identity"],source_identity=case["identity"],
                    semantic_identity=case["semantic_identity"],family=family,split=role,choices=choices,answer=answer,
                    source_answer=answer,source_split=split,source_row=case["source_row"],permutation=list(range(len(choices))),
                    permutation_preserved_reason="TRAIN-only augmentation; evaluation order unchanged"))
                excluded.add(case["identity"]);semantic_excluded.add(case["semantic_identity"])
            inventories[family][role]=details
    train_sources={c["source_identity"] for c in train};audit_ids={c["identity"] for c in audit}
    # The frozen exclusion inventory excludes new DEV/audit themselves by construction.
    original_exclusions=excluded-{c["identity"] for c in development}-audit_ids
    items=train+development
    validate_items(items,train_sources,original_exclusions,audit_ids,{c["semantic_identity"] for c in audit})
    if len(audit_ids)!=128 or Counter(c["family"] for c in audit)!={f:32 for f in DATASETS}:
        raise ValueError("Separate sealed audit must contain 128 unique balanced cases")
    tasks=dict(splits={"train":train,"development":development},scope="Recovery fitting and selection only; old64/128 diagnostic-only")
    audit_tasks=dict(splits={"audit":audit},scope="Sealed recovery audit; score once after global DEV-selected winner is frozen")
    exclusions=dict(identities=sorted(original_exclusions),reserved_identities=sorted(reserved),training_source_identities=sorted(train_sources),
        historical_files=[str(p) for p in sorted(historical)])
    audit_receipt=dict(identities=sorted(audit_ids),semantic_identities=sorted(c["semantic_identity"] for c in audit),
        families={f:32 for f in DATASETS},seed=CONFIG["audit_seed"],content_policy="Trainer verifies hashes/identities only")
    sources=[Path(__file__).resolve(),ROOT/"tests/structure_net/test_jspace_3b_broad_recovery_data.py",
        ROOT/"experiments/structure_net/jspace_3b_sanity_data.py",ROOT/"experiments/structure_net/jspace_recovery_expansion.py",
        ROOT/"experiments/structure_net/jspace_recovery_data.py",ROOT/"experiments/structure_net/jspace_bench_data.py"]
    inputs[str(PROTOCOL)]=sha(PROTOCOL)
    counts={s:sum(c["split"]==s for c in items) for s in ("train","development")}
    tokens={s:sum(len(c["input_ids"]) for c in items if c["split"]==s) for s in counts}
    total_tokens=sum(tokens.values());n=len(items)
    plan=dict(schema="jspace.3b-broad-recovery-data.v1",config=CONFIG,created_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha(p) for p in sources},input_sha256=inputs,model=model,label_ids=labels,
        counts=counts,token_counts=tokens,audit_count=128,audit_token_count=sum(len(c["input_ids"]) for c in audit),
        training_order=list(range(len(train))),training_order_policy="Deterministically shuffle within each family, then interleave active families; each case once",
        weighting="Uniform per case in both arms; unequal family pools mean late batches are not four-per-family",
        option_balance=balance,selection_inventory=inventories,confirmation_fingerprint=confirmation["fingerprint"],
        historical_file_count=len(historical),exclusion_count=len(original_exclusions),items_digest=digest(items),tasks_digest=digest(tasks),
        audit_tasks_digest=digest(audit_tasks),audit_identities_digest=digest(audit_receipt),exclusions_digest=digest(exclusions),
        content_position_policy="Three evenly spaced eligible lexical next-token targets within user content, excluding option labels/newlines, plus answer; original legacy four retained",
        payload_estimates=dict(native_logits_bf16_8_positions=n*8*151936*2,parent_logits_bf16_4_positions=n*4*151936*2,
            two_native_boundary_bf16_banks=total_tokens*2048*2*2,training_final_features_bf16=len(train)*1024*2),
        runtime_packages={name:version(name) for name in ("transformers","tokenizers","pyarrow")},
        final_confirmation_content_decoded=False,policy="Shared fixed data for both arms; fresh256 selects; separate128 audit once after winner freeze; final512 remains sealed")
    plan["fingerprint"]=digest(plan);root.mkdir(parents=True)
    for source in sources:
        destination=root/"producer"/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    for name,value in (("plan.json",plan),("items.json",dict(items=items)),("tasks.json",tasks),
                       ("audit_tasks.json",audit_tasks),("audit_identities.json",audit_receipt),("exclusions.json",exclusions)):
        write(root/name,value)
    write(root/"manifest.json",dict(status="frozen_unscored",fingerprint=plan["fingerprint"],artifacts={name:sha(root/name) for name in
        ("plan.json","items.json","tasks.json","audit_tasks.json","audit_identities.json","exclusions.json")}))
    return verify(root)


def verify(root):
    """Authenticate all bytes while never decoding sealed recovery-audit content."""
    root=Path(root);plan,manifest=read(root/"plan.json"),read(root/"manifest.json")
    if (plan["fingerprint"]!=digest({k:v for k,v in plan.items() if k!="fingerprint"}) or plan["config"]!=CONFIG
            or manifest["fingerprint"]!=plan["fingerprint"] or manifest["status"]!="frozen_unscored"):
        raise ValueError("Broad recovery data plan changed")
    for name,expected in manifest["artifacts"].items():
        if sha(root/name)!=expected:raise ValueError("Broad recovery artifact changed")
    for name,expected in plan["input_sha256"].items():
        if sha(name)!=expected:raise ValueError("Broad recovery input changed")
    for name,expected in plan["source_sha256"].items():
        if sha(name)!=expected or sha(root/"producer"/Path(name).relative_to(ROOT))!=expected:
            raise ValueError("Broad recovery producer changed")
    items,tasks,audit,exclusions=(read(root/name) for name in ("items.json","tasks.json","audit_identities.json","exclusions.json"))
    if (digest(items["items"])!=plan["items_digest"] or digest(tasks)!=plan["tasks_digest"]
            or digest(audit)!=plan["audit_identities_digest"] or digest(exclusions)!=plan["exclusions_digest"]
            or plan["training_order"]!=list(range(4608)) or len(audit["identities"])!=128):
        raise ValueError("Broad recovery rows/order changed")
    validate_items(items["items"],exclusions["training_source_identities"],exclusions["identities"],audit["identities"],audit["semantic_identities"])
    return dict(status="frozen_unscored",root=str(root.resolve()),fingerprint=plan["fingerprint"],counts=plan["counts"],
        audit_count=128,token_counts=plan["token_counts"],option_balance=plan["option_balance"],payload_estimates=plan["payload_estimates"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("freeze","verify"))
    parser.add_argument("--root",type=Path,default=ROOT/"data/experiments/jspace_3b_broad_recovery_data/20261115_v1")
    args=parser.parse_args();print(json.dumps(globals()[args.stage](args.root),allow_nan=False))
