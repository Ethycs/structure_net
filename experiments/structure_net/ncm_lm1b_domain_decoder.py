"""Task-domain decoder adaptation with the existing 64 sources held out."""
import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import torch

from experiments.structure_net import ncm_lm1b_decoder_repair as base
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

native, framed, semantics, q, ROOT = base.native, base.framed, base.semantics, base.q, base.ROOT
HYPOTHESIS = "ncm-lm1b-source-disjoint-domain-decoder-v1"
MODULE = "experiments.structure_net.ncm_lm1b_domain_decoder"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-domain-decoder-plan.md")
DATA = ROOT / "20261024_lm1b_domain_decoder_data_v1"
PILOT = ROOT / "20261024_lm1b_domain_decoder_pilot_v1"
TRAIN = ROOT / "20261024_lm1b_domain_decoder_train_v1"
WORKFLOW = ROOT / "20261024_lm1b_domain_decoder_workflow_v1"
PREVIOUS = ROOT / "20261024_lm1b_decoder_repair_results_v1"
PREVIOUS_HYPOTHESIS = "ncm-lm1b-fixed1024-decoder-repair-semantic-v1"
PREVIOUS_SHA = "9fcefdba0ef580599b0b226d762b898b180601756ebfd57513ba92e7c7441075"
FIRST_CACHE = ROOT / "20261024_lm1b_decoder_repair_grade_v1/judgment_cache.json"
FIRST_CACHE_SHA = "883d00182e9614ba999e0d1ab4182988cd519f7c956ec5ae7d23075415fc40fb"
SEMANTIC_PACKETS = ROOT / "20261024_lm1b_padding_semantic_packets_v1"
AUTHOR_DATA = base.DATA
_producing, _train, _rate, _native_augment = base.producing, base.train, base.rate, base.augment


@dataclass(frozen=True)
class Config(base.Config):
    corpus_seed: int = 83142
    sampler_seed: int = 84142
    augmentation_seed: int = 85142
    dropout_seed: int = 86142


def identities(rows):
    return {"case": {r["id"] for r in rows}, "session": {s["source_id"] for r in rows for s in r["spans"]},
        "group": {r["group_id"] for r in rows}, "source_hash": {r["source_sha256"] for r in rows},
        "turn_hash": {s["turn_text_sha256"] for r in rows for s in r["spans"]}}


def split(rows, heldout_ids):
    if len(rows) != 690 or len({r["id"] for r in rows}) != 690 or len(set(heldout_ids)) != 64:
        raise ValueError("all690 cases and exactly64 held-out IDs required")
    held = [r for r in rows if r["id"] in set(heldout_ids)]
    train = [r for r in rows if r["id"] not in set(heldout_ids)]
    if len(held) != 64 or len(train) != 626:
        raise ValueError("exact626/64 split required")
    a, b = identities(train), identities(held)
    if any(a[k] & b[k] for k in a):
        raise ValueError("codec fit and semantic sources overlap")
    return train, held, {k: {"train_unique": len(a[k]), "heldout_unique": len(b[k]), "overlap": 0} for k in a}


def augment(frame, rng, config=None):
    """Fixed complete corpus frames; no runtime crop or alteration of targets."""
    values = frame.tolist()
    active = [v for v in values if v != 0]
    if (len(values) != 128 or len(active) < 3 or active[0] != 101 or active[-1] != 101
            or any(v == 0 or not 0 <= v < 30522 for v in active[1:-1])
            or values[:len(active)] != active):
        raise ValueError("complete right-padded native CLS frame required")
    return frame.clone(), {"mode": "fixed_domain_mixture", "length": len(active) - 2}


def configure():
    for name in ("DATA", "PILOT", "TRAIN", "WORKFLOW", "HYPOTHESIS"):
        setattr(base, name, globals()[name])
    base.Config, base.augment, base.producing = Config, augment, producing


def producing(root, stage, inputs=(), config=None):
    files = [Path(__file__), PLAN, Path("tests/structure_net/test_ncm_lm1b_domain_decoder.py")]
    extra = [PREVIOUS / "campaign_results.json", PREVIOUS / (PREVIOUS_HYPOTHESIS + "_registry_readback.json"),
        FIRST_CACHE, SEMANTIC_PACKETS / "cases.json", *files]
    domain_config = {**(config or {}), **asdict(Config()),
        "hypothesis_id": HYPOTHESIS, "training_source": "626 task TRAIN sessions plus author TRAIN replay",
        "application_targets_used_for_gradient": True, "heldout64_targets_used_for_gradient": False,
        "validation_used_for_training": False, "semantic_heldout_cases": 64, "codec_fit_cases": 626,
        "corpus_mixture_counts": {"author_train": 3276, "task_summary": 6554, "task_source": 6554},
        "native_replay_augmentation": "precomputed baseline20percent full/80percent variable lexical prefix",
        "task_augmentation": "none; all words in all windows retained in the candidate pools",
        "runtime_augmentation": "none; uniform fixed-frame sampling with replacement",
        "intervention": "task-domain corpus and length distribution, with20percent native replay",
        "previous_generic_repair_rejected": True, "prior_first_judgments": 2266,
        "same64_previously_observed_development": True, "independent_test_claim": False}
    p = _producing(root, stage, [*inputs, *extra], domain_config)
    p["config"].update(domain_config)
    p["config"]["schema"] = "nal.ncm-lm1b-domain-decoder.v1"
    for i, f in enumerate(files):
        p["source_hashes"][str(f)] = file_hash(f)
        (root / "producer_source" / f"domain_{i}_{f.name}").write_bytes(f.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def prepare():
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only domain corpus preparation required")
    native.vendor.terminal(AUTHOR_DATA)
    if file_hash(native.CASES) != native.CASES_SHA or file_hash(FIRST_CACHE) != FIRST_CACHE_SHA:
        raise ValueError("unchanged original targets and latest first cache required")
    rows = q.read(native.CASES)["train"]
    chosen = q.read(SEMANTIC_PACKETS / "cases.json")
    train, held, proof = split(rows, [r["id"] for r in chosen])
    semantics.reference.verify_public(rows)
    tokenizer = native.tokenizer()
    tokenizer.model_max_length = 1_000_000
    pools, origins, texts = {"task_summary": [], "task_source": []}, [], []
    for row in train:
        for kind, field in (("task_summary", "target_text"), ("task_source", "source")):
            text = row[field]
            lexical = tokenizer.encode(text, add_special_tokens=False)
            full, frames = framed.pack(lexical)
            if len(full) != len(lexical) + 2 * len(frames):
                raise ValueError("complete lexical coverage required")
            start = len(pools[kind])
            pools[kind].extend(frames)
            origins.extend({"kind": kind, "pool_row": start + i, "case_id": row["id"],
                "source_id": row["source_id"], "group_id": row["group_id"], "source_sha256": row["source_sha256"],
                "text_sha256": text_hash(text), "window": i, "windows": len(frames)} for i in range(len(frames)))
            texts.append({"case_id": row["id"], "kind": kind, "text_sha256": text_hash(text),
                "lexical_tokens": len(lexical), "complete_frames": len(frames)})
    author = torch.load(AUTHOR_DATA / "frames.pt", map_location="cpu", weights_only=True)
    if len(author) != 16384:
        raise ValueError("verified fixed author TRAIN corpus required")
    rng = random.Random(Config().corpus_seed)
    augmentation = torch.Generator().manual_seed(Config().augmentation_seed)
    frames, selection = [], []
    for index in rng.sample(range(len(author)), 3276):
        frame, meta = _native_augment(author[index], augmentation)
        frames.append(frame.tolist()); selection.append({"kind": "author_train", "pool_row": index, "augmentation": meta})
    for kind in ("task_summary", "task_source"):
        if not pools[kind]: raise ValueError("nonempty domain candidate pool required")
        for _ in range(6554):
            index = rng.randrange(len(pools[kind]))
            frames.append(pools[kind][index]); selection.append({"kind": kind, "pool_row": index})
    if len(frames) != Config().corpus_frames:
        raise ValueError("fixed16384-frame mixture required")
    p = producing(DATA, "source_disjoint626_task_domain_corpus", [native.CASES,
        AUTHOR_DATA / "frames.pt", AUTHOR_DATA / "result.json", AUTHOR_DATA / "worker_terminal_receipt.json"], {"CPU_only": True})
    torch.save(torch.tensor(frames, dtype=torch.long), DATA / "frames.pt")
    write_json(DATA / "selection.json", selection)
    write_json(DATA / "origins.json", origins)
    write_json(DATA / "text_coverage.json", texts)
    write_json(DATA / "split.json", {"train_ids": [r["id"] for r in train], "heldout_ids": [r["id"] for r in held],
        "semantic_order": [r["id"] for r in chosen], "source_disjoint_proof": proof,
        "fit_identities": {k: sorted(v) for k, v in identities(train).items()},
        "heldout_identities": {k: sorted(v) for k, v in identities(held).items()}})
    native.vendor.verify(p)
    result = {"status": "source_disjoint_domain_corpus_prepared", "frames": len(frames), "fit_cases": 626,
        "heldout_cases": 64, "pool_frames": {k: len(v) for k, v in pools.items()}, "native_replay_frames": 3276,
        "source_disjoint_proof": proof, "frames_sha256": file_hash(DATA / "frames.pt"),
        "hashes": {n: file_hash(DATA / n) for n in ("selection.json", "origins.json", "text_coverage.json", "split.json")},
        "application_fit_targets_used": True, "heldout_targets_used_for_gradient": False,
        "validation_training": False, "task_accuracy": None, "producer_fingerprint": p["fingerprint"]}
    write_json(DATA / "result.json", result)
    print(json.dumps(result), flush=True)


def workflow():
    if (file_hash(PREVIOUS / "campaign_results.json") != PREVIOUS_SHA
            or q.read(PREVIOUS / (PREVIOUS_HYPOTHESIS + "_registry_readback.json"))["record"]["campaign_sha256"] != PREVIOUS_SHA):
        raise ValueError("registered generic decoder failure required")
    if any(p.exists() for p in (WORKFLOW, DATA, PILOT, TRAIN)):
        raise ValueError("fresh domain decoder workflow required")
    p = producing(WORKFLOW, "source_disjoint_domain_decoder_training")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    try:
        for stage, root, gpu in (("prepare", DATA, False), ("pilot", PILOT, True), ("train", TRAIN, True)):
            native.vendor.verify(p)
            command = [sys.executable, "-u", "-m", MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "2" if gpu else ""})
            journal.update(active_stage=stage, child_pid=child.pid); write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"stage": stage, "exit_code": code, "child_pid": child.pid, "observer_pid": os.getpid(),
                "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if root.exists(): write_json(root / "worker_terminal_receipt.json", receipt)
            journal["stages"].append(receipt)
            if code: raise RuntimeError("domain decoder stage failed: " + stage)
        native.vendor.verify(p)
        journal.update(status="decoder_repair_training_completed_quality_unmeasured", active_stage=None, child_pid=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


def train(pilot=False):
    _train(pilot)
    root = PILOT if pilot else TRAIN
    result = q.read(root / "result.json")
    result.update(application_targets_used_for_gradient=True, application_fit_cases=626,
        semantic_heldout_cases=64, heldout_targets_used_for_gradient=False,
        training_source="task-domain626 plus author TRAIN replay")
    write_json(root / "result.json", result)
    print(json.dumps({"domain_result": result}), flush=True)


# Read-only lifecycle helpers shared with the frozen generic trainer/evaluator.
frozen_hash, same, rate = base.frozen_hash, base.same, _rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "train", "workflow"))
    args = parser.parse_args()
    configure()
    {"prepare": prepare, "pilot": lambda: train(True), "train": train, "workflow": workflow}[args.stage]()
