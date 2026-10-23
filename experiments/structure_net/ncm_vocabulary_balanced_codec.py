"""Add vocabulary-balanced native-token exercises to the immutable decoder trainer."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_broad_decoder_transfer as engine
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.train_ncm_memory import save

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-vocabulary-balanced-codec-plan.md")
ORIGINAL_DATA, ORIGINAL_TRAIN = engine.DATA, engine.TRAIN
DATA = ROOT / "20261022_balanced_codec_data_v1"
PILOT = ROOT / "20261022_balanced_codec_pilot_v1"
TRAIN = ROOT / "20261022_balanced_codec_train_v1"
POLICY = {"length_seed": 5042, "content_seed": 6042, "minimum_length": 8, "maximum_length": 47,
          "content_vocabulary": 50256, "synthetic_per_natural": 2,
          "synthetic_kind": "native_token_ID_exercises_not_natural_summaries"}


def synthetic_targets(count, *, vocabulary=50256, length_seed=5042, content_seed=6042):
    """Balanced permutation stream; content IDs exclude reserved EOS and PAD."""
    lengths = torch.randint(8, 48, (count,), generator=torch.Generator().manual_seed(length_seed)).tolist()
    total = sum(lengths)
    generator = torch.Generator().manual_seed(content_seed)
    stream = torch.cat([torch.randperm(vocabulary, generator=generator) for _ in range((total+vocabulary-1)//vocabulary)])[:total].tolist()
    rows, offset = [], 0
    for i, length in enumerate(lengths):
        ids = stream[offset:offset+length]+[50256]+[50257]*(47-length)
        offset += length
        rows.append({"id": f"synthetic-native-{i:07d}", "target_kind": POLICY["synthetic_kind"], "native_target_ids": ids})
    return rows


def actual_exposure(raw, anchors):
    cfg = engine.Config()
    rng = torch.Generator().manual_seed(cfg.sampler_seed)
    bank = torch.tensor([r["native_target_ids"] for r in raw])
    other = torch.tensor([r["native_target_ids"] for r in anchors])
    counts = torch.zeros(50258, dtype=torch.int64)
    for _ in range(cfg.steps):
        # This is the unchanged engine's exact draw order and shape.
        a = torch.randint(len(raw), (cfg.raw_per_batch,), generator=rng)
        b = torch.randint(len(anchors), (cfg.anchor_per_batch,), generator=rng)
        ids = torch.cat([bank[a], other[b]]).reshape(-1)
        counts += torch.bincount(ids, minlength=50258)
    return counts


@contextmanager
def redirected():
    """Change configuration in this process only; restore the common engine."""
    saved = {key: getattr(engine, key) for key in ("DATA", "PILOT", "TRAIN", "PLAN", "freeze")}
    engine.DATA, engine.PILOT, engine.TRAIN, engine.PLAN = DATA, PILOT, TRAIN, PLAN
    original_freeze = saved["freeze"]
    def freeze(root, stage, inputs, config):
        p = original_freeze(root, stage, inputs, {**config, "token_exposure_policy": POLICY})
        source = Path(__file__)
        p["source_hashes"][str(source)] = file_hash(source)
        (root / "producer_source" / ("balanced_"+source.name)).write_bytes(source.read_bytes())
        p["hypothesis_id"] = "ncm-vocabulary-balanced-codec-v1"
        p["source_scope"] = "SHA-verified public natural text plus synthetic GPT2 native token-ID sequences"
        p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
        write_json(root / "producer.json", p)
        return p
    engine.freeze = freeze
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(engine, key, value)


def prepare():
    original, training = q.read(ORIGINAL_DATA / "result.json"), q.read(ORIGINAL_TRAIN / "result.json")
    if original["status"] != "passed" or training["status"] != "codec_transfer_gate_failed":
        raise ValueError("preserved natural-only failure and original data required")
    for root, result in ((ORIGINAL_DATA, original), (ORIGINAL_TRAIN, training)):
        verify(q.read(root / "producer.json"))
        for path, expected in result["output_hashes"].items():
            if file_hash(path) != expected:
                raise ValueError("original evidence changed")
    names = ("anchors.json", "raw_validation.json", "novel.json", "seen.json", "source_split.json", "exclusions.json")
    inputs = [ORIGINAL_DATA / n for n in (*names, "raw_train.json", "result.json", "producer.json")]
    inputs += [ORIGINAL_TRAIN / "result.json"]
    p = engine.freeze(DATA, "vocabulary_balanced_data_before_training", inputs, {**POLICY, "sampler_replay": True})
    raw = q.read(ORIGINAL_DATA / "raw_train.json")
    synthetic = synthetic_targets(2*len(raw))
    mixed = [item for i, row in enumerate(raw) for item in (row, synthetic[2*i], synthetic[2*i+1])]
    counts = actual_exposure(mixed, q.read(ORIGINAL_DATA / "anchors.json"))
    for name in names:
        (DATA / name).write_bytes((ORIGINAL_DATA / name).read_bytes())
    write_json(DATA / "raw_train.json", mixed)
    save(DATA / "actual_sampler_exposure.pt", counts)
    if not torch.equal(torch.load(DATA / "actual_sampler_exposure.pt", weights_only=True), counts):
        raise ValueError("exposure counts changed on reload")
    passed = bool(counts[:50256].gt(0).all())
    write_json(DATA / "exposure_summary.json", {"actual_positive_content_ids": int(counts[:50256].gt(0).sum()),
        "content_vocabulary": 50256, "minimum_actual_content_count": int(counts[:50256].min()),
        "maximum_actual_content_count": int(counts[:50256].max()), "draws_replayed": engine.Config().steps,
        "seed": engine.Config().sampler_seed, "natural_windows": len(raw), "synthetic_sequences": len(synthetic)})
    verify(p)
    result = {"status": "passed" if passed else "failed_actual_token_exposure_gate", "raw_training": len(mixed),
        "natural_windows": len(raw), "synthetic_sequences": len(synthetic), "anchors": original["anchors"],
        "validation": original["validation"], "novel": original["novel"], "all_content_ids_positive_in_actual_schedule": passed,
        "producer_fingerprint": p["fingerprint"], "target_aware": True, "task_accuracy": None,
        "trained": False, "final_test_opened": False,
        "output_hashes": {str(path): file_hash(path) for path in DATA.glob("*") if path.is_file() and path.name != "producer.json"}}
    write_json(DATA / "result.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "output_hashes"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "pilot", "train"])
    command = parser.parse_args().command
    with redirected():
        prepare() if command == "prepare" else engine.run(command == "pilot")
