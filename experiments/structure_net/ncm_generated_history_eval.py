"""Source-only two-branch NCM evaluation under the unchanged whole-summary rubric."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json

import torch

from experiments.structure_net import ncm_generated_history_study as training
from experiments.structure_net import ncm_word_final_eval as reference
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT = training.ROOT
PILOT = ROOT / "20261023_generated_history_inference_pilot_v1"
OUTPUTS = ROOT / "20261023_generated_history_outputs_v1"
PACKETS = ROOT / "20261023_generated_history_packets_v1"
GRADE = ROOT / "20261023_generated_history_grade_v1"


def producing(root, stage, inputs, config):
    files = [Path(__file__)]
    p = training.producing(root, stage, list(inputs)+files, {**config,
        "source_only": True, "oracle_target_codes": False, "two_independent_head_parameter_sets": True})
    p["source_hashes"][str(Path(__file__))] = file_hash(Path(__file__))
    (root / "producer_source" / "generated_history_eval.py").write_bytes(Path(__file__).read_bytes())
    p["fingerprint"] = reference.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def prior_cache():
    training.require_data()
    cache = reference.q.read(training.expanded.DEST / "judgment_cache.json")
    expected = reference.q.read(training.expanded.DEST / "result.json")["canonical_judgments"]
    if len(cache) != expected:
        raise ValueError("full latest first-judgment cache required")
    for key, entry in cache.items():
        if key != reference.first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("canonical judgment identity differs")
        reference.first.audit_row(entry["packet"], entry["role"], entry["row"])
    return cache


def configure():
    reference.PILOT, reference.OUTPUTS, reference.PACKETS, reference.GRADE = PILOT, OUTPUTS, PACKETS, GRADE
    reference.producing = producing
    reference.prior_cache = prior_cache


def checkpoint():
    reference.data.terminal(training.TRAIN)
    result = reference.q.read(training.TRAIN / "result.json")
    if (result["status"] != "generated_history_word_trained_pending_task_evaluation"
            or result["updates"] != training.Config().updates or not result["reload_exact"]
            or not result["source_only_composed_reload_exact"]
            or file_hash(training.TRAIN / "model.pt") != result["checkpoint_sha256"]):
        raise ValueError("observed fixed final word checkpoint required")
    state = torch.load(training.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    if (state["step"] != training.Config().updates or state["config"] != asdict(training.Config())
            or state["producer_fingerprint"] != result["producer_fingerprint"]
            or state["forecaster_checkpoint_sha256"] != training.require_forecaster()["checkpoint_sha256"]):
        raise ValueError("word model selection or frozen forecaster identity differs")
    forecaster, codec = training.load_forecaster()
    if training.parameter_hash(forecaster) != result["forecaster_parameter_sha256_unchanged"]:
        raise ValueError("evaluation concept forecaster differs from training")
    branch = training.core.independent_word_branch(forecaster).eval()
    set_source_frame(branch.generator, state["source_frame"], maximum=8192)
    branch.load_state_dict(state["model"], strict=True)
    if training.parameter_hash(branch, frozen_only=True) != result["frozen_word_branch_parameter_sha256"]:
        raise ValueError("unused word-branch modules changed")
    if reference.core.released.module_hash(codec) != result["codec_hash_unchanged"]:
        raise ValueError("original codec changed")
    return forecaster, branch, codec, state


@torch.no_grad()
def predict(forecaster, branch, row, manifest, tokenizer):
    batch = training.source_batch(row, manifest)
    for net in (forecaster, branch):
        if net.training:
            raise ValueError("both inference branches must be in eval mode")
        set_source_frame(net.generator, batch["source_ids"].shape[1], maximum=8192)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        history = training.core.own_history(forecaster, batch["features"], batch["mask"], batch["source_ids"])
        logits = training.core.word_logits(branch, batch["features"], batch["mask"],
            history["word_conditioning"], batch["source_ids"])
    raw = logits.argmax(-1)[0].cpu().tolist()
    widths = tuple(forecaster.generator.levels[:-1])
    coarse = [v[0].cpu().tolist() for v in history["coarse_codes"].split(widths, dim=1)]
    text, end = reference.decode_words(raw, tokenizer)
    return {**{key: row[key] for key in reference.IDENTITY}, "split": row["split"],
        "text": text, "raw_word_ids": raw, "coarse_codes": coarse,
        "global_EOS_position": end if end < len(raw) else None,
        "source_tokens": len(row["source_ids"]), "source_frame": row["source_frame"],
        "source_truncated": False, "fixed_generator_passes": 9, "AR_fallback": False,
        "oracle_target_codes": False, "output_repair": False}


def generate(pilot=False):
    reference.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    reference.data.terminal(reference.data.FEATURES)
    cases = reference.sources()
    all_cases = [r for rows in cases.values() for r in rows]
    selected = all_cases[:2] if pilot else all_cases
    manifest = reference.q.read(reference.data.FEATURES / "manifest.json")
    if file_hash(reference.data.FEATURES / "manifest.json") != reference.q.read(reference.data.FEATURES / "result.json")["manifest_sha256"]:
        raise ValueError("frozen development feature manifest changed")
    train_rows, _ = training.require_data()
    for cohort in cases.values():
        reference.data.prior.check_disjoint(train_rows, cohort)
    inputs = [training.TRAIN / name for name in ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
    inputs += [reference.SOURCES / name for name in ("producer.json", "result.json", "cases.json", "worker_terminal_receipt.json")]
    for row in selected:
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("frozen source feature changed")
        inputs.append(Path(entry["path"]))
    if not pilot:
        reference.data.terminal(PILOT)
        if reference.q.read(PILOT / "result.json")["status"] != "systems_source_only_inference_reload_passed":
            raise ValueError("actual composed inference/reload pilot required")
        inputs += [PILOT / name for name in ("producer.json", "result.json", "outputs.json", "worker_terminal_receipt.json")]
    root = PILOT if pilot else OUTPUTS
    p = producing(root, "generated_history_composed_inference_pilot" if pilot else "generated_history_composed_development_outputs",
        inputs, {"pilot": pilot, "cases": len(selected), "physical_gpu": 0,
            "precision": "FP32weights/BF16forward/FP32reconstruction",
            "retained_pilot_cases": 0 if pilot else 2, "output_selection": "single_deterministic_argmax"})
    forecaster, branch, codec, state = checkpoint()
    reference.data.prior.dependencies()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(reference.data.GPT2, local_files_only=True)
    before = training.parameter_hash(forecaster), training.parameter_hash(branch)
    frozen_codec = reference.core.released.module_hash(codec)
    torch.cuda.reset_peak_memory_stats()
    outputs = [] if pilot else reference.q.read(PILOT / "outputs.json")
    if not pilot:
        reference.require_outputs(outputs, selected[:2])
    for row in selected[len(outputs):]:
        outputs.append(predict(forecaster, branch, row, manifest, tokenizer))
        write_json(root / "outputs.json", outputs)
        if pilot or len(outputs) % 8 == 0 or len(outputs) == len(selected):
            print(json.dumps({"generated": len(outputs), "total": len(selected)}), flush=True)
    reference.require_outputs(outputs, selected)
    if pilot:
        forecaster_state = torch.load(training.bootstrap.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        with torch.no_grad():
            for net in (forecaster, branch):
                for value in net.state_dict().values():
                    value.zero_()
                net.generator._ncm_device_source_geometry = {}
        for net, saved in ((forecaster, forecaster_state), (branch, state)):
            set_source_frame(net.generator, saved["source_frame"], maximum=8192)
            net.load_state_dict(saved["model"], strict=True)
        if [predict(forecaster, branch, row, manifest, tokenizer) for row in selected] != outputs:
            raise ValueError("composed raw codes, word IDs or texts differ after reload")
    if (before != (training.parameter_hash(forecaster), training.parameter_hash(branch))
            or reference.core.released.module_hash(codec) != frozen_codec
            or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("inference parameter/codec identity or memory gate failed")
    reference.data.verify(p)
    result = {"status": "systems_source_only_inference_reload_passed" if pilot else "source_only_outputs_pending_task_grade",
        "cases": len(outputs), "counts": {s: sum(r["split"] == s for r in outputs) for s in reference.COHORTS},
        "pilot_outputs_retained_exactly": not pilot, "all_parameters_and_codec_unchanged": True,
        "output_sha256": file_hash(root / "outputs.json"), "checkpoint_sha256": file_hash(training.TRAIN / "model.pt"),
        "task_accuracy": None, "NCM_success_established": False, "final_test_opened": False,
        "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["pilot", "generate", "packets", "grade"])
    stage = parser.parse_args().stage
    configure()
    generate(stage == "pilot") if stage in ("pilot", "generate") else (
        reference.prepare_packets() if stage == "packets" else reference.grade())
