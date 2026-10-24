"""CPU oracle control: frozen neural decoding before versus after quantization."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_matched_clean_word as clean
from experiments.structure_net import ncm_broad_learned_codec as released
from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

ROOT = clean.ROOT
HYPOTHESIS = "ncm-continuous-codec-target-fidelity-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-continuous-codec-target-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-continuous-codec-target.md")
WORKFLOW = ROOT / "20261023_continuous_codec_target_workflow_v1"
PILOT = ROOT / "20261023_continuous_codec_target_pilot_v1"
MEASURE = ROOT / "20261023_continuous_codec_target_measure_v1"
RESULTS = ROOT / "20261023_continuous_codec_target_results_v1"
MODULE = "experiments.structure_net.ncm_continuous_codec_target"
ARMS = ("saved_quantized", "continuous_encoder")
EXPECTED_CODEC = "50fc5680ce0fb3c82764d865243a2e509d8f16dd8c051113b16080b5d763d680"


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("explicit CPU-only control; CUDA must not be initialized")
    torch.set_num_threads(2)


def inputs():
    cpu_only()
    _, _, _, selected = clean.require_inputs()
    return selected


def producing(destination, stage, extra=()):
    study = clean.recovery.study
    files = [Path(__file__), PLAN, Path(clean.__file__), clean.PLAN, Path(clean.diagnostic.__file__),
        Path(released.__file__), Path("experiments/structure_net/ncm_blockwise_codec.py"),
        Path("experiments/structure_net/ncm_frozen_body.py"),
        Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    files += sorted(study.data.VENDOR.rglob("*.py"))
    common = [study.expanded.DEST / name for name in ("cases.json", "manifest.json", "result.json")]
    common += [clean.diagnostic.DEST / name for name in ("result.json", "selection.json", "case_measurements.json")]
    common += [released.INITIAL / name for name in ("config.yaml", "tokenizer.pt")]
    p = study.data.freeze(destination, stage, list(extra) + common + files,
        {"hypothesis_id": HYPOTHESIS, "CPU_only": True, "threads": 2, "precision": "FP32", "constructor_seed": 42142,
         "autocast": False, "batch_cases": 1, "independent_blocks": 4, "word_frame": 192,
         "readout_word_vocab": 50257, "arms": ARMS, "expected_codec_hash": EXPECTED_CODEC,
         "oracle_target_information": True, "model_training_or_source_generation": False,
         "task_accuracy": None, "heldout_accuracy_measured": False,
         "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (destination / "producer_source" / f"continuous_target_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = study.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    return p


def load_codec():
    torch.manual_seed(42142)
    codec = released.load_codec(device="cpu").float().eval().requires_grad_(False)
    if module_hash(codec) != EXPECTED_CODEC:
        raise ValueError("unchanged original codec state required")
    return codec, BlockwiseCodec(codec, 4)


@torch.no_grad()
def latent_arms(facade, words, codes):
    if (words.ndim != 2 or words.shape[1] != facade.context_length
            or codes.shape != (len(words), sum(facade.vq.levels))
            or words.device.type != "cpu" or codes.device.type != "cpu"):
        raise ValueError("complete CPU word and saved-code frames required")
    original = facade.original
    if original.training or any(p.requires_grad for p in original.parameters()):
        raise ValueError("explicit frozen eval-mode original codec required")
    continuous = original.encode_z(words.reshape(len(words) * facade.blocks, original.context_length))
    continuous = continuous.reshape(len(words), facade.context_length, facade.embed_dim)
    quantized = facade.vq.idx_to_z_q(codes)
    if continuous.shape != quantized.shape or not torch.isfinite(continuous).all() or not torch.isfinite(quantized).all():
        raise ValueError("finite complete matched latent frames required")
    return dict(zip(ARMS, (quantized, continuous)))


@torch.no_grad()
def neural_logits(facade, latent, vocab_limit=50257):
    logits = facade.decoder(latent)
    if not 1 <= vocab_limit <= logits.shape[-1]:
        raise ValueError("explicit bounded neural word vocabulary required")
    logits = logits[..., :vocab_limit]
    if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise ValueError("finite FP32 neural decoder logits required")
    return logits


@torch.no_grad()
def measure_case(facade, row, tokenizer):
    target = torch.tensor([row["word_ids"]], dtype=torch.long)
    codes = torch.tensor([row["concept_ids"]], dtype=torch.long)
    active = [word for word in row["word_ids"] if word != 50257]
    if (not active or active[-1] != 50256
            or tokenizer.decode(active[:-1], skip_special_tokens=False) != row["target_text"]):
        raise ValueError("unchanged complete TRAIN target text/EOS required")
    values = {}
    for arm, latent in latent_arms(facade, target, codes).items():
        logits = neural_logits(facade, latent)
        loss = float(F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=50257))
        raw = logits.argmax(-1)
        text, end = clean.recovery.evaluation.reference.decode_words(raw[0].tolist(), tokenizer)
        values[arm] = {**clean.diagnostic.counts(raw, target), "CE": loss,
            "raw_word_ids": raw[0].tolist(), "text": text,
            "global_EOS_position": end if end < 192 else None,
            "exact_target_text": text == row["target_text"],
            "latent_sha256": clean.recovery.study.base.tensor_digest({"latent": latent})}
    return {"id": row["id"], "source_sha256": row["source_sha256"],
        "diagnostic_group": row["diagnostic_group"], "target_word_ids": row["word_ids"],
        "target_text": row["target_text"], "arms": values}


def aggregate(rows):
    result = {}
    for group in ("original_16", "additional_16", "all_32"):
        selected = rows if group == "all_32" else [row for row in rows if row["diagnostic_group"] == group]
        values = {}
        for arm in ARMS:
            total = sum(row["arms"][arm]["total"] for row in selected)
            correct = sum(row["arms"][arm]["correct"] for row in selected)
            values[arm] = {"correct": correct, "total": total,
                "token_agreement": correct / total if total else None,
                "exact_target_texts": sum(row["arms"][arm]["exact_target_text"] for row in selected)}
        result[group] = {"cases": len(selected), "arms": values}
    return result


def fidelity_gate(components):
    pairs = {}
    for group, cases in (("original_16", 16), ("additional_16", 16), ("all_32", 32)):
        value = components[group]
        correct, total = (value["arms"][ARMS[1]][key] for key in ("correct", "total"))
        if value["cases"] != cases or type(correct) is not int or type(total) is not int or not 0 <= correct <= total or total == 0:
            raise ValueError("complete fixed group counts required")
        pairs[group] = correct, total
    if any(pairs["all_32"][i] != pairs["original_16"][i] + pairs["additional_16"][i] for i in (0, 1)):
        raise ValueError("group counts must compose")
    return all(200 * correct >= 199 * total for correct, total in pairs.values())


def replay_case(item, source, tokenizer):
    if (item["id"] != source["id"] or item["target_word_ids"] != source["word_ids"]
            or item["target_text"] != source["target_text"] or item["source_sha256"] != source["source_sha256"]
            or item["diagnostic_group"] != source["diagnostic_group"]):
        raise ValueError("source, group or complete target binding differs")
    target = torch.tensor([source["word_ids"]])
    for arm in ARMS:
        value = item["arms"][arm]
        raw = value["raw_word_ids"]
        if len(raw) != 192 or any(type(word) is not int or not 0 <= word < 50257 for word in raw):
            raise ValueError("complete bounded raw word IDs required")
        counts = clean.diagnostic.counts(torch.tensor([raw]), target)
        text, end = clean.recovery.evaluation.reference.decode_words(raw, tokenizer)
        if (any(value[key] != counts[key] for key in ("correct", "total"))
                or value["text"] != text or value["global_EOS_position"] != (end if end < 192 else None)
                or value["exact_target_text"] != (text == source["target_text"])
                or len(value["latent_sha256"]) != 64 or not torch.isfinite(torch.tensor(value["CE"]))
                or value["CE"] < 0):
            raise ValueError("raw decoder IDs/counts/EOS/text/finite loss differ")


def run(pilot=False):
    selected = inputs()
    rows = clean.training_probes(selected) if pilot else selected
    destination = PILOT if pilot else MEASURE
    extra = []
    if not pilot:
        clean.recovery.study.data.terminal(PILOT)
        extra = [PILOT / name for name in ("producer.json", "result.json", "case_measurements.json", "worker_terminal_receipt.json")]
    p = producing(destination, "continuous_target_CPU_lifecycle_pilot" if pilot else "continuous_target_fixed32_TRAIN_control", extra)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(clean.recovery.study.data.GPT2, local_files_only=True)
    codec, facade = load_codec()
    values = [measure_case(facade, row, tokenizer) for row in rows]
    if module_hash(codec) != EXPECTED_CODEC or any(value.grad is not None for value in codec.parameters()):
        raise ValueError("original frozen codec changed")
    if pilot:
        del facade, codec
        gc.collect()
        codec, facade = load_codec()
        if values != [measure_case(facade, row, tokenizer) for row in rows]:
            raise ValueError("fresh codec reload changed latent digests, raw IDs or texts")
    else:
        retained = {row["id"]: row for row in clean.recovery.study.q.read(PILOT / "case_measurements.json")}
        if any(row != retained[row["id"]] for row in values if row["id"] in retained):
            raise ValueError("full measurement must retain exact lifecycle pilot outputs")
    if module_hash(codec) != EXPECTED_CODEC or torch.cuda.is_initialized():
        raise ValueError("codec identity or explicit CPU-only contract changed")
    components = aggregate(values)
    write_json(destination / "case_measurements.json", values)
    clean.recovery.study.data.verify(p)
    result = {"status": "systems_continuous_target_CPU_path_passed" if pilot else "continuous_target_fidelity_measured",
        "cases": len(rows), "components": components, "continuous_fit_gate": None if pilot else fidelity_gate(components),
        "original_codec_hash_unchanged": EXPECTED_CODEC, "fresh_reload_or_pilot_retention_exact": True,
        "CUDA_initialized": False, "task_accuracy": None, "heldout_accuracy_measured": False,
        "NCM_success_established": False, "final_test_opened": False, "speed_tested": False,
        "measurements_sha256": file_hash(destination / "case_measurements.json"), "producer_fingerprint": p["fingerprint"]}
    write_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)


def finalize():
    selected = inputs()
    if RESULTS.exists() or REPORT.exists():
        raise ValueError("fresh continuous-target archive and report required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(clean.recovery.study.data.GPT2, local_files_only=True)
    for location in (PILOT, MEASURE):
        clean.recovery.study.data.terminal(location)
    values = clean.recovery.study.q.read(MEASURE / "case_measurements.json")
    if [row["id"] for row in values] != [row["id"] for row in selected]:
        raise ValueError("exact complete32 TRAIN identities required")
    for item, source in zip(values, selected):
        replay_case(item, source, tokenizer)
    pilot = clean.recovery.study.q.read(PILOT / "result.json")
    retained = clean.recovery.study.q.read(PILOT / "case_measurements.json")
    if ([row["id"] for row in retained] != [row["id"] for row in clean.training_probes(selected)]
            or any(row != next(value for value in values if value["id"] == row["id"]) for row in retained)):
        raise ValueError("complete exact lifecycle pilot retention required")
    result = clean.recovery.study.q.read(MEASURE / "result.json")
    components = aggregate(values)
    gate = fidelity_gate(components)
    if (result["components"] != components or result["continuous_fit_gate"] != gate
            or result["measurements_sha256"] != file_hash(MEASURE / "case_measurements.json")
            or pilot["status"] != "systems_continuous_target_CPU_path_passed"
            or any(row["original_codec_hash_unchanged"] != EXPECTED_CODEC
                   or row["CUDA_initialized"] or not row["fresh_reload_or_pilot_retention_exact"] for row in (pilot, result))):
        raise ValueError("measured aggregation, original state or CPU lifecycle differs")
    receipts = [WORKFLOW / "stage_receipts" / (stage + ".json") for stage in ("pilot", "measure")]
    if any(clean.recovery.study.q.read(path)["exit_code"] != 0 for path in receipts):
        raise ValueError("both actual CPU child zero exits required")
    classification = "continuous_codec_target_fidelity_passed" if gate else "continuous_codec_target_fidelity_failed"
    artifacts = [Path(__file__), PLAN, WORKFLOW / "producer.json", *receipts]
    for location in (PILOT, MEASURE):
        artifacts += [path for path in location.rglob("*") if path.is_file() and "producer_source" not in path.parts]
    campaign = {"schema": "nal.ncm-continuous-codec-target.v1", "hypothesis_id": HYPOTHESIS,
        "classification": classification, "measurement": result, "components": components,
        "continuous_fit_gate": gate, "oracle_target_information": True, "task_accuracy": None,
        "heldout_accuracy_measured": False, "NCM_success_established": False, "confirmed": False,
        "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(path): file_hash(path) for path in dict.fromkeys(artifacts)}}
    RESULTS.mkdir()
    write_json(RESULTS / "campaign_results.json", campaign)
    lines = []
    for group, data in components.items():
        cells = [f"{data['arms'][arm]['correct']}/{data['arms'][arm]['total']} = {data['arms'][arm]['token_agreement']:.2%}" for arm in ARMS]
        lines.append("| " + group + " | " + " | ".join(cells) + " |")
    REPORT.write_text(f"""# Frozen neural decoder: quantized versus continuous target latents

Date:2026-10-23. Hypothesis {HYPOTHESIS}.
Classification **{classification}**. Campaign {RESULTS / 'campaign_results.json'};
SHA256 {file_hash(RESULTS / 'campaign_results.json')}.

| Same TRAIN targets | Saved quantized latents | Continuous encoder latents |
| --- | --- | --- |
""" + "\n".join(lines) + """

Both arms receive complete target-derived information. These are oracle TRAIN
token-reconstruction counts including EOS and excluding PAD, not summary task
accuracy or an information-theoretic bound. The continuous arm removes VQ but
keeps the original encoder normalization and SAME frozen neural decoder.
No true normalization statistics are used to invert tokens. A qualifying
199/200 fitting result in every group only supports considering continuous
latent supervision; no new source-only model, semantic quality or training
stage is admitted automatically.

The original codec stayed byte-identical. Computation was FP32 CPU, two threads,
with four independent48-position blocks and the same0..50256 vocabulary argmax
policy in both arms. This policy excludes PAD and unused output rows and differs
from earlier full-vocabulary codec diagnostics. Complete192-position outputs
and targets stayed intact. Fresh-load lifecycle parity and full pilot retention
passed, and all raw source/target bindings, decoding/counts and inclusive group
gates replayed. No CUDA, model training or judge request was used.

The full95% source-disjoint semantic summary goal remains unmet; both sealed
final cohorts remain unopened. No speed or compression claim is made.
""")


def register():
    campaign = clean.recovery.study.q.read(RESULTS / "campaign_results.json")
    register_records(RESULTS / "campaign_results.json", REPORT, RESULTS, {HYPOTHESIS: {
        "name": "Continuous target latent fidelity with the original NCM neural decoder",
        "description": "CPU-only32TRAIN oracle control compares saved VQ and raw encoder latents; no source-only summary quality or speed claim.",
        "confirmation_status": campaign["classification"], "task_accuracy_measured": False,
        "oracle_target_information": True, "continuous_fit_gate": campaign["continuous_fit_gate"],
        "NCM_success_established": False}})


def workflow():
    inputs()
    if any(path.exists() for path in (WORKFLOW, PILOT, MEASURE, RESULTS, REPORT,
            Path("data/hypothesis_registry") / (HYPOTHESIS + ".json"))):
        raise ValueError("fresh workflow and all destinations required")
    p = producing(WORKFLOW, "supervise_CPU_continuous_target_control")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "final_test_opened": False,
        "producer_fingerprint": p["fingerprint"]}
    try:
        for stage, destination in (("pilot", PILOT), ("measure", MEASURE), ("finalize", RESULTS), ("register", RESULTS)):
            if destination.exists() != (stage == "register"):
                raise ValueError("fresh stage destination required")
            clean.recovery.study.data.verify(p)
            command = [sys.executable, "-u", "-m", MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"status": "observed_worker_terminal", "stage": stage, "exit_code": code,
                "observer_pid": os.getpid(), "child_pid": child.pid, "observer": "subprocess.wait",
                "command": command, "destination": str(destination)}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if destination.exists():
                path = destination / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json")
                if path.exists():
                    raise ValueError("terminal receipt overwrite prohibited")
                write_json(path, receipt)
            journal["stages"].append(receipt)
            if code != 0:
                raise RuntimeError("actual CPU stage failure: " + stage)
            clean.recovery.study.data.verify(p)
        path = RESULTS / "campaign_results.json"
        receipt = clean.recovery.study.q.read(RESULTS / (HYPOTHESIS + "_registry_readback.json"))
        if receipt["status"] != "registered_and_read_back" or receipt["record"]["campaign_sha256"] != file_hash(path):
            raise ValueError("exact conservative control registry readback required")
        journal.update(status="continuous_target_CPU_control_completed", active_stage=None,
            classification=clean.recovery.study.q.read(path)["classification"], task_accuracy=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "pilot", "measure", "finalize", "register", "workflow"))
    args = parser.parse_args()
    cpu_only()
    if args.stage == "check":
        print(json.dumps({"fixed_TRAIN_cases": len(inputs()), "CPU_only": True, "arms": ARMS, "task_accuracy": None}))
    elif args.stage in ("pilot", "measure"):
        run(args.stage == "pilot")
    elif args.stage == "workflow":
        workflow()
    else:
        inputs()
        finalize() if args.stage == "finalize" else register()
