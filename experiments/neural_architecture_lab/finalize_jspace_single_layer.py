"""CPU-only, append-only closeout of the completed single-block recovery."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

import torch

from neural_architecture_lab.jspace_single_layer_record import JSpaceSingleLayerRecord, STUDY_ID, SCOPE, require
from .finalize_jspace_qwen_affine import ROOT, LOG, logger, read, sha, write

DEFAULT = ROOT / "data/experiments/jspace_single_block_distill/20261114_v2"
VERIFY = ROOT / "data/experiments/jspace_single_block_verify/20261114_v2"
ORACLE = ROOT / "data/experiments/jspace_update_projection_oracle/20261114_v1"
FAILURE = ROOT / "data/experiments/jspace_single_block_distill/20261114_v1"
REPORT = ROOT / "docs/08 - Analysis/jspace/2026-11-14_qwen-single-layer-recovery-result.md"
REGISTRY = ROOT / "data/hypothesis_registry/jspace_single_layer_recovery_20261114.json"
TYPED = ROOT / "src/neural_architecture_lab/jspace_single_layer_record.py"
TEST = ROOT / "tests/neural_architecture_lab/test_jspace_single_layer.py"


def tensor_inventory(state):
    tensors = {k:v for k,v in state.items() if isinstance(v, torch.Tensor)}
    storages = {}
    for value in tensors.values():
        storage = value.untyped_storage()
        storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return dict(tensor_bytes=sum(v.numel()*v.element_size() for v in tensors.values()),
        serialized_storage_bytes=sum(storages.values()), tensor_count=len(tensors),
        elements=sum(v.numel() for v in tensors.values()),
        tensors={k:dict(shape=list(v.shape), dtype=str(v.dtype), elements=v.numel(), bytes=v.numel()*v.element_size())
                 for k,v in tensors.items()})


def native_inventory(model_path):
    """Read safetensors shape/dtype headers, never the multi-GB weight payload."""
    model_path = Path(model_path); config = read(model_path / "config.json")
    require(config["tie_word_embeddings"] is True and config["num_hidden_layers"] == 36, "Pinned tied36-layer Qwen required")
    index_path = model_path / "model.safetensors.index.json"
    mapping = read(index_path)["weight_map"]
    sizes = {"BF16":2, "F16":2, "F32":4, "F64":8}
    tensors, files = {}, {}
    for filename in sorted(set(mapping.values())):
        path = model_path / filename
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            require(0 < size < 64*1024**2, "Invalid safetensors header length")
            raw = stream.read(size); header = json.loads(raw)
        files[filename] = dict(header_sha256=hashlib.sha256(raw).hexdigest(), file_bytes=path.stat().st_size)
        for name, value in header.items():
            if name == "__metadata__": continue
            require(mapping.get(name) == filename and value["dtype"] in sizes, "Unexpected native tensor or dtype")
            elements = math.prod(value["shape"]); width = sizes[value["dtype"]]
            require(value["data_offsets"][1]-value["data_offsets"][0] == elements*width, "Native tensor payload size mismatch")
            tensors[name] = dict(shape=value["shape"], dtype=value["dtype"], elements=elements, bytes=elements*width)
    require(set(tensors) == set(mapping), "Native shard index/header mismatch")
    if "lm_head.weight" in tensors:
        require(tensors["lm_head.weight"]["shape"] == tensors["model.embed_tokens.weight"]["shape"], "Tied readout shape differs")
        del tensors["lm_head.weight"]  # The configured model ties this to E.
    require(all(name in ("model.embed_tokens.weight", "model.norm.weight") or name.startswith("model.layers.") for name in tensors),
            "Unexpected persistent native state")
    blocks = {i:{k:v for k,v in tensors.items() if k.startswith(f"model.layers.{i}.")} for i in range(36)}
    require(all(blocks.values()), "Every native block must be represented")
    return dict(config=config, tensors=tensors, blocks=blocks, headers=files,
        parameter_count=sum(v["elements"] for v in tensors.values()),
        source_tensor_bytes=sum(v["bytes"] for v in tensors.values()),
        matched_fp32_bytes=sum(v["elements"]*4 for v in tensors.values()))


def storage_accounting(checkpoint, model_path):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["block"]
    require(state["wrapped._extra_state"] == dict(version=1, mode="softmax", metric_active=False), "Expected softmax without learned metric")
    prefix = "wrapped.original."
    parameters = {prefix+name for name in ("qkv_weight", "qkv_bias", "attention_output", "gate_up_weight", "mlp_output", "qkv_offset", "gate_up_offset")}
    buffers = {prefix+name for name in ("basis", "anchor", "anchor_energy", "qkv_shift", "attention_output_bias", "gate_up_shift")}
    require({k for k,v in state.items() if isinstance(v,torch.Tensor)} == parameters | buffers,
            "Selected state differs from declared full-head/no-polynomial compact block")
    require(all(v.dtype == torch.float32 for v in state.values() if isinstance(v,torch.Tensor)), "Selected FP32 state required")
    unpacked = tensor_inventory(state)
    removed = {prefix+"qkv_offset", prefix+"gate_up_offset"}
    # The frozen pack() folds offsets into equally-shaped FP32 shift buffers.
    # It changes no remaining shape/dtype in this no-polynomial, no-metric arm.
    packed = tensor_inventory({k:v for k,v in state.items() if k not in removed})
    packed["parameter_count"] = sum(state[k].numel() for k in parameters-removed)
    packed["fixed_geometry_and_buffer_bytes"] = sum(state[k].numel()*state[k].element_size() for k in buffers)
    packed["accounting"] = "Exact retained tensor shapes after two offsets fold into existing FP32 shift tables; no weight GEMM or dtype conversion needed to count"
    native = native_inventory(model_path); block = native["blocks"][0]
    block_elements = sum(v["elements"] for v in block.values())
    block_source_bytes = sum(v["bytes"] for v in block.values()); native_fp32 = native["matched_fp32_bytes"]
    require(all(sum(v["elements"] for v in row.values()) == block_elements for row in native["blocks"].values()),
            "Native blocks differ in parameter count")
    selected_fp32 = native_fp32-block_elements*4+packed["tensor_bytes"]
    mixed = native["source_tensor_bytes"]-block_source_bytes+packed["tensor_bytes"]
    return dict(changed_layers=[0], tied_embedding_readout_counted_once=True,
        selected_checkpoint_file_bytes=Path(checkpoint).stat().st_size, selected_unpacked=unpacked, packed_block=packed,
        native_block_parameters=block_elements, native_block_source_bytes=block_source_bytes,
        native_block_fp32_bytes=block_elements*4, native_parameters=native["parameter_count"],
        selected_whole_parameters=native["parameter_count"]-block_elements+packed["parameter_count"],
        native_source_tensor_bytes=native["source_tensor_bytes"], native_matched_fp32_bytes=native_fp32,
        block_matched_fp32_ratio=packed["tensor_bytes"]/(block_elements*4),
        whole_matched_fp32=dict(native_bytes=native_fp32, selected_bytes=selected_fp32, reduction_fraction=1-selected_fp32/native_fp32),
        whole_source_precision=dict(native_bytes=native["source_tensor_bytes"], selected_bytes=mixed,
            reduction_fraction=1-mixed/native["source_tensor_bytes"],
            scope="Original native checkpoint dtypes for unchanged weights plus measured FP32 compact block; an accounting comparison, not an evaluated mixed-precision deployment"),
        native_headers=native["headers"],
        scope="Persistent weight and saved geometry tensors only; optimizer, activations, KV caches, transient RoPE tables and serialization headers excluded from tensor counts")


def audit(root, verification, oracle, failed):
    from experiments.structure_net import jspace_single_block_verify as fresh
    from experiments.structure_net import jspace_single_block_reserved as reserved
    from experiments.structure_net import jspace_update_projection_oracle as projection
    root, verification, oracle, failed = map(lambda p:Path(p).resolve(), (root,verification,oracle,failed))
    vplan, checked = fresh.verify(verification)
    require(Path(vplan["campaign"]).resolve() == root, "Verifier belongs to another campaign")
    plan, policy = reserved.verify(root)
    runtime = reserved.verify_runtime_receipt(root, "train", plan, policy)
    vmanifest = read(verification / "verification/manifest.json"); vr = read(verification / "verification/result.json")
    require(vmanifest["status"] == "completed" and vmanifest["passed"] is True
            and vmanifest["fingerprint"] == vplan["fingerprint"]
            and vmanifest["result_sha256"] == sha(verification / "verification/result.json")
            and vr["checkpoint_sha256"] == vplan["checkpoint_sha256"]
            and vr["fingerprint"] == vplan["fingerprint"], "Fresh selected-checkpoint verification incomplete or changed")
    inputs = dict(checked["input_sha256"])
    for name in ("plan.json", "verification/manifest.json", "verification/result.json"):
        inputs[str(verification / name)] = sha(verification / name)
    train = checked["result"]; initial_path = root / "train/epoch_00_development.json"
    require(sha(initial_path) == train["history"][0]["development_sha256"], "Initial development control changed")
    inputs[str(initial_path)] = sha(initial_path)
    native_path = Path(plan["native_baseline"])
    require(sha(native_path) == plan["input_sha256"][str(native_path)], "Native baseline changed")
    inputs[str(native_path)] = sha(native_path)
    for history in train["history"]:
        path = root / "train" / f"epoch_{history['epoch']:02d}_development.json"
        require(sha(path) == history["development_sha256"], "Epoch development record changed")
        inputs[str(path)] = sha(path)
    oplan, _, _ = projection.verify(oracle)
    require(oplan["campaign_fingerprint"] == plan["fingerprint"]
            and oplan["input_sha256"][plan["fit_path"]] == plan["input_sha256"][plan["fit_path"]], "Oracle uses another fixed basis")
    controls = {}
    for arm in projection.ARMS:
        path = oracle / arm / "result.json"; manifest = read(oracle / arm / "manifest.json")
        require(manifest["status"] == "completed" and manifest["fingerprint"] == oplan["fingerprint"]
                and manifest["result_sha256"] == sha(path), "Incomplete projection control")
        controls[arm] = read(path)
        inputs[str(path)] = sha(path); inputs[str(path.parent / "manifest.json")] = sha(path.parent / "manifest.json")
    inputs[str(oracle / "plan.json")] = sha(oracle / "plan.json")
    failure = read(failed / "train/failure.json"); fplan = read(failed / "plan.json")
    require(fplan["fingerprint"] == plan["fingerprint"] == failure["fingerprint"], "v1 failure is not the same scientific plan")
    for path in (failed / "train/failure.json", failed / "plan.json"):
        inputs[str(path)] = sha(path)
    original = read(Path(plan["source"]) / "plan.json")
    value = dict(status="completed", confirmed=False, study_id=STUDY_ID, plan=plan, runtime_policy=policy,
        runtime_train=runtime, training=train, verification=vr, verification_plan=vplan,
        initial_records=read(initial_path)["records"], selected_records=checked["reference"]["records"],
        native_records=read(native_path)["records"], oracle=controls, oracle_plan=oplan, v1_failure=failure,
        storage=storage_accounting(checked["checkpoint"], original["model"]["path"]),
        native_weight_pins=original["model"],
        native_weight_hash_scope="Original producer hashed native shards and immutable file identities are checked by current parent verification; closeout reads actual safetensors headers without rehashing multi-GB payloads",
        artifact_sha256=inputs)
    return value, JSpaceSingleLayerRecord.from_evidence(value)


def sources():
    return [Path(__file__).resolve(), TYPED, TEST, Path(__file__).with_name("finalize_jspace_qwen_affine.py")]


def analyze(root, verification=VERIFY, oracle=ORACLE, failed=FAILURE):
    root = Path(root); require(not (root / "single_layer_analysis.json").exists(), "Analysis already exists")
    value, record = audit(root, verification, oracle, failed)
    archive = root / "single_layer_analysis_source"; require(not archive.exists(), "Analysis source archive already exists")
    for path in sources():
        target = archive / path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path,target)
    value.update(schema="jspace.single-layer-analysis.v1", created_at=datetime.now(timezone.utc).isoformat(),
        classification=record.classification, record=record.to_dict(), scope=SCOPE,
        analysis_source_sha256={str(path):sha(path) for path in sources()})
    write(root / "single_layer_analysis.json", value)
    return dict(status="analyzed", study_id=STUDY_ID, classification=record.classification,
                analysis_sha256=sha(root / "single_layer_analysis.json"))


def expected(root):
    root = Path(root); value = read(root / "single_layer_analysis.json")
    record = JSpaceSingleLayerRecord.from_evidence(value)
    require(value["record"] == record.to_dict(), "Typed record differs from evidence")
    for path, digest in {**value["artifact_sha256"], **value["analysis_source_sha256"]}.items():
        require(sha(path) == digest, "Changed closeout artifact/source: "+path)
    for path, digest in value["analysis_source_sha256"].items():
        require(sha(root / "single_layer_analysis_source" / Path(path).relative_to(ROOT)) == digest, "Changed archived closeout producer")
    return value, record


def report(root, report_path):
    root, report_path = Path(root), Path(report_path); value, record = expected(root)
    r, s = record, record.storage
    lines = ["# Qwen single-layer recovery", "", f"Status: completed exploratory development study. Hypothesis: **{STUDY_ID}**. Classification: `{r.classification}`. **Confirmed: false.**", "",
        f"Artifact: `{root.relative_to(ROOT)}/single_layer_analysis.json`; SHA-256 `{sha(root/'single_layer_analysis.json')}`.", "",
        f"The selected checkpoint scored **{r.selected['correct']}/64 ({100*r.primary_metric:.4f}%)**, compared with 46/64 before training and 50/64 for the native-equivalent control. The frozen acceptance threshold was 47/64. It {'passed' if r.quality_pass else 'failed'} that development gate. Only the first of 36 blocks was compressed.", "",
        "| Condition | Correct /64 | Accuracy | Gains / losses vs native |",
        "|---|---:|---:|---:|"]
    for label, row in (("Native-equivalent control",r.native),("Initial enriched block",r.initial),("Selected recovered block",r.selected),("Teacher-update projection oracle",r.projection_oracle)):
        pair = row.get("paired_to_native",dict(gains=0,losses=0))
        lines.append(f"| {label} | {row['correct']} | {100*row['accuracy']:.4f}% | {pair['gains']} / {pair['losses']} |")
    lines += ["", "| Family | Native | Initial | Selected |", "|---|---:|---:|---:|"]
    for family in sorted(r.native["by_family"]):
        lines.append(f"| {family} | {r.native['by_family'][family]['correct']}/16 | {r.initial['by_family'][family]['correct']}/16 | {r.selected['by_family'][family]['correct']}/16 |")
    lines += ["", f"All 12 epochs and 384 optimizer updates completed on 512 training cases. Accuracy-first selection chose epoch {r.selected_epoch}; relative feature MSE broke ties. The separate systems pilot did not initialize training. A fresh process authenticated and strictly reloaded the selected checkpoint, checked unpacked and packed feature metrics, then reproduced all selected predictions and choice logits within the frozen tolerance through the 35 original layers.", "",
        "The objective averages per-token squared output error divided by detached original teacher-update energy, then averages cases equally. The floor comes only from training. The full native residual is carried; the computed update has rank 1024, with native 128-dimensional attention heads, softmax, no learned metric, no polynomial replacement, and fixed geometry enriched using training data.", "",
        "| Feature diagnostic | Initial | Selected |", "|---|---:|---:|"]
    for key, number in r.normalized_features.items(): lines.append(f"| {key} | {r.initial_normalized_features[key]:.9g} | {number:.9g} |")
    initial_retained = r.initial_normalized_features["retained_relative_mse"]
    retained_reduction = 1-r.normalized_features["retained_relative_mse"]/initial_retained if initial_retained > 0 else None
    if retained_reduction is not None:
        lines += ["", f"The error within the retained subspace fell by {100*retained_reduction:.3f}%. Recovery gained {r.selected['paired_to_initial']['gains']} correct answers and lost {r.selected['paired_to_initial']['losses']} versus initialization. Selected predictions agree with the projection oracle on {r.projection_oracle['selected_prediction_agreement']}/64 cases."]
    lines += ["", f"Training median update energy: `{r.normalization['median_teacher_update_energy']:.9g}`; floor: `{r.normalization['floor']:.9g}`. The 48/64 teacher-update projection result is a closest-feature-fit surrogate, not a task-accuracy upper bound: nonteacher features could yield better answers. The matched cached-native boundary control scored 50/64.", "",
        "| Persistent tensor state | Bytes |", "|---|---:|",
        f"| Native block0, matched FP32 | {s['native_block_fp32_bytes']:,} |",
        f"| Actual selected unpacked block | {s['selected_unpacked']['tensor_bytes']:,} |",
        f"| Packed block, including U and saved geometry | {s['packed_block']['tensor_bytes']:,} |",
        f"| Entire native model, matched FP32 | {s['native_matched_fp32_bytes']:,} |",
        f"| Entire one-block-changed model, matched FP32 | {s['whole_matched_fp32']['selected_bytes']:,} |",
        f"| Original native checkpoint precision | {s['native_source_tensor_bytes']:,} |",
        f"| Source precision plus FP32 compact block | {s['whole_source_precision']['selected_bytes']:,} |", "",
        f"The packed block uses {100*s['block_matched_fp32_ratio']:.3f}% of native block0's FP32 tensor bytes. Across the full model, the matched-FP32 reduction is {100*s['whole_matched_fp32']['reduction_fraction']:.4f}%; the source-precision accounting reduction is {100*s['whole_source_precision']['reduction_fraction']:.4f}%. Tied embedding/readout E is counted once. These totals include the fixed basis and buffers; optimizer state, activations, KV caches and transient rotary tables are excluded. The source-precision hybrid is an accounting comparison, not an evaluated mixed-precision deployment. No speed benchmark was run for this checkpoint.", "",
        f"The earlier v1 attempt remains preserved: its allocator peak reached {r.resource_failure_lineage['peak_reserved_bytes']:,} bytes against a 4 GiB reservation, causing a MemoryError after completed epochs 0–4. That was a resource-limit failure, not evidence of a quality failure. v2 restarted the unchanged scientific 12-epoch protocol under a 7 GiB lease with a 256 MiB allocator margin; it did not resume v1 optimizer state.", "",
        "These 64 development questions have been inspected repeatedly. Passing this local gate does not establish held-out accuracy, equivalence, a fully compressed model, or faster generation. The remaining 35 blocks and original tied readout remain native. Further layer advances require their own explicit training and verification.", "",
        "CPU reproduction after source/report commit:", "", "```sh"]
    for stage in ("store","readback","closeout"):
        lines.append(f"CUDA_VISIBLE_DEVICES='' pixi run --locked --offline python -m experiments.neural_architecture_lab.finalize_jspace_single_layer {stage}")
    lines += ["```", ""]
    report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open("x") as stream:stream.write("\n".join(lines))
    return dict(report=str(report_path), sha256=sha(report_path))


def metadata(root, record):
    return dict(hypothesis_id=STUDY_ID, confirmed=False, confirmation_status=record.classification,
        accuracy_scope=SCOPE, primary_metric=record.primary_metric, accuracy=record.primary_metric,
        correct=record.selected["correct"], count=64, selected_epoch=record.selected_epoch, updates=384,
        quality_pass=record.quality_pass, independent_confirmations=0, compressed_blocks=1,
        native_correct=50, initial_correct=46, projection_oracle_correct=48,
        packed_block_tensor_bytes=record.storage["packed_block"]["tensor_bytes"],
        whole_matched_fp32_reduction=record.storage["whole_matched_fp32"]["reduction_fraction"],
        aggregate_sha256=sha(Path(root)/"single_layer_analysis.json"))


def readback(root, registry, report_path, key=None):
    _, record = expected(root)
    if key is None:
        saved = read(registry)
        require(saved["aggregate_sha256"] == sha(Path(root)/"single_layer_analysis.json")
                and saved["report_sha256"] == sha(report_path), "Registry/report provenance mismatch")
        key = saved["storage"]["result_hash"]
    db = logger(); wanted = metadata(root,record)
    actual = db.experiments_collection.get(ids=[key],include=["metadatas"])
    require(actual["ids"] == [key] and all(actual["metadatas"][0].get(k) == v for k,v in wanted.items()), "Experiment readback mismatch")
    hypothesis = db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(hypothesis["ids"] == [STUDY_ID], "Hypothesis not stored")
    row = hypothesis["metadatas"][0]
    require(row.get("tested") is True and row.get("confirmed") is False
            and row.get("confirmation_status") == record.classification and row.get("evidence_count") == 1
            and json.loads(row["direct_result_hashes_json"]) == [key], "Hypothesis readback mismatch")
    return dict(readback_verified=True,result_hash=key,**wanted)


def require_committed(paths):
    for path in paths:
        relative = str(Path(path).resolve().relative_to(ROOT))
        stored = subprocess.run(["git","show","HEAD:"+relative],cwd=ROOT,capture_output=True)
        require(stored.returncode == 0 and hashlib.sha256(stored.stdout).hexdigest() == sha(path),
                "Commit the exact closeout source/tests/report before registration: "+relative)


def store(root, registry, report_path):
    require(not Path(registry).exists() and Path(report_path).is_file(), "Registry exists or report missing")
    value, record = expected(root); require_committed([*sources(), Path(report_path)])
    from neural_architecture_lab.core import ExperimentResult
    hypothesis = dict(id=STUDY_ID,name="Single-layer Qwen recovery with enriched functional geometry",category="compression",
        tested=True,confirmed=False,evidence_count=1,direct_experiment_count=1,confirmation_status=record.classification,
        tested_scope=SCOPE,description="Original-teacher feature distillation of block0; cumulative native-suffix accuracy and independent reload.")
    experiment = ExperimentResult(experiment_id="jspace-single-layer-"+sha(Path(root)/"single_layer_analysis.json")[:16],
        hypothesis_id=STUDY_ID, metrics=dict(accuracy=record.primary_metric,native_accuracy=50/64,initial_accuracy=46/64,
            packed_block_tensor_bytes=record.storage["packed_block"]["tensor_bytes"]), primary_metric=record.primary_metric,
        model_architecture=[2048]*36, model_parameters=record.storage["selected_whole_parameters"],
        training_time=value["runtime_train"]["elapsed_seconds"], observations=[SCOPE,record.classification,
            "The residual stream remains2048 throughout; only block0's computed update has rank1024."])
    db = logger(); prior = db.hypotheses_collection.get(ids=[STUDY_ID],include=["metadatas"])
    require(not prior["ids"] or prior["metadatas"][0].get("tested") is not True, "Refusing to overwrite tested hypothesis")
    db.log_hypothesis(hypothesis); key = db.log_experiment_result(experiment)
    db.experiments_collection.update(ids=[key],metadatas=[metadata(root,record)])
    db.hypotheses_collection.update(ids=[STUDY_ID],metadatas=[dict(direct_result_hashes_json=json.dumps([key]),
        aggregate_sha256=sha(Path(root)/"single_layer_analysis.json"),report_path=str(report_path))])
    receipt = readback(root,registry,report_path,key)
    write(registry,dict(schema_version="nal.meta-hypothesis.v1",hypothesis=hypothesis,storage=receipt,
        aggregate_sha256=sha(Path(root)/"single_layer_analysis.json"),report_path=str(report_path),report_sha256=sha(report_path)))
    return receipt


def closeout(root, registry, report_path):
    root = Path(root); _, record = expected(root); receipt = read(root/"single_layer_registry_readback.json")
    require(receipt == readback(root,registry,report_path) and receipt["readback_verified"] is True, "Readback changed")
    event = dict(schema_version="jspace.research-log.v1",id=STUDY_ID+"-"+sha(root/"single_layer_analysis.json"),
        event="single_layer_recovery_completed",study_id=STUDY_ID,recorded_at=datetime.now(timezone.utc).isoformat(),
        confirmed=False,classification=record.classification,scope=SCOPE,record=record.to_dict(),
        registry_result_id=receipt["result_hash"],aggregate_sha256=sha(root/"single_layer_analysis.json"),
        registry_sha256=sha(registry),report_path=str(report_path),report_sha256=sha(report_path))
    with LOG.open("a+") as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX); stream.seek(0)
        require(not any(json.loads(line).get("id") == event["id"] for line in stream if line.strip()), "Study already logged")
        stream.write(json.dumps(event,sort_keys=True,allow_nan=False)+"\n");stream.flush();os.fsync(stream.fileno())
    write(root/"single_layer_finalization.json",dict(status="completed",study_id=STUDY_ID,confirmed=False,
        classification=record.classification,registry_readback=receipt,research_event=event))
    return dict(finalized=True,study_id=STUDY_ID)


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for CPU finalization")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("analyze","report","store","readback","closeout"))
    parser.add_argument("--root",type=Path,default=DEFAULT);parser.add_argument("--verification",type=Path,default=VERIFY)
    parser.add_argument("--oracle",type=Path,default=ORACLE);parser.add_argument("--failed-v1",type=Path,default=FAILURE)
    parser.add_argument("--report",type=Path,default=REPORT);parser.add_argument("--registry",type=Path,default=REGISTRY)
    args=parser.parse_args()
    if args.stage == "analyze": output=analyze(args.root,args.verification,args.oracle,args.failed_v1)
    elif args.stage == "report": output=report(args.root,args.report)
    else:
        output={"store":store,"readback":readback,"closeout":closeout}[args.stage](args.root,args.registry,args.report)
        if args.stage == "readback":write(args.root/"single_layer_registry_readback.json",output)
    print(json.dumps(output,allow_nan=False))
