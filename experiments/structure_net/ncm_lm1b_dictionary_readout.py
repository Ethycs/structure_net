"""Group-affine invariant embedding readout of fixed native NCM code latents."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_lm1b_codec as native
from experiments.structure_net import ncm_lm1b_framed_codec as framing
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT, q = native.ROOT, native.q
HYPOTHESIS = "ncm-lm1b-group-invariant-readout-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-group-readout-plan.md")
MODULE = "experiments.structure_net.ncm_lm1b_dictionary_readout"
WORKFLOW = ROOT / "20261024_lm1b_dictionary_readout_workflow_v1"
PILOT = ROOT / "20261024_lm1b_dictionary_readout_pilot_v1"
MEASURE = ROOT / "20261024_lm1b_dictionary_readout_measure_v1"
GROUPS = 4
VOCAB = 30522


def normalized_groups(values, groups=GROUPS):
    if groups <= 0 or values.ndim < 2 or values.shape[-1] % groups or not torch.isfinite(values).all():
        raise ValueError("finite complete group geometry required")
    x = values.float().reshape(*values.shape[:-1], groups, values.shape[-1] // groups)
    x = x - x.mean(-1, keepdim=True)
    return F.normalize(x, p=2, dim=-1, eps=1e-8).reshape(values.shape)


def nearest(latent, normalized_dictionary):
    """Read only predicted latents and pretrained embeddings, never true IDs/stats."""
    if latent.shape[-1] != normalized_dictionary.shape[-1] or normalized_dictionary.ndim != 2:
        raise ValueError("matching pretrained embedding dimensions required")
    return (normalized_groups(latent) @ normalized_dictionary.T).argmax(-1)


def producing(root, stage, inputs=(), config=None):
    files = [Path(__file__), PLAN, Path("tests/structure_net/test_ncm_lm1b_dictionary_readout.py")]
    paths = [framing.WORKFLOW / "result.json", framing.DATA / "cases.json", framing.RECON / "outputs.json",
        framing.RECON / "result.json", native.RELEASE / "files/lm1b/tokenizer.pt", *files, *inputs]
    p = native.producing(root, stage, paths, {**(config or {}), "readout_groups": GROUPS,
        "dictionary_vocabulary": VOCAB, "unknown_group_stats_used": False,
        "target_ids_received_by_prediction": False, "source_codes": "same saved native128 framed codec codes",
        "original_neural_decoder_outputs": "retained matched baseline", "new_weights": False,
        "one_output_frame_covers_all_labels": False, "output_policy": "single_nearest_group_centered_embedding",
        "continuous_encoder_control": "all raw IDs including padding must recover exactly",
        "no_decoder_training": True, "source_prediction": False})
    p["hypothesis_id"] = HYPOTHESIS
    p["config"]["hypothesis_id"] = HYPOTHESIS
    p["config"]["schema"] = "nal.ncm-lm1b-group-invariant-readout.v1"
    p["config"]["output_policy"] = "single_nearest_group_centered_embedding"
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"dictionary_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def inputs():
    done = q.read(framing.WORKFLOW / "result.json")
    if [(s["stage"], s["exit_code"]) for s in done["stages"]] != [(s, 0) for s in ("prepare", "native", "pilot", "measure")]:
        raise ValueError("complete original framed native control required")
    for root in (framing.DATA, framing.RECON):
        native.vendor.terminal(root)
    rows, outputs = q.read(framing.DATA / "cases.json"), q.read(framing.RECON / "outputs.json")
    if ([r["id"] for r in rows] != [r["id"] for r in outputs] or len(rows) != 690
            or file_hash(framing.RECON / "outputs.json") != q.read(framing.RECON / "result.json")["outputs_sha256"]):
        raise ValueError("all690 exact native-code cases required")
    return rows, {r["id"]: r for r in outputs}


@torch.no_grad()
def measure(pilot=False):
    native.require_gpu()
    rows, baseline = inputs()
    selected = q.read(framing.DATA / "selection.json")["pilot_ids"]
    if pilot:
        rows = [next(r for r in rows if r["id"] == identity) for identity in selected]
    else:
        native.vendor.terminal(PILOT)
    root = PILOT if pilot else MEASURE
    p = producing(root, "systems_group_readout_pilot" if pilot else "full690_group_invariant_code_readout",
        [] if pilot else [PILOT / n for n in ("producer.json", "result.json", "outputs.json", "worker_terminal_receipt.json")],
        {"pilot": pilot, "cases": len(rows), "physical_gpu": 2, "peak_limit_bytes": 4700000000})
    codec, t = native.load_codec(), native.tokenizer()
    enc, norm = codec.encoder, codec.pre_quant_norm
    if (enc.wpe is not None or len(enc.blocks) or len(enc.conv_blocks) or enc.down is not None
            or not isinstance(norm, torch.nn.GroupNorm) or norm.num_groups != GROUPS or norm.affine):
        raise ValueError("embedding-only encoder and affine-free global GroupNorm4 required")
    before = native.vendor.module_hash(codec)
    dictionary = normalized_groups(enc.wte.weight[:VOCAB].detach()).contiguous()
    retained = {} if pilot else {r["id"]: r for r in q.read(PILOT / "outputs.json")}
    torch.cuda.reset_peak_memory_stats()
    outputs = []
    for row in rows:
        source = baseline[row["id"]]
        # Production readout is computed before any clean-input control or scoring.
        codes = torch.tensor(source["codes"], device="cuda", dtype=torch.long)
        latent = codec.vq.idx_to_z_q(codes)
        prediction = nearest(latent, dictionary)
        if pilot and not torch.equal(prediction, nearest(latent, dictionary)):
            raise ValueError("group readout must repeat exactly")
        raw_frames = prediction.cpu().tolist()
        raw = [v for frame in raw_frames for v in frame]
        text, ends, bos = framing.render(raw_frames, t)
        # Target-aware positive control is separate from production prediction.
        correct_input = torch.tensor(row["packed_ids"], device="cuda", dtype=torch.long)
        clean_prediction = nearest(codec.encode_z(correct_input), dictionary)
        if not torch.equal(clean_prediction, correct_input):
            raise ValueError("continuous group-affine cancellation failed on native IDs")
        gold = row["target_ids"]
        expected_ends = [sum(v != 0 for v in frame) - 1 for frame in row["packed_ids"]]
        result = {"id": row["id"], "text": text, "raw_ids": raw, "source_code_sha256": q.digest(source["codes"]),
            "correct": sum(a == b for a, b in zip(raw, gold)), "total": len(gold),
            "exact_native_ids": raw[:len(gold)] == gold, "frame_EOS_positions": ends,
            "all_window_EOS_correct": ends == expected_ends, "all_frame_BOS_correct": all(bos),
            "clean_control_exact_all_positions": True, "clean_control_positions": correct_input.numel(),
            "neural_decoder_correct": source["correct"], "neural_decoder_exact": source["exact_native_ids"],
            "fits_one_native_frame": row["fits_one_native_frame"]}
        if row["id"] in retained and result != retained[row["id"]]:
            raise ValueError("fresh group readout changed retained pilot")
        outputs.append(result)
        if pilot or len(outputs) % 16 == 0 or len(outputs) == len(rows):
            write_json(root / "outputs.json", outputs)
        if pilot or len(outputs) % 64 == 0 or len(outputs) == len(rows):
            print(json.dumps({"read_out": len(outputs), "total": len(rows)}), flush=True)
    if native.vendor.module_hash(codec) != before or any(v.grad is not None for v in codec.parameters()):
        raise ValueError("fixed native weights changed")
    peak = torch.cuda.max_memory_allocated()
    if peak > 4700000000:
        raise ValueError("group readout exceeded peak limit")
    native.vendor.verify(p)
    result = {"status": "systems_group_readout_passed" if pilot else "all690_group_readout_measured_semantics_pending",
        "cases": len(outputs), "correct": sum(v["correct"] for v in outputs), "total": sum(v["total"] for v in outputs),
        "exact_native_ids": sum(v["exact_native_ids"] for v in outputs),
        "all_window_EOS_correct": sum(v["all_window_EOS_correct"] for v in outputs),
        "all_frame_BOS_correct": sum(v["all_frame_BOS_correct"] for v in outputs),
        "neural_decoder_correct": sum(v["neural_decoder_correct"] for v in outputs),
        "clean_encoder_exact_positions": sum(v["clean_control_positions"] for v in outputs),
        "codec_parameter_hash_unchanged": before, "new_weights": False, "model_training_executed": False,
        "no_target_statistics_in_prediction": True, "source_prediction": False, "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False, "peak_allocated_bytes": peak,
        "outputs_sha256": file_hash(root / "outputs.json"), "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def workflow():
    inputs()
    if any(p.exists() for p in (WORKFLOW, PILOT, MEASURE)):
        raise ValueError("fresh group readout workflow required")
    p = producing(WORKFLOW, "native_code_group_readout_control")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    try:
        for stage, root in (("pilot", PILOT), ("measure", MEASURE)):
            native.vendor.verify(p)
            command = [sys.executable, "-u", "-m", MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "2"})
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"stage": stage, "exit_code": code, "child_pid": child.pid, "observer_pid": os.getpid(),
                "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if root.exists():
                write_json(root / "worker_terminal_receipt.json", receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("group readout stage failed: " + stage)
        native.vendor.verify(p)
        journal.update(status="group_readout_control_completed_semantics_unmeasured", active_stage=None, child_pid=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("pilot", "measure", "workflow"))
    args = parser.parse_args()
    {"pilot": lambda: measure(True), "measure": measure, "workflow": workflow}[args.stage]()
