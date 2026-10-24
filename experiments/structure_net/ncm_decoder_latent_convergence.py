"""Continue saved oracle latent optimization with a fixed larger budget."""
import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_decoder_latent_refinement as prior
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT = prior.ROOT
HYPOTHESIS = "ncm-decoder-latent-convergence-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-decoder-latent-convergence-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-decoder-latent-convergence.md")
WORKFLOW = ROOT / "20261023_decoder_latent_convergence_workflow_v1"
PILOT = ROOT / "20261023_decoder_latent_convergence_pilot_v1"
MEASURE = ROOT / "20261023_decoder_latent_convergence_measure_v1"
RESULTS = ROOT / "20261023_decoder_latent_convergence_results_v1"
MODULE = "experiments.structure_net.ncm_decoder_latent_convergence"
UPSTREAM_SHA = "e3984e88a4d8844bab6fc680209855e115c9182ec8060dbdd2afa49141a6ca32"


@dataclass(frozen=True)
class Config(prior.Config):
    updates: int = 256
    prior_updates: int = 32

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.prior_updates < self.updates or self.pilot_updates >= self.updates - self.prior_updates:
            raise ValueError("fixed additional latent budget required")


def inputs():
    selected = prior.inputs()
    q = prior.clean.recovery.study.q
    path = prior.RESULTS / "campaign_results.json"
    campaign = q.read(path)
    receipt = q.read(prior.RESULTS / (prior.HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(path) != UPSTREAM_SHA or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != UPSTREAM_SHA
            or receipt["exact"]["campaign_sha256"] != UPSTREAM_SHA
            or campaign["classification"] != "oracle_refined_latent_fit_failed"):
        raise ValueError("exact registered32-step failed control required")
    for name in ("case_measurements.json", "latents.pt", "result.json", "worker_terminal_receipt.json", "producer.json"):
        item = prior.MEASURE / name
        if campaign["artifacts"].get(str(item)) != file_hash(item):
            raise ValueError("unchanged full32-step numerical evidence required")
    measured = q.read(prior.MEASURE / "case_measurements.json")
    if [r["id"] for r in selected] != [r["id"] for r in measured]:
        raise ValueError("same exact32 TRAIN identities required")
    return selected, measured


def producing(destination, stage, extra=()):
    common = [prior.RESULTS / "campaign_results.json", prior.RESULTS / (prior.HYPOTHESIS + "_registry_readback.json")]
    common += [prior.MEASURE / name for name in ("producer.json", "result.json", "case_measurements.json", "latents.pt", "worker_terminal_receipt.json")]
    files = [Path(__file__), PLAN]
    p = prior.producing(destination, stage, [*extra, *common, *files])
    p["hypothesis_id"] = HYPOTHESIS
    p["config"].update(**asdict(Config()), hypothesis_id=HYPOTHESIS,
        optimizer_reset=False,
        additional_updates=Config().updates - Config().prior_updates,
        selected_checkpoint="fixed_final256", upstream_campaign_sha256=UPSTREAM_SHA)
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (destination / "producer_source" / f"latent_convergence_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = prior.clean.recovery.study.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    return p


def continue_latent(facade, saved, target, additional, config=Config()):
    if (not 1 <= additional <= config.updates - config.prior_updates
            or int(saved["optimizer"]["state"][0]["step"]) != config.prior_updates):
        raise ValueError("exact32-step state and fixed additional budget required")
    latent = saved["refined"].to(target.device).clone().requires_grad_(True)
    initial = saved["initial"].to(target.device)
    opt = prior.optimizer(latent, config)
    # AdamW keeps step tensors on CPU. Clone to avoid mutating the input state.
    opt.load_state_dict(prior.to_cpu(saved["optimizer"]))
    trace = []
    target_hash = prior.clean.recovery.study.base.tensor_digest({"target": target})
    for step in range(config.prior_updates + 1, config.prior_updates + additional + 1):
        opt.zero_grad(set_to_none=True)
        logits = prior.neural_logits(facade, latent, config)
        ce = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=config.pad)
        anchor = F.mse_loss(latent, initial)
        loss = ce + config.anchor_weight * anchor
        if not torch.isfinite(loss):
            raise ValueError("finite latent objective required")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([latent], config.gradient_clip)
        if not torch.isfinite(norm) or not torch.count_nonzero(latent.grad):
            raise ValueError("positive finite latent gradient required")
        if any(p.grad is not None for p in facade.original.parameters()):
            raise ValueError("frozen decoder parameter gradient forbidden")
        opt.step()
        if not torch.isfinite(latent).all():
            raise ValueError("finite continued latent required")
        trace.append({"step": step, "word_CE": float(ce.detach()), "anchor_MSE": float(anchor.detach()),
            "gradient_norm": float(norm), "target_word_sha256": target_hash, "lr": config.lr})
        del logits, ce, anchor, loss
    return prior.to_cpu({"initial": initial, "refined": latent, "optimizer": opt.state_dict()}), trace


def run(pilot=False):
    selected, upstream = inputs()
    rows = prior.clean.training_probes(selected) if pilot else selected
    destination = PILOT if pilot else MEASURE
    extra = []
    if not pilot:
        prior.clean.recovery.study.data.terminal(PILOT)
        extra = [PILOT / n for n in ("producer.json", "result.json", "case_measurements.json", "worker_terminal_receipt.json")]
    p = producing(destination, "systems_lifecycle_only_not_quality_evidence" if pilot else "fixed256_total_TRAIN_latent_updates", extra)
    prior.gpu()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(prior.clean.recovery.study.data.GPT2, local_files_only=True)
    codec, facade = prior.load_codec()
    original = torch.load(prior.MEASURE / "latents.pt", map_location="cpu", weights_only=True)
    old = {row["id"]: row for row in upstream}
    additional = Config().pilot_updates if pilot else Config().updates - Config().prior_updates
    end_step = Config().prior_updates + additional
    values, latents = [], {}
    for row in rows:
        target = torch.tensor([row["word_ids"]], device="cuda")
        parent = old[row["id"]]
        saved = original[row["id"]]
        prior.replay_case(parent, row, saved, Config().prior_updates, tokenizer)
        for arm, key in zip(prior.ARMS, ("initial", "refined")):
            if prior.readout(facade, saved[key].cuda(), target, tokenizer) != parent["arms"][arm]:
                raise ValueError("same GPU must reproduce original and32-step readouts exactly")
        updated, trace = continue_latent(facade, saved, target, additional)
        value = deepcopy(parent)
        value["parent_refined_readout"] = deepcopy(parent["arms"][prior.ARMS[1]])
        value["trace"] += trace
        value["arms"][prior.ARMS[1]] = prior.readout(facade, updated["refined"].cuda(), target, tokenizer)
        value["final_anchor_MSE"] = float(F.mse_loss(updated["refined"], updated["initial"]))
        prior.replay_case(value, row, updated, end_step, tokenizer)
        values.append(value)
        latents[row["id"]] = updated
        write_json(destination / "progress.json", {"completed_cases": len(values), "total_cases": len(rows), "total_updates": end_step})
    torch.save(latents, destination / "latents.pt")
    stored = torch.load(destination / "latents.pt", map_location="cpu", weights_only=True)
    if not prior.same(stored, latents):
        raise ValueError("exact saved continuation latents and optimizer required")
    if pilot:
        del codec, facade
        gc.collect()
        torch.cuda.empty_cache()
        codec, facade = prior.load_codec()
        for row, value in zip(rows, values):
            saved = stored[row["id"]]
            target = torch.tensor([row["word_ids"]], device="cuda")
            opt = prior.optimizer(saved["refined"].cuda().requires_grad_(True))
            opt.load_state_dict(prior.to_cpu(saved["optimizer"]))
            if not prior.same(prior.to_cpu(opt.state_dict()), saved["optimizer"]):
                raise ValueError("fresh continued optimizer reload differs")
            if prior.readout(facade, saved["refined"].cuda(), target, tokenizer) != value["arms"][prior.ARMS[1]]:
                raise ValueError("fresh continued latent decoder reload differs")
        del opt
    peak = torch.cuda.max_memory_allocated()
    if (peak >= Config().peak_memory_limit or prior.module_hash(codec) != prior.continuous.EXPECTED_CODEC
            or any(p.grad is not None for p in codec.parameters())):
        raise ValueError("frozen codec or reserved CUDA memory limit failed")
    components = prior.aggregate(values)
    write_json(destination / "case_measurements.json", values)
    prior.clean.recovery.study.data.verify(p)
    result = {"status": "systems_latent_convergence_passed" if pilot else "oracle_latent_convergence_measured",
        "cases": len(rows), "total_updates": end_step, "additional_updates": additional, "components": components,
        "latent_fit_gate": None if pilot else prior.fidelity_gate(components),
        "codec_unchanged": True, "saved_reload_exact": True, "parent_state_retained": True,
        "peak_allocated_bytes": peak, "task_accuracy": None, "heldout_accuracy_measured": False,
        "final_test_opened": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"],
        "measurements_sha256": file_hash(destination / "case_measurements.json"), "latents_sha256": file_hash(destination / "latents.pt")}
    write_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)


def finalize():
    selected, upstream = inputs()
    if RESULTS.exists() or REPORT.exists():
        raise ValueError("fresh convergence aggregate and report required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(prior.clean.recovery.study.data.GPT2, local_files_only=True)
    old = {row["id"]: row for row in upstream}
    previous_latents = torch.load(prior.MEASURE / "latents.pt", map_location="cpu", weights_only=True)
    records = {}
    for root, rows, steps in ((PILOT, prior.clean.training_probes(selected), 34), (MEASURE, selected, Config().updates)):
        prior.clean.recovery.study.data.terminal(root)
        values = prior.clean.recovery.study.q.read(root / "case_measurements.json")
        saved = torch.load(root / "latents.pt", map_location="cpu", weights_only=True)
        result = prior.clean.recovery.study.q.read(root / "result.json")
        if [r["id"] for r in values] != [r["id"] for r in rows] or set(saved) != {r["id"] for r in rows}:
            raise ValueError("complete fixed convergence identities required")
        for value, row in zip(values, rows):
            prior.replay_case(value, row, saved[row["id"]], steps, tokenizer)
            parent = old[row["id"]]
            if (value["trace"][:32] != parent["trace"] or value["parent_refined_readout"] != parent["arms"][prior.ARMS[1]]
                    or value["arms"][prior.ARMS[0]] != parent["arms"][prior.ARMS[0]]
                    or not torch.equal(saved[row["id"]]["initial"], previous_latents[row["id"]]["initial"])):
                raise ValueError("unchanged original latent, first32 trace and readouts required")
        if (result["components"] != prior.aggregate(values) or result["total_updates"] != steps
                or result["additional_updates"] != steps - 32 or result["cases"] != len(rows)
                or result["measurements_sha256"] != file_hash(root / "case_measurements.json")
                or result["latents_sha256"] != file_hash(root / "latents.pt")
                or not all(result[key] for key in ("codec_unchanged", "saved_reload_exact", "parent_state_retained"))
                or result["peak_allocated_bytes"] >= Config().peak_memory_limit):
            raise ValueError("convergence aggregate or lifecycle controls differ")
        records[str(root)] = values, result
    values, measured = records[str(MEASURE)]
    pilot = records[str(PILOT)][1]
    if pilot["status"] != "systems_latent_convergence_passed" or pilot["latent_fit_gate"] is not None:
        raise ValueError("systems-only successful convergence pilot required")
    gate = prior.fidelity_gate(measured["components"])
    if measured["status"] != "oracle_latent_convergence_measured" or measured["latent_fit_gate"] != gate:
        raise ValueError("fixed256-update convergence gate differs")
    receipts = [WORKFLOW / "stage_receipts" / (stage + ".json") for stage in ("pilot", "measure")]
    if any(prior.clean.recovery.study.q.read(p)["exit_code"] != 0 for p in receipts):
        raise ValueError("two actual CUDA zero-exit receipts required")
    classification = "oracle_latent_convergence_fit_passed" if gate else "oracle_latent_convergence_fit_failed"
    artifacts = [Path(__file__), PLAN, WORKFLOW / "producer.json", prior.RESULTS / "campaign_results.json", *receipts]
    for root in (PILOT, MEASURE):
        artifacts += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts and p.name != "progress.json"]
    campaign = {"schema": "nal.ncm-decoder-latent-convergence.v1", "hypothesis_id": HYPOTHESIS,
        "classification": classification, "measurement": measured, "components": measured["components"],
        "prior32_components": prior.aggregate(upstream), "latent_fit_gate": gate,
        "oracle_target_information": True, "model_weights_trained": False, "task_accuracy": None,
        "heldout_accuracy_measured": False, "NCM_success_established": False, "confirmed": False,
        "final_test_opened": False, "speed_tested": False, "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(artifacts)}}
    RESULTS.mkdir()
    write_json(RESULTS / "campaign_results.json", campaign)
    lines = []
    for group, component in measured["components"].items():
        pairs = [component["arms"][prior.ARMS[0]], campaign["prior32_components"][group]["arms"][prior.ARMS[1]], component["arms"][prior.ARMS[1]]]
        lines.append("| " + group + " | " + " | ".join(f"{x['correct']}/{x['total']} = {x['token_agreement']:.2%}" for x in pairs) + " |")
    REPORT.write_text(f"""# Fixed longer optimization of NCM teacher latents

Date: 2026-10-23. Hypothesis `{HYPOTHESIS}`.
Classification **{classification}**. Campaign `{RESULTS / 'campaign_results.json'}`;
SHA256 `{file_hash(RESULTS / 'campaign_results.json')}`.

| Same TRAIN targets | Continuous encoder | 32 latent updates | 256 total latent updates |
| --- | --- | --- | --- |
""" + "\n".join(lines) + """

This exploratory follow-up was specified after the32-step fitting miss.
It continues each exact saved latent and AdamW state for224 further updates,
without resetting moments or changing the learning rate, objective, targets or
frozen original decoder. Only the fixed final256-step tensors are selected.
The original32-step failure remains unchanged. The two-case systems pilot adds
two updates and does not supply a quality estimate or initialize the full run.

The audit replayed the complete inherited32-step traces and every added target
hash, rate, finite loss and gradient, full latent and optimizer state, raw
192-word decoding/counts and all subgroup gates. Fresh decoder/optimizer reloads
were exact and the same-GPU baseline and32-step readouts stayed identical.

These are target-aware TRAIN reconstruction measurements. They cannot establish
source-only summary accuracy or a speed gain. Passing the99.5% fitting gate in
each group supports a new dense-target distillation experiment; it does not
admit any student training automatically. The95% source-disjoint complete-summary
goal remains unmet and both final cohorts remain unopened.
""")


def register():
    campaign = prior.clean.recovery.study.q.read(RESULTS / "campaign_results.json")
    prior.register_records(RESULTS / "campaign_results.json", REPORT, RESULTS, {HYPOTHESIS: {
        "name": "Fixed longer optimization of frozen NCM decoder input latents",
        "description": "32TRAIN targets, exact32-step latent/AdamW continuation to256; no student quality claim.",
        "confirmation_status": campaign["classification"], "task_accuracy_measured": False,
        "oracle_target_information": True, "latent_fit_gate": campaign["latent_fit_gate"], "NCM_success_established": False}})


def workflow():
    inputs()
    if any(p.exists() for p in (WORKFLOW, PILOT, MEASURE, RESULTS, REPORT,
            Path("data/hypothesis_registry") / (HYPOTHESIS + ".json"))):
        raise ValueError("fresh convergence destinations required")
    p = producing(WORKFLOW, "supervise_fixed_latent_continuation")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"], "final_test_opened": False}
    try:
        for stage, dest in (("pilot", PILOT), ("measure", MEASURE), ("finalize", RESULTS), ("register", RESULTS)):
            if dest.exists() != (stage == "register"):
                raise ValueError("fresh convergence stage destination required")
            prior.clean.recovery.study.data.verify(p)
            command = [sys.executable, "-u", "-m", MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": Config().physical_gpu if stage in ("pilot", "measure") else ""})
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"status": "observed_worker_terminal", "stage": stage, "exit_code": code,
                "observer_pid": os.getpid(), "child_pid": child.pid, "observer": "subprocess.wait", "command": command, "destination": str(dest)}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if dest.exists():
                terminal = dest / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json")
                if terminal.exists():
                    raise ValueError("terminal receipt overwrite prohibited")
                write_json(terminal, receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("actual convergence stage failure: " + stage)
            prior.clean.recovery.study.data.verify(p)
        archive = RESULTS / "campaign_results.json"
        receipt = prior.clean.recovery.study.q.read(RESULTS / (HYPOTHESIS + "_registry_readback.json"))
        if receipt["status"] != "registered_and_read_back" or receipt["record"]["campaign_sha256"] != file_hash(archive):
            raise ValueError("exact registered convergence result required")
        journal.update(status="latent_convergence_completed", active_stage=None,
            classification=prior.clean.recovery.study.q.read(archive)["classification"], task_accuracy=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "pilot", "measure", "finalize", "register", "workflow"))
    args = parser.parse_args()
    if args.stage == "check":
        print(json.dumps({"fixed_TRAIN_cases": len(inputs()[0]), "config": asdict(Config()), "task_accuracy": None}))
    elif args.stage in ("pilot", "measure"):
        run(args.stage == "pilot")
    elif args.stage == "workflow":
        workflow()
    else:
        finalize() if args.stage == "finalize" else register()
