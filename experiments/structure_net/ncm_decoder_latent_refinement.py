"""Oracle TRAIN control: refine latents through an unchanged NCM decoder."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_continuous_codec_target as continuous
from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

clean = continuous.clean
ROOT = clean.ROOT
HYPOTHESIS = "ncm-decoder-latent-refinement-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-decoder-latent-refinement-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-decoder-latent-refinement.md")
WORKFLOW = ROOT / "20261023_decoder_latent_refinement_workflow_v1"
PILOT = ROOT / "20261023_decoder_latent_refinement_pilot_v1"
MEASURE = ROOT / "20261023_decoder_latent_refinement_measure_v1"
RESULTS = ROOT / "20261023_decoder_latent_refinement_results_v1"
MODULE = "experiments.structure_net.ncm_decoder_latent_refinement"
ARMS = ("continuous_encoder", "refined_encoder")
UPSTREAM_SHA = "14c1b4227eb63c10218dffbb957403454a4f1b1873e74667e1f959c6ffa6383e"


@dataclass(frozen=True)
class Config:
    updates: int = 32
    pilot_updates: int = 2
    lr: float = .02
    beta1: float = .9
    beta2: float = .99
    eps: float = 1e-8
    anchor_weight: float = .001
    gradient_clip: float = 1.
    constructor_seed: int = 43142
    word_vocab: int = 50257
    pad: int = 50257
    physical_gpu: str = "2"
    peak_memory_limit: int = 2_000_000_000

    def __post_init__(self):
        if (not 1 <= self.pilot_updates <= self.updates or self.lr <= 0
                or not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1 or self.eps <= 0
                or self.anchor_weight <= 0 or self.gradient_clip <= 0
                or self.word_vocab < 2 or self.pad != self.word_vocab
                or self.peak_memory_limit <= 0):
            raise ValueError("invalid fixed latent refinement configuration")


def bound_targets(training, selection, measured):
    by_id = {row["id"]: row for row in training}
    if (len(training) != 690 or len(by_id) != 690 or len(selection) != 32
            or len({row["id"] for row in selection}) != 32
            or [row["id"] for row in selection] != [row["id"] for row in measured]
            or any(sum(row["diagnostic_group"] == group for row in selection) != 16
                   for group in ("original_16", "additional_16"))):
        raise ValueError("same complete TRAIN population and ordered32 identities required")
    result = []
    for identity, previous in zip(selection, measured):
        row = {**by_id[identity["id"]], "diagnostic_group": identity["diagnostic_group"]}
        if (any(row[key] != value for key, value in identity.items())
                or any(row[key] != previous[key] for key in ("id", "source_sha256", "diagnostic_group", "target_text"))
                or row["word_ids"] != previous["target_word_ids"]):
            raise ValueError("bound complete TRAIN target differs")
        result.append(row)
    return result


def inputs():
    clean.dependencies()
    clean.recovery.configure()
    study = clean.recovery.study
    path = continuous.RESULTS / "campaign_results.json"
    campaign = study.q.read(path)
    receipt = study.q.read(continuous.RESULTS / (continuous.HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(path) != UPSTREAM_SHA or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != UPSTREAM_SHA
            or receipt["exact"]["campaign_sha256"] != UPSTREAM_SHA):
        raise ValueError("unchanged registered continuous-target control required")
    producer_path = continuous.WORKFLOW / "producer.json"
    measured_path = continuous.MEASURE / "case_measurements.json"
    for item in (producer_path, measured_path):
        if campaign["artifacts"].get(str(item)) != file_hash(item):
            raise ValueError("registered continuous control binding changed")
    producer = study.q.read(producer_path)
    cases = study.expanded.DEST / "cases.json"
    selected = clean.diagnostic.DEST / "selection.json"
    for item in (cases, selected):
        if producer["input_hashes"].get(str(item)) != file_hash(item):
            raise ValueError("registered TRAIN cases or selection changed")
    return bound_targets(study.q.read(cases)["train"], study.q.read(selected), study.q.read(measured_path))


def producing(destination, stage, extra=()):
    study = clean.recovery.study
    old = study.q.read(continuous.WORKFLOW / "producer.json")
    files = [Path(__file__), PLAN, Path(clean.diagnostic.__file__)]
    files += [Path(name) for name in old["source_hashes"]]
    files = list(dict.fromkeys(files))
    common = [Path(name) for name in old["input_hashes"]]
    common += [continuous.RESULTS / "campaign_results.json",
        continuous.RESULTS / (continuous.HYPOTHESIS + "_registry_readback.json")]
    p = study.data.freeze(destination, stage, [*extra, *common, *files],
        {**asdict(Config()), "hypothesis_id": HYPOTHESIS, "precision": "FP32_no_autocast",
         "matmul_TF32": False, "cuDNN_TF32": False, "SDPA": "math_only",
         "cases": 32, "independent_blocks": 4, "word_frame": 192,
         "codec_hash": continuous.EXPECTED_CODEC, "arms": ARMS,
         "oracle_target_information": True, "optimized_variable": "per_case_latent_only",
         "codec_parameters_frozen": True, "task_accuracy": None,
         "heldout_accuracy_measured": False, "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (destination / "producer_source" / f"latent_refinement_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = study.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    return p


def gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != Config().physical_gpu or not torch.cuda.is_available():
        raise ValueError("physicalGPU2 must be selected before startup; no CPU fallback")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(Config().constructor_seed)
    torch.cuda.manual_seed_all(Config().constructor_seed)
    torch.cuda.reset_peak_memory_stats()


def load_codec():
    codec = continuous.released.load_codec(device="cpu").float().eval().requires_grad_(False)
    if module_hash(codec) != continuous.EXPECTED_CODEC:
        raise ValueError("unchanged original codec required")
    codec.cuda()
    return codec, BlockwiseCodec(codec, 4)


def neural_logits(facade, latent, config=Config()):
    original = facade.original
    if original.training or any(p.requires_grad for p in original.parameters()):
        raise ValueError("explicit frozen eval codec required")
    if latent.dtype != torch.float32 or latent.shape[1:] != (facade.context_length, facade.embed_dim):
        raise ValueError("complete FP32 latent frame required")
    logits = facade.decoder(latent)[..., :config.word_vocab]
    if logits.shape[-1] != config.word_vocab or not torch.isfinite(logits).all():
        raise ValueError("finite bounded word logits required")
    return logits


@torch.no_grad()
def encoder_latent(facade, target):
    if target.ndim != 2 or target.shape[1] != facade.context_length:
        raise ValueError("complete target word frame required")
    codec = facade.original
    value = codec.encode_z(target.reshape(len(target) * facade.blocks, codec.context_length))
    return value.reshape(len(target), facade.context_length, facade.embed_dim)


def optimizer(latent, config=Config()):
    return torch.optim.AdamW([latent], lr=config.lr, betas=(config.beta1, config.beta2),
        eps=config.eps, weight_decay=0.)


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value


def same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def refine(facade, initial, target, updates, config=Config()):
    if (not 1 <= updates <= config.updates or initial.requires_grad
            or target.shape != initial.shape[:2] or not (target != config.pad).any()):
        raise ValueError("detached complete initial latents, active targets and fixed budget required")
    latent = initial.detach().clone().requires_grad_(True)
    opt = optimizer(latent, config)
    trace = []
    target_hash = clean.recovery.study.base.tensor_digest({"target": target})
    for step in range(1, updates + 1):
        opt.zero_grad(set_to_none=True)
        logits = neural_logits(facade, latent, config)
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
            raise ValueError("frozen decoder received parameter gradients")
        opt.step()
        if not torch.isfinite(latent).all():
            raise ValueError("finite refined latent required")
        trace.append({"step": step, "word_CE": float(ce.detach()), "anchor_MSE": float(anchor.detach()),
            "gradient_norm": float(norm), "target_word_sha256": target_hash, "lr": config.lr})
        del logits, ce, anchor, loss
    return to_cpu({"initial": initial, "refined": latent, "optimizer": opt.state_dict()}), trace


@torch.no_grad()
def readout(facade, latent, target, tokenizer):
    logits = neural_logits(facade, latent)
    raw = logits.argmax(-1)
    ids = raw[0].tolist()
    text, end = clean.recovery.evaluation.reference.decode_words(ids, tokenizer)
    active = target[0][target[0] != Config().pad].tolist()
    target_text = tokenizer.decode(active[:-1], skip_special_tokens=False)
    return {**clean.diagnostic.counts(raw, target), "raw_word_ids": ids, "text": text,
        "global_EOS_position": end if end < 192 else None, "exact_target_text": text == target_text,
        "CE": float(F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=Config().pad)),
        "logits_sha256": clean.recovery.study.base.tensor_digest({"logits": logits}),
        "latent_sha256": clean.recovery.study.base.tensor_digest({"latent": latent})}


def aggregate(rows):
    result = {}
    for group in ("original_16", "additional_16", "all_32"):
        selected = rows if group == "all_32" else [row for row in rows if row["diagnostic_group"] == group]
        arms = {}
        for arm in ARMS:
            total = sum(row["arms"][arm]["total"] for row in selected)
            correct = sum(row["arms"][arm]["correct"] for row in selected)
            arms[arm] = {"correct": correct, "total": total, "token_agreement": correct / total if total else None,
                "exact_target_texts": sum(row["arms"][arm]["exact_target_text"] for row in selected)}
        result[group] = {"cases": len(selected), "arms": arms}
    return result


def fidelity_gate(components):
    converted = {group: {**value, "arms": {continuous.ARMS[1]: value["arms"][ARMS[1]]}}
        for group, value in components.items()}
    return continuous.fidelity_gate(converted)


def validate_saved(saved, trace, target, updates, config=Config()):
    for key in ("initial", "refined"):
        if saved[key].shape != (1, 192, 384) or saved[key].dtype != torch.float32 or not torch.isfinite(saved[key]).all():
            raise ValueError("complete finite saved latent required")
    opt = saved["optimizer"]
    expected = optimizer(saved["refined"].clone().requires_grad_(True), config).param_groups[0]
    if len(opt["param_groups"]) != 1 or opt["param_groups"][0] != {**expected, "params": [0]} or set(opt["state"]) != {0}:
        raise ValueError("unchanged one-latent optimizer configuration required")
    state = opt["state"][0]
    if int(state["step"]) != updates or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("complete fixed latent optimizer step required")
    for key in ("exp_avg", "exp_avg_sq"):
        if state[key].shape != saved["refined"].shape or not torch.isfinite(state[key]).all():
            raise ValueError("complete finite latent optimizer moments required")
    if (state["exp_avg_sq"] < 0).any():
        raise ValueError("nonnegative latent optimizer second moments required")
    if len(trace) != updates:
        raise ValueError("complete latent update trace required")
    target_hash = clean.recovery.study.base.tensor_digest({"target": target})
    for step, row in enumerate(trace, 1):
        if (row["step"] != step or row["lr"] != config.lr or row["target_word_sha256"] != target_hash
                or any(not math.isfinite(row[k]) or row[k] < 0 for k in ("word_CE", "anchor_MSE", "gradient_norm"))
                or row["gradient_norm"] == 0):
            raise ValueError("fixed actual target, rate and finite positive gradient trace required")


def run(pilot=False):
    selected = inputs()
    rows = clean.training_probes(selected) if pilot else selected
    destination = PILOT if pilot else MEASURE
    extra = []
    if not pilot:
        clean.recovery.study.data.terminal(PILOT)
        extra = [PILOT / name for name in ("producer.json", "result.json", "case_measurements.json", "worker_terminal_receipt.json")]
    p = producing(destination, "systems_lifecycle_only_not_quality_evidence" if pilot else "fixed32_TRAIN_latent_refinement", extra)
    gpu()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(clean.recovery.study.data.GPT2, local_files_only=True)
    codec, facade = load_codec()
    updates = Config().pilot_updates if pilot else Config().updates
    values, latents = [], {}
    for row in rows:
        target = torch.tensor([row["word_ids"]], device="cuda")
        active = target[0][target[0] != Config().pad].tolist()
        if active[-1] != 50256 or tokenizer.decode(active[:-1], skip_special_tokens=False) != row["target_text"]:
            raise ValueError("exact complete target/EOS binding required")
        initial = encoder_latent(facade, target)
        baseline = readout(facade, initial, target, tokenizer)
        saved, trace = refine(facade, initial, target, updates)
        refined = readout(facade, saved["refined"].cuda(), target, tokenizer)
        validate_saved(saved, trace, target.cpu(), updates)
        latents[row["id"]] = saved
        anchor = float(F.mse_loss(saved["refined"], saved["initial"]))
        value = {"id": row["id"], "source_sha256": row["source_sha256"], "diagnostic_group": row["diagnostic_group"],
            "target_word_ids": row["word_ids"], "target_text": row["target_text"],
            "arms": dict(zip(ARMS, (baseline, refined))), "trace": trace, "final_anchor_MSE": anchor}
        values.append(value)
        write_json(destination / "progress.json", {"completed_cases": len(values), "total_cases": len(rows), "updates_per_case": updates})
    if module_hash(codec) != continuous.EXPECTED_CODEC or any(p.grad is not None for p in codec.parameters()):
        raise ValueError("original frozen codec changed")
    torch.save(latents, destination / "latents.pt")
    stored = torch.load(destination / "latents.pt", map_location="cpu", weights_only=True)
    if not same(stored, latents):
        raise ValueError("exact saved latent and optimizer reload required")
    if pilot:
        del facade, codec
        gc.collect()
        torch.cuda.empty_cache()
        codec, facade = load_codec()
        for row, value in zip(rows, values):
            target = torch.tensor([row["word_ids"]], device="cuda")
            saved = stored[row["id"]]
            latent = saved["refined"].cuda().requires_grad_(True)
            opt = optimizer(latent)
            opt.load_state_dict(saved["optimizer"])
            if not same(to_cpu(opt.state_dict()), saved["optimizer"]):
                raise ValueError("fresh actual optimizer reload differs")
            for arm, key in zip(ARMS, ("initial", "refined")):
                if readout(facade, saved[key].cuda(), target, tokenizer) != value["arms"][arm]:
                    raise ValueError("fresh decoder reload changed exact pilot output")
        del latent, opt
    else:
        previous = {row["id"]: row for row in clean.recovery.study.q.read(PILOT / "case_measurements.json")}
        if any(row["arms"][ARMS[0]] != previous[row["id"]]["arms"][ARMS[0]] for row in values if row["id"] in previous):
            raise ValueError("same GPU baseline pilot retention differs")
    peak = torch.cuda.max_memory_allocated()
    if peak >= Config().peak_memory_limit or module_hash(codec) != continuous.EXPECTED_CODEC:
        raise ValueError("frozen codec identity or reserved CUDA memory bound failed")
    components = aggregate(values)
    write_json(destination / "case_measurements.json", values)
    clean.recovery.study.data.verify(p)
    result = {"status": "systems_latent_refinement_passed" if pilot else "oracle_latent_refinement_measured",
        "cases": len(rows), "updates_per_case": updates, "components": components,
        "latent_fit_gate": None if pilot else fidelity_gate(components), "codec_hash": continuous.EXPECTED_CODEC,
        "codec_unchanged": True, "codec_parameter_gradients": False, "saved_reload_exact": True,
        "peak_allocated_bytes": peak, "GPU": torch.cuda.get_device_name(), "task_accuracy": None,
        "heldout_accuracy_measured": False, "final_test_opened": False, "speed_tested": False,
        "measurements_sha256": file_hash(destination / "case_measurements.json"),
        "latents_sha256": file_hash(destination / "latents.pt"), "producer_fingerprint": p["fingerprint"]}
    write_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)


def replay_case(item, row, saved, updates, tokenizer):
    converted = {**item, "arms": dict(zip(continuous.ARMS, (item["arms"][arm] for arm in ARMS)))}
    continuous.replay_case(converted, row, tokenizer)
    validate_saved(saved, item["trace"], torch.tensor([row["word_ids"]]), updates)
    for arm, key in zip(ARMS, ("initial", "refined")):
        if item["arms"][arm]["latent_sha256"] != clean.recovery.study.base.tensor_digest({"latent": saved[key]}):
            raise ValueError("raw latent digest differs")
        if len(item["arms"][arm]["logits_sha256"]) != 64:
            raise ValueError("raw logits digest missing")
    if item["final_anchor_MSE"] != float(F.mse_loss(saved["refined"], saved["initial"])):
        raise ValueError("saved latent displacement differs")


def finalize():
    selected = inputs()
    if RESULTS.exists() or REPORT.exists():
        raise ValueError("fresh latent control aggregate and report required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(clean.recovery.study.data.GPT2, local_files_only=True)
    records = {}
    for root, rows, updates in ((PILOT, clean.training_probes(selected), Config().pilot_updates), (MEASURE, selected, Config().updates)):
        clean.recovery.study.data.terminal(root)
        data = clean.recovery.study.q.read(root / "case_measurements.json")
        saved = torch.load(root / "latents.pt", map_location="cpu", weights_only=True)
        result = clean.recovery.study.q.read(root / "result.json")
        if [r["id"] for r in data] != [r["id"] for r in rows] or set(saved) != {r["id"] for r in rows}:
            raise ValueError("complete fixed TRAIN identities required")
        for item, row in zip(data, rows):
            replay_case(item, row, saved[row["id"]], updates, tokenizer)
        if (result["components"] != aggregate(data) or result["cases"] != len(rows) or result["updates_per_case"] != updates
                or result["measurements_sha256"] != file_hash(root / "case_measurements.json")
                or result["latents_sha256"] != file_hash(root / "latents.pt")
                or result["codec_hash"] != continuous.EXPECTED_CODEC or not result["codec_unchanged"]
                or result["codec_parameter_gradients"] or not result["saved_reload_exact"]
                or result["peak_allocated_bytes"] >= Config().peak_memory_limit):
            raise ValueError("raw aggregate, frozen codec or resource control differs")
        records[str(root)] = data, result
    values, measured = records[str(MEASURE)]
    previous, pilot = records[str(PILOT)]
    if pilot["status"] != "systems_latent_refinement_passed" or pilot["latent_fit_gate"] is not None:
        raise ValueError("systems-only successful lifecycle pilot required")
    if any(row["arms"][ARMS[0]] != next(value for value in values if value["id"] == row["id"])["arms"][ARMS[0]] for row in previous):
        raise ValueError("exact same-GPU baseline pilot retention required")
    gate = fidelity_gate(measured["components"])
    if measured["status"] != "oracle_latent_refinement_measured" or measured["latent_fit_gate"] != gate:
        raise ValueError("fixed refinement classification differs")
    receipts = [WORKFLOW / "stage_receipts" / (name + ".json") for name in ("pilot", "measure")]
    if any(clean.recovery.study.q.read(path)["exit_code"] != 0 for path in receipts):
        raise ValueError("both actual CUDA child zero exits required")
    classification = "oracle_refined_latent_fit_passed" if gate else "oracle_refined_latent_fit_failed"
    artifacts = [Path(__file__), PLAN, WORKFLOW / "producer.json", *receipts]
    for root in (PILOT, MEASURE):
        artifacts += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts and p.name != "progress.json"]
    campaign = {"schema": "nal.ncm-decoder-latent-refinement.v1", "hypothesis_id": HYPOTHESIS,
        "classification": classification, "measurement": measured, "components": measured["components"],
        "latent_fit_gate": gate, "oracle_target_information": True, "model_weights_trained": False,
        "task_accuracy": None, "heldout_accuracy_measured": False, "NCM_success_established": False,
        "confirmed": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(artifacts)}}
    RESULTS.mkdir()
    write_json(RESULTS / "campaign_results.json", campaign)
    lines = []
    for group, value in measured["components"].items():
        cells = [f"{value['arms'][arm]['correct']}/{value['arms'][arm]['total']} = {value['arms'][arm]['token_agreement']:.2%}" for arm in ARMS]
        lines.append("| " + group + " | " + " | ".join(cells) + " |")
    REPORT.write_text(f"""# Refining target latents through the frozen NCM decoder

Date: 2026-10-23. Hypothesis `{HYPOTHESIS}`.
Classification **{classification}**. Campaign `{RESULTS / 'campaign_results.json'}`;
SHA256 `{file_hash(RESULTS / 'campaign_results.json')}`.

| Same TRAIN targets | Continuous encoder | Refined encoder |
| --- | --- | --- |
""" + "\n".join(lines) + """

Each arm receives complete target-derived information. The intervention optimizes
only per-example192-by384 latent tensors for32 fixed AdamW updates through the
same frozen original decoder. The objective is nonpadding word cross entropy
plus.001 times mean squared displacement from the continuous encoder latent.
All codec parameters remain unchanged; no source-to-summary student is trained.

These are TRAIN oracle token-reconstruction measurements, including EOS and
excluding PAD. Passing199/200 in each subgroup and the union establishes usable
teacher targets on these32 known cases. It does not establish that a student
can predict these targets from a source. Any larger label construction, latent
distillation or source-only training requires a separate prospective experiment.

The two-case lifecycle pilot used only two updates and supplies no quality
estimate. All raw words, final latents, optimizer states,32-step traces, source
bindings, exact counts and group gates replayed. Same-GPU baseline pilot outputs
and fresh decoder/latent/optimizer reloads were exact. The FP32 CUDA control used
math SDPA with TF32 off on physicalGPU2, separately from the fixed dropout study
on physicalGPU0. Shared-device wall time is not an inference benchmark.

The95% source-disjoint semantic summary goal remains unmet. Both final cohorts
remain unopened; no speed or compression benefit is established here.
""")


def register():
    campaign = clean.recovery.study.q.read(RESULTS / "campaign_results.json")
    register_records(RESULTS / "campaign_results.json", REPORT, RESULTS, {HYPOTHESIS: {
        "name": "Teacher latent refinement through the unchanged NCM decoder",
        "description": "Fixed32TRAIN oracle targets; optimize latents only, not model weights. No source-only quality claim.",
        "confirmation_status": campaign["classification"], "task_accuracy_measured": False,
        "oracle_target_information": True, "latent_fit_gate": campaign["latent_fit_gate"],
        "NCM_success_established": False}})


def workflow():
    inputs()
    if any(p.exists() for p in (WORKFLOW, PILOT, MEASURE, RESULTS, REPORT,
            Path("data/hypothesis_registry") / (HYPOTHESIS + ".json"))):
        raise ValueError("fresh workflow and all destinations required")
    p = producing(WORKFLOW, "supervise_frozen_decoder_latent_control")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [],
        "producer_fingerprint": p["fingerprint"], "final_test_opened": False}
    try:
        for stage, dest in (("pilot", PILOT), ("measure", MEASURE), ("finalize", RESULTS), ("register", RESULTS)):
            if dest.exists() != (stage == "register"):
                raise ValueError("fresh stage destination required")
            clean.recovery.study.data.verify(p)
            command = [sys.executable, "-u", "-m", MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": Config().physical_gpu if stage in ("pilot", "measure") else ""})
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"status": "observed_worker_terminal", "stage": stage, "exit_code": code,
                "observer_pid": os.getpid(), "child_pid": child.pid, "observer": "subprocess.wait",
                "command": command, "destination": str(dest)}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if dest.exists():
                terminal = dest / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json")
                if terminal.exists():
                    raise ValueError("terminal receipt overwrite prohibited")
                write_json(terminal, receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("actual latent-control stage failure: " + stage)
            clean.recovery.study.data.verify(p)
        path = RESULTS / "campaign_results.json"
        receipt = clean.recovery.study.q.read(RESULTS / (HYPOTHESIS + "_registry_readback.json"))
        if receipt["status"] != "registered_and_read_back" or receipt["record"]["campaign_sha256"] != file_hash(path):
            raise ValueError("exact registered latent-control result required")
        journal.update(status="latent_control_completed", active_stage=None,
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
    if args.stage == "check":
        print(json.dumps({"fixed_TRAIN_cases": len(inputs()), "config": asdict(Config()), "task_accuracy": None}))
    elif args.stage in ("pilot", "measure"):
        run(args.stage == "pilot")
    elif args.stage == "workflow":
        workflow()
    else:
        inputs()
        finalize() if args.stage == "finalize" else register()
