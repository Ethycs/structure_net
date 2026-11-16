"""Reservation-backed complete-response recovery for existing compact Qwen arms.

Prepare freezes data/order/initializer/source before pilot or science. A real
pilot is discarded; scientific execution reconstructs that original initializer
and creates a fresh CPU FP32 optimizer. Existing sparse-MC studies stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch

from experiments.neural_architecture_lab.distillation_recovery import module
from . import jspace_3b_response_cache as cache
from . import jspace_3b_response_eval as evaluation
from . import jspace_3b_response_judge as judge
from . import jspace_3b_response_model as model_source
from . import jspace_3b_response_train as training
from . import jspace_3b_final_evaluate as execution
from . import jspace_qwen3_train as offload

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/07 - Status Reports/jspace/2026-11-16_qwen3b-complete-response-recovery.md"
STANDARD = ROOT / "docs/05 - Standards/NAL-STD-DISTILLATION-v1.md"
SCHEMA = "jspace.3b-complete-response-campaign.v1"
SEED = 20261116
nal = module("distillation_recovery")
read, write, sha, identity = cache.read, cache.write, cache.sha, cache.identity


def training_windows(records, *, budget=100_000, accumulation=16, seed=SEED):
    """Round-robin shuffled families without replacement; retain entire windows."""
    if budget < 1 or accumulation < 1:
        raise ValueError("Positive token budget and accumulation required")
    families = {}
    for i, record in enumerate(records):
        if record.split == "train":
            name = record.provenance.get("source_family", record.task)
            families.setdefault(name, []).append(i)
    rng = random.Random(seed)
    for name in sorted(families):
        rng.shuffle(families[name])
    order, position = [], dict.fromkeys(families, 0)
    while any(position[name] < len(values) for name, values in families.items()):
        for name in sorted(families):
            if position[name] < len(families[name]):
                order.append(families[name][position[name]])
                position[name] += 1
    windows, total, included = [], 0, []
    for start in range(0, len(order), accumulation):
        indices = order[start:start + accumulation]
        # A final smaller whole-example window is allowed only if needed to
        # reach the prospective budget; no response is cut at the budget edge.
        total += sum(len(records[i].response_ids) for i in indices)
        windows.append(indices); included.extend(indices)
        if total >= budget:
            break
    if total < budget:
        raise ValueError("Admitted unique TRAIN responses cannot cover the requested token rung")
    return dict(windows=windows, order=included, nominal_supervised_tokens=budget,
                actual_supervised_tokens=total, examples=len(included), seed=seed,
                accumulation_examples=accumulation,
                input_tokens=sum(len(records[i].input_ids) for i in included),
                family_examples=dict(Counter(records[i].provenance.get("source_family", records[i].task)
                                             for i in included)),
                mixture="Per-family deterministic shuffle, round-robin examples, no replacement")


def checkpoint_boundaries(windows, records, *, thresholds=(25_000, 50_000, 75_000, 100_000)):
    pending = list(thresholds); boundaries = []; total = 0
    for update, indices in enumerate(windows, 1):
        total += sum(len(records[i].response_ids) for i in indices)
        crossed = []
        while pending and total >= pending[0]:
            crossed.append(pending.pop(0))
        if crossed:
            boundaries.append(dict(update=update, supervised_tokens=total, crossed_thresholds=crossed))
    if pending:
        raise ValueError("Frozen training windows do not reach all declared checkpoints")
    return boundaries


def pilot_windows(records, scientific_order, *, accumulation=16, updates=2):
    """Cover worst observed lengths and every family before ordinary examples."""
    train = [i for i, record in enumerate(records) if record.split == "train"]
    needed = accumulation * updates
    if len(train) < needed:
        raise ValueError("Pilot needs two complete independent TRAIN accumulation windows")
    chosen = []
    def add(index):
        if index not in chosen:
            chosen.append(index)
    add(max(train, key=lambda i:(len(records[i].input_ids), -i)))
    add(max(train, key=lambda i:(len(records[i].response_ids), -i)))
    families = {}
    for index in train:
        families.setdefault(records[index].provenance.get("source_family", records[index].task), []).append(index)
    if len(families) + 2 > needed:
        raise ValueError("Pilot cannot cover every admitted family within the fixed window count")
    for family in sorted(families):
        add(families[family][0])
    for index in [*scientific_order, *train]:
        if len(chosen) == needed:
            break
        if index not in train:
            raise ValueError("Pilot input order contains non-TRAIN material")
        add(index)
    return [chosen[start:start+accumulation] for start in range(0, needed, accumulation)]


def _source_files():
    explicit = [Path(__file__), PROTOCOL, STANDARD,
                ROOT/"tests/structure_net/test_jspace_3b_response_campaign.py",
                ROOT/"experiments/neural_architecture_lab/distillation_recovery.py"]
    explicit.extend(ROOT/"src/neural_architecture_lab"/(name+".py") for name in
                    ("distillation_data", "distillation_training", "distillation_recovery", "core", "local_scheduler"))
    explicit.extend(ROOT/"experiments/structure_net"/("jspace_3b_response_"+name+".py")
                    for name in ("cache", "train", "model", "eval", "judge"))
    return cache.prior.source.inherited.local_import_closure(explicit)


def prepare(root, arm, source_plan, checkpoint, cache_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Preparation is CPU-only with CUDA hidden")
    if arm not in {"progressive", "all36"}:
        raise ValueError("Choose progressive or all36")
    root, source_plan, checkpoint, cache_root = [Path(p).resolve() for p in
                                                (root, source_plan, checkpoint, cache_root)]
    if root.exists():
        raise FileExistsError(root)
    cp, records = cache.verify(cache_root, live=False)
    previous = read(source_plan)
    model_source.progressive.source.check_plan(previous)
    cut = previous["cut"] if arm == "progressive" else 0
    expected_schema = model_source.progressive.SCHEMA if arm == "progressive" else model_source.joint.SCHEMA
    if previous.get("schema") != expected_schema or cut not in (0, cp["cut"]):
        raise ValueError("Source model depth does not match its native response cache")
    if (previous["model"]["revision"] != cp["model"]["revision"] or
            Path(previous["model"]["path"]).resolve() != Path(cp["model"]["path"]).resolve()):
        raise ValueError("Source model and frozen native teacher differ")
    order = training_windows(records)
    configuration = asdict(training.ResponseConfig(max_supervised_tokens=order["actual_supervised_tokens"]))
    inputs = {str(path): sha(path) for path in (source_plan, checkpoint, cache_root/"plan.json")}
    source_files = _source_files()
    p = dict(schema=SCHEMA, arm=arm, cut=cut, source_plan=str(source_plan),
             source_fingerprint=previous["fingerprint"], checkpoint=str(checkpoint),
             checkpoint_sha256=inputs[str(checkpoint)], cache_root=str(cache_root),
             cache_fingerprint=cp["fingerprint"], corpus_fingerprint=cp["records_sha256"],
             model=cp["model"], training=order, configuration=configuration,
             checkpoints=checkpoint_boundaries(order["windows"], records),
             generation_indices=cp["generation_indices"], max_new_tokens=128,
             execution="BF16 unpacked Torch compact stack",
             parity_reference="Independent HF wrapper with same-weight unpacked compact stack",
             packed_deployment_parity="Separate gate; prior packed comparisons failed and are not certified here",
             judge=asdict(judge.JudgeConfig(expected_returned_model="codex_sdk/gpt-5.6-sol")),
             judging_execution="One asynchronous CPU worker, immutable checkpoint snapshots, all joined before selection",
             gates=dict(maximum_drop_pp=5., maximum_family_drop_pp=10., minimum_generation_rate=.5,
                        maximum_native_nll_ratio=1.10),
             selection="Eligible: higher semantic task success, lower response NLL, earlier update",
             failure_policy="Retain initializer and diagnostic best; no automatic extra epochs",
             pilot_updates=2, pilot_windows=pilot_windows(records, order["order"]), seed=SEED, confirmed=False,
             input_sha256=inputs, input_identity={path: identity(path) for path in inputs},
             source_sha256={str(path): sha(path) for path in sorted(source_files)})
    p["fingerprint"] = cache.data.digest(p)
    root.mkdir(parents=True)
    write(root/"plan.json", p)
    for path, digest in p["source_sha256"].items():
        source = Path(path)
        target = root/"producer"/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != digest:
            raise ValueError("Producer archive differs from prepared source")
    return dict(status="prepared", arm=arm, fingerprint=p["fingerprint"], cut=cut,
                examples=order["examples"], supervised_tokens=order["actual_supervised_tokens"],
                updates=len(order["windows"]), native_cache_complete=(cache_root/"native/manifest.json").is_file())


def verify(root, *, live=True):
    root = Path(root); p = read(root/"plan.json")
    body = dict(p); fingerprint = body.pop("fingerprint")
    if p.get("schema") != SCHEMA or cache.data.digest(body) != fingerprint:
        raise ValueError("Campaign plan changed")
    for path, pinned in p["input_identity"].items():
        if identity(path) != pinned:
            raise ValueError("Campaign admitted input changed: "+path)
    for path, digest in p["source_sha256"].items():
        if sha(root/"producer"/Path(path).relative_to(ROOT)) != digest or live and sha(path) != digest:
            raise ValueError("Campaign producer changed: "+path)
    cp, records = cache.verify(p["cache_root"], live=False)
    if cp["fingerprint"] != p["cache_fingerprint"] or cp["records_sha256"] != p["corpus_fingerprint"]:
        raise ValueError("Campaign cache/corpus changed")
    expected = training_windows(records, budget=p["training"]["nominal_supervised_tokens"],
                                accumulation=p["training"]["accumulation_examples"], seed=p["seed"])
    if expected != p["training"] or checkpoint_boundaries(expected["windows"], records) != p["checkpoints"]:
        raise ValueError("Training membership/order or evaluation boundaries changed")
    if pilot_windows(records, expected["order"]) != p["pilot_windows"]:
        raise ValueError("Representative pilot windows changed")
    return p, cp, records


def _progress(directory, event):
    value = dict(event, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with (directory/"progress.jsonl").open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")
        stream.flush()
    print(json.dumps(value, allow_nan=False), flush=True)


def _identities(p):
    return dict(study=p["fingerprint"], teacher=p["model"]["revision"],
                initializer=p["checkpoint_sha256"], corpus=p["corpus_fingerprint"],
                cache=p["cache_fingerprint"])


def _open_model(p, guard):
    values = model_source.build_student(p["arm"], p["source_plan"], p["checkpoint"], "cuda:0", guard)
    stack, norm, embedding, rotary, proof = values
    if proof["checkpoint_sha256"] != p["checkpoint_sha256"] or proof["cut"] != p["cut"]:
        raise ValueError("Constructed initializer differs from prepared model")
    return values


def _pilot_report(trainer, changes, updates):
    # The optimizer's master mask is in trainer.parameters order. Norm storage
    # may not move in BF16 even though its FP32 master changed.
    masks = [row["optimizer"]["master_changed_by_parameter"] for row in updates]
    if any(len(mask) != len(trainer.parameters) for mask in masks):
        raise ValueError("Optimizer master mask differs from registered trainables")
    changed = {id(p): any(mask[i] for mask in masks) for i, p in enumerate(trainer.parameters)}
    block_master = [any(changed[id(p)] for p in block.parameters()) for block in trainer.stack.blocks]
    norm_master = any(changed[id(p)] for p in trainer.norm.parameters())
    gradients = all(all(row["gradients_present_by_block"]) for row in updates)
    finite = all(math.isfinite(row["loss"]) and math.isfinite(row["optimizer"]["unclipped_gradient_norm"])
                 for row in updates)
    passed = finite and gradients and all(block_master) and norm_master and all(changes["bf16_changed_by_block"])
    return dict(passed=passed, finite_gradients=finite, all_block_gradients=gradients,
                fp32_master_changed_by_block=block_master, norm_fp32_master_changed=norm_master, **changes)


@torch.no_grad()
def _replay_checkpoint(trainer, checkpoint, batch):
    def logits():
        hidden = trainer.model.hidden_forward(batch)
        position = int(batch["loss_mask"].nonzero()[0, 1])
        return trainer.model.readout(hidden[:, position]).detach().cpu()
    trainer.model.eval()
    expected = logits()
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    if payload.get("schema") != "jspace.complete-response-checkpoint.v1":
        raise ValueError("Pilot checkpoint format changed")
    trainer.stack.load_state_dict(payload["stack"], strict=True)
    trainer.norm.load_state_dict(payload["norm"], strict=True)
    del payload
    actual = logits()
    same = torch.equal(actual, expected)
    if not same:
        raise ValueError("Saved pilot checkpoint failed full-vocabulary replay")
    return dict(passed=True, full_vocabulary_exact=True, vocabulary=actual.shape[-1],
                scope="Strict saved-state in-process replay; scientific worker reconstructs fresh initializer")


def _phase_start(root, phase, limit_mib):
    root = Path(root).resolve(); p, cp, records = verify(root)
    directory = root/phase
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir()
    return root, p, cp, records, directory


def pilot(root, *, limit_mib):
    root, p, cp, records, directory = _phase_start(root, "pilot", limit_mib)
    start = time.monotonic(); trainer = dataset = None; lease = None
    try:
        lease, guard = cache.reservation(limit_mib); torch.set_num_threads(2)
        _progress(directory, dict(phase="pilot_initialization", arm=p["arm"], lease=lease["id"]))
        dataset = cache.ResponseDataset(p["cache_root"], cut=p["cut"])
        native_receipt = read(Path(p["cache_root"])/"native/result.json")
        if not native_receipt["teacher_unchanged"] or not native_receipt["full_vocabulary_parity"]["passed"]:
            raise ValueError("Original teacher cache lacks immutability/full-vocabulary checks")
        with execution.arithmetic_policy():
            stack, norm, embedding, rotary, proof = _open_model(p, guard)
            student = training.ResponseModel(stack, norm, embedding, rotary, guard=guard)
            parity = model_source.check_unpacked_execution_parity(student, p["model"]["path"],
                records[p["generation_indices"][0]].prompt_ids, guard=guard)
            write(directory/"path_parity.json", parity)
            if not parity["passed"]:
                raise ValueError("Same-checkpoint full-vocabulary inference parity failed")
            del student
            trainer = training.ResponseTrainer(stack, norm, embedding, rotary, dataset,
                teacher_id=p["model"]["revision"], config=training.ResponseConfig(**p["configuration"]), guard=guard)
            before = trainer.parameter_fingerprints(); rows = []
            for indices in p["pilot_windows"]:
                row = trainer.train_window(indices); rows.append(row)
                _progress(directory, dict(phase="pilot_update", **row))
            changes = trainer.change_report(before)
            updates = _pilot_report(trainer, changes, rows)
            if not updates["passed"]:
                write(directory/"update_checks.json", updates)
                raise ValueError("Actual per-block BF16/master/gradient pilot checks failed")
            checkpoint = trainer.save_checkpoint(directory/"discarded_pilot.pt", identities=_identities(p))
            replay = _replay_checkpoint(trainer, checkpoint, dataset.batch([p["pilot_windows"][0][0]]))
            fixed = offload.fixed_state(stack, embedding, rotary) == trainer.fixed
            if not fixed:
                raise ValueError("Pilot changed fixed geometry/readout")
            result = dict(status="systems_passed", fingerprint=p["fingerprint"], lease_id=lease["id"],
                allocation=lease["allocation"], allocator_mib=limit_mib, updates=updates,
                full_vocabulary_path_parity=parity, save_reload=replay, fixed_geometry_unchanged=fixed,
                teacher_unchanged=native_receipt["teacher_unchanged"], memory=trainer.memory,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.monotonic()-start,
                checkpoint=checkpoint, scientific_initialization=p["checkpoint"], pilot_discarded=True,
                pilot_indices=[i for window in p["pilot_windows"] for i in window],
                longest_input=max(len(records[i].input_ids) for window in p["pilot_windows"] for i in window),
                longest_response=max(len(records[i].response_ids) for window in p["pilot_windows"] for i in window),
                device_capability=list(torch.cuda.get_device_capability(0)),
                confirmed=False, source_proof=proof)
            write(directory/"result.json", result)
            write(directory/"manifest.json", dict(status="completed", fingerprint=p["fingerprint"],
                artifacts={"result.json":sha(directory/"result.json"), "discarded_pilot.pt":checkpoint["sha256"],
                           "path_parity.json":sha(directory/"path_parity.json")}))
            return result
    except BaseException as error:
        write(directory/"failure.json", dict(status="failed", error_type=type(error).__name__,
            error=str(error), lease=lease, elapsed_seconds=time.monotonic()-start))
        raise
    finally:
        if trainer is not None: trainer.close()
        if dataset is not None: dataset.close()


def checked_pilot(root, p):
    directory = Path(root)/"pilot"; manifest = read(directory/"manifest.json")
    if manifest["status"] != "completed" or manifest["fingerprint"] != p["fingerprint"]:
        raise ValueError("Successful matching actual-model pilot required before scientific recovery")
    # Large pilot weights are not consumed by science. Their retained digest is
    # evidence; the small receipt and parity are authenticated at admission.
    for name in ("result.json", "path_parity.json"):
        if sha(directory/name) != manifest["artifacts"][name]:
            raise ValueError("Pilot receipt changed")
    result = read(directory/"result.json")
    if (result["fingerprint"] != p["fingerprint"] or result["status"] != "systems_passed" or not result["pilot_discarded"]
            or not result["updates"]["passed"] or not result["save_reload"]["passed"]):
        raise ValueError("Pilot failed required scientific admission gates")
    return result


def combine_gates(development, native_development, canaries, semantics, mc, gates, *, task_checks):
    nll = development["ce"]
    nll_passed = math.isfinite(nll) and nll <= gates["maximum_native_nll_ratio"] * native_development["nll"]
    checks = dict(response_nll=nll_passed, canaries=canaries["passed"],
                  semantic=semantics["passed"], multiple_choice=mc["eligibility"]["passed"],
                  generative_tasks=task_checks["passed"])
    return dict(passed=all(checks.values()), checks=checks, student_nll=nll,
                native_nll=native_development["nll"], maximum_native_nll_ratio=gates["maximum_native_nll_ratio"],
                confirmed=False)


def select_checkpoint(candidates):
    if not candidates:
        raise ValueError("Retained initializer and candidates required")
    def key(row):
        rate = row.get("semantic_task_success")
        return (-(rate if rate is not None else -1.), row["nll"], row["update"])
    eligible = [row for row in candidates if row["eligibility"]["passed"]]
    best = min(eligible, key=key) if eligible else None
    diagnostic = min(candidates, key=key)
    initial = next((row for row in candidates if row["update"] == 0), None)
    if initial is None:
        raise ValueError("Initializer must remain retained")
    return dict(status="exploratory_quality_pass" if best else "quality_gate_failed", selected=best,
                diagnostic_best=diagnostic, initializer=initial, fallback=initial if best is None else None,
                confirmed=False, promoted=False)


def _complete_evaluation(directory, output, snapshot):
    """CPU/network-only completion, bound to copied saved-checkpoint outputs."""
    update = snapshot["update"]
    _progress(directory, dict(phase="semantic_judge", update=update,
                             pairs=len(snapshot["generation"])))
    semantics = judge.judge_pairs(snapshot["native_corpus"], snapshot["generation"],
        output/"semantic_judge", config=judge.JudgeConfig(**snapshot["judge"]), contexts=snapshot["contexts"])
    eligibility = combine_gates(snapshot["development"], snapshot["native_development"],
        snapshot["canary_gate"], semantics["semantic_gate"], snapshot["mc"], snapshot["gates"],
        task_checks=snapshot["task_checks"])
    semantic_rate = semantics["semantic_gate"].get("rates", {}).get("task_success", {}).get("student")
    result = dict(update=update, supervised_tokens=snapshot["supervised_tokens"], checkpoint=snapshot["checkpoint"],
                  nll=snapshot["development"]["ce"], teacher_kl=snapshot["development"]["forward_kl"],
                  semantic_task_success=semantic_rate, eligibility=eligibility,
                  canary_gate=snapshot["canary_gate"], semantic_gate=semantics["semantic_gate"],
                  generative_task_gates=snapshot["task_checks"].get("gates", {}),
                  multiple_choice_eligibility=snapshot["mc"]["eligibility"],
                  evaluation_path=str(output), confirmed=False)
    write(output/"result.json", result)
    _progress(directory, dict(phase="evaluation_complete", update=update, **eligibility))
    return result


def queue_judgment(pool, directory, output, snapshot):
    # Never capture the live trainer/model/dataset. Later optimizer updates may
    # run concurrently, but cannot change any text, metric or checkpoint bound
    # to this asynchronous judgment.
    return pool.submit(_complete_evaluation, directory, output, deepcopy(snapshot))


def collect_judgments(pending, candidates, *, block=False):
    remaining = []
    for future in pending:
        if block or future.done():
            candidates.append(future.result())
        else:
            remaining.append(future)
    return remaining


def _evaluate(trainer, p, cp, directory, checkpoint, tokenizer, guard, judge_pool):
    update = trainer.updates; output = directory/f"evaluation_{update:04d}"; output.mkdir()
    _progress(directory, dict(phase="response_development", update=update))
    development, rows = trainer.development()
    write(output/"development.json", dict(**development, records=rows))
    _progress(directory, dict(phase="old_mc_development", update=update))
    mc = model_source.old_mc_evaluate(p["arm"], p["source_plan"], trainer.model, guard)
    write(output/"multiple_choice.json", mc)
    _progress(directory, dict(phase="generation", update=update))
    texts = {row["example_id"]:row for row in map(json.loads,
              (Path(cp["corpus"])/"text.jsonl").read_text().splitlines())}
    native = read(Path(p["cache_root"])/"native/generation.json")
    generation = []
    with model_source.attach_native_prefix(trainer.model, p["model"]["path"], guard=guard):
        canaries = evaluation.evaluate_canaries(trainer.model, tokenizer, device="cuda:0",
                                                eos_token_ids=cp["generation_eos_token_ids"])
        for index in p["generation_indices"]:
            record = trainer.dataset.records[index]; source = texts[record.example_id]
            generation.append(evaluation.generate_record(trainer.model, record, tokenizer, device="cuda:0",
                source_text=source.get("context", source.get("prompt_text", "")),
                eos_token_ids=cp["generation_eos_token_ids"]))
            _progress(directory, dict(phase="generation", update=update, completed=len(generation),
                                     total=len(p["generation_indices"])))
    write(output/"generation.json", dict(canaries=canaries, corpus=generation))
    _progress(directory, dict(phase="generative_task_checks", update=update))
    from . import jspace_3b_response_task_metrics as task_metrics
    task_checks = task_metrics.grade_generated_tasks(native["corpus"], generation, texts)
    write(output/"generative_task_checks.json", task_checks)
    canary_gate = evaluation.generation_gate(native["canaries"], canaries,
        maximum_drop_pp=p["gates"]["maximum_drop_pp"], minimum_student_rate=p["gates"]["minimum_generation_rate"])
    contexts = {row["example_id"]:dict(prompt=row["prompt_text"], source=row["context"],
                  reference=row["response"], task=row["task"]) for row in texts.values()
                  if row["example_id"] in {r["example_id"] for r in generation}}
    native_dev = read(Path(p["cache_root"])/"native/development.json")
    snapshot = dict(update=update, supervised_tokens=trainer.supervised_tokens, checkpoint=checkpoint,
        development=development, native_development={"nll":native_dev["nll"]}, canary_gate=canary_gate,
        native_corpus=native["corpus"], generation=generation, contexts=contexts,
        mc={"eligibility":mc["eligibility"]}, task_checks={"passed":task_checks["passed"], "gates":task_checks.get("gates", {})},
        gates=p["gates"], judge=p["judge"])
    write(output/"judgment_snapshot.json", snapshot)
    _progress(directory, dict(phase="semantic_judge_queued", update=update, pairs=len(generation)))
    return queue_judgment(judge_pool, directory, output, snapshot)


def _ledger(root, p, result, trainer, selected, training_seconds):
    experiment = nal.Experiment(id=f"{p['arm']}.{p['fingerprint'][:12]}",
        hypothesis_id="JSPACE-QWEN3B-RESPONSE-RECOVERY-01", name="Complete-response compressed Qwen recovery",
        parameters=dict(fingerprint=p["fingerprint"], arm=p["arm"], configuration=p["configuration"]), seed=p["seed"])
    record = nal.ExperimentResult(experiment_id=experiment.id, hypothesis_id=experiment.hypothesis_id,
        metrics=dict(eligible=float(result["status"] == "exploratory_quality_pass"),
                     response_nll=float(selected["nll"]), supervised_tokens=float(trainer.supervised_tokens)),
        primary_metric=float(selected["semantic_task_success"] or 0.),
        model_architecture=[len(trainer.stack.blocks), trainer.stack.width],
        model_parameters=sum(parameter.numel() for parameter in trainer.parameters),
        training_time=training_seconds, model_checkpoint=selected["checkpoint"]["path"],
        observations=[result["status"], "Exploratory DEV only; no independent confirmation or deployment claim",
                      "model_parameters counts trainable compact parameters, excluding frozen embedding/native prefix; not a deployed-size claim"],
        status=nal.ExperimentStatus.COMPLETED)
    path = nal.ExperimentResultLedger(root).save(experiment, record, attempts=1, device_id=0)
    loaded = nal.ExperimentResultLedger(root).load(experiment)
    if loaded is None or loaded.metrics != record.metrics or loaded.model_checkpoint != record.model_checkpoint:
        raise ValueError("NAL result ledger readback differs")
    return str(path)


def _reload_selected(trainer, p, selected):
    """Strictly restore the retained candidate and replay its complete DEV loss."""
    checkpoint = selected["checkpoint"]
    if sha(checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Consumed selected checkpoint changed")
    offload.require_host_memory(trainer.memory["snapshot_bytes"], reserve_bytes=1024**3)
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    if (payload.get("schema") != "jspace.complete-response-checkpoint.v1"
            or payload.get("identities") != _identities(p)
            or payload.get("updates") != selected["update"]):
        raise ValueError("Selected saved checkpoint provenance differs")
    trainer.stack.load_state_dict(payload["stack"], strict=True)
    trainer.norm.load_state_dict(payload["norm"], strict=True)
    del payload
    if offload.fixed_state(trainer.stack, trainer.readout, trainer.rotary) != trainer.fixed:
        raise ValueError("Selected checkpoint reload changed fixed geometry/readout")
    metrics, rows = trainer.development()
    expected = read(Path(selected["evaluation_path"])/"development.json")
    if dict(**metrics, records=rows) != expected:
        raise ValueError("Selected checkpoint complete-response DEV replay differs")
    return dict(passed=True, checkpoint_sha256=checkpoint["sha256"], update=selected["update"],
                strict_state_load=True, complete_response_dev_exact=True, fixed_geometry_unchanged=True,
                in_process=True, fresh_process_export_validation_pending=True)


def run(root, *, limit_mib):
    root, p, cp, records, directory = _phase_start(root, "train", limit_mib)
    start = time.monotonic(); trainer = dataset = judge_pool = None; lease = None
    try:
        pilot_receipt = checked_pilot(root, p)
        if pilot_receipt["peak_reserved_bytes"] > limit_mib*1024**2:
            raise ValueError("Scientific allocation is below measured representative pilot peak")
        lease, guard = cache.reservation(limit_mib); torch.set_num_threads(2)
        _progress(directory, dict(phase="scientific_initialization", arm=p["arm"], lease=lease["id"],
                                 pilot_weights_consumed=False))
        dataset = cache.ResponseDataset(p["cache_root"], cut=p["cut"])
        with execution.arithmetic_policy():
            stack, norm, embedding, rotary, proof = _open_model(p, guard)
            trainer = training.ResponseTrainer(stack, norm, embedding, rotary, dataset,
                teacher_id=p["model"]["revision"], config=training.ResponseConfig(**p["configuration"]), guard=guard)
            if trainer.updates or trainer.supervised_tokens or trainer.visited:
                raise ValueError("Scientific optimizer/counters must start fresh")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(p["model"]["path"], local_files_only=True, trust_remote_code=False)
            judge_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="response-semantic-judge")
            checkpoint = trainer.save_checkpoint(directory/"update_0000.pt", identities=_identities(p))
            candidates = []
            pending = [_evaluate(trainer, p, cp, directory, checkpoint, tokenizer, guard, judge_pool)]
            boundaries = {row["update"] for row in p["checkpoints"]}
            training_seconds = 0.
            for indices in p["training"]["windows"]:
                row = trainer.train_window(indices); training_seconds += row["train_window_seconds"]
                _progress(directory, dict(phase="training", **row))
                if trainer.updates in boundaries:
                    pending = collect_judgments(pending, candidates)
                    checkpoint = trainer.save_checkpoint(directory/f"update_{trainer.updates:04d}.pt", identities=_identities(p))
                    pending.append(_evaluate(trainer, p, cp, directory, checkpoint, tokenizer, guard, judge_pool))
            if trainer.supervised_tokens != p["training"]["actual_supervised_tokens"] or trainer.visited != set(p["training"]["order"]):
                raise ValueError("Executed token count/TRAIN membership differs from frozen first rung")
            _progress(directory, dict(phase="await_semantic_judgments", pending=len(pending)))
            pending = collect_judgments(pending, candidates, block=True)
            if pending:
                raise RuntimeError("Every checkpoint judgment must resolve before selection")
            selection = select_checkpoint(candidates)
            write(directory/"selection.json", selection)
            selected = selection["selected"] or selection["diagnostic_best"]
            _progress(directory, dict(phase="selected_checkpoint_reload", selected_update=selected["update"]))
            reload = _reload_selected(trainer, p, selected)
            write(directory/"selected_reload.json", reload)
            result = dict(**selection, fingerprint=p["fingerprint"], candidates=candidates,
                updates=trainer.updates, supervised_tokens=trainer.supervised_tokens,
                input_tokens=trainer.input_tokens, memory=trainer.memory,
                cpu_optimizer=trainer.optimizer.memory_stats(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                lease_id=lease["id"], allocation=lease["allocation"], allocator_mib=limit_mib,
                pilot_lease_id=pilot_receipt["lease_id"], initializer_proof=proof,
                fresh_scientific_initializer=True, training_seconds=training_seconds,
                elapsed_seconds=time.monotonic()-start, selected_reload=reload)
            result["nal_ledger"] = _ledger(root, p, result, trainer, selected, training_seconds)
            write(directory/"result.json", result)
            write(directory/"manifest.json", dict(status="completed", fingerprint=p["fingerprint"],
                artifacts={"result.json":sha(directory/"result.json"), "selection.json":sha(directory/"selection.json"),
                           "selected_reload.json":sha(directory/"selected_reload.json")},
                checkpoints={row["checkpoint"]["path"]:row["checkpoint"]["sha256"] for row in candidates}))
            return result
    except BaseException as error:
        write(directory/"failure.json", dict(status="failed", error_type=type(error).__name__, error=str(error),
            lease=lease, elapsed_seconds=time.monotonic()-start, checkpoints_preserved=True))
        raise
    finally:
        if judge_pool is not None: judge_pool.shutdown(wait=True, cancel_futures=True)
        if trainer is not None: trainer.close()
        if dataset is not None: dataset.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=("prepare", "pilot", "run", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("progressive", "all36"))
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--limit-mib", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        if not all((args.arm, args.source_plan, args.checkpoint, args.cache_root)):
            parser.error("prepare requires --arm --source-plan --checkpoint --cache-root")
        result = prepare(args.root, args.arm, args.source_plan, args.checkpoint, args.cache_root)
    elif args.phase in {"pilot", "run"}:
        if args.limit_mib is None:
            parser.error("GPU phases require explicit allocator --limit-mib below leased capacity")
        result = (pilot if args.phase == "pilot" else run)(args.root, limit_mib=args.limit_mib)
    else:
        p, _, _ = verify(args.root)
        result = dict(status="prepared", fingerprint=p["fingerprint"], arm=p["arm"])
    print(json.dumps({key: result[key] for key in ("status", "fingerprint", "arm", "updates", "supervised_tokens")
                      if key in result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
