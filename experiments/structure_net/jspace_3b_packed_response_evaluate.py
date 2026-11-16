"""Pack, export, reload and evaluate immutable response-recovery snapshots.

This is a separate inference study. It never changes training weights, plans,
checkpoint selection, or producer files. Quality uses the actual packed reload;
packed-versus-unpacked logit equality is not a scientific admission criterion.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

import torch

from experiments.neural_architecture_lab.distillation_recovery import module
from . import jspace_3b_response_cache as cache
from . import jspace_3b_response_campaign as campaign
from . import jspace_3b_response_eval as evaluation
from . import jspace_3b_response_judge as judge
from . import jspace_3b_response_model as source_model
from . import jspace_3b_response_train as training
from . import jspace_3b_response_task_metrics as task_metrics
from . import jspace_3b_final_evaluate as execution
from . import jspace_3b_packed_response_model as packed_source

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/"docs/07 - Status Reports/jspace/2026-11-16_qwen3b-packed-response-evaluation.md"
SCHEMA = "jspace.3b-packed-response-evaluation.v1"
nal = module("distillation_recovery")
read, write, sha, identity = cache.read, cache.write, cache.sha, cache.identity


def checkpoint_metadata(path, source_plan):
    """Read the complete saved container, without allocating another weight copy."""
    before = identity(path)
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = campaign._identities(source_plan)
    if (payload.get("schema") != "jspace.complete-response-checkpoint.v1"
            or payload.get("identities") != expected
            or payload.get("config") != source_plan["configuration"]
            or not isinstance(payload.get("stack"), dict) or not payload["stack"]
            or not isinstance(payload.get("norm"), dict) or not payload["norm"]):
        raise ValueError("Complete response checkpoint and matching campaign identities/configuration required")
    update = payload.get("updates")
    expected_tokens = {0:0, **{row["update"]:row["supervised_tokens"] for row in source_plan["checkpoints"]}}
    if type(update) is not int or update not in expected_tokens or payload.get("supervised_tokens") != expected_tokens[update]:
        raise ValueError("Only the frozen saved initializer/checkpoint boundaries may be tested")
    expected_visited = sorted(i for window in source_plan["training"]["windows"][:update] for i in window)
    if payload.get("visited") != expected_visited:
        raise ValueError("Snapshot TRAIN membership differs from its frozen update boundary")
    if not all(isinstance(value, torch.Tensor) for state in (payload["stack"], payload["norm"])
               for key, value in state.items() if not key.endswith("_extra_state")):
        raise ValueError("Saved compact state contains an unexpected non-tensor")
    result = dict(schema=payload["schema"], update=update, supervised_tokens=payload["supervised_tokens"],
        input_tokens=payload["input_tokens"], identities=payload["identities"],
        fixed_state=payload["fixed_state"], stack_entries=len(payload["stack"]), norm_entries=len(payload["norm"]))
    del payload
    if identity(path) != before:
        raise ValueError("Checkpoint changed while being admitted")
    return result


def _sources():
    paths = [Path(__file__), Path(packed_source.__file__), PROTOCOL,
             ROOT/"tests/structure_net/test_jspace_3b_packed_response_evaluate.py",
             ROOT/"experiments/neural_architecture_lab/distillation_recovery.py"]
    paths.extend(ROOT/"src/neural_architecture_lab"/(name+".py") for name in
                 ("core", "local_scheduler", "distillation_recovery", "distillation_data", "distillation_training"))
    paths.extend(path for path in (ROOT/"experiments/structure_net/jspace_3b_packed_response_watch.py",
                                   ROOT/"tests/structure_net/test_jspace_3b_packed_response_watch.py") if path.is_file())
    return cache.prior.source.inherited.local_import_closure(paths)


def prepare(root, campaign_root, checkpoint, *, prefix_mode="resident", unpacked_report=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Packed study preparation requires CUDA hidden")
    if prefix_mode not in {"resident", "cpu"}:
        raise ValueError("Declare resident or CPU-resident streamed native-prefix weights")
    root, campaign_root, checkpoint = [Path(p).resolve() for p in (root, campaign_root, checkpoint)]
    if root.exists():
        raise FileExistsError(root)
    parent, cp, records = campaign.verify(campaign_root, live=False)
    metadata = checkpoint_metadata(checkpoint, parent)
    before = identity(checkpoint); checkpoint_sha = sha(checkpoint)
    if identity(checkpoint) != before:
        raise ValueError("Checkpoint changed while hashing; wait for the durable saved receipt")
    cache_root = Path(parent["cache_root"])
    native_manifest = read(cache_root/"native/manifest.json")
    if native_manifest["status"] != "completed" or native_manifest["fingerprint"] != cp["fingerprint"]:
        raise ValueError("The original native cache and evaluation must be completed")
    inputs = {str(campaign_root/"plan.json"):sha(campaign_root/"plan.json"),
              str(checkpoint):checkpoint_sha,
              str(cache_root/"plan.json"):sha(cache_root/"plan.json"),
              str(cache_root/"native/manifest.json"):sha(cache_root/"native/manifest.json")}
    comparison = None
    if unpacked_report is not None:
        unpacked_report = Path(unpacked_report).resolve(); report = read(unpacked_report)
        if report.get("checkpoint", {}).get("sha256") != checkpoint_sha or report.get("update") != metadata["update"]:
            raise ValueError("Contextual unpacked report is not this exact saved checkpoint")
        inputs[str(unpacked_report)] = sha(unpacked_report)
        comparison = str(unpacked_report)
    sources = _sources()
    p = dict(schema=SCHEMA, source_campaign=str(campaign_root), source_fingerprint=parent["fingerprint"],
        arm=parent["arm"], cut=parent["cut"], source_plan=parent["source_plan"],
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha, checkpoint_metadata=metadata,
        cache_root=str(cache_root), cache_fingerprint=cp["fingerprint"], model=parent["model"],
        prefix_mode=prefix_mode, generation_indices=parent["generation_indices"],
        generation_eos_token_ids=cp["generation_eos_token_ids"], max_new_tokens=128,
        gates=parent["gates"], judge=dict(parent["judge"], workers=1), warmup_calls=2, timing_repeats=5,
        judging_concurrency=dict(parent_workers=parent["judge"]["workers"], packed_workers=1,
            reason="Reduce added gateway load after exposed parent HTTP429/connection failures",
            unchanged="model, temperature, rubric, quality gates, one attempt per pair"),
        timing_index=parent["generation_indices"][0], unpacked_report=comparison,
        packing_policy="Export structured packed state; destroy initial model; load fresh packed state before scoring",
        quality_policy="Original native task/generation gates; no packed-versus-unpacked full-logit equality gate",
        device_policy="Single externally reserved GPU; record actual backend and allocator; no automatic CPU-prefix fallback",
        confirmed=False, input_sha256=inputs, input_identity={path:identity(path) for path in inputs},
        source_sha256={str(path):sha(path) for path in sorted(sources)})
    p["fingerprint"] = cache.data.digest(p)
    root.mkdir(parents=True)
    write(root/"plan.json", p)
    for path, digest in p["source_sha256"].items():
        destination = root/"producer"/Path(path).relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
        if sha(destination) != digest:
            raise ValueError("Packed sidecar producer archive differs")
    return dict(status="prepared", fingerprint=p["fingerprint"], arm=p["arm"],
                update=metadata["update"], prefix_mode=prefix_mode)


def verify(root, *, live=True):
    root = Path(root); p = read(root/"plan.json")
    body = dict(p); fingerprint = body.pop("fingerprint")
    if p.get("schema") != SCHEMA or cache.data.digest(body) != fingerprint:
        raise ValueError("Packed evaluation plan changed")
    for path, pinned in p["input_identity"].items():
        if identity(path) != pinned:
            raise ValueError("Admitted packed-evaluation input changed: "+path)
    for path, digest in p["source_sha256"].items():
        if sha(root/"producer"/Path(path).relative_to(ROOT)) != digest or live and sha(path) != digest:
            raise ValueError("Packed evaluation producer changed: "+path)
    parent, cp, records = campaign.verify(p["source_campaign"], live=False)
    if parent["fingerprint"] != p["source_fingerprint"] or cp["fingerprint"] != p["cache_fingerprint"]:
        raise ValueError("Source campaign/cache identity changed")
    return p, parent, cp, records


def _progress(directory, **event):
    event["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (directory/"progress.jsonl").open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(event, allow_nan=False), flush=True)


@torch.inference_mode()
def packed_probe(model, prompt_ids, *, prefixes=None):
    """Small systems vectors; these are not task-accuracy observations."""
    device = model.embedding.device
    history = list(prompt_ids); rows = []
    for step in range(3):
        current = list(prefixes[step]) if prefixes is not None else list(history)
        logits = model.next_logits(torch.tensor([current], dtype=torch.long, device=device)).detach().cpu()
        if logits.ndim != 1 or not torch.isfinite(logits).all():
            raise ValueError("Packed executor produced nonfinite or malformed vocabulary logits")
        rows.append(dict(input_ids=current, logits=logits))
        history.append(int(logits.argmax()))
    return rows


def reload_parity(reference, actual):
    if len(reference) != 3 or len(actual) != 3:
        raise ValueError("Prompt and two generated-prefix reload checks required")
    rows = []
    for before, after in zip(reference, actual):
        if before["input_ids"] != after["input_ids"] or not torch.equal(before["logits"], after["logits"]):
            raise ValueError("Fresh packed export reload changed its own full-vocabulary inference")
        rows.append(dict(input_tokens=len(before["input_ids"]), vocabulary=before["logits"].numel(),
                         exact=True, greedy_id=int(after["logits"].argmax())))
    return dict(passed=True, fresh_structural_reload=True, full_vocabulary_exact=True, records=rows,
                scope="Packed-before-export versus reloaded packed artifact; no unpacked equality requirement")


@torch.inference_mode()
def warm_timing(model, prompt_ids, *, warmup_calls=2, repeats=5):
    if warmup_calls < 1 or repeats < 1 or not prompt_ids:
        raise ValueError("A fixed prompt, warmup and timed repetitions are required")
    device = model.embedding.device
    prefix = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    next_token = int(model.next_logits(prefix).argmax())
    extended = torch.cat((prefix, torch.tensor([[next_token]], device=device)), dim=1)
    def synchronize():
        if device.type == "cuda": torch.cuda.synchronize(device)
    rows = {}
    for name, ids in (("full_prompt_next_logit", prefix), ("extended_prefix_next_logit", extended)):
        for _ in range(warmup_calls): model.next_logits(ids)
        synchronize(); durations = []
        for _ in range(repeats):
            synchronize(); start = time.perf_counter()
            model.next_logits(ids); synchronize()
            durations.append(1000*(time.perf_counter()-start))
        rows[name] = dict(input_tokens=ids.shape[1], median_ms=statistics.median(durations),
                          samples_ms=durations, repeats=repeats)
    return dict(**rows, warmup_calls=warmup_calls, use_cache=False,
                scope="Warm end-to-end full-prefix recomputation, including guard/readout; no KV-cached decode or speedup claim")


def combine_quality(response, native_response, canary_gate, semantic_gate, mc, task_checks, limits):
    native_nll, student_nll = native_response["nll"], response["nll"]
    checks = dict(response_nll=math.isfinite(student_nll) and student_nll <= limits["maximum_native_nll_ratio"]*native_nll,
                  generation_canaries=canary_gate["passed"], semantic=semantic_gate["passed"],
                  old_multiple_choice=mc["eligibility"]["passed"], executable_tasks=task_checks["passed"])
    return dict(passed=all(checks.values()), checks=checks, native_nll=native_nll, student_nll=student_nll,
                native_mc_correct=mc["native_correct"], student_mc_correct=mc["metrics"]["correct"],
                mc_count=mc["native_count"], mc_accuracy=mc["metrics"]["accuracy"],
                allowed_accuracy_drop_pp=limits["maximum_drop_pp"], confirmed=False,
                full_logit_parity_required=False,
                scope="Packed task/generation DEV evaluation under the original native-relative screens")


def _ledger(root, p, result, model, export_receipt, elapsed):
    experiment = nal.Experiment(id=f"packed.{p['arm']}.u{p['checkpoint_metadata']['update']}.{p['fingerprint'][:10]}",
        hypothesis_id="JSPACE-QWEN3B-PACKED-RESPONSE-01", name="Packed snapshot task evaluation",
        parameters=dict(fingerprint=p["fingerprint"], checkpoint=p["checkpoint_sha256"], prefix_mode=p["prefix_mode"]), seed=20261116)
    quality = result["quality"]
    record = nal.ExperimentResult(experiment_id=experiment.id, hypothesis_id=experiment.hypothesis_id,
        metrics=dict(eligible=float(quality["passed"]), mc_accuracy=quality["mc_accuracy"],
                     response_nll=quality["student_nll"], serialized_bytes=float(export_receipt["serialized_bytes"])),
        primary_metric=quality["mc_accuracy"], model_architecture=[model.cut, len(model.stack.blocks), model.stack.width],
        model_parameters=sum(parameter.numel() for parameter in model.parameters()), training_time=0.,
        model_checkpoint=str(Path(root)/"run/export"), observations=[result["status"],
            "Inference evaluation only; no retraining or independent confirmation",
            "model_parameters covers exported compact modules; pinned original prefix is additional",
            f"Evaluation wall time {elapsed:.3f} seconds"], status=nal.ExperimentStatus.COMPLETED)
    path = nal.ExperimentResultLedger(root).save(experiment, record, attempts=1, device_id=0)
    restored = nal.ExperimentResultLedger(root).load(experiment)
    if restored is None or restored.metrics != record.metrics:
        raise ValueError("Packed evaluation NAL ledger readback differs")
    return str(path)


def run(root, *, limit_mib=5376):
    root = Path(root).resolve(); p, parent, cp, records = verify(root)
    directory = root/"run"
    if directory.exists(): raise FileExistsError(directory)
    directory.mkdir(); started = time.monotonic(); lease = None; dataset = None; phase = "reservation"
    try:
        lease, guard = cache.reservation(limit_mib); torch.set_num_threads(2)
        _progress(directory, phase="packed_construction", arm=p["arm"], update=p["checkpoint_metadata"]["update"])
        dataset = cache.ResponseDataset(p["cache_root"], cut=p["cut"])
        with execution.arithmetic_policy():
            phase = "packed_construction"
            model, construction = packed_source.build_packed_student(p["arm"], p["source_plan"],
                p["checkpoint"], device="cuda:0", guard=guard)
            identities = dict(study=p["fingerprint"], campaign=p["source_fingerprint"],
                checkpoint=p["checkpoint_sha256"], teacher=p["model"]["revision"], cache=p["cache_fingerprint"])
            phase = "packed_export"
            export_receipt = packed_source.export_packed_snapshot(model, directory/"export", identities=identities)
            write(directory/"export_receipt.json", export_receipt)
            prompt_ids = records[p["timing_index"]].prompt_ids
            phase = "packed_reload_reference"
            with packed_source.attach_prefix(model, p["model"]["path"], mode=p["prefix_mode"]):
                reference = packed_probe(model, prompt_ids)
            # All task measurements below consume the independent artifact load.
            del model; gc.collect(); torch.cuda.empty_cache(); guard()
            phase = "packed_reload"
            model, reload_receipt = packed_source.load_packed_snapshot(directory/"export", device="cuda:0", guard=guard)
            if reload_receipt["packed_state_sha256"] != export_receipt["packed_state_sha256"]:
                raise ValueError("Reloaded packed tensor identity differs from export")
            write(directory/"reload_receipt.json", reload_receipt)
            with packed_source.attach_prefix(model, p["model"]["path"], mode=p["prefix_mode"]):
                actual = packed_probe(model, prompt_ids, prefixes=[row["input_ids"] for row in reference])
                parity = reload_parity(reference, actual)
                timing = warm_timing(model, prompt_ids, warmup_calls=p["warmup_calls"], repeats=p["timing_repeats"])
            del reference, actual
            write(directory/"packed_reload_parity.json", parity)
            write(directory/"warm_timing.json", dict(timing, prefix_mode=p["prefix_mode"]))
            phase = "response_development"
            _progress(directory, phase=phase, cases=len(dataset.development))
            by_id = {record.example_id:i for i, record in enumerate(dataset.records)}
            def prepared_batch(batch):
                return dataset.batch([by_id[name] for name in batch["example_ids"]])
            teacher = training.CachedHiddenTeacher(model.embedding, teacher_id=p["model"]["revision"], guard=guard)
            response = evaluation.response_losses(model, (records[i] for i in dataset.development),
                pad_token_id=dataset.pad_token_id, max_length=416, teacher_provider=teacher,
                prepare_batch=prepared_batch, token_chunk_size=8)
            write(directory/"development.json", response)
            phase = "packed_multiple_choice"
            _progress(directory, phase=phase, cases=256)
            mc = packed_source.packed_old_mc_evaluate(p["arm"], p["source_plan"], model, guard)
            write(directory/"multiple_choice.json", mc)
            phase = "packed_generation"
            _progress(directory, phase=phase, cases=len(p["generation_indices"])+6)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(p["model"]["path"], local_files_only=True, trust_remote_code=False)
            texts = {row["example_id"]:row for row in map(json.loads,
                     (Path(cp["corpus"])/"text.jsonl").read_text().splitlines())}
            generation = []
            with packed_source.attach_prefix(model, p["model"]["path"], mode=p["prefix_mode"]) as prefix_receipt:
                generation_started = time.monotonic()
                canaries = evaluation.evaluate_canaries(model, tokenizer, device="cuda:0",
                    eos_token_ids=p["generation_eos_token_ids"], max_new_tokens=p["max_new_tokens"])
                for index in p["generation_indices"]:
                    record = records[index]; context = texts[record.example_id]
                    row = evaluation.generate_record(model, record, tokenizer, device="cuda:0",
                        source_text=context.get("context", context.get("prompt_text", "")),
                        eos_token_ids=p["generation_eos_token_ids"], max_new_tokens=p["max_new_tokens"])
                    generation.append(row)
                    _progress(directory, phase=phase, completed=len(generation), total=len(p["generation_indices"]))
                generation_seconds = time.monotonic()-generation_started
            generation_tokens = sum(len(row["output_ids"]) for row in [*canaries["records"], *generation])
            generation_receipt = dict(canaries=canaries, corpus=generation,
                execution="Packed, exported and freshly reloaded", use_cache=False,
                seconds=generation_seconds, output_tokens_including_eos=generation_tokens,
                cachefree_generation_tokens_per_second=generation_tokens/generation_seconds,
                native_prefix=prefix_receipt,
                timing_scope="Cache-free whole-prefix regeneration including readout; not KV-cached deployment decode")
            write(directory/"generation.json", generation_receipt)
            phase = "executable_task_checks"
            native_generation = read(Path(p["cache_root"])/"native/generation.json")
            task_checks = task_metrics.grade_generated_tasks(native_generation["corpus"], generation, texts,
                maximum_drop_pp=p["gates"]["maximum_drop_pp"], minimum_student_rate=p["gates"]["minimum_generation_rate"])
            write(directory/"generative_task_checks.json", task_checks)
            canary_gate = evaluation.generation_gate(native_generation["canaries"], canaries,
                maximum_drop_pp=p["gates"]["maximum_drop_pp"], minimum_student_rate=p["gates"]["minimum_generation_rate"])
            contexts = {row["example_id"]:dict(prompt=texts[row["example_id"]]["prompt_text"],
                source=texts[row["example_id"]]["context"], reference=texts[row["example_id"]]["response"],
                task=texts[row["example_id"]]["task"]) for row in generation}
            phase = "semantic_judge"
            _progress(directory, phase=phase, pairs=len(generation))
            semantic = judge.judge_pairs(native_generation["corpus"], generation, directory/"semantic_judge",
                                        config=judge.JudgeConfig(**p["judge"]), contexts=contexts)
            native_response = read(Path(p["cache_root"])/"native/development.json")
            quality = combine_quality(response, native_response, canary_gate, semantic["semantic_gate"],
                                      mc, task_checks, p["gates"])
            comparison = None if p["unpacked_report"] is None else read(p["unpacked_report"])
            result = dict(status="exploratory_quality_pass" if quality["passed"] else "quality_gate_failed",
                fingerprint=p["fingerprint"], arm=p["arm"], update=p["checkpoint_metadata"]["update"],
                nll=response["nll"], teacher_kl=response["kl"],
                semantic_task_success=semantic["semantic_gate"].get("rates", {}).get("task_success", {}).get("student"),
                eligibility=quality, checkpoint={"path":p["checkpoint"], "sha256":p["checkpoint_sha256"]},
                packed_artifact_path=str(directory/"export"),
                checkpoint_sha256=p["checkpoint_sha256"], quality=quality, canary_gate=canary_gate,
                semantic_gate=semantic["semantic_gate"], generative_task_gates=task_checks["gates"],
                multiple_choice_eligibility=mc["eligibility"], packed_export=export_receipt,
                packed_reload=reload_receipt, packed_reload_parity=parity, construction=construction,
                warm_timing=timing, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                prefix_mode=p["prefix_mode"], lease_id=lease["id"], allocation=lease["allocation"],
                allocator_mib=limit_mib, device_name=torch.cuda.get_device_name(0),
                device_capability=list(torch.cuda.get_device_capability(0)),
                elapsed_seconds=time.monotonic()-started, contextual_unpacked_report=comparison,
                contextual_unpacked_can_rescue_failure=False, full_logit_parity_gate=False,
                fresh_process_confirmation_pending=True, confirmed=False, promoted=False)
            result["nal_ledger"] = _ledger(root, p, result, model, export_receipt, result["elapsed_seconds"])
            write(directory/"result.json", result)
            small_files = [path for path in directory.iterdir() if path.is_file() and path.suffix==".json"]
            write(directory/"manifest.json", dict(status="completed", fingerprint=p["fingerprint"],
                artifacts={path.name:sha(path) for path in small_files}, packed_files=export_receipt["files"]))
            return result
    except BaseException as error:
        write(directory/"failure.json", dict(status="failed", fingerprint=p["fingerprint"], phase=phase,
            error_type=type(error).__name__, error=str(error), lease=lease, prefix_mode=p["prefix_mode"],
            allocator_mib=limit_mib, elapsed_seconds=time.monotonic()-started,
            quality_claim=False, artifacts_preserved=True, automatic_fallback=False))
        raise
    finally:
        if dataset is not None: dataset.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--unpacked-report", type=Path)
    parser.add_argument("--prefix-mode", choices=("resident", "cpu"), default="resident")
    parser.add_argument("--limit-mib", type=int, default=5376)
    args = parser.parse_args()
    if args.phase=="prepare":
        if args.campaign_root is None or args.checkpoint is None:
            parser.error("prepare requires --campaign-root and --checkpoint")
        result = prepare(args.root, args.campaign_root, args.checkpoint,
                         prefix_mode=args.prefix_mode, unpacked_report=args.unpacked_report)
    elif args.phase=="run":
        result = run(args.root, limit_mib=args.limit_mib)
    else:
        p, _, _, _ = verify(args.root)
        result = dict(status="prepared", fingerprint=p["fingerprint"], update=p["checkpoint_metadata"]["update"])
    print(json.dumps({key:result[key] for key in ("status", "fingerprint", "arm", "update") if key in result},
                     allow_nan=False), flush=True)


if __name__ == "__main__": main()
