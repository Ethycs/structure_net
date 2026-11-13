"""Actual prefix removal and synchronized bounded-task latency measurement."""
import gc
import json
import statistics
import time

import torch

from .jspace_active_math import flatten_state, head_scores, move, tensor_bytes
from .jspace_relation_capture import _prompt
from .jspace_relation_study import file_hash, save_tensor, write_json
from .jspace_relation_tasks import COLOURS
from .jspace_runtime import generate_answer


def truncate_model(model, layer):
    if not 0 <= layer < len(model.model.layers)-1:
        raise ValueError("A proper prefix is required")
    model.model.layers = torch.nn.ModuleList(list(model.model.layers[:layer+1]))
    model.model.norm = torch.nn.Identity()  # Training features are pre-final-norm.
    model.lm_head = torch.nn.Identity()
    model.config.num_hidden_layers = layer+1
    return model


def prefix_answer(model, tokenizer, prompt, head, scope):
    encoded = tokenizer(prompt, return_tensors="pt")
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        output = model.model(input_ids=encoded["input_ids"].to(device),
                             attention_mask=encoded["attention_mask"].to(device), use_cache=False)
        values = flatten_state(output.last_hidden_state[0], scope).float()[None, :]
        return int(head_scores(head, values).argmax(1).item())


def timed(function, prompts):
    for prompt in prompts[:4]:
        function(prompt)
    durations, answers = [], []
    for repeat in range(5):
        for index, prompt in enumerate(prompts):
            torch.cuda.synchronize()
            started = time.perf_counter()
            answer = function(prompt)
            torch.cuda.synchronize()
            durations.append(time.perf_counter()-started)
            answers.append({"repeat": repeat, "index": index, "answer": answer})
    return {"median_seconds": statistics.median(durations), "mean_seconds": statistics.mean(durations),
            "calls": len(durations), "seconds": durations, "answers": answers}


def benchmark(root):
    from .jspace_active_study import Stage, CONFIG, checked_manifest
    checked_manifest(root, "evaluation")
    analysis = json.loads((root/"analysis.json").read_text())
    if not analysis["candidate_quality_gate"]:
        raise ValueError("Do not benchmark a quality-rejected candidate")
    if analysis["fit_sha256"] != file_hash(root/"fit.pt"):
        raise ValueError("Frozen head changed")
    stage = Stage(root, "benchmark")
    selection = analysis["benchmark_candidate"]
    key, name = str(selection["layer"]), selection["head"]
    head = torch.load(root/"fit.pt", map_location="cpu", weights_only=True)["heads"][key][name]
    pairs = stage.tasks["pairs"]["evaluation"][:32]
    prompts = [_prompt(stage.tokenizer, pair["source"]) for pair in pairs]
    teacher = timed(lambda p: generate_answer(stage.model, stage.tokenizer, p, max_new_tokens=8)["answer"], prompts)
    stage.event("teacher_benchmark", {"calls": teacher["calls"]})
    teacher_peak = torch.cuda.max_memory_reserved()
    if teacher_peak > sum(stage.lease["allocation"].values())*1024**2:
        raise RuntimeError("Teacher exceeded reserved memory")
    full_parameters = sum(p.numel() for p in stage.model.parameters())
    truncate_model(stage.model, int(key))
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device_head = move(head, "cuda:0")
    prefix = timed(lambda p: prefix_answer(stage.model, stage.tokenizer, p, device_head, CONFIG["sites"][key]), prompts)
    expected = analysis["head_predictions"][key][name][:len(pairs)]
    if any(row["answer"] != expected[row["index"]] for row in prefix["answers"]):
        raise ValueError("Actual shortened model disagrees with the frozen captured-feature predictions")
    stage.event("prefix_benchmark", {"calls": prefix["calls"]})
    remaining_parameters = sum(p.numel() for p in stage.model.parameters())
    save_tensor(stage.path/"export_head.pt", head)
    result = {"candidate": selection, "teacher": teacher, "prefix": prefix,
              "median_task_latency_speedup": teacher["median_seconds"]/prefix["median_seconds"],
              "full_backbone_parameters": full_parameters, "remaining_backbone_parameters": remaining_parameters,
              "removed_backbone_parameters": full_parameters-remaining_parameters,
              "head_numeric_bytes": tensor_bytes(head), "export_head_file_bytes": (stage.path/"export_head.pt").stat().st_size,
              "teacher_peak_reserved_bytes": teacher_peak, "prefix_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "gpu_predictions_match_frozen_head": True, "batch_size": 1, "warmup_prompts_per_arm": 4, "repetitions": 5,
              "scope": "four-colour lookup classification; teacher completes one answer, prefix returns one label; not general token throughput",
              "limitations": ["one_device_one_run", "sequential_teacher_then_prefix_timing", "warm_resident_weights", "32_reused_evaluation_prompts"]}
    write_json(stage.path/"result.json", result)
    write_json(stage.path/"deployment.json", {"base_checkpoint": stage.plan["model"], "prefix_last_block": int(key),
                "head": name, "head_sha256": file_hash(stage.path/"export_head.pt"), "labels": list(COLOURS),
                "scope": "specialized four-colour classifier; fixed JSON and 21-token question suffix",
                "loader": "jspace_active_benchmark.truncate_model + prefix_answer; no deployed service"})
    return stage.finish("quality_qualified_prefix_benchmarked", measured=result)
