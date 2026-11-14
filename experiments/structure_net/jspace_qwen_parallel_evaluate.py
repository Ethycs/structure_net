"""Final held-out quality, deployed size, and paired runtime for all-block Qwen.

No checkpoint selection occurs here. GPU execution requires the campaign lease;
native and compact models are loaded consecutively, never together. The primary
runtime includes dynamic prefill, cache transfer, and 32 generated tokens, with
only the autoregressive decode step captured in a CUDA graph.
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .jspace_qwen_parallel_model import build_teacher, state_dict_cpu


def exact_binomial_p(wins, losses):
    """Two-sided exact paired sign/McNemar test, conditional on discordance."""
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (wins, losses)):
        raise ValueError("Discordance counts must be nonnegative integers")
    n = wins + losses
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)


def paired_quality(native, compact):
    """Paired accuracy counts; family loss gates use net loss, not discordance."""
    if not native or len(native) != len(compact):
        raise ValueError("Expected nonempty paired evaluation")
    identities = [r["identity"] for r in native]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate evaluation identity")
    for a, b in zip(native, compact):
        if any(a[k] != b[k] for k in ("identity", "family", "answer")):
            raise ValueError("Unpaired evaluation identity, family, or answer")

    def summarize(pairs):
        n = len(pairs)
        native_correct = sum(a["prediction"] == a["answer"] for a, _ in pairs)
        compact_correct = sum(b["prediction"] == b["answer"] for _, b in pairs)
        wins = sum(a["prediction"] != a["answer"] and b["prediction"] == b["answer"] for a, b in pairs)
        losses = sum(a["prediction"] == a["answer"] and b["prediction"] != b["answer"] for a, b in pairs)
        return dict(count=n, native_correct=native_correct, compact_correct=compact_correct,
                    native_accuracy=native_correct/n, compact_accuracy=compact_correct/n,
                    accuracy_drop=(native_correct-compact_correct)/n,
                    net_lost_correct=native_correct-compact_correct, wins=wins, losses=losses,
                    agreement=sum(a["prediction"] == b["prediction"] for a, b in pairs)/n,
                    exact_binomial_p=exact_binomial_p(wins, losses))

    pairs = list(zip(native, compact))
    return {"overall": summarize(pairs), "by_family": {
        family: summarize([(a, b) for a, b in pairs if a["family"] == family])
        for family in sorted({r["family"] for r in native})}}


def acceptance_gates(quality, native_bytes, compact_bytes, native_before, compact_rate,
                     native_after, *, reload_exact, graph_static_exact, config):
    values = (native_bytes, compact_bytes, native_before, compact_rate, native_after)
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("State sizes and measured rates must be finite and positive")
    before_after_mean = (native_before + native_after)/2
    speedup = compact_rate/before_after_mean
    reduction = 1-compact_bytes/native_bytes
    drop = quality["overall"]["accuracy_drop"]
    family_drops = [v["accuracy_drop"] for v in quality["by_family"].values()]
    if not family_drops or not all(math.isfinite(v) for v in [drop, *family_drops]):
        raise ValueError("Invalid quality measurements")
    gates = {
        "overall_quality": drop <= config["maximum_accuracy_drop"] + 1e-12,
        "every_family_quality": all(v <= config["maximum_family_drop"] + 1e-12 for v in family_drops),
        "deployed_size": reduction + 1e-12 >= config["minimum_size_reduction"],
        "whole_model_speed": speedup + 1e-12 >= config["minimum_speedup"],
        "packed_reload_exact": bool(reload_exact),
        "graph_static_tokens_exact": bool(graph_static_exact),
    }
    return dict(passed=all(gates.values()), gates=gates, measured_size_reduction=reduction,
                whole_model_speedup=speedup, native_baseline_tokens_per_second=before_after_mean,
                native_after_over_before=native_after/native_before,
                baseline_aggregation="arithmetic mean of native before/after wall-clock token rates",
                thresholds={k: config[k] for k in ("maximum_accuracy_drop", "maximum_family_drop",
                            "minimum_size_reduction", "minimum_speedup")})


def tensor_bytes(model):
    """Resident parameters/buffers, deduplicated across the two collections too."""
    seen, total = set(), 0
    for value in (*model.parameters(), *model.buffers()):
        if id(value) not in seen:
            seen.add(id(value))
            total += value.numel()*value.element_size()
    return total


def evaluation_cases(tasks):
    rows = tasks["splits"]["evaluation"]
    families = {r["family"] for r in rows}
    if len(rows) != 512 or len(families) != 4 or any(sum(r["family"] == f for r in rows) != 128 for f in families):
        raise ValueError("Final evaluation requires 512 rows, 128 per each of four families")
    identities = [r["identity"] for r in rows]
    excluded = set(tasks.get("previous_identities", []))
    for name, split in tasks["splits"].items():
        if name != "evaluation":
            excluded.update(r["identity"] for r in split)
    if len(set(identities)) != len(rows) or set(identities) & excluded:
        raise ValueError("Fresh evaluation has duplicate or previously used identities")
    return rows


def runtime_cases(tasks):
    rows = tasks["splits"]["development"]
    selected = []
    for family in sorted({r["family"] for r in tasks["splits"]["evaluation"]}):
        candidates = [r for r in rows if r["family"] == family][:2]
        if len(candidates) != 2:
            raise ValueError("Runtime requires two development prompts per family")
        selected.extend(candidates)
    if len(selected) != 8 or any(len(r["input_ids"])+32 > 416 for r in selected):
        raise ValueError("Runtime prompts violate eight-prompt/416-position protocol")
    return selected


@torch.inference_mode()
def score_cases(model, cases, label_ids, *, guard=lambda: None, event=lambda **kw: None):
    records = []
    device = model.lm_head.weight.device
    for index, case in enumerate(cases):
        guard()
        ids = torch.tensor([case["input_ids"]], device=device)
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
        count = len(case["choices"])
        if not 0 < count <= len(label_ids):
            raise ValueError("Invalid number of answer choices")
        logits = F.linear(hidden, model.lm_head.weight[label_ids[:count]])[0].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        records.append(dict(identity=case["identity"], family=case["family"], answer=case["answer"],
                            prediction=int(logits.argmax()), logits=logits.cpu().tolist()))
        if index % 16 == 15 or index+1 == len(cases):
            event(completed=index+1, total=len(cases))
    return records


def _timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = call()
    torch.cuda.synchronize()
    return output, time.perf_counter()-start


@torch.inference_mode()
def runtime(model, tokenizer, cases, *, seed, guard=lambda: None):
    from .jspace_qwen_affine import graph_decoder

    decoder = graph_decoder(model)
    setup_seconds = decoder.capture()
    inputs = [torch.tensor([c["input_ids"]], device=model.lm_head.weight.device) for c in cases]
    static_outputs, graph_outputs = [], []
    for ids in inputs:
        guard()
        eager = decoder.generate(ids, False)
        captured = decoder.generate(ids, True)
        if not torch.equal(eager, captured):
            raise ValueError("CUDA graph and uncaptured static-cache tokens differ")
        static_outputs.append(eager.cpu().tolist())
        graph_outputs.append(captured.cpu().tolist())

    def hf_generate(ids):
        return model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, use_cache=True,
            min_new_tokens=32, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)[0, -32:]

    rows, outputs = [], {"eager_static": static_outputs, "cuda_graph": graph_outputs}
    for arm in ("cuda_graph", "eager_static", "hf_eager"):
        generate = hf_generate if arm == "hf_eager" else lambda ids: decoder.generate(ids, arm == "cuda_graph")
        generate(inputs[0]); torch.cuda.synchronize()
        arm_outputs = [None]*len(cases)
        for repeat in range(3 if arm == "cuda_graph" else 1):
            order = list(range(len(cases))); random.Random(seed+repeat).shuffle(order)
            for index in order:
                guard()
                tokens, seconds = _timed(lambda: generate(inputs[index]))
                if tokens.numel() != 32:
                    raise ValueError("Generation did not return exactly 32 tokens")
                rows.append(dict(arm=arm, repeat=repeat, identity=cases[index]["identity"], tokens=32, seconds=seconds))
                if repeat == 0:
                    arm_outputs[index] = tokens.cpu().tolist()
        outputs[arm] = arm_outputs

    # This prefill is deliberately eager in BOTH graph and static controls.
    # A separate prefill graph would change the experiment, so do not imply one.
    prefill = []
    for repeat in range(3):
        for index, ids in enumerate(inputs):
            guard()
            _, seconds = _timed(lambda: decoder.prefill(ids))
            prefill.append(dict(repeat=repeat, identity=cases[index]["identity"],
                                input_tokens=ids.numel(), seconds=seconds))
    rates = {}
    for arm in ("cuda_graph", "eager_static", "hf_eager"):
        samples = [r for r in rows if r["arm"] == arm]
        seconds = sum(r["seconds"] for r in samples)
        rates[arm] = dict(tokens_per_second=sum(r["tokens"] for r in samples)/seconds,
                          total_seconds=seconds, samples=samples)
    value = dict(timings=rates, graph_capture_seconds=setup_seconds,
                 graph_static_exact=True, generated_tokens=outputs,
                 hf_graph_tokens_equal=[a == b for a, b in zip(outputs["hf_eager"], outputs["cuda_graph"])],
                 prefill=dict(mode="shared eager dynamic prefill including reset/copy into static cache",
                              input_tokens_per_second=sum(r["input_tokens"] for r in prefill)/sum(r["seconds"] for r in prefill),
                              samples=prefill),
                 scope="batch 1; eight fixed development prompts; 32 tokens including prefill; three graph repeats; capture setup excluded")
    del decoder
    gc.collect(); torch.cuda.empty_cache()
    return value


def export_packed(path, model, *, fingerprint):
    stack = model.model.layers[-1]
    if len(model.model.layers) != 1 or not stack.packed or not stack.stripped or len(stack.layer_indices) != 36:
        raise ValueError("Export requires the complete packed, stripped 36-block stack")
    payload = dict(schema="jspace.qwen-allblocks-packed.v1", fingerprint=fingerprint,
                   stack=state_dict_cpu(stack), norm=state_dict_cpu(model.model.norm))
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


def reload_packed(path, model, *, fingerprint):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload["schema"] != "jspace.qwen-allblocks-packed.v1" or payload["fingerprint"] != fingerprint:
        raise ValueError("Packed export provenance mismatch")
    model.model.layers[-1].load_state_dict(payload["stack"], strict=True)
    model.model.norm.load_state_dict(payload["norm"], strict=True)


def evaluate(root, *, setup=None):
    """Run inside a reservation, or accept the already-guarded gpu_setup tuple."""
    from . import jspace_qwen_parallel as campaign

    root = Path(root)
    plan, lease, guard = campaign.gpu_setup(root) if setup is None else setup
    if setup is not None and campaign.verify(root, model=True) != plan:
        raise ValueError("Provided setup plan does not match the frozen campaign")
    if plan["systems_only"]:
        raise ValueError("A systems-only root cannot produce research acceptance gates")
    guard()
    dependencies = {}
    for index in range(6):
        group = campaign.group_path(root, index)
        for phase in ("basis", "recovery"):
            campaign.checked(root, group/phase)
            dependencies[str((group/phase/"manifest.json").relative_to(root))] = campaign.sha(group/phase/"manifest.json")
        recovery = group/"recovered_unpacked.pt"
        recovery_sha = campaign.sha(recovery)
        if recovery_sha != campaign.sha(group/"recovery/recovered_unpacked.pt"):
            raise ValueError("Accepted group checkpoint differs from its recovery artifact")
        dependencies[str(recovery.relative_to(root))] = recovery_sha
        for layer in campaign.group_layers(index):
            block = group/f"block_{layer:02d}"
            campaign.checked(root, block/"fit_result")
            dependencies[str((block/"fit_result/manifest.json").relative_to(root))] = campaign.sha(block/"fit_result/manifest.json")
            for name in ("fit.pt", "relation.pt"):
                dependencies[str((block/name).relative_to(root))] = campaign.sha(block/name)
    tasks = json.loads((root/"tasks.json").read_text())
    cases, prompts = evaluation_cases(tasks), runtime_cases(tasks)
    path = root/"evaluation"
    if (root/"campaign_results.json").exists():
        raise FileExistsError(root/"campaign_results.json")
    path.mkdir()
    campaign.write(path/"inputs.json", dict(fingerprint=plan["fingerprint"], artifacts=dependencies,
                                           task_sha256=campaign.sha(root/"tasks.json")))

    def log(kind, **values):
        campaign.event(path/"events.jsonl", kind, **values)

    results = {}
    for tag, previous in (("native_before", None), ("compact", 5), ("native_after", None)):
        guard()
        model, tokenizer = build_teacher(root, previous)
        size = tensor_bytes(model)
        if tag != "native_after":
            records = score_cases(model, cases, plan["label_ids"], guard=guard,
                                  event=lambda **kw: log("score", arm=tag, **kw))
            campaign.write(path/f"{tag}_fresh.json", records)
        timing = runtime(model, tokenizer, prompts, seed=plan["config"]["seed"], guard=guard)
        campaign.write(path/f"{tag}_timing.json", timing)
        results[tag] = dict(tensor_bytes=size, runtime=timing)
        if tag == "compact":
            export_packed(path/"packed.pt", model, fingerprint=plan["fingerprint"])
            log("packed_export", bytes=(path/"packed.pt").stat().st_size)
        log("arm_complete", arm=tag, tensor_bytes=size,
            graph_tokens_per_second=timing["timings"]["cuda_graph"]["tokens_per_second"])
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()

    # Fresh construction, then strict packed-state load; no original GPU model
    # remains resident and the complete evaluation is repeated independently.
    model, tokenizer = build_teacher(root, 5)
    reload_packed(path/"packed.pt", model, fingerprint=plan["fingerprint"])
    reloaded = score_cases(model, cases, plan["label_ids"], guard=guard,
                           event=lambda **kw: log("reload_score", **kw))
    campaign.write(path/"reloaded_fresh.json", reloaded)
    native = json.loads((path/"native_before_fresh.json").read_text())
    compact = json.loads((path/"compact_fresh.json").read_text())
    reload_exact = reloaded == compact and tensor_bytes(model) == results["compact"]["tensor_bytes"]
    if results["native_before"]["tensor_bytes"] != results["native_after"]["tensor_bytes"]:
        raise ValueError("Native model state size changed between controls")
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    quality = paired_quality(native, compact)
    rate = lambda tag: results[tag]["runtime"]["timings"]["cuda_graph"]["tokens_per_second"]
    gates = acceptance_gates(quality, results["native_before"]["tensor_bytes"], results["compact"]["tensor_bytes"],
        rate("native_before"), rate("compact"), rate("native_after"), reload_exact=reload_exact,
        graph_static_exact=all(r["runtime"]["graph_static_exact"] for r in results.values()), config=plan["config"])
    guard()
    peak = torch.cuda.max_memory_reserved()
    if peak > sum(lease["allocation"].values())*1024**2:
        raise RuntimeError("Evaluation exceeded GPU reservation")
    result = dict(study_id=campaign.STUDY_ID, fingerprint=plan["fingerprint"], status="completed",
        confirmed=False, targets_met=gates["passed"],
        classification="all_targets_met_on_restricted_benchmark" if gates["passed"] else "all_block_targets_not_jointly_met",
        scope=plan["config"]["scope"], evaluation_count=len(cases),
        quality=quality, acceptance=gates, results=results, packed_reload_exact=reload_exact,
        packed_sha256=campaign.sha(path/"packed.pt"), packed_file_bytes=(path/"packed.pt").stat().st_size,
        peak_reserved_bytes=peak, lease_id=lease["id"],
        limitations="One fixed exploratory checkpoint; exact binomial p values describe paired discordances, not non-inferiority proof. Resident tensor size excludes serving caches and allocator overhead.")
    campaign.write(path/"result.json", result)
    campaign.complete(root, path, plan, confirmed=False, targets_met=gates["passed"], lease_id=lease["id"],
                      peak_reserved_bytes=peak, packed_reload_exact=reload_exact)
    campaign.write(root/"campaign_results.json", result)
    return result
