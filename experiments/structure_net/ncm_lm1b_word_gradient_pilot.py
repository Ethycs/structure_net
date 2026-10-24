"""Representative CUDA lifecycle for a word-loss gradient across hard NCM codes."""
import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_lm1b_grounded_training as training
from experiments.structure_net import ncm_lm1b_word_gradient as objective
from experiments.structure_net.ncm_lm1b_decoder_repair import same
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, head, q, ROOT = training.native, training.head, training.q, training.ROOT
HYPOTHESIS = "ncm-lm1b-word-gradient-lifecycle-v1"
DEST = ROOT / "20261024_lm1b_word_gradient_cuda_pilot_v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-word-gradient-pilot-plan.md")
TEST = Path("tests/structure_net/test_ncm_lm1b_word_gradient.py")
BASE_SHA = "bb239a482db67d67d2d7f410588fada7b6782c42474eb9e8d1dffdd8bb7d1490"
BASE_ATTENTION = "38af3e583bde80a09981556860a282e7133cb8d20a7c1f9faffdd15f31c6307d"
GROUPS = ("concepts.source_memory.", "concepts.adapters.", "word_memory.", "word_adapters.")


def optimizer(model):
    return torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-4,
        betas=(.9, .95), eps=1e-8, weight_decay=.01)


def gradients(model):
    return {n: None if v.grad is None else v.grad.detach().cpu().clone()
        for n, v in model.named_parameters() if v.requires_grad}


def active_groups(values):
    return {prefix: any(n.startswith(prefix) and v is not None and bool(v.abs().sum() > 0)
        for n, v in values.items()) for prefix in GROUPS}


def frozen_grads_absent(model):
    return all(v.grad is None for module in (model.codec, model.concepts.generator) for v in module.parameters())


def frozen_states(model):
    return {"generator": native.vendor.module_hash(model.concepts.generator), "codec": native.vendor.module_hash(model.codec)}


@torch.no_grad()
def probes(model, rows, features):
    was_training = model.training
    model.eval()
    output = {}
    try:
        for row in rows:
            value = features[row["id"]].unsqueeze(0).cuda()
            mask = torch.ones(value.shape[:2], dtype=torch.bool, device="cuda")
            output[row["id"]] = {k: v.cpu() for k, v in model.generate(value, mask).items()}
    finally:
        model.train(was_training)
    return output


def run():
    training.require_gpu()
    if file_hash(training.TRAIN / "model.pt") != BASE_SHA:
        raise ValueError("exact final4096 warm start required")
    for root in (training.DATA, training.CACHE, training.TRAIN):
        native.vendor.terminal(root)
    if q.read(training.WORKFLOW / "tool_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("actual completed full-training tool required")
    population, manifest = q.read(training.DATA / "cases.json"), q.read(training.DATA / "manifest.json")
    training.disjoint(population, q.read(training.DEV))
    ids = q.read(training.DATA / "selection.json")["pilot_ids"]
    rows = [next(r for r in population if r["id"] == identity) for identity in ids]
    codes = {r["id"]: r["codes"] for r in q.read(training.CACHE / "outputs.json")}
    sources = [Path(__file__), PLAN, TEST, Path(objective.__file__), Path(training.__file__),
        Path(head.__file__), Path(head.concepts.__file__), Path(training.objective.__file__),
        *sorted(native.vendor.VENDOR.rglob("*.py"))]
    inputs = [training.DATA / n for n in ("cases.json", "manifest.json", "selection.json", "result.json")]
    inputs += [training.DEV, training.CACHE / "outputs.json", training.TRAIN / "model.pt", training.TRAIN / "result.json",
        training.WORKFLOW / "tool_terminal_receipt.json", *[Path(manifest[r["id"]]["path"]) for r in rows], *sources]
    config = {"hypothesis_id": HYPOTHESIS, "schema": "nal.ncm-lm1b-word-gradient-pilot.v1",
        "systems_only_not_task_quality": True, "base_checkpoint_sha256": BASE_SHA, "probes": ids,
        "gradient_diagnostic_microbatch": 2, "training_microbatch": 4, "updates": 2,
        "neighborhood_seed": 105142, "dropout_seed": 104142, "surrogate_temperature": 1.,
        "auxiliary_concept_weight": .25, "auxiliary_word_weight": 1.,
        "hard_forward_exact": True, "backward_is_biased_surrogate": True,
        "precision": "FP32diagnostic;FP32weights/BF16training;FP32source-onlyhistory", "TF32": False,
        "peak_limit_bytes": 7_500_000_000, "new_parameters": 0, "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False}
    p = native.vendor.freeze(DEST, "word_gradient_hard_forward_and_reload_systems_only", inputs, config)
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(sources):
        p["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / f"word_gradient_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", p)
    features = training.load_features(rows, manifest)
    f, m, _, words = training.batch(rows, [0, 1], features, codes)
    model = head.from_domain(device="cuda")
    base = torch.load(training.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model.load_attention_state(base["attention"])
    if training.attention_hash(model) != BASE_ATTENTION or sum(v.numel() for v in model.parameters() if v.requires_grad) != 11796480:
        raise ValueError("exact warm attention state and unchanged trainable parameter set required")
    del base
    frozen = frozen_states(model)
    if frozen != {"generator": "60316ea14c56e6276be581112a2a1d81437e3ad8bb4ea3ddf8a98b946805e26a", "codec": head.DOMAIN_CODEC_HASH}:
        raise ValueError("frozen pretrained native states differ")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        history = model.concepts.generate_codes(f, m)
    hard = model.word_logits(f, m, history)
    hard_cpu = hard.detach().cpu().clone()
    F.cross_entropy(hard.flatten(0, 1), words.flatten(), ignore_index=0).backward()
    hard_grad = gradients(model)
    if any(active_groups(hard_grad)[name] for name in GROUPS[:2]) or not all(active_groups(hard_grad)[name] for name in GROUPS[2:]):
        raise ValueError("hard discrete baseline must have word-only attention gradients")
    if not frozen_grads_absent(model):
        raise ValueError("old frozen states received baseline gradients")
    model.zero_grad(set_to_none=True)
    del hard
    bridged, concept_logits = objective.word_logits(model, f, m, history)
    bridged_cpu = bridged.detach().cpu().clone()
    if not torch.equal(hard_cpu, bridged_cpu):
        raise ValueError("surrogate changed native hard forward word logits")
    F.cross_entropy(bridged.flatten(0, 1), words.flatten(), ignore_index=0).backward()
    bridge_grad = gradients(model)
    if not all(active_groups(bridge_grad).values()) or not frozen_grads_absent(model):
        raise ValueError("word-only loss must reach both new source-attention stages, no old weights")
    if any(v is not None and not torch.isfinite(v).all() for v in bridge_grad.values()):
        raise ValueError("nonfinite surrogate gradients")
    if training.attention_hash(model) != BASE_ATTENTION or frozen_states(model) != frozen:
        raise ValueError("gradient diagnostic changed model weights")
    torch.save({"hard_word_logits": hard_cpu, "bridged_word_logits": bridged_cpu,
        "source_only_codes": history.cpu()}, DEST / "forward_parity.pt")
    torch.save({"hard": hard_grad, "bridged": bridge_grad}, DEST / "word_only_gradients.pt")
    write_json(DEST / "gradient_proof.json", {"hard_active_groups": active_groups(hard_grad),
        "bridged_active_groups": active_groups(bridge_grad), "word_logits_bitwise_equal": True,
        "old_frozen_gradients_absent": True, "diagnostic_optimizer_steps": 0})
    model.zero_grad(set_to_none=True)
    del bridged, concept_logits, history, hard_grad, bridge_grad, hard_cpu, bridged_cpu
    f, m, c, words = training.batch(rows, [0, 1, 0, 1], features, codes)
    model.train()
    opt = optimizer(model)
    neighbor = torch.Generator().manual_seed(105142)
    torch.manual_seed(104142)
    torch.cuda.manual_seed_all(104142)
    trace = []
    for step in (1, 2):
        opt.zero_grad(set_to_none=True)
        losses = objective.backward_microbatch(model, f, m, c, words, neighbor, 1024)
        active = active_groups(gradients(model))
        if not all(active.values()) or not frozen_grads_absent(model):
            raise ValueError("full combined objective gradient routing invalid")
        norm = torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1.)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite combined word-gradient update")
        opt.step()
        trace.append({"step": step, "losses": losses, "gradient_norm": float(norm), "active_groups": active})
    expected = probes(model, rows, features)
    final_attention = training.attention_hash(model)
    if final_attention == BASE_ATTENTION or frozen_states(model) != frozen:
        raise ValueError("new attention must update while native states stay exact")
    state = {"attention": model.attention_state(), "optimizer": opt.state_dict(), "CPU_rng": torch.get_rng_state(),
        "CUDA_rng": torch.cuda.get_rng_state_all(), "neighbor_rng": neighbor.get_state(), "updates": 2,
        "producer_fingerprint": p["fingerprint"]}
    torch.save(state, DEST / "model.pt")
    model.close()
    del state, model, opt, losses
    gc.collect()
    torch.cuda.empty_cache()
    saved = torch.load(DEST / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model = head.from_domain(device="cuda")
    model.load_attention_state(saved["attention"])
    opt = optimizer(model)
    opt.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["CPU_rng"])
    torch.cuda.set_rng_state_all(saved["CUDA_rng"])
    neighbor.set_state(saved["neighbor_rng"])
    if (not same(probes(model, rows, features), expected) or not same(opt.state_dict(), saved["optimizer"])
            or not torch.equal(torch.get_rng_state(), saved["CPU_rng"])
            or not same(torch.cuda.get_rng_state_all(), saved["CUDA_rng"])
            or not torch.equal(neighbor.get_state(), saved["neighbor_rng"])
            or training.attention_hash(model) != final_attention or frozen_states(model) != frozen
            or model._word_bound is not None or model.concepts._bound is not None):
        raise ValueError("fresh hard-generation/optimizer/allRNG/frozen-state reload failed")
    peak = torch.cuda.max_memory_allocated()
    if peak > 7_500_000_000:
        raise ValueError("representative word-gradient lifecycle exceeds7.5GB")
    native.vendor.verify(p)
    write_json(DEST / "trace.json", trace)
    torch.save(expected, DEST / "source_only_reload_outputs.pt")
    result = {"status": "systems_word_gradient_hard_forward_and_reload_passed_not_task_quality",
        "updates": 2, "training_microbatch": 4, "source_ids": ids, "new_trainable_parameters": 11796480,
        "baseline_word_loss_concept_gradient_absent": True, "bridged_word_loss_concept_gradient_nonzero": True,
        "hard_forward_word_logits_bitwise_equal": True, "old_frozen_states_unchanged": frozen,
        "new_attention_updated": True, "initial_attention_sha256": BASE_ATTENTION, "final_attention_sha256": final_attention,
        "exact_hard_generation_optimizer_and_allRNG_reload": True, "peak_allocated_bytes": peak,
        "checkpoint_sha256": file_hash(DEST / "model.pt"), "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(DEST / "result.json", result)
    model.close()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.parse_args()
    run()
