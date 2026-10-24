"""Two-update native-NCM CUDA lifecycle; original Llama body stays unloaded."""
import gc
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import torch

from experiments.structure_net import ncm_lm1b_grounded_training as training
from experiments.structure_net import ncm_lm1b_native_finetuning as mechanism
from experiments.structure_net import ncm_lm1b_word_gradient as objective
from experiments.structure_net.ncm_lm1b_decoder_repair import same
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, head, q, ROOT = training.native, training.head, training.q, training.ROOT
HYPOTHESIS = "ncm-lm1b-native-finetuning-lifecycle-v2"
DEST = ROOT / "20261024_lm1b_native_finetuning_cuda_pilot_v2"
WARM = ROOT / "20261024_lm1b_word_gradient_training_train_v1"
WARM_WORKFLOW = ROOT / "20261024_lm1b_word_gradient_training_workflow_v1"
WARM_SHA = "c1f1b7c332bc5328b4269498e7d023aaeef731ef8731db2713a34787a91a01a7"
WARM_MODEL = "682f0fe40bceb8c1229784e85a9a1bf3814a3bb9534999e8e15471ac30896997"
WARM_ATTENTION = "da4c0878a44c591c6216bdef6826f68213883377eb78fdd4c6e6cb06e3c1bbd0"
DEV_OUTPUTS = ROOT / "20261024_lm1b_word_gradient_development_outputs_v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-native-finetuning-pilot-v2-plan.md")
AUDITOR = Path("experiments/neural_architecture_lab/finalize_ncm_lm1b_native_finetuning_pilot_v2.py")
ENCODER = "80d53f9dc37c428e5a4f6ea863374a219d1c9864c912dca8a18a2e1d5ac6b2c2"
VQ = "34e1825b089e1239f99debe8461db38574be9e4ea93dc568b9e95bd921594ddf"
GROUPS = ("concepts.generator.", "concepts.codec.decoder.", "concepts.source_memory.",
          "concepts.adapters.", "word_memory.", "word_adapters.", "concepts.generator.l0_queries")


def stepwise_logits(model, features, mask, codes):
    """TRAIN-only consistency probe: force the same histories as joint forward."""
    generator = model.concepts.generator
    existed = "_sample_level" in generator.__dict__
    original = generator.__dict__.get("_sample_level")
    recorded, offset = [], 0

    def forced(instance, logits, temperature, top_k, top_p):
        nonlocal offset
        index = len(recorded)
        if index >= len(generator.levels) or logits.shape[:2] != (len(codes), generator.levels[index]):
            raise ValueError("exact native sequential level schedule required")
        recorded.append(logits.detach().cpu().clone())
        width = generator.levels[index]
        result = codes[:, offset:offset + width].clone()
        offset += width
        return result

    generator._sample_level = MethodType(forced, generator)
    try:
        with torch.no_grad():
            forced_codes = model.concepts.generate_codes(features, mask)
        if offset != sum(generator.levels) or not torch.equal(forced_codes, codes):
            raise ValueError("complete unchanged TRAIN-only forced history required")
        return torch.cat(recorded, dim=1), forced_codes.cpu()
    finally:
        if existed:
            generator._sample_level = original
        else:
            del generator._sample_level


def protected(model):
    return {"encoder": native.vendor.module_hash(model.codec.encoder), "VQ": native.vendor.module_hash(model.codec.vq)}


def optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=.01)


def trainable_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted((n, p) for n, p in model.named_parameters() if p.requires_grad):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def probes(model, rows, features, codes):
    model.eval()
    values = []
    with torch.no_grad():
        for i in range(2):
            f, m, _, _ = training.batch(rows, [i], features, codes)
            values.append({k: v.cpu().tolist() for k, v in model.generate(f, m).items()})
    return values


def main():
    training.require_gpu()
    native.vendor.terminal(WARM)
    native.vendor.terminal(DEV_OUTPUTS)  # PhysicalGPU0 becomes free only after baseline generation exits.
    diagnostic = ROOT / "20261024_lm1b_native_parity_diagnostic_v1"
    native.vendor.terminal(diagnostic)
    observed = q.read(diagnostic / "result.json")
    if (observed["optimizer_steps"] != 0 or not observed["hard_generation_equal"]
            or not observed["logit_comparison"]["concept_logits"]["allclose_1e4"]
            or not observed["logit_comparison"]["word_logits"]["bitwise_equal"]
            or observed["after_model_sha256"] != WARM_MODEL):
        raise ValueError("actual zero-update parity diagnosis required")
    if q.read(WARM_WORKFLOW / "tool_terminal_receipt.json")["exit_code"] != 0 or file_hash(WARM / "model.pt") != WARM_SHA:
        raise ValueError("actual final1024 warm checkpoint and whole-tool exit required")
    if DEST.exists():
        raise ValueError("fresh native fine-tuning pilot root required")
    rows = q.read(training.DATA / "cases.json")
    ids = q.read(training.DATA / "selection.json")["pilot_ids"]
    chosen = [next(r for r in rows if r["id"] == key) for key in ids]
    manifest = q.read(training.DATA / "manifest.json")
    codes = {r["id"]: r["codes"] for r in q.read(training.CACHE / "outputs.json")}
    if len(rows) != 690 or len(codes) != 690 or len(chosen) != 2:
        raise ValueError("unchanged complete690 target population and representative two sources required")
    files = [Path(__file__), PLAN, AUDITOR, Path(mechanism.__file__), Path(objective.__file__), Path(objective.original.__file__),
        Path(head.__file__), Path(head.concepts.__file__), Path(training.__file__), Path(native.__file__), Path(same.__code__.co_filename),
        Path("tests/structure_net/test_ncm_lm1b_native_finetuning.py"),
        Path("tests/structure_net/test_ncm_lm1b_stepwise_consistency_probe.py"),
        Path("tests/neural_architecture_lab/test_ncm_lm1b_native_finetuning_audit.py")]
    files += sorted(native.vendor.VENDOR.rglob("*.py"))
    inputs = [WARM / n for n in ("model.pt", "result.json", "worker_terminal_receipt.json")]
    inputs += [WARM_WORKFLOW / "tool_terminal_receipt.json", *[training.DATA / n for n in ("cases.json", "manifest.json", "selection.json")],
        training.CACHE / "outputs.json", head.DOMAIN_CHECKPOINT,
        *[Path(manifest[r["id"]]["path"]) for r in chosen], DEV_OUTPUTS / "worker_terminal_receipt.json", DEV_OUTPUTS / "result.json"]
    inputs += [native.RELEASE / "files" / name for name in native.FILES]
    inputs.append(native.RELEASE / "files/lm1b/small/config.yaml")
    inputs += [ROOT / "20261024_lm1b_native_finetuning_cuda_pilot_v1/worker_terminal_receipt.json",
        ROOT / "20261024_lm1b_native_parity_diagnostic_v1/result.json", ROOT / "20261024_lm1b_native_parity_diagnostic_v1/worker_terminal_receipt.json", diagnostic / "producer.json"]
    p = native.vendor.freeze(DEST, "native_NCM_finetuning_lifecycle_not_task_quality", inputs + files,
        {"hypothesis_id": HYPOTHESIS, "systems_only": True, "updates": 2, "microbatch": 4,
        "trainable_parameters": 127732736, "precision": "FP32weights/BF16training/FP32parityandgeneration", "TF32": False,
        "optimizer": "AdamW1e-5,.9/.95,eps1e-8,wd.01,clip1", "dropout_seed": 112142, "neighbor_seed": 111142,
        "corruption_absolute_schedule": "5120+step;maximum.25percaseuniform;rootprotected",
        "loss": "clean.5C+.7W;noisy.5C+.3W;self.25C+1Wwithwordgrad;accum1",
        "no_new_random_weights": True, "Llama_body_loaded": False, "Llama_MLPs_updated": False,
        "encoder_and_VQ_frozen": True, "native_generator_and_word_decoder_trainable": True,
        "pilot_not_full_training_initializer": True, "peak_limit_bytes": 8_000_000_000,
        "source_ids": ids, "task_accuracy": None, "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / f"native_finetune_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", p)
    features = training.load_features(chosen, manifest)
    model = head.from_domain(device="cuda").eval()
    state = torch.load(WARM / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model.load_attention_state(state["attention"])
    if training.attention_hash(model) != WARM_ATTENTION or native.vendor.module_hash(model) != WARM_MODEL:
        raise ValueError("fixed warm model or attention differs")
    f, m, c, _ = training.batch(chosen, [0, 1], features, codes)
    with torch.no_grad():
        before = {k: v.cpu().clone() for k, v in model(f, m, c).items()}
        generated = {k: v.cpu().clone() for k, v in model.generate(f, m).items()}
    stepped, forced_codes = stepwise_logits(model, f, m, c)
    torch.save({"teacher_logits": before["concept_logits"], "step_logits": stepped,
        "forced_codes": forced_codes, "target_codes": c.cpu()}, DEST / "teacher_stepwise_parity.pt")
    consistency = {"same_history_max_abs_logit_difference": float((stepped - before["concept_logits"]).abs().max()),
        "same_history_argmax_disagreements": int((stepped.argmax(-1) != before["concept_logits"].argmax(-1)).sum()),
        "same_history_logits_allclose": torch.allclose(stepped, before["concept_logits"], rtol=1e-4, atol=1e-4)}
    write_json(DEST / "teacher_stepwise_parity.json", consistency)
    if not consistency["same_history_logits_allclose"]:
        raise ValueError("joint-training and stepwise-generation logits differ on identical TRAIN histories; no native updates admitted")
    del stepped, forced_codes
    mechanism.enable(model).eval()
    with torch.no_grad():
        after = {k: v.cpu().clone() for k, v in model(f, m, c).items()}
        new_generated = {k: v.cpu().clone() for k, v in model.generate(f, m).items()}
    initial_parity = {k: {"bitwise_equal": torch.equal(before[k], after[k]),
        "max_abs_difference": float((before[k] - after[k]).abs().max()),
        "allclose_1e4": torch.allclose(before[k], after[k], rtol=1e-4, atol=1e-4),
        "argmax_disagreements": int((before[k].argmax(-1) != after[k].argmax(-1)).sum())} for k in before}
    torch.save({"before": before, "after": after, "generated_before": generated, "generated_after": new_generated}, DEST / "parity.pt")
    write_json(DEST / "initial_forward_parity.json", initial_parity)
    if (not initial_parity["concept_logits"]["allclose_1e4"] or not initial_parity["word_logits"]["bitwise_equal"]
            or not same(generated, new_generated) or native.vendor.module_hash(model) != WARM_MODEL
            or protected(model) != {"encoder": ENCODER, "VQ": VQ}
            or sum(v.numel() for v in model.parameters() if v.requires_grad) != 127732736):
        raise ValueError("initial native allclose/wordexact/hardexact/weight/protected-codec contract differs")
    del f, m, c, before, after, generated, new_generated, state
    initial = trainable_hash(model)
    initial_native = {"generator": native.vendor.module_hash(model.concepts.generator), "word_decoder": native.vendor.module_hash(model.codec.decoder)}
    root_initial = model.concepts.generator.l0_queries.detach().cpu().clone()
    opt = optimizer(model)
    neighbor = torch.Generator().manual_seed(111142)
    torch.manual_seed(112142)
    torch.cuda.manual_seed_all(112142)
    torch.cuda.reset_peak_memory_stats()
    trace = []
    for step in (1, 2):
        model.train()
        opt.zero_grad(set_to_none=True)
        f, m, c, w = training.batch(chosen, [0, 1, 0, 1], features, codes)
        loss = objective.backward_microbatch(model, f, m, c, w, neighbor, 5120 + step, accumulation=1, temperature=1.)
        if any(v.grad is not None for module in (model.codec.encoder, model.codec.vq) for v in module.parameters()):
            raise ValueError("protected native target codec received gradients")
        gradients = {n: None if v.grad is None else v.grad.detach().cpu().clone() for n, v in model.named_parameters() if v.requires_grad}
        if any(v is not None and not torch.isfinite(v).all() for v in gradients.values()):
            raise ValueError("nonfinite full native gradients")
        active = {prefix: any(n.startswith(prefix) and v is not None and bool(v.abs().sum() > 0) for n, v in gradients.items()) for prefix in GROUPS}
        if not all(active.values()):
            raise ValueError("native generator/root/word decoder or source-attention gradients inactive")
        if step == 2:
            torch.save(gradients, DEST / "full_native_gradients.pt")
        del gradients
        norm = torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1.)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite full native gradient norm")
        opt.step()
        trace.append({"step": step, "gradient_norm": float(norm), "active_groups": active, "losses": loss})
        write_json(DEST / "trace.json", trace)
        print(json.dumps({"native_updates": step, "self_word_loss": loss["self_word_loss"]}), flush=True)
    del f, m, c, w, norm
    if protected(model) != {"encoder": ENCODER, "VQ": VQ} or torch.equal(root_initial, model.concepts.generator.l0_queries.detach().cpu()):
        raise ValueError("frozen target geometry changed or learned root did not update")
    final = trainable_hash(model)
    final_native = {"generator": native.vendor.module_hash(model.concepts.generator), "word_decoder": native.vendor.module_hash(model.codec.decoder)}
    if any(initial_native[k] == final_native[k] for k in initial_native):
        raise ValueError("both native generator and word decoder must actually update")
    reload_probes = probes(model, chosen, features, codes)
    saved = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "optimizer": opt.state_dict(),
        "CPU_rng": torch.get_rng_state(), "CUDA_rng": torch.cuda.get_rng_state_all(), "neighbor_rng": neighbor.get_state(),
        "step": 2, "producer_fingerprint": p["fingerprint"]}
    torch.save(saved, DEST / "model.pt")
    # Discard the GPU optimizer dictionary before constructing a genuinely fresh model.
    model.close()
    del model, opt, saved
    gc.collect()
    torch.cuda.empty_cache()
    saved = torch.load(DEST / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model = mechanism.enable(head.from_domain(device="cuda"))
    model.load_state_dict(saved["model"], strict=True)
    opt = optimizer(model)
    opt.load_state_dict(saved["optimizer"])
    neighbor.set_state(saved["neighbor_rng"])
    torch.set_rng_state(saved["CPU_rng"])
    torch.cuda.set_rng_state_all(saved["CUDA_rng"])
    if (probes(model, chosen, features, codes) != reload_probes or not same(model.state_dict(), saved["model"])
            or not same(opt.state_dict(), saved["optimizer"]) or trainable_hash(model) != final
            or not torch.equal(torch.get_rng_state(), saved["CPU_rng"])
            or not same(torch.cuda.get_rng_state_all(), saved["CUDA_rng"])
            or not torch.equal(neighbor.get_state(), saved["neighbor_rng"]) or protected(model) != {"encoder": ENCODER, "VQ": VQ}):
        raise ValueError("exact full native/attention/optimizer/allRNG/hard-generation reload failed")
    peak = torch.cuda.max_memory_allocated()
    if peak > 8_000_000_000:
        raise ValueError("native fine-tuning lifecycle exceeds8GBplan")
    write_json(DEST / "reload_probes.json", reload_probes)
    native.vendor.verify(p)
    result = {"status": "systems_native_NCM_finetuning_and_reload_passed_task_accuracy_unmeasured", "updates": 2,
        "trainable_parameters": 127732736, "protected_codec_hashes": protected(model), "native_root_query_updated": True,
        "initial_native_forward_allclose_and_hard_generation_exact": True, "initial_logit_parity": initial_parity, "strict_fullstate_optimizer_allRNG_hard_generation_reload": True,
        "initial_trainable_sha256": initial, "final_trainable_sha256": final, "trainable_weights_updated": initial != final,
        "initial_native_hashes": initial_native, "final_native_hashes": final_native,
        **consistency,
        "checkpoint_sha256": file_hash(DEST / "model.pt"), "peak_allocated_bytes": peak, "task_accuracy": None,
        "Llama_body_loaded": False, "final_test_opened": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(DEST / "result.json", result)
    model.close()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
