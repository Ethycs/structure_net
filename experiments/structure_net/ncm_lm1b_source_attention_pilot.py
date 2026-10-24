"""Real released-generator attachment lifecycle; systems evidence only."""
import gc
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_lm1b_source_attention as attention
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native = attention.release
q, ROOT = native.q, native.ROOT
DEST = ROOT / "20261024_lm1b_source_attention_cuda_pilot_v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-source-attention-cuda-pilot.md")
HYPOTHESIS = "ncm-lm1b-source-attention-native-CUDA-lifecycle-v1"
CASES = native.CASES
MANIFEST = ROOT / "20261023_reviewed_draft_data_v1/manifest.json"
TARGETS = ROOT / "20261024_lm1b_framed_codec_reconstruction_v1/outputs.json"


def freeze():
    rows, manifest, targets = q.read(CASES)["train"], q.read(MANIFEST), q.read(TARGETS)
    by_id = {v["id"]: v for v in targets}
    eligible = [r for r in rows if by_id[r["id"]]["fits_one_native_frame"]]
    selected = [eligible[0], min(eligible, key=lambda r: (-manifest[r["id"]]["source_tokens"], r["id"]))]
    if len({v["id"] for v in selected}) != 2:
        raise ValueError("distinct first/longest TRAIN systems probes required")
    paths = [Path(__file__), PLAN, Path(attention.__file__), Path("tests/structure_net/test_ncm_lm1b_source_attention.py")]
    paths += [CASES, MANIFEST, TARGETS, *[Path(manifest[r["id"]]["path"]) for r in selected]]
    paths += [native.RELEASE / "files" / n for n in (*native.FILES, "lm1b/small/config.yaml")]
    p = native.producing(DEST, "systems_real_native_generator_source_attention_lifecycle", paths,
        {"hypothesis_id": HYPOTHESIS, "classification": "systems_lifecycle_only_not_quality_evidence",
        "native_generator_frozen": True, "native_codec_frozen": True, "original_Llama_loaded": False,
        "trainable": "new source K/V and three zero-output Q/output attention projections only",
        "new_trainable_parameters": 8257536, "adapter_seed": 68142, "dropout_seed": 88142,
        "steps": 2, "optimizer": "AdamW1e-3,beta(.9,.95),weight_decay0,clip1", "physical_gpu": 0,
        "probes": [r["id"] for r in selected], "task_accuracy": None, "speed_tested": False,
        "single_window_selection_for_systems_only": True, "future_task_denominators_unchanged": True})
    p["hypothesis_id"] = HYPOTHESIS
    p["config"].update(hypothesis_id=HYPOTHESIS, schema="nal.ncm-lm1b-native-source-attention-pilot.v1",
        model_training_executed=True, dropout="native frozen eval; new attention0.1 while training")
    for i, path in enumerate(paths[:4]):
        p["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / f"native_attention_pilot_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", p)
    return p, selected, manifest, by_id


def sample(row, manifest, targets):
    entry = manifest[row["id"]]
    if file_hash(entry["path"]) != entry["sha256"]:
        raise ValueError("original frozen-body source features changed")
    saved = torch.load(entry["path"], map_location="cpu", weights_only=True)
    if (saved["id"] != row["id"] or saved["source_sha256"] != row["source_sha256"]
            or saved["native_prompt_sha256"] != row["native_prompt_sha256"]
            or saved["features"].shape != (entry["source_tokens"], 3072)):
        raise ValueError("exact full source-feature identity/shape required")
    features = saved["features"].unsqueeze(0).cuda()
    mask = torch.ones(features.shape[:2], device="cuda", dtype=torch.bool)
    codes = torch.tensor(targets[row["id"]]["codes"], device="cuda", dtype=torch.long)
    if codes.shape != (1, sum(native.LEVELS)):
        raise ValueError("complete single-frame TRAIN systems target required")
    return features, mask, codes


def native_logits(head, codes):
    prefix = torch.full((len(codes), 1), 101, device="cuda", dtype=torch.long)
    x = torch.cat((head.codec.encoder.wte(prefix), head.generator.build_l0_query_input(len(codes)),
        head.codec.vq.idx_to_ncm_input(codes)), dim=1)
    return head.generator(x)


def optimizer(head):
    return torch.optim.AdamW([v for v in head.parameters() if v.requires_grad], lr=1e-3,
        betas=(.9, .95), weight_decay=0.)


def main():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 8_000_000_000):
        raise ValueError("physicalGPU0 RTX3060 with8GBfree required; no CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    p, rows, manifest, targets = freeze()
    head = attention.from_release(device="cuda")
    trainable = sum(v.numel() for v in head.parameters() if v.requires_grad)
    if trainable != 8257536: raise ValueError("only expected new attention parameters may train")
    frozen = {"generator": native.vendor.module_hash(head.generator), "codec": native.vendor.module_hash(head.codec)}
    if frozen != {"generator": "60316ea14c56e6276be581112a2a1d81437e3ad8bb4ea3ddf8a98b946805e26a",
            "codec": "9aace450a49a7dfb44ba1eb00a19974a50d67062ad9f2202a6a2361ea4344e32"}:
        raise ValueError("exact released native generator and codec required")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for row in rows:
            features, mask, codes = sample(row, manifest, targets)
            unconditioned = native_logits(head, codes)
            attached = head.teacher_logits(features, mask, codes)
            if unconditioned.shape != (1, sum(native.LEVELS), 8192) or not torch.equal(unconditioned, attached):
                raise ValueError("zero-output adapter must preserve actual native full641 logits exactly")
    del unconditioned, attached
    opt = optimizer(head)
    torch.manual_seed(88142); torch.cuda.manual_seed_all(88142)
    head.train()
    traces = []
    for step, row in enumerate(rows, 1):
        features, mask, codes = sample(row, manifest, targets)
        opt.zero_grad(set_to_none=True)
        logits = head.teacher_logits(features, mask, codes)
        loss = F.cross_entropy(logits.flatten(0, 1), codes.flatten())
        if not torch.isfinite(loss): raise ValueError("nonfinite native adapter loss")
        loss.backward()
        if any(v.grad is not None for m in (head.generator, head.codec) for v in m.parameters()):
            raise ValueError("released body received adapter gradients")
        nonzero = [n for n, v in head.named_parameters() if v.requires_grad and v.grad is not None and v.grad.abs().sum() > 0]
        if not any("out.weight" in n for n in nonzero) or (step == 2 and not any(n.startswith("source_memory.") for n in nonzero)):
            raise ValueError("actual new attention gradient paths inactive")
        norm = torch.nn.utils.clip_grad_norm_([v for v in head.parameters() if v.requires_grad], 1.)
        if not torch.isfinite(norm): raise ValueError("nonfinite adapter gradients")
        opt.step(); traces.append({"step": step, "case_id": row["id"], "loss": float(loss.detach()),
            "nonzero_gradient_names": nonzero, "gradient_norm": float(norm)})
    head.eval()
    with torch.no_grad(): expected = head.teacher_logits(features, mask, codes).cpu()
    generated = head.generate_codes(features, mask).cpu()
    saved = {"adapter": head.adapter_state(), "optimizer": opt.state_dict(),
        "CPU_rng": torch.get_rng_state(), "CUDA_rng": torch.cuda.get_rng_state_all()}
    torch.save(saved, DEST / "adapter.pt")
    if any(native.vendor.module_hash(getattr(head, k)) != v for k, v in frozen.items()) or head._bound is not None:
        raise ValueError("frozen native state drift or stale source memory")
    head.close(); del head, opt, logits, loss
    gc.collect(); torch.cuda.empty_cache()
    head = attention.from_release(device="cuda")
    saved = torch.load(DEST / "adapter.pt", map_location="cpu", weights_only=True)
    head.load_adapter_state(saved["adapter"]); opt = optimizer(head); opt.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["CPU_rng"]); torch.cuda.set_rng_state_all(saved["CUDA_rng"])
    head.eval()
    with torch.no_grad(): actual = head.teacher_logits(features, mask, codes).cpu()
    if not torch.equal(actual, expected) or not torch.equal(head.generate_codes(features, mask).cpu(), generated):
        raise ValueError("actual native teacher/source-only generation reload differs")
    from experiments.structure_net.ncm_lm1b_decoder_repair import same
    if (not same(opt.state_dict(), saved["optimizer"]) or head._bound is not None
            or not torch.equal(torch.get_rng_state(), saved["CPU_rng"])
            or not same(torch.cuda.get_rng_state_all(), saved["CUDA_rng"])):
        raise ValueError("optimizer reload or source context cleanup failed")
    if torch.cuda.max_memory_allocated() > 8_000_000_000:
        raise ValueError("source-attention native pilot exceeds8GBplan")
    native.vendor.verify(p)
    write_json(DEST / "trace.json", traces)
    torch.save({"teacher_logits": expected, "source_only_generated_codes": generated}, DEST / "reload_outputs.pt")
    result = {"status": "systems_native_source_attention_gradients_and_reload_passed_not_quality",
        "hypothesis_id": HYPOTHESIS, "trainable_new_parameters": trainable, "released_states_unchanged": frozen,
        "native_post_convolutions_and_shared_heads_preserved": True, "zero_output_full641_logit_parity_exact": True,
        "teacher_logits_and_source_only_generation_reload_exact": True, "real_backward_updates": 2,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "source_tokens": [manifest[r["id"]]["source_tokens"] for r in rows],
        "checkpoint_sha256": file_hash(DEST / "adapter.pt"), "task_accuracy": None, "speed_tested": False,
        "final_test_opened": False, "producer_fingerprint": p["fingerprint"]}
    write_json(DEST / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
