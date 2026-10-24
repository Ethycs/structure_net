"""Record each initial native-unfreezing parity condition; never optimize weights."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_lm1b_native_finetuning_pilot as previous

DEST = previous.ROOT / "20261024_lm1b_native_parity_diagnostic_v1"


def main():
    previous.training.require_gpu()
    if DEST.exists():
        raise ValueError("fresh diagnostic root required")
    previous.native.vendor.terminal(previous.WARM)
    previous.native.vendor.terminal(previous.DEV_OUTPUTS)
    rows = previous.q.read(previous.training.DATA / "cases.json")
    ids = previous.q.read(previous.training.DATA / "selection.json")["pilot_ids"]
    chosen = [next(r for r in rows if r["id"] == identity) for identity in ids]
    manifest = previous.q.read(previous.training.DATA / "manifest.json")
    codes = {r["id"]: r["codes"] for r in previous.q.read(previous.training.CACHE / "outputs.json")}
    inputs = [previous.WARM / "model.pt", *[previous.training.DATA / n for n in ("cases.json", "selection.json", "manifest.json")],
        previous.training.CACHE / "outputs.json", *[Path(manifest[r["id"]]["path"]) for r in chosen],
        Path(__file__), Path(previous.__file__), previous.PLAN, Path(previous.mechanism.__file__),
        previous.DEST / "worker_terminal_receipt.json"]
    p = previous.native.vendor.freeze(DEST, "zero_update_native_unfreezing_parity_diagnostic", inputs,
        {"optimizer_steps": 0, "source_ids": ids, "task_accuracy": None, "Llama_body_loaded": False,
            "initial_failed_pilot_preserved": True, "no_semantic_judgments": True})
    features = previous.training.load_features(chosen, manifest)
    model = previous.head.from_domain(device="cuda").eval()
    saved = torch.load(previous.WARM / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model.load_attention_state(saved["attention"])
    f, m, c, _ = previous.training.batch(chosen, [0, 1], features, codes)
    with torch.no_grad():
        before = {k: v.cpu().clone() for k, v in model(f, m, c).items()}
        generated_before = {k: v.cpu().clone() for k, v in model.generate(f, m).items()}
    baseline = {"model_sha256": previous.native.vendor.module_hash(model), "protected": previous.protected(model),
        "trainable_parameters": sum(v.numel() for v in model.parameters() if v.requires_grad)}
    previous.mechanism.enable(model).eval()
    with torch.no_grad():
        after = {k: v.cpu().clone() for k, v in model(f, m, c).items()}
        generated_after = {k: v.cpu().clone() for k, v in model.generate(f, m).items()}
    comparison = {k: {"bitwise_equal": torch.equal(before[k], after[k]),
        "max_abs_difference": float((before[k] - after[k]).abs().max()),
        "allclose_1e4": torch.allclose(before[k], after[k], rtol=1e-4, atol=1e-4),
        "argmax_disagreements": int((before[k].argmax(-1) != after[k].argmax(-1)).sum())} for k in before}
    torch.save({"before": before, "after": after, "generated_before": generated_before, "generated_after": generated_after}, DEST / "parity.pt")
    result = {"status": "native_unfreezing_zero_update_parity_conditions_recorded", "baseline": baseline,
        "after_model_sha256": previous.native.vendor.module_hash(model), "expected_model_sha256": previous.WARM_MODEL,
        "after_protected": previous.protected(model), "expected_protected": {"encoder": previous.ENCODER, "VQ": previous.VQ},
        "after_trainable_parameters": sum(v.numel() for v in model.parameters() if v.requires_grad),
        "expected_trainable_parameters": 127732736, "logit_comparison": comparison,
        "hard_generation_equal": previous.same(generated_before, generated_after),
        "model_training": model.training, "native_generator_training": model.concepts.generator.training,
        "decoder_training": model.codec.decoder.training, "source_concepts_training": model.concepts.training,
        "optimizer_steps": 0, "task_accuracy": None, "Llama_body_loaded": False}
    previous.native.vendor.verify(p)
    previous.write_json(DEST / "result.json", result)
    model.close()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
