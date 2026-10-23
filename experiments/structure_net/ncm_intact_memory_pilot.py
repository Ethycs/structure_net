"""Real-GPU identity/gradient/lifecycle checks; no NCM or task-quality claim."""
import copy
from dataclasses import fields
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_intact_language_control as language
from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_readout_control as accepted
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_intact_memory import IntactMemoryReadout, NativeMemoryCorrection, passage_mask
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_parent_control import native_inputs
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-intact-memory-pilot-plan.md")
DEST = ROOT / "20261023_intact_memory_pilot_v1"
MEMORY_CODE = Path("experiments/structure_net/ncm_intact_memory.py")


def main():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or not torch.cuda.is_available() or "2060 SUPER" not in torch.cuda.get_device_name().upper()
            or torch.cuda.mem_get_info()[0] < 3_000_000_000):
        raise ValueError("planned GPU2 headroom required")
    cases, audit_inputs = accepted.selected_cases()
    cases = cases[:4]
    for root in (accepted.CAPACITY, language.VERIFIED):
        verify(q.read(root / "producer.json"))
    inputs = [accepted.CAPACITY / n for n in ("producer.json", "source_features.pt")]
    inputs += [language.VERIFIED / n for n in ("producer.json", "result.json", "file_checks.json")]
    inputs += [p for p in language.MODEL.iterdir() if p.is_file()] + audit_inputs
    p = freeze(DEST, "systems_lifecycle_only_not_quality_evidence", inputs + [Path(__file__), MEMORY_CODE, PLAN],
        {"physical_gpu": 2, "seed": 42, "batch": 2, "steps": 2, "lr": 1e-4, "max_fraction": .1,
         "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
         "precision": "FP16 native; FP32 correction", "NCM_model": False, "task_accuracy": None})
    # Keep the outcome-relevant new sources in the immutable producing record.
    for path in (Path(__file__), MEMORY_CODE, PLAN):
        p["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / ("memory_" + path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", p)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(language.MODEL, local_files_only=True)
    native = AutoModelForCausalLM.from_pretrained(language.MODEL, local_files_only=True,
        torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if len(native.model.layers) != 16:
        raise ValueError("full native language stack required")
    before = module_hash(native)
    if before != q.read(language.PILOT / "result.json")["full_model_hash_unchanged"]:
        raise ValueError("native baseline precision/weights differ")
    stored = torch.load(accepted.CAPACITY / "source_features.pt", map_location="cpu", weights_only=True)
    features = stored["features"][:4].clone().cuda()
    originals = features.clone()
    masks = passage_mask(tokenizer, cases).cuda()
    correction = NativeMemoryCorrection(2048, 3072).cuda()
    wrapped = IntactMemoryReadout(native, correction)
    targets, _, _ = accepted.targets_for(cases, tokenizer, "cuda")
    # Source/concept memory contains no target-derived vectors in this pilot.
    plan = torch.randn(4, 4, 3072, device="cuda")
    x, _ = native_inputs(cases[:2], tokenizer, "cuda")
    with torch.no_grad():
        base_logits = native(**x, logits_to_keep=1).logits
        baseline = native.generate(**x, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        memory = correction.prepare(features[:2], masks[:2], plan[:2])
        copies = [getattr(memory, f.name).clone() for f in fields(memory)]
        with wrapped.conditioned(memory):
            conditioned_logits = native(**x, logits_to_keep=1).logits
            conditioned = native.generate(**x, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        if not torch.equal(base_logits, conditioned_logits) or not torch.equal(baseline, conditioned):
            raise ValueError("zero correction changed the native language path")
        if any(not torch.equal(getattr(memory, f.name), v) for f, v in zip(fields(memory), copies)):
            raise ValueError("conditioning mutated memory")
    write_json(DEST / "initial_greedy_ids.json", {"baseline": baseline[:, 768:].tolist(),
        "conditioned": conditioned[:, 768:].tolist(), "systems_only": True})
    del baseline, conditioned, base_logits, conditioned_logits, memory, copies
    optimizer = torch.optim.AdamW(correction.parameters(), lr=1e-4, weight_decay=.01)
    trace = []
    for step in range(2):
        rows = slice(2*step, 2*step+2)
        chunk = cases[rows]
        inputs, _ = native_inputs(chunk, tokenizer, "cuda")
        labels = targets[rows]
        prev = labels[:, :-1].masked_fill(labels[:, :-1].eq(-100), tokenizer.eos_token_id)
        ids = torch.cat((inputs["input_ids"], prev), 1)
        mask = torch.cat((inputs["attention_mask"], labels[:, :-1].ne(-100).long()), 1)
        positions = (mask.cumsum(-1)-1).clamp_min(0)
        with torch.no_grad():
            hidden = native.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                use_cache=False).last_hidden_state[:, 767:895].detach()
        memory = correction.prepare(features[rows], masks[rows], plan[rows])
        optimizer.zero_grad(set_to_none=True)
        with wrapped.conditioned(memory):
            logits = native.lm_head(hidden)
        loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())
        if not torch.isfinite(loss):
            raise ValueError("nonfinite loss")
        loss.backward()
        if not any(v.grad is not None and v.grad.abs().sum() > 0 for v in correction.parameters()):
            raise ValueError("no new-module gradient")
        if any(v.grad is not None for v in native.parameters()):
            raise ValueError("frozen native weights received gradients")
        norm = torch.nn.utils.clip_grad_norm_(correction.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        trace.append({"step": step+1, "loss": loss.item(), "gradient_norm": float(norm)})
        del logits, loss, memory
    if module_hash(native) != before or not torch.equal(features, originals):
        raise ValueError("native weights or source features changed")
    if wrapped._active or native.lm_head._forward_pre_hooks:
        raise ValueError("conditioning hooks leaked")
    torch.save({"correction": correction.state_dict(), "optimizer": optimizer.state_dict(),
                "producer_fingerprint": p["fingerprint"]}, DEST / "model.pt")
    with torch.no_grad():
        expected = correction(hidden, correction.prepare(features[2:], masks[2:], plan[2:]))
        delta = (expected-hidden).float().norm(dim=-1)
        if not (delta <= hidden.float().norm(dim=-1)*.1+2e-2).all():
            raise ValueError("FP16 correction bound violated")
        restored = copy.deepcopy(correction)
        state = torch.load(DEST / "model.pt", map_location="cuda", weights_only=True)
        restored.load_state_dict(state["correction"])
        actual = restored(hidden, restored.prepare(features[2:], masks[2:], plan[2:]))
        if not torch.equal(expected, actual) or not torch.isfinite(actual).all():
            raise ValueError("new-module save/reload differs")
    verify(p)
    result = {"status": "passed", "systems_only": True, "cases": 4, "updates": 2, "trace": trace,
        "initial_logits_and_greedy_exact": True, "native_full_hash_unchanged": before,
        "native_weights_frozen": True, "features_immutable": True, "reload_exact": True,
        "native_blocks": 16, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "trainable_parameters": sum(v.numel() for v in correction.parameters()),
        "NCM_model": False, "task_accuracy": None, "speed_tested": False,
        "producer_fingerprint": p["fingerprint"]}
    write_json(DEST / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
