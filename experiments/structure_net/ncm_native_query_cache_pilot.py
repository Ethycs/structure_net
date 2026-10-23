"""Full native3B source-cache/query numerical and training-lifecycle control."""
import importlib.metadata
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_readout_control as accepted
from experiments.structure_net.ncm_frozen_body import MODEL_PATH, module_hash
from experiments.structure_net.ncm_matched_parent_control import EXPECTED_BODY
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_parent_control import native_inputs
from experiments.structure_net.ncm_native_query_cache import NativeQueryBackbone
from experiments.structure_net.ncm_native_readout import frozen_hash, restore_trainable, trainable_state
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-native-query-cache-plan.md")
DEST = ROOT / "20261023_native_query_cache_pilot_v1"


def main():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 7_500_000_000):
        raise ValueError("planned GPU0 headroom required")
    cases, audits = accepted.selected_cases()
    cases = cases[:4]
    paths = [Path(__file__), PLAN, Path("experiments/structure_net/ncm_native_query_cache.py"),
        Path("experiments/structure_net/ncm_native_readout.py"), Path(accepted.__file__),
        Path("experiments/structure_net/ncm_native_parent_control.py")]
    inputs = audits+[accepted.QUALITY/n for n in ("selected.json", "result.json", "producer.json")]
    inputs += [p for p in MODEL_PATH.iterdir() if p.is_file()]
    p = freeze(DEST, "systems_lifecycle_only_not_quality_evidence", inputs+paths,
        {"rank": 8, "native_blocks": 28, "physical_gpu": 0, "seed": 42, "batch": 2,
         "steps": 2, "lr": 1e-4, "precision": "BF16 native; FP32 new query adapters", "attention": "sdpa",
         "relative_l2_max": .02, "top1_min": .95, "full_target_frame": 128,
         "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
         "task_accuracy": None, "NCM_model": False, "uses_additional_1B_decoder": False})
    for path in paths:
        p["source_hashes"][str(path)] = file_hash(path)
        (DEST/"producer_source"/("native_query_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k,v in p.items() if k != "fingerprint"})
    write_json(DEST/"producer.json", p)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    native = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if (module_hash(native.model) != EXPECTED_BODY or len(native.model.layers) != 28
            or native.lm_head.weight.data_ptr() != native.model.embed_tokens.weight.data_ptr()):
        raise ValueError("intact original native body/word-head identity differs")
    targets, _, _ = accepted.targets_for(cases, tokenizer, "cuda")
    previous = targets[:, :-1].masked_fill(targets[:, :-1].eq(-100), tokenizer.eos_token_id)
    inputs, _ = native_inputs(cases[:2], tokenizer, "cuda")
    ids = torch.cat((inputs["input_ids"], previous[:2]), 1)
    mask = torch.cat((inputs["attention_mask"], torch.ones_like(previous[:2])), 1)
    with torch.no_grad():
        oracle = native(input_ids=ids, attention_mask=mask,
            position_ids=(mask.cumsum(-1)-1).clamp_min(0), logits_to_keep=128).logits
    model = NativeQueryBackbone(native, rank=8)
    frozen = frozen_hash(model)
    memories = []
    for start in range(0, 4, 2):
        x, _ = native_inputs(cases[start:start+2], tokenizer, "cuda")
        memories.append(model.source_cache(**x))
    snapshots = [[tuple(t.clone() for t in pair) for pair in memory.keys_values] for memory in memories]
    with torch.no_grad():
        actual = model.query_logits(memories[0], model.predecessor_embeddings(memories[0], previous[:2]))
        relative = float((actual.float()-oracle.float()).norm()/oracle.float().norm().clamp_min(1e-8))
        top1 = float(actual.argmax(-1).eq(oracle.argmax(-1)).float().mean())
        if not torch.isfinite(actual).all() or relative > .02 or top1 < .95:
            raise ValueError("native full/cached oracle contract failed")
        if not torch.equal(actual, model.query_logits(memories[0], model.predecessor_embeddings(memories[0], previous[:2]))):
            raise ValueError("repeated native query differs")
    write_json(DEST/"oracle_control.json", {"relative_l2": relative, "top1_agreement": top1,
        "stock_argmax": oracle.argmax(-1).tolist(), "cached_argmax": actual.argmax(-1).tolist(),
        "teacher_forced_only_not_free_summary_quality": True})
    del oracle, actual
    parameters = [v for v in model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01)
    trace = []
    for i in range(2):
        optimizer.zero_grad(set_to_none=True)
        rows = slice(i*2, i*2+2)
        logits = model.query_logits(memories[i], model.predecessor_embeddings(memories[i], previous[rows]))
        loss = F.cross_entropy(logits.float().flatten(0,1), targets[rows].flatten())
        if not torch.isfinite(loss): raise ValueError("nonfinite native query loss")
        loss.backward()
        if not any(v.grad is not None and v.grad.abs().sum()>0 for v in parameters):
            raise ValueError("missing query-adapter gradients")
        if any(v.grad is not None for v in model.parameters() if not v.requires_grad):
            raise ValueError("original weights received gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        trace.append({"step": i+1, "loss": loss.item(), "gradient_norm": float(norm)})
        del logits, loss
    if frozen_hash(model) != frozen: raise ValueError("original native weights changed")
    for memory, snapshot in zip(memories, snapshots):
        if any(not torch.equal(t,s) for pair,old in zip(memory.keys_values,snapshot) for t,s in zip(pair,old)):
            raise ValueError("source cache mutated")
    state = trainable_state(model)
    torch.save({"trainable": state, "optimizer": optimizer.state_dict(), "producer_fingerprint": p["fingerprint"]}, DEST/"model.pt")
    with torch.no_grad():
        query = model.predecessor_embeddings(memories[0], previous[:2])
        expected = model.query_logits(memories[0], query)
        restore_trainable(model, torch.load(DEST/"model.pt", map_location="cpu", weights_only=True)["trainable"])
        actual = model.query_logits(memories[0], query)
        if not torch.equal(expected, actual): raise ValueError("native query reload differs")
    if any(v.enabled for v in model.adapters): raise ValueError("query adapter context leaked")
    verify(p)
    result = {"status": "passed", "systems_only": True, "cases": 4, "updates": 2, "trace": trace,
        "native_blocks": 28, "trainable_parameters": sum(v.numel() for v in parameters),
        "oracle_relative_l2": relative, "oracle_top1_agreement": top1, "source_cache_immutable": True,
        "original_weights_frozen": True, "original_frozen_hash_unchanged": frozen, "reload_exact": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "NCM_model": False,
        "task_accuracy": None, "speed_tested": False, "final_test_opened": False,
        "uses_additional_1B_decoder": False, "producer_fingerprint": p["fingerprint"]}
    write_json(DEST/"result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
