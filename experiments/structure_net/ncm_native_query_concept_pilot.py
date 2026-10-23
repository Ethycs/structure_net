"""Two-update lifecycle of actual generated/noisy native NCM concepts."""
import importlib.metadata
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_readout_control as accepted
from experiments.structure_net.ncm_frozen_body import MODEL_PATH,module_hash
from experiments.structure_net.ncm_matched_parent_control import EXPECTED_BODY
from experiments.structure_net.ncm_memory_data import file_hash,write_json
from experiments.structure_net.ncm_native_concepts import ConceptConfig,conditional_target,encode
from experiments.structure_net.ncm_native_parent_control import native_inputs
from experiments.structure_net.ncm_native_query_cache import NativeQueryBackbone
from experiments.structure_net.ncm_native_query_concepts import NativeQueryConcepts
from experiments.structure_net.ncm_native_query_training import actual_history,expert_gains
from experiments.structure_net.ncm_native_readout import frozen_hash,restore_trainable,trainable_state
from experiments.structure_net.ncm_paper_data import ROOT,freeze,verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-native-query-concept-pilot-plan.md")
DEST = ROOT/"20261023_native_query_concept_pilot_v1"
BOOKS = ROOT/"20261023_native_concepts_fit_v1"
CACHE_CONTROL = ROOT/"20261023_native_query_cache_pilot_v2"
RATES = (.5,.05,1/30,1/60,0.)


def check_cache(memory,copy):
    if any(not torch.equal(x.cpu(),y) for pair,old in zip(memory.keys_values,copy) for x,y in zip(pair,old)):
        raise ValueError("source cache changed during NCM levels")


@torch.no_grad()
def measure(net,cases,tokenizer):
    net.eval(); output = []
    for i in range(0,len(cases),2):
        x,_ = native_inputs(cases[i:i+2],tokenizer,"cuda")
        memory = net.backbone.source_cache(**x)
        snapshot = [tuple(v.cpu().clone() for v in pair) for pair in memory.keys_values]
        a,words = net.generate(memory)
        b,repeated = net.generate(memory)
        if not torch.equal(words,repeated) or any(not torch.equal(u,v) for u,v in zip(a,b)):
            raise ValueError("actual free NCM output not repeatable")
        check_cache(memory,snapshot)
        for j,(case,row) in enumerate(zip(cases[i:i+2],words.tolist())):
            active = row[:row.index(tokenizer.eos_token_id)+1] if tokenizer.eos_token_id in row else row
            output.append({"id":case["id"],"raw_parallel_native_ids":row,"active_native_ids":active,
                "text":tokenizer.decode(active,skip_special_tokens=True).strip(),
                "codes":[v[j].tolist() for v in a],"systems_only":True,"task_accuracy":None})
    return output


def main():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 7_500_000_000):
        raise ValueError("planned GPU0 headroom required")
    cases,audits = accepted.selected_cases(); cases = cases[:4]
    for path in (BOOKS,CACHE_CONTROL):
        if q.read(path/"worker_terminal_receipt.json")["exit_code"] != 0:
            raise ValueError("component workers must be terminal")
        verify(q.read(path/"producer.json"))
    if q.read(CACHE_CONTROL/"result.json")["status"] != "passed":
        raise ValueError("native cache component gate failed")
    if q.read(BOOKS/"result.json")["status"] != "representation_invariants_passed":
        raise ValueError("fixed concept component gate failed")
    paths = [Path(__file__),PLAN,Path("experiments/structure_net/ncm_native_query_cache.py"),
        Path("experiments/structure_net/ncm_native_query_concepts.py"),Path("experiments/structure_net/ncm_native_query_training.py"),
        Path("experiments/structure_net/ncm_native_concepts.py"),Path("experiments/structure_net/ncm_native_readout.py"),
        Path(accepted.__file__),Path("experiments/structure_net/ncm_native_parent_control.py")]
    inputs = audits+[accepted.QUALITY/n for n in ("selected.json","result.json","producer.json")]
    inputs += [BOOKS/n for n in ("concepts.pt","producer.json","result.json")]
    inputs += [CACHE_CONTROL/n for n in ("producer.json","result.json","worker_terminal_receipt.json")]
    inputs += [p for p in MODEL_PATH.iterdir() if p.is_file()]
    p = freeze(DEST,"systems_lifecycle_only_not_quality_evidence",inputs+paths,
        {"hypothesis_id":"ncm-native-query-concept-lifecycle-v1","NCM_model":True,"native_blocks":28,
         "rank":8,"hint_rank":16,"seed":42,"corruption_seed":4042,"rates":RATES,"batch":2,
         "steps":2,"lr":1e-4,"clip":1.,"weight_decay":.01,"native_frame":128,"physical_gpu":0,
         "precision":"BF16native/FP32new","torch":torch.__version__,"cuda":torch.version.cuda,
         "transformers":importlib.metadata.version("transformers"),
         "allocator":os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),"allow_tf32":False,"task_accuracy":None,"paper_replication":False})
    for path in paths:
        p["source_hashes"][str(path)] = file_hash(path)
        (DEST/"producer_source"/("query_concept_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
    write_json(DEST/"producer.json",p)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    native = AutoModelForCausalLM.from_pretrained(MODEL_PATH,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if module_hash(native.model) != EXPECTED_BODY or len(native.model.layers) != 28:
        raise ValueError("original native body changed")
    book_state = torch.load(BOOKS/"concepts.pt",map_location="cpu",weights_only=True)
    cfg = ConceptConfig(**book_state["config"])
    net = NativeQueryConcepts(NativeQueryBackbone(native,rank=8),book_state["books"],cfg.levels,cfg.canvas,16)
    frozen = frozen_hash(net)
    book_before = [b.cpu().clone() for b in net.books]
    targets,_,_ = accepted.targets_for(cases,tokenizer,"cuda")
    with torch.no_grad():
        embedding = native.model.embed_tokens(targets.masked_fill(targets.eq(-100),tokenizer.eos_token_id)).float()
    clean = encode(embedding,net.books,cfg)[0]
    initial = measure(net,cases,tokenizer)
    write_json(DEST/"initial_free_outputs.json",initial)
    parameters = [v for v in net.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01)
    generator = torch.Generator().manual_seed(4042)
    trace = []
    for step in range(2):
        rows = slice(2*step,2*step+2)
        x,_ = native_inputs(cases[rows],tokenizer,"cuda")
        memory = net.backbone.source_cache(**x)
        snapshot = [tuple(v.cpu().clone() for v in pair) for pair in memory.keys_values]
        optimizer.zero_grad(set_to_none=True)
        if step == 0:
            level = 0
            prior = net.book_0.new_zeros(2,cfg.canvas,native.config.hidden_size)
            labels = conditional_target(embedding[rows],prior,net.books[0],net.levels[0])
            gains = None
        else:
            level = len(net.levels)
            net.eval()
            codes,prior = actual_history(net,memory,level,generator,RATES)
            gains = expert_gains(embedding[rows],net.books,net.levels,codes,[v[rows] for v in clean])
            labels = targets[rows]
        net.train()
        logits = net.level_logits(memory,prior,level)
        if not torch.isfinite(logits).all(): raise ValueError("nonfinite NCM logits")
        loss = F.cross_entropy(logits.float().flatten(0,1),labels.flatten())
        if not torch.isfinite(loss): raise ValueError("nonfinite NCM loss")
        loss.backward()
        expected = net.heads[0].weight if step == 0 else net.hint_b
        if expected.grad is None or not expected.grad.abs().sum() > 0:
            raise ValueError("missing root-head/current-hint gradient")
        if any(v.grad is not None for v in net.parameters() if not v.requires_grad):
            raise ValueError("original native weights received gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(v).all() for v in parameters): raise ValueError("nonfinite new parameter")
        check_cache(memory,snapshot)
        trace.append({"step":step+1,"level":level,"loss":loss.item(),"gradient_norm":float(norm),
            "actual_noisy_history":step == 1,"same_history_expert_gains":gains.tolist() if gains is not None else None})
        write_json(DEST/"trace.json",trace)
        del logits,loss,memory,snapshot,prior
    net.eval()
    if frozen_hash(net) != frozen: raise ValueError("original native weights changed")
    if any(not torch.equal(b.cpu(),old) for b,old in zip(net.books,book_before)):
        raise ValueError("fixed concept books changed")
    final = measure(net,cases,tokenizer)
    write_json(DEST/"final_free_outputs.json",final)
    torch.save({"trainable":trainable_state(net),"optimizer":optimizer.state_dict(),
        "producer_fingerprint":p["fingerprint"]},DEST/"model.pt")
    with torch.no_grad():
        for value in parameters: value.zero_()
    restore_trainable(net,torch.load(DEST/"model.pt",map_location="cpu",weights_only=True)["trainable"])
    if final != measure(net,cases,tokenizer): raise ValueError("actual NCM save/reload differs")
    if any(v.enabled for v in net.backbone.adapters): raise ValueError("query context leaked")
    verify(p)
    result = {"status":"passed","systems_only":True,"NCM_model":True,"updates":2,"cases":4,
        "trace":trace,"native_blocks":28,"levels":net.levels,"canvas":net.canvas,
        "original_weights_frozen":True,"original_frozen_hash_unchanged":frozen,"books_unchanged":True,
        "source_cache_immutable":True,"free_repeat_and_reload_exact":True,
        "trainable_parameters":sum(v.numel() for v in parameters),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "task_accuracy":None,"NCM_success_established":False,"final_test_opened":False,"speed_tested":False,
        "teacher_AR_or_verifier_at_inference":False,"uses_additional_decoder":False,
        "producer_fingerprint":p["fingerprint"]}
    write_json(DEST/"result.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
