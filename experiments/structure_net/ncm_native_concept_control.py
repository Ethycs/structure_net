"""Fit native residual concepts on public raw text, never on task-test labels."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body import MODEL_PATH
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_native_concepts import ConceptConfig, check_repair, encode, expand, fit_book, pool
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-native-concept-interface-plan.md")
SOURCE = ROOT/"20261022_expanded_data_v2/train_sources.json"
PREP = ROOT/"20261023_native_concepts_prepare_v1"
PILOT = ROOT/"20261023_native_concepts_pilot_v1"
FIT = ROOT/"20261023_native_concepts_fit_v1"


def producing(root, stage, inputs, config):
    p = freeze(root,stage,inputs,{**config,"physical_gpu":2,"precision":"FP32","source_batch":16,
        "torch":torch.__version__,"tf32":False,"seed":42,"cuda_reduction_bitwise_repeatability":False,
        "text_decoder":False,"student_trained":False})
    for path in (Path(__file__),PLAN,Path("experiments/structure_net/ncm_native_concepts.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root/"producer_source"/("native_concepts_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-native-concept-interface-v1"
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
    write_json(root/"producer.json",p)
    return p


def prepare():
    from transformers import AutoTokenizer
    verify(q.read(SOURCE.parent/"producer.json"))
    sources = sorted(q.read(SOURCE),key=lambda r:text_hash("native-concept-bank-v1:"+r["id"]))[:576]
    if len(sources)!=576 or any(len({r[k] for r in sources})!=576 for k in ("id","source_id","group_id","source_sha256")):
        raise ValueError("unique512+64 training-source groups required")
    verify_public(sources)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    windows = []
    for row in sources:
        full = tokenizer.encode(row["source"],add_special_tokens=False)
        if not full: raise ValueError("empty representation source")
        ids = full[:128]+[tokenizer.eos_token_id]*max(0,128-len(full))
        windows.append({"id":row["id"],"source_id":row["source_id"],"group_id":row["group_id"],
            "source_sha256":row["source_sha256"],"ids":ids,"raw_native_tokens":len(full),
            "representation_window_truncated":len(full)>128,"task_source_truncated":False})
    inputs = [SOURCE,*[p for p in MODEL_PATH.iterdir() if p.is_file() and p.suffix not in (".safetensors",)]]
    p = producing(PREP,"raw_public_native_window_preparation",inputs,{"train":512,"diagnostic":64})
    write_json(PREP/"windows.json",windows)
    verify(p); write_json(PREP/"result.json",{"status":"prepared","train":512,"diagnostic":64,
        "task_labels_used":False,"producer_fingerprint":p["fingerprint"]})
    print(json.dumps({"status":"prepared","train":512,"diagnostic":64}),flush=True)


def pooled_vectors(ids, word, books, codes, width, config):
    vectors = []
    for i in range(0,len(ids),16):
        target = F.embedding(ids[i:i+16].cuda(),word)
        reconstruction = torch.zeros_like(target)
        for book,previous in zip(books,codes):
            reconstruction.add_(expand(book[previous[i:i+16].cuda()],config.canvas))
        vectors.append(pool(target-reconstruction,width).flatten(0,1).cpu())
    return torch.cat(vectors)


def assess(ids, word, books, config):
    outputs = [[] for _ in books]; errors = []; initial = []; gains = []
    for i in range(0,len(ids),16):
        target = F.embedding(ids[i:i+16].cuda(),word)
        codes,_,mse = encode(target,books,config)
        checks = check_repair(target,books,config)
        gains.append(checks["mean_repair_gain"])
        for out,value in zip(outputs,codes): out.append(value.cpu())
        errors.append(mse.cpu()); initial.append(target.square().mean((1,2)).cpu())
    codes = [torch.cat(values) for values in outputs]; mse = torch.cat(errors)
    return {"cases":len(ids),"initial_mse":float(torch.cat(initial).mean()),
        "mse_by_level":mse.mean(0).tolist(),"active_entries_by_level":[len(torch.unique(c)) for c in codes],
        "mean_same_history_repair_gain":sum(gains)/len(gains),"invariants_passed":True,
        "task_accuracy":None},codes


@torch.no_grad()
def fit(pilot):
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="2" or os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID" or
        not torch.cuda.is_available() or "2060 SUPER" not in torch.cuda.get_device_name().upper() or
        torch.cuda.mem_get_info()[0]<3_000_000_000):
        raise ValueError("planned GPU2 and3GB headroom required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    verify(q.read(PREP/"producer.json"))
    cfg = replace(ConceptConfig(),iterations=2) if pilot else ConceptConfig()
    root = PILOT if pilot else FIT
    if not pilot and (q.read(PILOT/"result.json")["status"]!="passed" or q.read(PILOT/"worker_terminal_receipt.json")["exit_code"]!=0):
        raise ValueError("terminal native concept GPU pilot required")
    index_path = MODEL_PATH/"model.safetensors.index.json"
    tensor_name = "model.embed_tokens.weight"
    weight_path = MODEL_PATH/q.read(index_path)["weight_map"][tensor_name]
    original = q.read(ROOT/"20261023_native_parent_pilot_v1/producer.json")
    if original["input_hashes"][str(weight_path)]!=file_hash(weight_path):
        raise ValueError("native embedding source no longer matches verified parent")
    p = producing(root,"systems_lifecycle_only_not_quality_evidence" if pilot else "native_embedding_residual_codebooks",
        [PREP/"windows.json",PREP/"producer.json",weight_path,index_path],asdict(cfg))
    if not pilot and q.read(PILOT/"producer.json")["source_hashes"]!=p["source_hashes"]:
        raise ValueError("native concept producer changed after pilot")
    from safetensors import safe_open
    with safe_open(weight_path,framework="pt",device="cpu") as handle:
        word = handle.get_tensor(tensor_name).float().cuda()
    if word.shape!=(128256,3072) or not torch.isfinite(word).all():
        raise ValueError("unexpected native word table")
    windows = q.read(PREP/"windows.json")
    train = torch.tensor([r["ids"] for r in windows[:64 if pilot else 512]])
    diagnostic = torch.tensor([r["ids"] for r in windows[512:]])
    books,codes = [],[]
    for level,width in enumerate(cfg.levels):
        vectors = pooled_vectors(train,word,books,codes,width,cfg)
        book = fit_book(vectors,cfg,level,"cuda")
        # Match the streamed fit's FP32 assignment, without changing chunk shape.
        from experiments.structure_net.ncm_native_concepts import nearest
        labels = torch.cat([nearest(vectors[i:i+cfg.chunk].cuda(),book,cfg.chunk).cpu()
            for i in range(0,len(vectors),cfg.chunk)]).reshape(len(train),width)
        books.append(book); codes.append(labels)
        print(json.dumps({"level":level,"width":width,"active_entries":len(torch.unique(labels))}),flush=True)
        del vectors
    train_metrics,train_codes = assess(train,word,books,cfg)
    held_metrics,held_codes = assess(diagnostic,word,books,cfg)
    torch.save({"books":[b.cpu() for b in books],"config":asdict(cfg),"embedding_tensor":tensor_name,
        "embedding_file_sha256":file_hash(weight_path),"producer_fingerprint":p["fingerprint"],
        "text_decoder":False},root/"concepts.pt")
    restored = torch.load(root/"concepts.pt",weights_only=True)
    reloaded = [b.cuda() for b in restored["books"]]
    if any(not torch.equal(a,b) for a,b in zip(books,reloaded)):
        raise ValueError("native concept center reload differs")
    for ids,expected in ((train,train_codes),(diagnostic,held_codes)):
        actual = assess(ids,word,reloaded,cfg)[1]
        if any(not torch.equal(a,b) for a,b in zip(expected,actual)):
            raise ValueError("native concept assignment reload differs")
    torch.save({"train":train_codes,"diagnostic":held_codes},root/"assignments.pt")
    verify(p)
    result = {"status":"passed" if pilot else "representation_invariants_passed","train":train_metrics,
        "diagnostic":held_metrics,"zero_code_identity":True,"assignment_reload_exact":True,
        "original_embedding_file_unchanged":True,"text_decoder":False,"student_trained":False,
        "student_training_admitted":False,"NCM_success_established":False,"task_accuracy":None,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"producer_fingerprint":p["fingerprint"],
        "final_test_opened":False,"speed_tested":False}
    write_json(root/"result.json",result); print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("command",choices=["prepare","pilot","fit"])
    command = parser.parse_args().command
    prepare() if command=="prepare" else fit(command=="pilot")
