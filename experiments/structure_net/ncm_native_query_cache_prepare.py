"""Offline frozen source-KV and sparse native-teacher preparation on CUDA."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_query_dataset as data
from experiments.structure_net.ncm_frozen_body import MODEL_PATH,module_hash
from experiments.structure_net.ncm_matched_parent_control import EXPECTED_BODY
from experiments.structure_net.ncm_memory_data import file_hash,write_json
from experiments.structure_net.ncm_native_concepts import ConceptConfig,encode
from experiments.structure_net.ncm_native_query_cache import NativeQueryBackbone
from experiments.structure_net.ncm_native_query_distillation import cpu_prefix,pack_prefix,native_teacher_support
from experiments.structure_net.ncm_native_readout import frozen_hash
from experiments.structure_net.ncm_paper_data import ROOT,freeze,verify

PILOT = ROOT/"20261023_native_query_neighborhood_cache_pilot_v1"
CACHE = ROOT/"20261023_native_query_neighborhood_cache_v1"
BOOKS = ROOT/"20261023_native_concepts_fit_v1"
NCM_PILOT = ROOT/"20261023_native_query_concept_pilot_v1"


def main(pilot):
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 7_500_000_000):
        raise ValueError("planned GPU0 headroom required")
    for root in (data.DEST,BOOKS,NCM_PILOT):
        if q.read(root/"worker_terminal_receipt.json")["exit_code"] != 0:
            raise ValueError("terminal dataset/component worker required")
        verify(q.read(root/"producer.json"))
    if q.read(NCM_PILOT/"result.json")["status"] != "passed":
        raise ValueError("actual NCM lifecycle failed")
    extra = []
    if not pilot:
        if (q.read(PILOT/"result.json")["status"] != "passed"
                or q.read(PILOT/"worker_terminal_receipt.json")["exit_code"] != 0):
            raise ValueError("terminal cache/teacher CUDA pilot required")
        verify(q.read(PILOT/"producer.json"))
        extra += [PILOT/n for n in ("producer.json","result.json","worker_terminal_receipt.json")]
    cohorts = q.read(data.DEST/"cases.json")
    if pilot: cohorts = {k:v[:4] for k,v in cohorts.items()}
    root = PILOT if pilot else CACHE
    inputs = [data.DEST/n for n in ("producer.json","result.json","cases.json","worker_terminal_receipt.json")]
    inputs += [BOOKS/n for n in ("concepts.pt","producer.json","result.json")]
    inputs += [NCM_PILOT/n for n in ("producer.json","result.json","worker_terminal_receipt.json")]
    inputs += [p for p in MODEL_PATH.iterdir() if p.is_file()]+extra
    paths = [Path(__file__),data.PLAN,Path(data.__file__),Path("experiments/structure_net/ncm_native_query_cache.py"),
        Path("experiments/structure_net/ncm_native_query_distillation.py"),Path("experiments/structure_net/ncm_native_concepts.py"),
        Path("experiments/structure_net/ncm_native_readout.py")]
    p = freeze(root,"systems_cache_teacher_only" if pilot else "frozen_native_neighborhood_cache",inputs+paths,
        {"pilot":pilot,"counts":{k:len(v) for k,v in cohorts.items()},"teacher_topk":32,"teacher_topk_renormalized":True,
         "physical_gpu":0,"seed":42,"batch":2,"native_precision":"BF16","code_precision":"FP32","allow_tf32":False,
         "torch":torch.__version__,"cuda":torch.version.cuda,"transformers":importlib.metadata.version("transformers"),
         "allocator":os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),"task_accuracy":None})
    for path in paths:
        p["source_hashes"][str(path)] = file_hash(path)
        (root/"producer_source"/("cache_prepare_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
    write_json(root/"producer.json",p)
    if not pilot and q.read(PILOT/"producer.json")["source_hashes"] != p["source_hashes"]:
        raise ValueError("cache/teacher implementation changed after pilot")
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    native = AutoModelForCausalLM.from_pretrained(MODEL_PATH,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if module_hash(native.model) != EXPECTED_BODY: raise ValueError("original native body differs")
    backbone = NativeQueryBackbone(native,rank=8)
    frozen = frozen_hash(backbone)
    saved = torch.load(BOOKS/"concepts.pt",map_location="cpu",weights_only=True)
    cfg = ConceptConfig(**saved["config"])
    books = [v.cuda().float() for v in saved["books"]]
    (root/"cases").mkdir()
    manifest = {"schema":"nal.native-query-cache.v1","producer_fingerprint":p["fingerprint"],
                "cohorts":{},"cases":{},"native_eos":tokenizer.eos_token_id}
    completed = 0
    with torch.no_grad():
        for cohort,cases in cohorts.items():
            manifest["cohorts"][cohort] = [c["id"] for c in cases]
            for start in range(0,len(cases),2):
                chunk = cases[start:start+2]
                x = data.source_inputs(chunk,tokenizer.eos_token_id,"cuda")
                targets = data.target_tensor(chunk,"cuda")
                memory = backbone.source_cache(**x)
                indices,probabilities = native_teacher_support(backbone,memory,targets,tokenizer.eos_token_id,32)
                if not torch.isfinite(probabilities).all() or not torch.allclose(probabilities.sum(-1),torch.ones_like(targets,dtype=torch.float32)):
                    raise ValueError("invalid normalized sparse teacher")
                embedding = native.model.embed_tokens(targets.masked_fill(targets.eq(-100),tokenizer.eos_token_id)).float()
                clean = encode(embedding,books,cfg)[0]
                first = []
                for i,case in enumerate(chunk):
                    if len(case["id"]) != 64 or any(c not in "0123456789abcdef" for c in case["id"]):
                        raise ValueError("safe hash case identifier required")
                    payload = {"id":case["id"],"prefix":cpu_prefix(memory,i),"targets":targets[i].cpu().clone(),
                        "teacher_indices":indices[i].cpu().clone(),"teacher_probabilities":probabilities[i].cpu().clone(),
                        "clean_codes":[v[i].cpu().clone() for v in clean],"kind":case["kind"],
                        "window_eos_injected":case["window_eos_injected"],"producer_fingerprint":p["fingerprint"]}
                    path = root/"cases"/(case["id"]+".pt")
                    torch.save(payload,path)
                    entry = {"path":str(path),"sha256":file_hash(path),"bytes":path.stat().st_size,
                        "cohort":cohort,"source_frame":case["source_frame"],"source_sha256":case["source_sha256"]}
                    if case["id"] in manifest["cases"]: raise ValueError("duplicate cache identity")
                    manifest["cases"][case["id"]] = entry
                    if start == 0:
                        reloaded = torch.load(path,map_location="cpu",weights_only=True)
                        if not torch.equal(reloaded["targets"],payload["targets"]): raise ValueError("cache target reload differs")
                        first.append(reloaded["prefix"])
                if start == 0:
                    packed = pack_prefix(first,"cuda")
                    if any(not torch.equal(a,b) for pair,loaded in zip(memory.keys_values,packed.keys_values) for a,b in zip(pair,loaded)):
                        raise ValueError("CPU prefix values changed")
                    a,b = native_teacher_support(backbone,packed,targets,tokenizer.eos_token_id,32)
                    if not torch.equal(a,indices) or not torch.equal(b,probabilities):
                        raise ValueError("CPU-packed source teacher differs")
                    del packed,a,b
                completed += len(chunk)
                if completed % 64 == 0 or start+2 >= len(cases):
                    write_json(root/"progress.json",{"completed":completed,"cohort":cohort})
                    print(json.dumps({"cache_cases":completed,"cohort":cohort}),flush=True)
                del memory,indices,probabilities,embedding,clean,targets
    if frozen_hash(backbone) != frozen or any(v.grad is not None for v in backbone.parameters()):
        raise ValueError("frozen teacher/source body changed")
    if any(v.enabled for v in backbone.adapters): raise ValueError("student query adapters leaked into teacher")
    write_json(root/"manifest.json",manifest)
    if len(manifest["cases"]) != sum(len(v) for v in cohorts.values()): raise ValueError("cache count differs")
    verify(p)
    result = {"status":"passed" if pilot else "prepared_and_verified","pilot":pilot,
        "cases":completed,"counts":{k:len(v) for k,v in cohorts.items()},
        "manifest_sha256":file_hash(root/"manifest.json"),"cache_bytes":sum(v["bytes"] for v in manifest["cases"].values()),
        "frozen_body_hash_unchanged":frozen,"CPU_cache_roundtrip_exact":True,"teacher_topk":32,
        "teacher_topk_renormalized":True,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "task_accuracy":None,"NCM_success_established":False,"speed_tested":False,
        "producer_fingerprint":p["fingerprint"]}
    write_json(root/"result.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=["pilot","full"])
    main(parser.parse_args().command == "pilot")
