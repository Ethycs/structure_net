"""An intact native decoder language control, not an NCM success substitute."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_readout_transfer as previous
from experiments.structure_net.ncm_frozen_body import MODEL_PATH as PARENT_MODEL, module_hash
from experiments.structure_net.ncm_matched_parent_control import ValidateScores
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_parent_control import native_inputs
from experiments.structure_net.ncm_paper_data import ROOT, PREP, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-intact-language-control-plan.md")
INPUTS = ROOT/"20261023_intact_decoder_inputs_v1"
VERIFIED = INPUTS/"verified"
MODEL = VERIFIED/"model"
CACHED = Path("/home/rabbit/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6")
PILOT = ROOT/"20261023_intact_decoder_pilot_v1"
OUTPUTS = ROOT/"20261023_intact_decoder_outputs_v1"
JUDGE_PILOT = ROOT/"20261023_intact_decoder_judge_pilot_v1"
GRADING = ROOT/"20261023_intact_decoder_grading_v1"
PRIMARY = ROOT/"20261023_intact_decoder_primary_v1"


def producing(root,stage,inputs,config):
    p = freeze(root,stage,inputs,{**config,"NCM_model":False,"intact_pretrained_decoder":True,
        "physical_gpu":2,"precision":"FP16","batch":2,"seed":42,"max_native_output_tokens":128})
    for path in (Path(__file__),PLAN,Path(previous.__file__),Path("experiments/structure_net/ncm_native_parent_control.py"),
        Path("experiments/structure_net/ncm_source_readout_grade.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root/"producer_source"/("intact_decoder_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-intact-language-control-v1"
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


def prepare():
    from transformers import AutoTokenizer
    meta = q.read(INPUTS/"metadata.json")
    if meta["sha"]!=CACHED.name: raise ValueError("official revision mismatch")
    entries = {r["rfilename"]:r for r in meta["siblings"]}
    files = sorted(p for p in CACHED.iterdir() if p.is_file()); checks = {}
    for path in files:
        entry = entries[path.name]
        if entry.get("lfs"):
            actual = file_hash(path); expected = entry["lfs"]["sha256"]
        else:
            content = path.read_bytes()
            actual = hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest(); expected = entry["blobId"]
        if actual!=expected: raise ValueError("cached official file mismatch: "+path.name)
        checks[path.name] = {"verified_identity":actual,"sha256":file_hash(path),"bytes":path.stat().st_size}
    p = producing(VERIFIED,"official_cached_decoder_provenance",[INPUTS/"metadata.json",*files],{"revision":CACHED.name})
    MODEL.mkdir()
    for path in files: (MODEL/path.name).symlink_to(path)
    small = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    parent = AutoTokenizer.from_pretrained(PARENT_MODEL,local_files_only=True)
    cases = q.read(PREP/"development_sources.json")
    _,a = native_inputs(cases,small,"cpu"); _,b = native_inputs(cases,parent,"cpu")
    compatibility = {"vocab_maps_identical":small.get_vocab()==parent.get_vocab(),
        "added_tokens_identical":small.get_added_vocab()==parent.get_added_vocab(),"native_prompt_ids_all64_identical":a==b}
    write_json(VERIFIED/"file_checks.json",checks); write_json(VERIFIED/"tokenizer_compatibility.json",compatibility)
    verify(p); write_json(VERIFIED/"result.json",{"status":"verified","revision":CACHED.name,
        "producer_fingerprint":p["fingerprint"],"compatibility":compatibility})
    print(json.dumps({"status":"verified","compatibility":compatibility}),flush=True)


@torch.no_grad()
def generate(pilot):
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="2" or os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID"
        or not torch.cuda.is_available() or "2060 SUPER" not in torch.cuda.get_device_name().upper()
        or torch.cuda.mem_get_info()[0]<3_000_000_000): raise ValueError("planned GPU2 with3GB free required")
    if q.read(VERIFIED/"result.json")["status"]!="verified": raise ValueError("verified intact model required")
    verify(q.read(VERIFIED/"producer.json"))
    if not pilot and (q.read(PILOT/"result.json")["status"]!="passed" or q.read(PILOT/"worker_terminal_receipt.json")["exit_code"]!=0):
        raise ValueError("terminal intact language pilot required")
    cases = q.read(PREP/"development_sources.json")[:4 if pilot else 64]; verify_public(cases)
    root = PILOT if pilot else OUTPUTS
    p = producing(root,"systems_lifecycle_only_not_quality_evidence" if pilot else "intact_native_language_development",
        [PREP/"development_sources.json",VERIFIED/"producer.json",VERIFIED/"result.json",*[f for f in MODEL.iterdir() if f.is_file()]],
        {"pilot":pilot,"cases":len(cases),"attention":"sdpa"})
    if not pilot and any(q.read(PILOT/"producer.json")[key]!=p[key] for key in ("source_hashes","input_hashes")):
        raise ValueError("intact decoder producer changed after pilot")
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float16,
        attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if len(model.model.layers)!=16 or model.lm_head.weight.data_ptr()!=model.model.embed_tokens.weight.data_ptr():
        raise ValueError("intact native model identity differs")
    before = module_hash(model)
    def run():
        outputs = []
        for i in range(0,len(cases),2):
            chunk = cases[i:i+2]; x,ids = native_inputs(chunk,tokenizer,"cuda")
            raw = model.generate(**x,max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.eos_token_id,
                logits_processor=[ValidateScores()])[:,768:].tolist()
            outputs.extend({**case,"text":tokenizer.decode(words,skip_special_tokens=True).strip(),
                "native_output_ids":words,"native_prompt_ids":prompt,"NCM_model":False,"source_truncated":False}
                for case,prompt,words in zip(chunk,ids,raw))
        return outputs
    outputs = run()
    if pilot and outputs!=run(): raise ValueError("intact greedy repetition differs")
    if not pilot and outputs[:4]!=q.read(PILOT/"outputs.json"): raise ValueError("intact first-four output changed")
    if module_hash(model)!=before or any(v.grad is not None for v in model.parameters()):
        raise ValueError("intact model weights changed")
    write_json(root/"outputs.json",outputs)
    if q.read(root/"outputs.json")!=outputs: raise ValueError("intact output reload differs")
    verify(p)
    result = {"status":"passed" if pilot else "generated_pending_task_grading","cases":len(cases),
        "output_sha256":file_hash(root/"outputs.json"),"producer_fingerprint":p["fingerprint"],
        "full_model_hash_unchanged":before,"native_blocks":16,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "NCM_model":False,"NCM_success_established":False,"task_accuracy":None,"trained":False,
        "final_test_opened":False,"speed_tested":False}
    write_json(root/"result.json",result); print(json.dumps(result),flush=True)


def grade(command):
    # Reuse the immutable first-judgment protocol in this process only.
    previous.PARENT_GRADING = previous.GRADING
    previous.OUTPUTS,previous.PILOT,previous.JUDGE_PILOT,previous.GRADING,previous.PRIMARY = OUTPUTS,PILOT,JUDGE_PILOT,GRADING,PRIMARY
    previous.producing = producing
    previous.grade(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("command",choices=["prepare","pilot","generate","judge-pilot","grade","primary"])
    command = parser.parse_args().command
    if command=="prepare": prepare()
    elif command in ("pilot","generate"): generate(command=="pilot")
    else: grade(command)
