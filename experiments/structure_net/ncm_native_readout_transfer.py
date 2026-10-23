"""Measure source-disjoint transfer of the fixed native AR readout reference."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_readout_control as c
from experiments.structure_net.ncm_frozen_body import MODEL_PATH, module_hash
from experiments.structure_net.ncm_matched_parent_control import EXPECTED_BODY
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_parent_control import native_inputs, GRADING as PARENT_GRADING
from experiments.structure_net.ncm_native_readout import NativeReadout, restore_trainable
from experiments.structure_net.ncm_paper_data import ROOT, PREP, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-native-readout-transfer-plan.md")
PILOT = ROOT/"20261023_native_readout_transfer_pilot_v1"
OUTPUTS = ROOT/"20261023_native_readout_transfer_outputs_v1"
JUDGE_PILOT = ROOT/"20261023_native_readout_transfer_judge_pilot_v1"
GRADING = ROOT/"20261023_native_readout_transfer_grading_v1"
PRIMARY = ROOT/"20261023_native_readout_transfer_primary_v1"
NOT_BEFORE = datetime(2026,10,23,9,50,tzinfo=timezone.utc)


def producing(root, stage, inputs, config):
    p = freeze(root,stage,inputs,{**config,"NCM_model":False,"source_layers":[1,14,28],
        "readout_blocks":[0,7,14,21,27],"batch":2,"max_native_output_tokens":128,
        "torch":torch.__version__,"transformers":importlib.metadata.version("transformers")})
    for path in (Path(__file__),PLAN,Path(c.__file__),Path("experiments/structure_net/ncm_native_readout.py"),
        Path("experiments/structure_net/ncm_native_parent_control.py"),Path("experiments/structure_net/ncm_source_readout_grade.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root/"producer_source"/("transfer_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-native-readout-transfer-v1"
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


@torch.no_grad()
def generate(pilot):
    cfg = c.Config()
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID"
        or not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<7_500_000_000):
        raise ValueError("planned GPU0 headroom required")
    if q.read(c.CAPACITY/"result.json")["status"]!="capacity_gate_passed" or q.read(c.CAPACITY/"worker_terminal_receipt.json")["exit_code"]!=0:
        raise ValueError("terminal passed AR capacity endpoint required")
    verify(q.read(c.CAPACITY/"producer.json"))
    if not pilot and (q.read(PILOT/"result.json")["status"]!="passed" or q.read(PILOT/"worker_terminal_receipt.json")["exit_code"]!=0):
        raise ValueError("matching terminal transfer pilot required")
    cases = q.read(PREP/"development_sources.json")[:4 if pilot else 64]; verify_public(cases)
    root = PILOT if pilot else OUTPUTS
    inputs = [PREP/"development_sources.json",*[c.CAPACITY/n for n in ("result.json","producer.json","model.pt","source_features.pt","outputs.json")],
        *[p for p in MODEL_PATH.iterdir() if p.is_file()]]
    p = producing(root,"systems_lifecycle_only_not_quality_evidence" if pilot else "fixed_native_AR_readout_development",
        inputs,{"pilot":pilot,"cases":len(cases),"precision":"BF16","attention":"sdpa","seed":42})
    if not pilot and any(q.read(PILOT/"producer.json")[k]!=p[k] for k in ("source_hashes","input_hashes")):
        raise ValueError("transfer producer changed after pilot")
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    language = AutoModelForCausalLM.from_pretrained(MODEL_PATH,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if module_hash(language.model)!=EXPECTED_BODY: raise ValueError("original body changed")
    features,masks = [],[]
    for i in range(0,len(cases),2):
        x,_ = native_inputs(cases[i:i+2],tokenizer,"cuda")
        output = language.model(**x,output_hidden_states=True,use_cache=False)
        features.append(torch.stack([output.hidden_states[k] for k in cfg.source_layers],2).cpu())
        masks.append(x["attention_mask"].cpu()); del output,x
    features,masks = torch.cat(features),torch.cat(masks)
    if module_hash(language.model)!=EXPECTED_BODY: raise ValueError("body changed during extraction")
    torch.save({"features":features,"masks":masks},root/"source_features.pt")
    language.model.layers = torch.nn.ModuleList([language.model.layers[i] for i in cfg.readout_blocks])
    language.config = copy.deepcopy(language.config); language.config.num_hidden_layers = len(cfg.readout_blocks)
    language.model.config = language.config
    for i,layer in enumerate(language.model.layers): layer.self_attn.layer_idx = i
    gc.collect(); torch.cuda.empty_cache()
    head = NativeReadout(language,rank=cfg.rank).eval()
    checkpoint = torch.load(c.CAPACITY/"model.pt",map_location="cpu",weights_only=True)
    restore_trainable(head,checkpoint["trainable"]); del checkpoint
    before = module_hash(head)
    if pilot:
        training,_ = c.selected_cases(); saved = torch.load(c.CAPACITY/"source_features.pt",weights_only=True)
        _,_,sequences = c.targets_for(training,tokenizer,"cpu")
        free,raw = c.evaluate(head,saved["features"],saved["masks"],training,sequences,tokenizer)
        shuffled,raw_shuffled = c.evaluate(head,saved["features"],saved["masks"],training,sequences,tokenizer,True)
        original = q.read(c.CAPACITY/"result.json")
        if free!=original["free"] or shuffled!=original["shuffled"] or raw!=q.read(c.CAPACITY/"outputs.json"):
            raise ValueError("capacity replay differs; do not rewrite original endpoint")
        write_json(root/"capacity_free_replay.json",raw); write_json(root/"capacity_shuffled_replay.json",raw_shuffled)
        del saved
    def run():
        results = []
        for i in range(0,len(cases),2):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                values = head.generate(features[i:i+2].cuda(),masks[i:i+2].cuda(),tokenizer.bos_token_id,tokenizer.eos_token_id)
            results.extend({**case,"text":tokenizer.decode(ids,skip_special_tokens=True).strip(),
                "native_output_ids":ids,"NCM_model":False,"source_truncated":False} for case,ids in zip(cases[i:i+2],values))
        return results
    outputs = run()
    if pilot and outputs!=run(): raise ValueError("transfer greedy repetition differs")
    if not pilot and outputs[:4]!=q.read(PILOT/"outputs.json"):
        raise ValueError("first-four transfer outputs changed")
    if module_hash(head)!=before: raise ValueError("fixed native readout parameters changed")
    write_json(root/"outputs.json",outputs)
    if q.read(root/"outputs.json")!=outputs: raise ValueError("transfer output reload differs")
    verify(p)
    result = {"status":"passed" if pilot else "generated_pending_task_grading","cases":len(cases),
        "capacity_replay_exact":True if pilot else None,"frozen_readout_unchanged":True,"trained":False,
        "output_sha256":file_hash(root/"outputs.json"),"producer_fingerprint":p["fingerprint"],
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"NCM_model":False,"task_accuracy":None,
        "NCM_success_established":False,"final_test_opened":False,"speed_tested":False}
    write_json(root/"result.json",result); print(json.dumps(result),flush=True)


def grade(kind):
    from experiments.structure_net import ncm_source_readout_grade as g
    from experiments.structure_net.ncm_native_label_quality import dependencies
    dependencies()
    if q.read(OUTPUTS/"worker_terminal_receipt.json")["exit_code"]!=0:
        raise ValueError("fixed generation must be terminal")
    verify(q.read(OUTPUTS/"producer.json"))
    if file_hash(OUTPUTS/"outputs.json")!=q.read(OUTPUTS/"result.json")["output_sha256"]:
        raise ValueError("fixed outputs changed")
    role = "primary" if kind=="primary" else "secondary"
    destination = PRIMARY if kind=="primary" else JUDGE_PILOT if kind=="judge-pilot" else GRADING
    if kind=="grade":
        if q.read(JUDGE_PILOT/"result.json")["endpoint"]["valid"]!=4 or q.read(JUDGE_PILOT/"worker_terminal_receipt.json")["exit_code"]!=0:
            raise ValueError("terminal four-valid-judgment pilot required")
    if kind=="primary":
        prior = q.read(GRADING/"result.json")["endpoint"]
        if prior["status"]!="completed" or prior["passed"]<61 or q.read(GRADING/"worker_terminal_receipt.json")["exit_code"]!=0:
            raise ValueError("terminal passing secondary gate required")
        if datetime.now(timezone.utc)<NOT_BEFORE: raise ValueError("provider reset has not occurred")
    seed_roots = [PARENT_GRADING]+([JUDGE_PILOT] if kind=="grade" else [GRADING] if kind=="primary" else [])
    inputs = [OUTPUTS/n for n in ("outputs.json","result.json","producer.json")]
    cache = {}
    for root in seed_roots:
        source = root/"judgment_cache.json"; inputs.append(source)
        for key,entry in q.read(source).items():
            g.audit_row(entry["packet"],entry["role"],entry["row"])
            if key!=g.packet_key(entry["packet"],entry["role"]): raise ValueError("prior cache key differs")
            inputs.extend(map(Path,entry["row"]["attempts"]))
            if key not in cache: g.remember(cache,key,entry)
    p = producing(destination,"systems_lifecycle_only_not_quality_evidence" if kind=="judge-pilot" else "native_AR_readout_task_accuracy",
        inputs,{"role":role,"systems_only":kind=="judge-pilot","minimum_full_passes":61})
    write_json(destination/"judgment_cache.json",cache)
    g.DEST,g.freeze = destination,producing
    cases = q.read(OUTPUTS/"outputs.json")[:4 if kind=="judge-pilot" else 64]
    endpoint = g.grade_condition(cases,"native_readout",role,cache,inputs)
    verify(p)
    result = {"status":endpoint["status"],"endpoint":endpoint,"systems_only":kind=="judge-pilot",
        "NCM_model":False,"NCM_success_established":False,"final_test_opened":False,"speed_tested":False,
        "producer_fingerprint":p["fingerprint"]}
    write_json(destination/"result.json",result); print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("command",choices=["pilot","generate","judge-pilot","grade","primary"])
    command = parser.parse_args().command
    generate(command=="pilot") if command in ("pilot","generate") else grade(command)
