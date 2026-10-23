"""Train a small native-word readout on accepted examples as a capacity control."""
import argparse
import copy
from dataclasses import asdict, dataclass
import gc
import importlib.metadata
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body import MODEL_PATH, module_hash
from experiments.structure_net.ncm_matched_parent_control import EXPECTED_BODY
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_parent_control import native_inputs
from experiments.structure_net.ncm_native_readout import NativeReadout, frozen_hash, restore_trainable, trainable_state
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_source_readout_grade import audit_row

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-native-readout-recovery-plan.md")
QUALITY = Path("data/experiments/ncm_recipe_update/20261022_quality_v2")
PILOT = ROOT / "20261023_native_readout_pilot_v1"
CAPACITY = ROOT / "20261023_native_readout_capacity_v1"


@dataclass(frozen=True)
class Config:
    source_layers: tuple = (1,14,28)
    readout_blocks: tuple = (0,7,14,21,27)
    rank: int = 16
    steps: int = 1024
    batch: int = 2
    accumulation: int = 4
    lr: float = 1e-4
    weight_decay: float = .01
    seed: int = 42
    sampler_seed: int = 2042
    source_limit: int = 768
    target_limit: int = 128
    cache_relative_l2_tolerance: float = .02
    cache_top1_agreement_minimum: float = .95


def selected_cases():
    from experiments.structure_net.ncm_native_label_quality import dependencies
    dependencies(); import tiktoken
    cases = q.read(QUALITY/"selected.json")["train"]
    if len(cases) != 32 or q.read(QUALITY/"result.json")["status"] != "passed":
        raise ValueError("original32 accepted training examples required")
    verify(q.read(QUALITY/"producer.json")); verify_public(cases)
    packets = {p["case_id"]:p for p in q.read(QUALITY/"packets.json")}
    enc = tiktoken.get_encoding("cl100k_base"); inputs = []
    for case in cases:
        packet = packets[case["id"]]; row_path = QUALITY/"rows"/(case["id"]+".json")
        row = q.read(row_path); audit_row(packet,"primary",row)
        if (packet["candidates"][0]["text"] != case["target"] or row["status"] != "valid" or
            not q.passes(row["evaluations"][0],case["target"],len(enc.encode(case["target"],disallowed_special=())))):
            raise ValueError("original accepted label judgment differs")
        inputs.extend([row_path,*map(Path,row["attempts"])])
    return cases,inputs


def producing(root, pilot, extra):
    cfg = Config()
    inputs = [QUALITY/n for n in ("selected.json","result.json","producer.json","packets.json")]
    inputs += [p for p in MODEL_PATH.iterdir() if p.is_file()]+extra
    p = freeze(root,"systems_lifecycle_only_not_quality_evidence" if pilot else "native_word_training_capacity_only",
        inputs,{**asdict(cfg),"pilot":pilot,"NCM_model":False,"training_only":True,"precision":"BF16",
        "physical_gpu":0,"attention":"sdpa","native_base_weights_frozen":True,
        "torch":torch.__version__,"transformers":importlib.metadata.version("transformers")})
    for path in (Path(__file__),PLAN,Path("experiments/structure_net/ncm_native_readout.py"),
                 Path("experiments/structure_net/ncm_native_parent_control.py"),
                 Path("experiments/structure_net/ncm_native_label_quality.py"),
                 Path("experiments/structure_net/ncm_source_readout_grade.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root/"producer_source"/("native_head_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-native-readout-capacity-v1"
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
    write_json(root/"producer.json",p)
    return p


def targets_for(cases, tokenizer, device):
    sequences = [tokenizer.encode(r["target"],add_special_tokens=False)+[tokenizer.eos_token_id] for r in cases]
    if any(len(ids)>128 or tokenizer.decode(ids[:-1],skip_special_tokens=True) != r["target"] for ids,r in zip(sequences,cases)):
        raise ValueError("native target round trip or128-token capacity failed")
    target = torch.full((len(cases),128),-100,dtype=torch.long,device=device)
    for i,ids in enumerate(sequences): target[i,:len(ids)] = torch.tensor(ids,device=device)
    previous = torch.cat((torch.full((len(cases),1),tokenizer.bos_token_id,dtype=torch.long,device=device),
        target[:,:-1].masked_fill(target[:,:-1].eq(-100),tokenizer.eos_token_id)),1)
    return target,previous,sequences


def logits_loss(head, features, masks, previous, targets):
    with torch.autocast("cuda",dtype=torch.bfloat16):
        logits,_ = head.teacher_logits(features,masks,previous)
        if not torch.isfinite(logits).all(): raise ValueError("nonfinite native head logits")
        loss = F.cross_entropy(logits.float().flatten(0,1),targets.flatten())
    if not torch.isfinite(loss): raise ValueError("nonfinite native head loss")
    return loss


@torch.no_grad()
def evaluate(head, features, masks, cases, sequences, tokenizer, shuffled=False):
    head.eval(); outputs = []
    if shuffled: features,masks = features.roll(1,0),masks.roll(1,0)
    for i in range(0,len(cases),2):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            values = head.generate(features[i:i+2].cuda(),masks[i:i+2].cuda(),tokenizer.bos_token_id,tokenizer.eos_token_id)
        for case,ids,gold in zip(cases[i:i+2],values,sequences[i:i+2]):
            text = tokenizer.decode(ids,skip_special_tokens=True).strip()
            outputs.append({"id":case["id"],"text":text,"native_output_ids":ids,"exact":ids==gold,
                "terminated":bool(ids and ids[-1]==tokenizer.eos_token_id),
                "correct":sum(i<len(ids) and ids[i]==x for i,x in enumerate(gold)),"total":len(gold)})
    correct,total = sum(r["correct"] for r in outputs),sum(r["total"] for r in outputs)
    return {"cases":len(cases),"correct":correct,"total":total,"token_accuracy":correct/total,
        "exact":sum(r["exact"] for r in outputs),"task_accuracy":None,"training_only":True},outputs


def main(pilot):
    cfg = Config(); cases,extra = selected_cases()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("select physicalGPU0 before startup")
    if not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name() or torch.cuda.mem_get_info()[0]<7_500_000_000:
        raise ValueError("planned GPU0 headroom required")
    root = PILOT if pilot else CAPACITY
    if not pilot and (q.read(PILOT/"result.json")["status"] != "passed" or q.read(PILOT/"worker_terminal_receipt.json")["exit_code"] != 0):
        raise ValueError("terminal real-memory head pilot required")
    p = producing(root,pilot,extra)
    if not pilot and q.read(PILOT/"producer.json")["source_hashes"] != p["source_hashes"]:
        raise ValueError("head producer changed after pilot")
    cases = cases[:4] if pilot else cases
    torch.manual_seed(cfg.seed); torch.cuda.reset_peak_memory_stats()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    language = AutoModelForCausalLM.from_pretrained(MODEL_PATH,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if module_hash(language.model) != EXPECTED_BODY: raise ValueError("original body identity changed")
    features,masks = [],[]
    for i in range(0,len(cases),2):
        inputs,_ = native_inputs(cases[i:i+2],tokenizer,"cuda")
        with torch.inference_mode():
            output = language.model(**inputs,output_hidden_states=True,use_cache=False)
            value = torch.stack([output.hidden_states[k] for k in cfg.source_layers],dim=2).cpu()
        features.append(value.clone()); masks.append(inputs["attention_mask"].cpu().clone())
        del output,value,inputs
    features,masks = torch.cat(features),torch.cat(masks)
    if module_hash(language.model) != EXPECTED_BODY: raise ValueError("body changed during source extraction")
    torch.save({"features":features,"masks":masks},root/"source_features.pt")
    if not torch.equal(torch.load(root/"source_features.pt",weights_only=True)["features"],features):
        raise ValueError("source feature reload differs")
    language.model.layers = torch.nn.ModuleList([language.model.layers[i] for i in cfg.readout_blocks])
    language.config = copy.deepcopy(language.config); language.config.num_hidden_layers = len(cfg.readout_blocks)
    language.model.config = language.config
    for i,layer in enumerate(language.model.layers): layer.self_attn.layer_idx = i
    gc.collect(); torch.cuda.empty_cache()
    head = NativeReadout(language,rank=cfg.rank)
    frozen = frozen_hash(head)
    target,previous,sequences = targets_for(cases,tokenizer,"cpu")
    parameters = [x for x in head.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=cfg.lr,weight_decay=cfg.weight_decay)
    sampler = torch.Generator().manual_seed(cfg.sampler_seed)
    trace = []; steps = 2 if pilot else cfg.steps
    for step in range(steps):
        head.train(); optimizer.zero_grad(set_to_none=True); value = 0.
        for _ in range(1 if pilot else cfg.accumulation):
            pick = torch.randint(len(cases),(cfg.batch,),generator=sampler)
            loss = logits_loss(head,features[pick].cuda(),masks[pick].cuda(),previous[pick].cuda(),target[pick].cuda())
            scaled = loss/(1 if pilot else cfg.accumulation); scaled.backward(); value += scaled.detach().item()
        if (not any(x.grad is not None and x.grad.abs().sum()>0 for x in parameters) or
            any(x.grad is not None for x in head.parameters() if not x.requires_grad)):
            raise ValueError("native head gradient routing failed")
        norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True); optimizer.step()
        trace.append({"step":step+1,"loss":value,"gradient_norm":float(norm)})
        if (step+1)%64 == 0: print(json.dumps(trace[-1]),flush=True)
    if pilot:
        # Full128-token target backward exercises the intended activation ceiling.
        optimizer.zero_grad(set_to_none=True)
        worst = target[:2].clone().masked_fill(target[:2].eq(-100),tokenizer.eos_token_id)
        logits_loss(head,features[:2].cuda(),masks[:2].cuda(),previous[:2].cuda(),worst.cuda()).backward()
        optimizer.zero_grad(set_to_none=True)
        head.eval()
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            x,mask,words = features[:2].cuda(),masks[:2].cuda(),previous[:2,:8].cuda()
            full,_ = head.teacher_logits(x,mask,words)
            first,past = head.teacher_logits(x,mask,words[:,:1],use_cache=True)
            incremental = [first]; cmask = torch.cat((mask,torch.ones_like(words[:,:1])),1)
            for i in range(1,8):
                v,past,cmask = head.cached_step(words[:,i:i+1],past,cmask); incremental.append(v)
            incremental = torch.cat(incremental,1)
            relative = float((full.float()-incremental.float()).norm()/full.float().norm().clamp_min(1e-8))
            agreement = float(full.argmax(-1).eq(incremental.argmax(-1)).float().mean())
            if relative>cfg.cache_relative_l2_tolerance or agreement<cfg.cache_top1_agreement_minimum:
                raise ValueError("native BF16 cached/full numerical contract failed")
    else: relative,agreement = None,None
    state = trainable_state(head)
    torch.save({"trainable":state,"optimizer":optimizer.state_dict(),"sampler":sampler.get_state(),
        "step":steps,"producer_fingerprint":p["fingerprint"],"NCM_model":False},root/"model.pt")
    restore_trainable(head,torch.load(root/"model.pt",weights_only=True)["trainable"])
    reloaded = trainable_state(head)
    if any(not torch.equal(v,reloaded[k]) for k,v in state.items()):
        raise ValueError("native trainable checkpoint differs")
    free,outputs = evaluate(head,features,masks,cases,sequences,tokenizer)
    if pilot and outputs != evaluate(head,features,masks,cases,sequences,tokenizer)[1]:
        raise ValueError("native greedy repetition differs")
    shuffled = None if pilot else evaluate(head,features,masks,cases,sequences,tokenizer,True)[0]
    if frozen_hash(head) != frozen: raise ValueError("native base weights changed")
    gate = not pilot and free["token_accuracy"]>=.98 and free["exact"]>=31 and free["token_accuracy"]-shuffled["token_accuracy"]>=.05
    write_json(root/"trace.json",trace); write_json(root/"outputs.json",outputs); verify(p)
    result = {"status":"passed" if pilot else ("capacity_gate_passed" if gate else "native_readout_capacity_failed"),
        "steps":steps,"free":free,"shuffled":shuffled,"cache_relative_l2":relative,"cache_top1_agreement":agreement,
        "frozen_native_parameters_unchanged":True,"original_body_changed":False,"trainable_parameters":sum(x.numel() for x in parameters),
        "frozen_parameters":sum(x.numel() for x in head.parameters() if not x.requires_grad),
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"checkpoint_reload_exact":True,
        "producer_fingerprint":p["fingerprint"],"task_accuracy":None,"training_only":True,
        "NCM_model":False,"NCM_success_established":False,"scaled_student_training_admitted":False,
        "final_test_opened":False,"speed_tested":False}
    write_json(root/"result.json",result); print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("command",choices=["pilot","capacity"])
    main(parser.parse_args().command == "pilot")
