"""Training-only capacity of an actual native coarse-to-fine NCM adaptation."""
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
from experiments.structure_net import ncm_native_readout_control as ar
from experiments.structure_net import ncm_native_concept_control as concept
from experiments.structure_net.ncm_frozen_body import MODEL_PATH, module_hash
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_concepts import ConceptConfig, conditional_target, encode, expand
from experiments.structure_net.ncm_native_parallel import NativeConceptGenerator, prior_for_level
from experiments.structure_net.ncm_native_readout import NativeReadout, frozen_hash, restore_trainable, trainable_state
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-native-parallel-capacity-plan.md")
PILOT = ROOT/"20261023_native_parallel_pilot_v1"
CAPACITY = ROOT/"20261023_native_parallel_capacity_v1"


@dataclass(frozen=True)
class Config:
    steps: int = 1024
    coarse_warmup: int = 256
    batch: int = 2
    accumulation: int = 4
    lr: float = 1e-4
    weight_decay: float = .01
    seed: int = 42
    sampler_seed: int = 2042
    corruption_seed: int = 3042
    level_seed: int = 4042
    self_history_probability: float = .5
    word_level_probability: float = .5
    corruption_rates: tuple = (.5,.05,1/30,1/60,0.)


def producing(root,pilot,extra):
    inputs = [ar.CAPACITY/n for n in ("model.pt","result.json","producer.json","source_features.pt","worker_terminal_receipt.json")]
    inputs += [concept.FIT/n for n in ("concepts.pt","result.json","producer.json","worker_terminal_receipt.json")]
    inputs += [p for p in MODEL_PATH.iterdir() if p.is_file()]+extra
    p = freeze(root,"systems_lifecycle_only_not_quality_evidence" if pilot else "native_parallel_training_capacity_only",
        inputs,{**asdict(Config()),"pilot":pilot,"physical_gpu":0,"precision":"BF16","torch":torch.__version__,
        "transformers":importlib.metadata.version("transformers"),
        "NCM_model":True,"unchanged_paper_replication":False,"training_only":True})
    for path in (Path(__file__),PLAN,Path(ar.__file__),Path(concept.__file__),
        Path("experiments/structure_net/ncm_native_parallel.py"),Path("experiments/structure_net/ncm_native_concepts.py"),
        Path("experiments/structure_net/ncm_native_readout.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root/"producer_source"/("native_parallel_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-native-parallel-capacity-v1"
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


@torch.no_grad()
def noisy_prior(net,codes,level,generator,rates):
    changed = [value.clone() for value in codes]
    for i in range(level):
        mask = torch.rand(changed[i].shape,generator=generator)<rates[i]
        replacements = torch.randint(len(net.books[i]),changed[i].shape,generator=generator)
        changed[i] = torch.where(mask.to(changed[i].device),replacements.to(changed[i].device),changed[i])
    return prior_for_level(net.books,changed,level,net.canvas)


@torch.no_grad()
def predicted_prior(net,features,masks,level):
    previous = net.training; net.eval()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        prior = net.queries.new_zeros(len(features),net.canvas,net.queries.shape[1])
        if level:
            cache = net.prefix_cache(features,masks)
            for i,book in enumerate(net.books[:level]):
                labels = net.cached_level_logits(cache,masks,prior,i).argmax(-1)
                prior = prior+expand(book[labels],net.canvas)
    net.train(previous)
    return prior.detach()


@torch.no_grad()
def measure(net,features,masks,cases,sequences,tokenizer,mode="free",free_codes=None):
    net.eval(); outputs = []; collected = [[] for _ in net.levels]
    if mode=="source_shuffle": features,masks = features.roll(1,0),masks.roll(1,0)
    for start in range(0,len(cases),2):
        x,mask = features[start:start+2].cuda(),masks[start:start+2].cuda()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            if mode=="concept_shuffle":
                codes = [c.roll(1,0)[start:start+2].cuda() for c in free_codes]
                prior = prior_for_level(net.books,codes,len(net.levels),net.canvas)
                cache = net.prefix_cache(x,mask)
                words = net.cached_level_logits(cache,mask,prior,len(net.levels)).argmax(-1)
            else:
                codes,words = net.generate(x,mask,use_hierarchy=mode!="zero")
        for out,ids in zip(collected,codes): out.append(ids.cpu())
        for case,row,gold in zip(cases[start:start+2],words.tolist(),sequences[start:start+2]):
            ids = row[:row.index(tokenizer.eos_token_id)+1] if tokenizer.eos_token_id in row else row
            outputs.append({"id":case["id"],"text":tokenizer.decode(ids,skip_special_tokens=True).strip(),
                "native_output_ids":ids,"exact":ids==gold,"terminated":ids[-1]==tokenizer.eos_token_id,
                "correct":sum(i<len(ids) and ids[i]==v for i,v in enumerate(gold)),"total":len(gold)})
    correct,total = sum(r["correct"] for r in outputs),sum(r["total"] for r in outputs)
    metrics = {"cases":len(cases),"correct":correct,"total":total,"token_accuracy":correct/total,
        "exact":sum(r["exact"] for r in outputs),"task_accuracy":None,"training_only":True}
    return metrics,outputs,[torch.cat(c) for c in collected] if mode!="zero" else []


def main(pilot):
    cfg = Config()
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID"
        or not torch.cuda.is_available() or torch.cuda.mem_get_info()[0]<6_000_000_000):
        raise ValueError("planned GPU0 with6GB free required; do not overlap AR generation")
    for root,status in ((ar.CAPACITY,"capacity_gate_passed"),(concept.FIT,"representation_invariants_passed")):
        if q.read(root/"result.json")["status"]!=status or q.read(root/"worker_terminal_receipt.json")["exit_code"]!=0:
            raise ValueError("terminal passed component control required")
        verify(q.read(root/"producer.json"))
    cases,extra = ar.selected_cases(); cases = cases[:4] if pilot else cases
    root = PILOT if pilot else CAPACITY
    if not pilot and (q.read(PILOT/"result.json")["status"]!="passed" or q.read(PILOT/"worker_terminal_receipt.json")["exit_code"]!=0):
        raise ValueError("terminal native parallel CUDA pilot required")
    p = producing(root,pilot,extra)
    if not pilot and q.read(PILOT/"producer.json")["source_hashes"]!=p["source_hashes"]:
        raise ValueError("native parallel producer changed after pilot")
    torch.manual_seed(cfg.seed); torch.backends.cuda.matmul.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    language = AutoModelForCausalLM.from_pretrained(MODEL_PATH,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").eval().requires_grad_(False)
    language.model.layers = torch.nn.ModuleList([language.model.layers[i] for i in ar.Config().readout_blocks])
    language.config = copy.deepcopy(language.config); language.config.num_hidden_layers = 5; language.model.config = language.config
    for i,layer in enumerate(language.model.layers): layer.self_attn.layer_idx = i
    language.cuda(); readout = NativeReadout(language,rank=16)
    initial = torch.load(ar.CAPACITY/"model.pt",map_location="cpu",weights_only=True)
    restore_trainable(readout,initial["trainable"]); del initial; gc.collect()
    saved = torch.load(concept.FIT/"concepts.pt",weights_only=True)
    net = NativeConceptGenerator(readout,saved["books"],tuple(saved["config"]["levels"])).cuda()
    book_before = [book.clone() for book in net.books]; frozen = frozen_hash(net.readout)
    source = torch.load(ar.CAPACITY/"source_features.pt",weights_only=True)
    features,masks = source["features"][:len(cases)],source["masks"][:len(cases)]
    targets,_,sequences = ar.targets_for(cases,tokenizer,"cpu")
    ids = targets.masked_fill(targets.eq(-100),tokenizer.eos_token_id).cuda()
    with torch.no_grad(): embedding = language.model.embed_tokens(ids).float()
    concept_cfg = ConceptConfig(); clean = encode(embedding,net.books,concept_cfg)[0]
    sampler = torch.Generator().manual_seed(cfg.sampler_seed)
    corruption = torch.Generator().manual_seed(cfg.corruption_seed)
    levels = torch.Generator().manual_seed(cfg.level_seed)
    parameters = [value for value in net.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=cfg.lr,weight_decay=cfg.weight_decay)
    trace = []; steps = 2 if pilot else cfg.steps; nlevels = len(net.levels)
    for step in range(steps):
        net.train(); optimizer.zero_grad(set_to_none=True)
        if pilot: level = 0 if step==0 else nlevels
        elif step<cfg.coarse_warmup: level = int(torch.randint(nlevels,(),generator=levels))
        else: level = nlevels if torch.rand((),generator=levels)<cfg.word_level_probability else int(torch.randint(nlevels,(),generator=levels))
        value = 0.; self_count = 0; accumulation = 1 if pilot else cfg.accumulation
        for _ in range(accumulation):
            pick = torch.randint(len(cases),(cfg.batch,),generator=sampler)
            x,mask = features[pick].cuda(),masks[pick].cuda()
            use_self = (pilot and level==nlevels) or (not pilot and step>=cfg.coarse_warmup and torch.rand((),generator=levels)<cfg.self_history_probability)
            prior = predicted_prior(net,x,mask,level) if use_self else noisy_prior(net,[v[pick] for v in clean],level,corruption,cfg.corruption_rates)
            self_count += bool(use_self)
            with torch.no_grad():
                labels = targets[pick].cuda() if level==nlevels else conditional_target(embedding[pick],prior,net.books[level],net.levels[level])
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits = net.level_logits(x,mask,prior,level)
                if not torch.isfinite(logits).all(): raise ValueError("nonfinite native concept logits")
                loss = F.cross_entropy(logits.float().flatten(0,1),labels.flatten())/accumulation
            if not torch.isfinite(loss): raise ValueError("nonfinite native concept loss")
            loss.backward(); value += loss.detach().item()
        if net.queries.grad is None or net.queries.grad.abs().sum()==0 or net.readout.fuser.weight.grad is None:
            raise ValueError("native concept interface gradients missing")
        if any(v.grad is not None for v in net.readout.parameters() if not v.requires_grad):
            raise ValueError("native base gradient leakage")
        norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True); optimizer.step()
        trace.append({"step":step+1,"level":level,"loss":value,"gradient_norm":float(norm),"self_history_microbatches":self_count})
        if (step+1)%64==0: print(json.dumps(trace[-1]),flush=True)
    net.eval(); relative = agreement = None
    if pilot:
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            x,mask = features[:2].cuda(),masks[:2].cuda()
            prior = prior_for_level(net.books,[c[:2] for c in clean],nlevels,128)
            cache = net.prefix_cache(x,mask); before_cache = [(k.clone(),v.clone()) for k,v in cache]
            a = net.level_logits(x,mask,prior,nlevels); b = net.cached_level_logits(cache,mask,prior,nlevels)
            relative = float((a.float()-b.float()).norm()/a.float().norm().clamp_min(1e-8))
            agreement = float(a.argmax(-1).eq(b.argmax(-1)).float().mean())
            if relative>.02 or agreement<.95 or any(not torch.equal(k,a) or not torch.equal(v,b) for (k,v),(a,b) in zip(cache,before_cache)):
                raise ValueError("native concept source-cache contract failed")
    state = trainable_state(net)
    torch.save({"trainable":state,"optimizer":optimizer.state_dict(),"sampler":sampler.get_state(),
        "corruption":corruption.get_state(),"levels":levels.get_state(),"producer_fingerprint":p["fingerprint"]},root/"model.pt")
    restore_trainable(net,torch.load(root/"model.pt",map_location="cpu",weights_only=True)["trainable"])
    reloaded = trainable_state(net)
    if any(not torch.equal(value,reloaded[name]) for name,value in state.items()): raise ValueError("native concept reload differs")
    free,outputs,codes = measure(net,features,masks,cases,sequences,tokenizer)
    if pilot and outputs!=measure(net,features,masks,cases,sequences,tokenizer)[1]: raise ValueError("native concept greedy repeat differs")
    ablations = {}
    if not pilot:
        for mode in ("source_shuffle","concept_shuffle","zero"):
            metrics,raw,_ = measure(net,features,masks,cases,sequences,tokenizer,mode,codes)
            ablations[mode] = metrics; write_json(root/(mode+"_outputs.json"),raw)
    if frozen_hash(net.readout)!=frozen or any(not torch.equal(a,b) for a,b in zip(net.books,book_before)):
        raise ValueError("native base or concept book changed")
    status = "passed"
    if not pilot:
        if free["token_accuracy"]<.98 or free["exact"]<31: status = "native_parallel_text_capacity_failed"
        elif free["token_accuracy"]-ablations["source_shuffle"]["token_accuracy"]<.05: status = "native_parallel_source_dependence_failed"
        elif free["token_accuracy"]-ablations["concept_shuffle"]["token_accuracy"]<.05: status = "native_parallel_concepts_unused"
        else: status = "native_parallel_capacity_passed"
    write_json(root/"trace.json",trace); write_json(root/"outputs.json",outputs); torch.save(codes,root/"predicted_concepts.pt")
    verify(p)
    result = {"status":status,"steps":steps,"free":free,"ablations":ablations,"cache_relative_l2":relative,
        "cache_top1_agreement":agreement,"frozen_native_base_and_books":True,"checkpoint_reload_exact":True,
        "NCM_model":True,"NCM_success_established":False,"task_accuracy":None,"training_only":True,
        "student_training_admitted":False,"final_test_opened":False,"speed_tested":False,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"producer_fingerprint":p["fingerprint"]}
    write_json(root/"result.json",result); print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("command",choices=["pilot","capacity"])
    main(parser.parse_args().command=="pilot")
