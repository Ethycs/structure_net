"""Frozen-native NCM: broad language distillation, then accepted-summary SFT."""
import argparse
from dataclasses import asdict,dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_query_dataset as data
from experiments.structure_net import ncm_native_query_cache_prepare as cache_preparation
from experiments.structure_net.ncm_frozen_body import MODEL_PATH,module_hash
from experiments.structure_net.ncm_matched_parent_control import EXPECTED_BODY
from experiments.structure_net.ncm_memory_data import file_hash,write_json
from experiments.structure_net.ncm_native_concepts import ConceptConfig,conditional_target
from experiments.structure_net.ncm_native_parent_control import native_inputs
from experiments.structure_net.ncm_native_query_cache import NativeQueryBackbone
from experiments.structure_net.ncm_native_query_concepts import NativeQueryConcepts
from experiments.structure_net.ncm_native_query_distillation import pack_prefix,sparse_word_loss,conditional_teacher_history
from experiments.structure_net.ncm_native_query_training import actual_history
from experiments.structure_net.ncm_native_word_neighborhood import native_word_logits,initial_state,parallel_generate,noise_width,special_policy
from experiments.structure_net.ncm_native_readout import frozen_hash,trainable_state,restore_trainable
from experiments.structure_net.ncm_paper_data import ROOT,PREP,freeze,verify

PILOT = ROOT/"20261023_native_query_neighborhood_pilot_v1"
TRAIN = ROOT/"20261023_native_query_neighborhood_train_v1"
OUTPUTS = ROOT/"20261023_native_query_neighborhood_outputs_v1"
GRADE = ROOT/"20261023_native_query_neighborhood_grade_v1"
AMENDMENT = Path("docs/07 - Status Reports/2026-10-23_ncm-native-word-neighborhood-amendment.md")


@dataclass(frozen=True)
class Config:
    language_steps: int = 1024
    summary_steps: int = 1024
    coarse_warmup: int = 128
    batch: int = 2
    accumulation: int = 4
    lr: float = 1e-4
    weight_decay: float = .01
    clip: float = 1.
    actual_history_probability: float = .5
    word_level_probability: float = .5
    language_replay_probability: float = .25
    language_teacher_weight: float = .5
    summary_teacher_weight: float = .1
    native_center_probability: float = .5
    native_center_loss_weight: float = 10.
    native_noise_rms: float = .1
    word_refinements: int = 4
    vocabulary_chunk: int = 2048
    corruption_rates: tuple = (.5,.05,1/30,1/60,0.)


def streams():
    return {name:torch.Generator().manual_seed(seed) for name,seed in
        (("sampler",2042),("history",3042),("level",4042),("task",5042),("corruption",6042))}


def require_gpu():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 7_500_000_000):
        raise ValueError("planned GPU0 with7.5GBfree required")


def producing(root,stage,inputs,config):
    paths = [Path(__file__),data.PLAN,AMENDMENT,Path(data.__file__),Path(cache_preparation.__file__),
        Path("experiments/structure_net/ncm_native_query_concepts.py"),Path("experiments/structure_net/ncm_native_query_cache.py"),
        Path("experiments/structure_net/ncm_native_query_training.py"),Path("experiments/structure_net/ncm_native_query_distillation.py"),
        Path("experiments/structure_net/ncm_native_word_neighborhood.py"),
        Path("experiments/structure_net/ncm_native_readout.py"),Path("experiments/structure_net/ncm_native_concepts.py"),
        Path("experiments/structure_net/ncm_native_parent_control.py")]
    p = freeze(root,stage,inputs+paths,{**asdict(Config()),**config,"NCM_model":True,
        "original_weights_frozen":True,"native_blocks":28,"word_frame":128,"seed":42,"physical_gpu":0,
        "precision":"BF16native/FP32new","allow_tf32":False,"teacher_topk":32,"teacher_topk_renormalized":True,
        "torch":torch.__version__,"cuda":torch.version.cuda,"transformers":importlib.metadata.version("transformers"),
        "allocator":os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),"paper_replication":False})
    for path in paths:
        p["source_hashes"][str(path)] = file_hash(path)
        (root/"producer_source"/("neighborhood_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
    write_json(root/"producer.json",p)
    return p


def model_and_tokenizer():
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    native = AutoModelForCausalLM.from_pretrained(MODEL_PATH,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if module_hash(native.model) != EXPECTED_BODY: raise ValueError("original native body identity differs")
    books = torch.load(cache_preparation.BOOKS/"concepts.pt",map_location="cpu",weights_only=True)
    cfg = ConceptConfig(**books["config"])
    net = NativeQueryConcepts(NativeQueryBackbone(native,rank=8),books["books"],cfg.levels,cfg.canvas,16).eval()
    return net,tokenizer


def load_cache(root):
    verify(q.read(root/"producer.json"))
    if q.read(root/"worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("cache worker must be terminal")
    result = q.read(root/"result.json")
    if result["status"] not in ("passed","prepared_and_verified") or result["manifest_sha256"] != file_hash(root/"manifest.json"):
        raise ValueError("cache manifest invalid")
    manifest = q.read(root/"manifest.json")
    loaded = {}
    for i,(key,entry) in enumerate(manifest["cases"].items(),1):
        if file_hash(entry["path"]) != entry["sha256"]: raise ValueError("cache file changed")
        value = torch.load(entry["path"],map_location="cpu",weights_only=True)
        if value["id"] != key or value["producer_fingerprint"] != manifest["producer_fingerprint"]:
            raise ValueError("cache provenance changed")
        loaded[key] = value
        if i%128 == 0: print(json.dumps({"CPU_cache_loaded":i,"total":len(manifest["cases"])}),flush=True)
    return {name:[loaded[key] for key in keys] for name,keys in manifest["cohorts"].items()},manifest


def cache_digest(cohorts):
    h = hashlib.sha256()
    for name,examples in cohorts.items():
        h.update(name.encode())
        for row in examples:
            prefix = row["prefix"]
            values = [v for pair in prefix["keys_values"] for v in pair]+[prefix["source_mask"],prefix["anchor_token"]]
            for value in values:
                h.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def batch_loss(net,examples,level,generators,config,tokenizer,step,total,forced_center=None):
    device = net.backbone.language.device
    memory = pack_prefix([e["prefix"] for e in examples],device)
    targets = torch.stack([e["targets"] for e in examples]).to(device)
    with torch.no_grad():
        embedding = net.backbone.language.model.embed_tokens(targets.masked_fill(targets.eq(-100),tokenizer.eos_token_id)).float()
    word = level == len(net.levels)
    center = (forced_center if forced_center is not None else float(torch.rand((),generator=generators["history"]))<config.native_center_probability) if word else False
    actual = word or float(torch.rand((),generator=generators["history"])) < config.actual_history_probability
    if actual:
        _,prior = actual_history(net,memory,level,generators["corruption"],config.corruption_rates)
    else:
        _,prior = conditional_teacher_history(net,embedding,level,generators["corruption"],config.corruption_rates)
    width = 0; rollout = 0
    if word:
        if center:
            state = targets.masked_fill(targets.eq(-100),tokenizer.eos_token_id)
        else:
            with torch.no_grad():
                state = initial_state(net,prior,tokenizer,config.vocabulary_chunk)
                rollout = int(torch.randint(config.word_refinements,(1,),generator=generators["corruption"]))
                for _ in range(rollout): state = native_word_logits(net,memory,prior,state).argmax(-1)
                width = noise_width(step,total,config.coarse_warmup)
                position = torch.arange(net.canvas,device=targets.device)[None]
                start = (targets.ne(-100).sum(-1,keepdim=True)-width).clamp_min(0)
                state = torch.where(position>=start,state,targets.masked_fill(targets.eq(-100),tokenizer.eos_token_id))
        logits = native_word_logits(net,memory,prior,state,config.native_noise_rms if center else 0.,generators["corruption"])
    else: logits = net.level_logits(memory,prior,level)
    if not torch.isfinite(logits).all(): raise ValueError("nonfinite NCM training logits")
    if level < len(net.levels):
        labels = conditional_target(embedding,prior,net.books[level],net.levels[level])
        loss = F.cross_entropy(logits.float().flatten(0,1),labels.flatten())
        static = torch.stack([e["clean_codes"][level] for e in examples]).to(device)
        changed = float(labels.ne(static).float().mean())
    else:
        index = torch.stack([e["teacher_indices"] for e in examples]).to(device)
        probability = torch.stack([e["teacher_probabilities"] for e in examples]).to(device)
        weight = config.language_teacher_weight if examples[0]["kind"] == "language" else config.summary_teacher_weight
        loss = sparse_word_loss(logits,targets,index,probability,weight,[e["window_eos_injected"] for e in examples])
        changed = None
    if not torch.isfinite(loss): raise ValueError("nonfinite NCM training objective")
    unscaled = float(loss.detach())
    if word and center: loss = loss*config.native_center_loss_weight
    return loss,{"actual_history":actual,"conditional_expert_change":changed,"native_center":center if word else None,
                 "noise_window":width,"trajectory_passes":rollout,"unscaled_loss":unscaled}


@torch.no_grad()
def evaluate(net,examples,tokenizer):
    rows = []
    for start in range(0,len(examples),2):
        chunk = examples[start:start+2]
        memory = pack_prefix([e["prefix"] for e in chunk],"cuda")
        codes,words = parallel_generate(net,memory,tokenizer,passes=Config().word_refinements,chunk=Config().vocabulary_chunk)
        for i,(e,full) in enumerate(zip(chunk,words.tolist())):
            stops,_ = special_policy(net,tokenizer)
            end = next((j+1 for j,value in enumerate(full) if value in stops),len(full))
            active = full[:end]
            gold = e["targets"][e["targets"].ne(-100)].tolist()
            rows.append({"id":e["id"],"raw_parallel_ids":full,"active_ids":active,
                "text":tokenizer.decode(active,skip_special_tokens=True).strip(),"codes":[c[i].tolist() for c in codes],
                "correct":sum(j<len(active) and active[j]==v for j,v in enumerate(gold)),"total":len(gold)})
    correct,total = sum(r["correct"] for r in rows),sum(r["total"] for r in rows)
    return {"cases":len(rows),"correct":correct,"total":total,"token_recovery":correct/total,
            "task_accuracy":None,"target_conditioned_history":False},rows


def checkpoint(path,net,optimizer,generators,step,producer,config):
    if path.exists(): raise ValueError("checkpoint overwrite forbidden")
    state = {"trainable":trainable_state(net),"optimizer":optimizer.state_dict(),"step":step,
        "generator_states":{k:v.get_state() for k,v in generators.items()},"config":asdict(config),
        "producer_fingerprint":producer["fingerprint"],"torch_rng":torch.get_rng_state(),
        "cuda_rng":torch.cuda.get_rng_state_all()}
    torch.save(state,path)


def check_reload(path,net,optimizer,generators,examples,tokenizer):
    before = evaluate(net,examples,tokenizer)
    states = {k:v.get_state().clone() for k,v in generators.items()}
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad: p.zero_()
    state = torch.load(path,map_location="cpu",weights_only=True)
    restore_trainable(net,state["trainable"])
    optimizer.load_state_dict(state["optimizer"])
    for k,g in generators.items():
        g.set_state(state["generator_states"][k])
        if not torch.equal(g.get_state(),states[k]): raise ValueError("generator reload differs")
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    if before != evaluate(net,examples,tokenizer): raise ValueError("NCM optimizer/weight reload output differs")


def train(pilot):
    require_gpu(); config = Config()
    cache_root = cache_preparation.PILOT if pilot else cache_preparation.CACHE
    if not pilot:
        if (q.read(PILOT/"result.json")["status"] != "passed"
                or q.read(PILOT/"worker_terminal_receipt.json")["exit_code"] != 0):
            raise ValueError("terminal mixed-task distillation pilot required")
        verify(q.read(PILOT/"producer.json"))
    cohorts,manifest = load_cache(cache_root)
    root = PILOT if pilot else TRAIN
    steps = 4 if pilot else config.language_steps+config.summary_steps
    accumulation = 1 if pilot else config.accumulation
    inputs = [cache_root/n for n in ("producer.json","result.json","manifest.json","worker_terminal_receipt.json")]
    inputs += [cache_preparation.BOOKS/n for n in ("concepts.pt","producer.json","result.json")]
    inputs += [data.DEST/n for n in ("producer.json","cases.json","result.json")]
    inputs += [p for p in MODEL_PATH.iterdir() if p.is_file()]
    if not pilot: inputs += [PILOT/n for n in ("producer.json","result.json","worker_terminal_receipt.json")]
    p = producing(root,"systems_mixed_task_distillation_only" if pilot else "native_query_language_and_summary_training",
        inputs,{"pilot":pilot,"total_updates":steps,"accumulation":accumulation,
                "counts":{k:len(v) for k,v in cohorts.items()},"module_mode":"eval with gradients enabled"})
    if not pilot and q.read(PILOT/"producer.json")["source_hashes"] != p["source_hashes"]:
        raise ValueError("trainer changed after pilot")
    torch.cuda.reset_peak_memory_stats()
    net,tokenizer = model_and_tokenizer()
    frozen = frozen_hash(net); books = [b.cpu().clone() for b in net.books]
    cache_before = cache_digest(cohorts)
    parameters = [v for v in net.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=config.lr,weight_decay=config.weight_decay)
    generators = streams()
    (root/"checkpoints").mkdir()
    trace = []
    for step in range(steps):
        if pilot:
            task = ("language_train","language_train","summary_train","summary_train")[step]
            level = (0,5,2,5)[step]
        else:
            raw = step < config.language_steps or float(torch.rand((),generator=generators["task"])) < config.language_replay_probability
            task = "language_train" if raw else "summary_train"
            word = step >= config.coarse_warmup and float(torch.rand((),generator=generators["level"])) < config.word_level_probability
            level = len(net.levels) if word else int(torch.randint(len(net.levels),(1,),generator=generators["level"]))
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.; observations = []
        examples = cohorts[task]
        for _ in range(accumulation):
            pick = torch.randint(len(examples),(config.batch,),generator=generators["sampler"]).tolist()
            forced = (step==1) if pilot and level==len(net.levels) else None
            loss,observation = batch_loss(net,[examples[i] for i in pick],level,generators,config,tokenizer,step,steps,forced)
            scaled = loss/accumulation
            loss_value += float(scaled.detach())
            scaled.backward(); observations.append(observation)
            del loss,scaled
        if any(v.grad is not None for v in net.parameters() if not v.requires_grad):
            raise ValueError("original native parameter received gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters,config.clip,error_if_nonfinite=True)
        if pilot and not float(norm)>0: raise ValueError("pilot missing new-parameter gradients")
        optimizer.step()
        changes = [o["conditional_expert_change"] for o in observations if o["conditional_expert_change"] is not None]
        row = {"step":step+1,"task":task,"level":level,"loss":loss_value,"gradient_norm":float(norm),
            "actual_history_fraction":sum(o["actual_history"] for o in observations)/len(observations),
            "conditional_expert_change":sum(changes)/len(changes) if changes else None,
            "micro_observations":observations}
        trace.append(row)
        with (root/"trace.jsonl").open("a") as handle: handle.write(json.dumps(row,allow_nan=False)+"\n")
        if pilot or (step+1)%32 == 0:
            write_json(root/"progress.json",row); print(json.dumps(row),flush=True)
        if (step+1)%256 == 0 or step+1 == steps:
            checkpoint(root/"checkpoints"/f"step{step+1:04d}.pt",net,optimizer,generators,step+1,p,config)
        if not pilot and step+1 in (config.language_steps,steps):
            metrics,rows = evaluate(net,cohorts["language_diagnostic"],tokenizer)
            write_json(root/f"language_diagnostic_step{step+1}.json",{"metrics":metrics,"outputs":rows,
                "mechanism_only_not_summary_accuracy":True})
    if frozen_hash(net) != frozen: raise ValueError("original native weights changed")
    if any(not torch.equal(b.cpu(),old) for b,old in zip(net.books,books)): raise ValueError("fixed concept books changed")
    if cache_digest(cohorts) != cache_before: raise ValueError("CPU source cache mutated")
    checkpoint(root/"model.pt",net,optimizer,generators,steps,p,config)
    probe = cohorts["summary_train"][:4]
    check_reload(root/"model.pt",net,optimizer,generators,probe,tokenizer)
    metrics,rows = evaluate(net,probe,tokenizer)
    write_json(root/"reload_probe.json",{"metrics":metrics,"outputs":rows,"training_only":True})
    if any(v.enabled for v in net.backbone.adapters): raise ValueError("adapter context leaked")
    verify(p)
    result = {"status":"passed" if pilot else "trained_pending_task_evaluation","pilot":pilot,
        "updates":steps,"NCM_model":True,"task_accuracy":None,"NCM_success_established":False,
        "fixed_parallel_passes":len(net.levels)+config.word_refinements,"word_refinements":config.word_refinements,
        "original_frozen_hash_unchanged":frozen,"original_weights_frozen":True,"books_unchanged":True,
        "CPU_cache_digest_unchanged":cache_before,"checkpoint_generator_reload_exact":True,
        "trainable_parameters":sum(v.numel() for v in parameters),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "checkpoint_sha256":file_hash(root/"model.pt"),"final_test_opened":False,"speed_tested":False,
        "uses_AR_fallback_or_external_verifier":False,"producer_fingerprint":p["fingerprint"]}
    write_json(root/"result.json",result)
    print(json.dumps(result),flush=True)


@torch.no_grad()
def generate():
    require_gpu()
    if (q.read(TRAIN/"result.json")["status"] != "trained_pending_task_evaluation"
            or q.read(TRAIN/"worker_terminal_receipt.json")["exit_code"] != 0):
        raise ValueError("terminal complete fixed training run required")
    verify(q.read(TRAIN/"producer.json"))
    cases = q.read(PREP/"development_sources.json")
    from experiments.structure_net.ncm_recipe_teacher import verify_public
    verify_public(cases)
    train_groups = {c["group_id"] for rows in q.read(data.DEST/"cases.json").values() for c in rows}
    if train_groups&{c["group_id"] for c in cases} or len(cases) != 64:
        raise ValueError("fixed source-disjoint64case development set required")
    p = producing(OUTPUTS,"fixed_nine_pass_NCM_development",[TRAIN/n for n in ("model.pt","producer.json","result.json")]
        +[PREP/"development_sources.json",cache_preparation.BOOKS/"concepts.pt",data.DEST/"cases.json"]
        +[path for path in MODEL_PATH.iterdir() if path.is_file()],{"cases":64,"batch":2,"fixed_forward_passes":9})
    net,tokenizer = model_and_tokenizer()
    restore_trainable(net,torch.load(TRAIN/"model.pt",map_location="cpu",weights_only=True)["trainable"])
    before = frozen_hash(net)
    outputs = []
    stops,_ = special_policy(net,tokenizer)
    for start in range(0,64,2):
        chunk = cases[start:start+2]
        x,_ = native_inputs(chunk,tokenizer,"cuda")
        memory = net.backbone.source_cache(**x)
        codes,words = parallel_generate(net,memory,tokenizer,Config().word_refinements,chunk=Config().vocabulary_chunk)
        for i,(case,full) in enumerate(zip(chunk,words.tolist())):
            end = next((j+1 for j,value in enumerate(full) if value in stops),len(full))
            outputs.append({**case,"text":tokenizer.decode(full[:end],skip_special_tokens=True).strip(),
                "raw_parallel_ids":full,"native_output_ids":full[:end],"codes":[v[i].tolist() for v in codes],
                "source_truncated":False,"NCM_model":True,"fixed_forward_passes":9,"AR_fallback":False})
    if frozen_hash(net) != before: raise ValueError("fixed development inference changed original weights")
    write_json(OUTPUTS/"outputs.json",outputs)
    verify(p)
    result = {"status":"generated_pending_task_grade","cases":64,"NCM_model":True,
        "task_accuracy":None,"NCM_success_established":False,"fixed_forward_passes":9,
        "output_sha256":file_hash(OUTPUTS/"outputs.json"),"final_test_opened":False,"speed_tested":False,
        "producer_fingerprint":p["fingerprint"]}
    write_json(OUTPUTS/"result.json",result); print(json.dumps(result),flush=True)


def grade():
    from experiments.structure_net import ncm_budget128_profile as profile
    from experiments.structure_net import ncm_source_readout_grade as first
    from experiments.structure_net.ncm_native_label_quality import dependencies
    dependencies(); profile.install()
    if q.read(OUTPUTS/"worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("generation worker must be terminal")
    verify(q.read(OUTPUTS/"producer.json"))
    qualifications = []
    for role in ("secondary","primary"):
        root = ROOT/("20261023_budget128_judge_"+role+"_v1")
        verify(q.read(root/"producer.json"))
        if q.read(root/"result.json")["status"] != "passed": raise ValueError("qualified128token judges required")
        qualifications += [root/"producer.json",root/"result.json"]
    inputs = [OUTPUTS/n for n in ("outputs.json","producer.json","result.json")]+qualifications+[Path(profile.__file__),profile.PLAN]
    p = producing(GRADE,"whole_summary_NCM_source_disjoint_development",inputs,{"rubric":profile.RUBRIC,"cases":64,
        "secondary_then_primary_joint_minimum":61,"one_first_judgment_per_exact_packet_role":True})
    cache = {}; write_json(GRADE/"judgment_cache.json",cache)
    first.DEST = GRADE; first.freeze = producing
    cases = q.read(OUTPUTS/"outputs.json")
    secondary = first.grade_condition(cases,"summary","secondary",cache,inputs)
    primary = None
    if secondary["valid"] == 64 and secondary["passed"] >= 61:
        primary = first.grade_condition(cases,"summary","primary",cache,inputs)
    passed = primary is not None and primary["valid"] == 64 and primary["passed"] >= 61
    verify(p)
    result = {"status":"completed","secondary":secondary,"primary":primary,
        "classification":"short_development_gate_passed_final_required" if passed else "NCM_summary_development_failed",
        "NCM_model":True,"development_gate":passed,"NCM_success_established":False,
        "final_test_opened":False,"speed_tested":False,"producer_fingerprint":p["fingerprint"]}
    write_json(GRADE/"result.json",result); print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=["pilot","train","generate","grade"])
    command = parser.parse_args().command
    if command in ("pilot","train"): train(command=="pilot")
    elif command == "generate": generate()
    else: grade()
