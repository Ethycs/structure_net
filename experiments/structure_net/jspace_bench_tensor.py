"""JSPACE-BENCH-TENSOR-01: common task basis, Groebner compilation, retained residual."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from .jspace_bench_data import DATASETS, make_tasks, sha, write
from .jspace_groebner_tensor import (features, fit_coefficients, anchor_indices,
                                    graph_certificate, PolynomialResidualMLP)
from .jspace_lens import load_small_qwen


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data/experiments/jspace_bench_tensor/20261113_v1"
DATA_ROOT = ROOT / "data/benchmarks/jspace_four_tasks_20261113"
PROTOCOL = ROOT / "docs/07 - Status Reports/jspace/2026-11-13_jspace-bench-tensor-plan.md"
PARENT = ROOT / "data/experiments/jspace_attention_transport/20261113_v1/plan.json"
STUDY_ID = "JSPACE-BENCH-TENSOR-01"
CONFIG = {"seed": 861, "layers": [8,17,26], "train_per_task": 32, "dev_per_task": 16,
          "eval_per_task": 64, "gradient_per_task": 16, "sample_tokens": 16, "max_tokens": 384,
          "anchors": 16, "ridge": .01, "removed_counts": [128,512,1024],
          "shared_relation_nrmse": .5, "generation_tokens": 32,
          "development_agreement": 63/64, "development_kl": .02,
          "evaluation_agreement": 254/256, "minimum_speedup": 1.02,
          "dtype": "bfloat16", "scope": "prospective_single_checkpoint_exploratory"}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def save(path, value):
    if path.exists():
        raise FileExistsError(path)
    torch.save(value,path)


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU stages require CUDA_VISIBLE_DEVICES=''")


def sources():
    paths=[Path(__file__), ROOT/"experiments/structure_net/jspace_bench_data.py",
           ROOT/"experiments/structure_net/jspace_groebner_tensor.py", ROOT/"experiments/structure_net/jspace_lens.py",
           ROOT/"tests/structure_net/test_jspace_bench_tensor.py", PROTOCOL, ROOT/"pixi.lock",ROOT/"pyproject.toml"]
    return paths + sorted((ROOT/"src/structure_net_resources").glob("*.py"))


def prepare(root):
    cpu_only()
    from transformers import AutoTokenizer
    if root.exists():
        raise FileExistsError(root)
    model=json.loads(PARENT.read_text())["model"]
    tokenizer=AutoTokenizer.from_pretrained(model["path"],local_files_only=True,trust_remote_code=False)
    tasks=make_tasks(DATA_ROOT,tokenizer,CONFIG)
    labels=[tokenizer.encode(chr(65+i),add_special_tokens=False) for i in range(4)]
    if any(len(x)!=1 for x in labels):raise ValueError("Choice labels must be single tokens")
    value={"study_id":STUDY_ID,"config":CONFIG,"model":model,"label_ids":[x[0] for x in labels],
           "task_digest":digest(tasks),"sources":{str(p):sha(p) for p in sources()},
           "input_sha256":{str(PARENT):sha(PARENT),str(DATA_ROOT/"manifest.json"):sha(DATA_ROOT/"manifest.json")},
           "created_at":datetime.now(timezone.utc).isoformat(),"conformance":CONFIG["scope"]}
    value["scientific_fingerprint"]=digest(value)
    root.mkdir(parents=True)
    for path in sources():
        target=root/"producer"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    write(root/"tasks.json",tasks);write(root/"plan.json",value)
    return {"fingerprint":value["scientific_fingerprint"],"counts":{k:len(v) for k,v in tasks["splits"].items()}}


def verify(root, model=False):
    value=json.loads((root/"plan.json").read_text())
    if value["config"]!=CONFIG or value["study_id"]!=STUDY_ID or value["scientific_fingerprint"]!=digest({k:v for k,v in value.items() if k!="scientific_fingerprint"}):
        raise ValueError("Scientific fingerprint changed")
    for name,expected in value["sources"].items():
        if sha(name)!=expected or sha(root/"producer"/Path(name).relative_to(ROOT))!=expected:
            raise ValueError("Frozen producer changed: "+name)
    for name,expected in value["input_sha256"].items():
        if sha(name)!=expected:raise ValueError("Input changed")
    if digest(json.loads((root/"tasks.json").read_text()))!=value["task_digest"]:
        raise ValueError("Benchmark changed")
    if model:
        for name,expected in value["model"]["files"].items():
            if sha(Path(value["model"]["path"])/name)!=expected:raise ValueError("Checkpoint changed")
    return value


def checked(root, name):
    plan=verify(root);value=json.loads((root/f"{name}_manifest.json").read_text())
    if value["status"]!="completed" or value["fingerprint"]!=plan["scientific_fingerprint"]:
        raise ValueError("Incomplete stage")
    for file,expected in value["artifacts"].items():
        if sha(root/file)!=expected:raise ValueError("Stage artifact changed")
    return value


class Stage:
    def __init__(self,root,name):
        from structure_net_resources.client import require_gpu_lease
        self.lease=require_gpu_lease([0]);self.guard=require_gpu_lease
        self.plan=verify(root,model=True);self.root=root;self.name=name
        self.path=root/name;self.path.mkdir(exist_ok=False)
        self.started=time.perf_counter();self.counts=Counter()
        torch.set_num_threads(4);torch.manual_seed(CONFIG["seed"])
        torch.cuda.reset_peak_memory_stats()
        self.model,self.tokenizer=load_small_qwen(self.plan["model"]["path"],device="cuda:0",dtype=torch.bfloat16)
        self.tasks=json.loads((root/"tasks.json").read_text())["splits"]
        self.label_ids=torch.tensor(self.plan["label_ids"],device="cuda:0")
        self.original_parameters=sum(x.numel() for x in self.model.parameters())
        write(self.path/"lease.json",{k:self.lease[k] for k in ("id","holder","state","ready","allocation","end")})

    def event(self,kind,**value):
        self.counts[kind]+=1
        row={"kind":kind,"elapsed_seconds":time.perf_counter()-self.started,**value}
        with (self.path/"events.jsonl").open("a") as stream:stream.write(json.dumps(row,allow_nan=False)+"\n")
        if self.counts[kind]==1 or self.counts[kind]%16==0:
            self.guard([0]);print(json.dumps(row,allow_nan=False),flush=True)

    def finish(self,names,**extra):
        self.guard([0]);verify(self.root)
        peak=torch.cuda.max_memory_reserved()
        if peak>sum(self.lease["allocation"].values())*1024**2:raise RuntimeError("Reservation memory exceeded")
        files=[p for p in self.path.rglob("*") if p.is_file()]+[self.root/n for n in names]
        value={"study_id":STUDY_ID,"status":"completed","fingerprint":self.plan["scientific_fingerprint"],
               "stage":self.name,"elapsed_seconds":time.perf_counter()-self.started,"counts":dict(self.counts),
               "peak_reserved_bytes":peak,"lease_id":self.lease["id"],"model_parameters":self.original_parameters,
               "torch":str(torch.__version__),"gpu":torch.cuda.get_device_name(0),
               "artifacts":{str(p.relative_to(self.root)):sha(p) for p in files},**extra}
        write(self.root/f"{self.name}_manifest.json",value)
        return {k:v for k,v in value.items() if k!="artifacts"}


def score(stage,case):
    ids=torch.tensor([case["input_ids"]],device="cuda:0")
    hidden=stage.model.model(input_ids=ids,use_cache=False).last_hidden_state[:,-1]
    count=len(case["choices"])
    logits=F.linear(hidden,stage.model.lm_head.weight[stage.label_ids[:count]])[0].float()
    if not torch.isfinite(logits).all():raise FloatingPointError("Nonfinite choice logits")
    return logits


def collect_case(stage,case,gradient):
    states={};handles=[]
    if gradient:
        def root_hook(module,args):args[0].requires_grad_(True)
        handles.append(stage.model.model.layers[min(CONFIG["layers"])].mlp.register_forward_pre_hook(root_hook))
    for layer in CONFIG["layers"]:
        def remember(module,args,layer=layer):states[layer]=args[0]
        handles.append(stage.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(remember))
    try:
        with torch.set_grad_enabled(gradient):
            logits=score(stage,case)
            positions=torch.linspace(0,len(case["input_ids"])-1,CONFIG["sample_tokens"],device="cuda:0").long().unique()
            activations={k:v[0,positions].detach().cpu() for k,v in states.items()}
            gradients={}
            if gradient:
                others=[i for i in range(len(logits)) if i!=case["answer"]]
                margin=logits[case["answer"]]-logits[others].mean()
                grads=torch.autograd.grad(margin,[states[k] for k in CONFIG["layers"]])
                for k,g in zip(CONFIG["layers"],grads):
                    gradients[k]=torch.stack([g[0,-1].float(),g[0].float().mean(0)]).cpu()
            return activations,gradients,{"identity":case["identity"],"family":case["family"],"answer":case["answer"],
                                         "prediction":int(logits.argmax()),"logits":logits.detach().cpu().tolist()}
    finally:
        for h in handles:h.remove()


def capture(root,shakedown=False):
    if not shakedown:checked(root,"shakedown")
    stage=Stage(root,"shakedown" if shakedown else "capture")
    packed={};records={}
    for split in (("train",) if shakedown else ("train","development")):
        rows=stage.tasks[split]
        if shakedown:rows=[max([r for r in rows if r["family"]==f],key=lambda r:len(r["input_ids"])) for f in DATASETS]
        acts={k:[] for k in CONFIG["layers"]};grads={k:[] for k in CONFIG["layers"]};family=[];jfamily=[];counts=Counter();records[split]=[]
        for case in rows:
            gradient=split=="train" and counts[case["family"]]<CONFIG["gradient_per_task"]
            a,g,record=collect_case(stage,case,gradient)
            for k in acts:
                acts[k].append(a[k])
                if gradient:grads[k].append(g[k])
            family += [case["family"]]*len(a[CONFIG["layers"][0]])
            if gradient:jfamily += [case["family"]]*2
            records[split].append(record);counts[case["family"]]+=1
            stage.event(split,completed=len(records[split]),total=len(rows),gradient=gradient)
        packed[split]={"activations":{str(k):torch.cat(v) for k,v in acts.items()},"families":family,
                       "jacobians":{str(k):torch.cat(v) for k,v in grads.items() if v},"jacobian_families":jfamily}
    save(stage.path/"features.pt",packed);write(stage.path/"records.json",records)
    # Exact reload of captured inputs and task Jacobians.
    restored=torch.load(stage.path/"features.pt",map_location="cpu",weights_only=True)
    for split in packed:
        for key,value in packed[split]["activations"].items():
            assert torch.equal(value,restored[split]["activations"][key])
    if shakedown:
        original=stage.model.model.layers[CONFIG["layers"][0]].mlp
        n=original.down_proj.weight.shape[1];k=CONFIG["anchors"]
        dummy={"anchors":list(range(k)),"mean":torch.zeros(n),"scale":torch.ones(n),
               "coefficients":{"1":torch.zeros(n,1+k),"2":torch.zeros(n,1+k+k*(k+1)//2)}}
        replacement=PolynomialResidualMLP(original,dummy,list(range(k,2*k)))
        x=torch.randn(1,4,original.gate_proj.in_features,device="cuda:0",dtype=torch.bfloat16)
        with torch.inference_mode():
            before=replacement(x)
            save(stage.path/"replacement_state.pt",{k:v.cpu() for k,v in replacement.state_dict().items()})
            replacement.load_state_dict(torch.load(stage.path/"replacement_state.pt",map_location="cpu",weights_only=True))
            assert torch.equal(before,replacement(x)) and torch.isfinite(before).all()
            stage.model.model.layers[CONFIG["layers"][0]].mlp=replacement
            row=stage.tasks["train"][0]
            output=stage.model.generate(input_ids=torch.tensor([row["input_ids"]],device="cuda:0"),
                max_new_tokens=4,min_new_tokens=4,do_sample=False,use_cache=True,pad_token_id=stage.tokenizer.eos_token_id)
            assert output.shape[1]==len(row["input_ids"])+4
            stage.model.model.layers[CONFIG["layers"][0]].mlp=original
        write(stage.path/"replacement_lifecycle.json",{"finite":True,"state_reload_exact":True,"cache_generation_passed":True})
    return stage.finish([],scope="systems_lifecycle_only_not_quality_evidence" if shakedown else CONFIG["scope"])


def redundancy_report(errors, separate_errors, config):
    threshold=config["shared_relation_nrmse"]
    eligible=errors<=threshold
    counts=eligible.sum(1);uses=eligible.sum(0)
    return {"definition":"same normalized shared graph generator passes held-out activation reconstruction on each counted task",
            "nrmse_threshold":threshold,"task_order":list(DATASETS),
            "per_task_shared_basis_elements":counts.tolist(),
            "per_task_separate_fit_elements":(separate_errors<=threshold).sum(1).tolist(),
            "total_task_occurrences":int(counts.sum()),"distinct_shared_elements":int((uses>0).sum()),
            "redundant_occurrences":int(counts.sum()-(uses>0).sum()),
            "redundant_fraction":float((counts.sum()-(uses>0).sum())/counts.sum()) if counts.sum() else 0.,
            "shared_by_at_least_two_tasks":int((uses>=2).sum()),"shared_by_all_tasks":int((uses==len(DATASETS)).sum()),
            "pairwise_shared_counts":(eligible.int()@eligible.int().T).tolist(),
            "scope":"approximate cross-task reuse of identical generators in a fixed chart; not ideal dimension or task accuracy"}


def fit(root):
    checked(root,"capture");stage=Stage(root,"fit")
    captures=torch.load(root/"capture/features.pt",map_location="cpu",weights_only=True)
    results={};summaries={}
    for layer in CONFIG["layers"]:
        key=str(layer)
        train=captures["train"]["activations"][key].float().cuda()
        dev=captures["development"]["activations"][key].float().cuda()
        mean=train.mean(0);scale=train.std(0).clamp_min(1e-5)
        z=(train-mean)/scale;zd=(dev-mean)/scale
        jac=captures["train"]["jacobians"][key].cuda()
        anchors,singular,bank=anchor_indices(jac,scale,CONFIG["anchors"])
        coefficients={str(d):fit_coefficients(z,anchors,d,CONFIG["ridge"]) for d in (1,2)}
        predicted=features(zd[:,anchors],2)@coefficients["2"].T
        errors=[];separate_errors=[];exact_same=[]
        masks={}
        for family in DATASETS:
            mask=torch.tensor([f==family for f in captures["development"]["families"]],device="cuda:0")
            masks[family]=mask
            denominator=zd[mask].square().mean(0).clamp_min(1e-6)
            errors.append(((predicted[mask]-zd[mask]).square().mean(0)/denominator).sqrt())
            tm=torch.tensor([f==family for f in captures["train"]["families"]],device="cuda:0")
            own=fit_coefficients(z[tm],anchors,2,CONFIG["ridge"])
            separate_errors.append(((features(zd[mask][:,anchors],2)@own.T-zd[mask]).square().mean(0)/denominator).sqrt())
            exact_same.append((own==coefficients["2"]).all(1))
        errors=torch.stack(errors);separate_errors=torch.stack(separate_errors)
        # Anchor identities are definitions, not discovered cross-task redundancies.
        errors[:,anchors]=float("inf");separate_errors[:,anchors]=float("inf")
        sensitivity=(jac*scale).square().mean(0).sqrt()
        ranking_score=errors.max(0).values*sensitivity.clamp_min(1e-8)
        ranking=ranking_score.argsort().cpu().tolist()
        redundancy=redundancy_report(errors,separate_errors,CONFIG)
        exact=torch.stack(exact_same);exact[:,anchors]=False
        redundancy["exact_task_fit_matches_to_common_per_task"]=exact.sum(1).tolist()
        chosen=ranking[:max(CONFIG["removed_counts"])]
        certificate=graph_certificate(coefficients["2"],anchors,chosen,2)
        result={"anchors":anchors,"mean":mean.cpu(),"scale":scale.cpu(),"J_bank":bank.cpu(),
                "J_singular_values":singular.cpu(),"ranking":ranking,
                "coefficients":{k:v.cpu() for k,v in coefficients.items()}}
        results[key]=result
        summaries[key]={"anchors":anchors,"redundancy":redundancy,"certificate":certificate,
                        "selected_1024_per_task_median_nrmse":errors[:,chosen].median(1).values.cpu().tolist(),
                        "captured_J_rows":len(jac),"calibration_tokens":len(z),"development_tokens":len(zd)}
        stage.event("layer",layer=layer,shared_all=redundancy["shared_by_all_tasks"],redundant=redundancy["redundant_occurrences"])
        del train,dev,z,zd,jac,predicted,coefficients,own,errors,separate_errors,bank
        gc.collect();torch.cuda.empty_cache()
    save(root/"fit.pt",results);write(root/"fit.json",summaries)
    return stage.finish(["fit.pt","fit.json"])


def accuracy(rows):
    return {"overall":sum(r["prediction"]==r["answer"] for r in rows)/len(rows),
            "correct":sum(r["prediction"]==r["answer"] for r in rows),"count":len(rows),
            "by_task":{f:{"correct":sum(r["prediction"]==r["answer"] for r in rows if r["family"]==f),
                           "count":sum(r["family"]==f for r in rows)} for f in DATASETS}}


def comparison(original,changed):
    if [r["identity"] for r in original]!=[r["identity"] for r in changed]:raise ValueError("Unpaired evaluation")
    divergence=[]
    for a,b in zip(original,changed):
        p=torch.tensor(a["logits"]).double().log_softmax(0);q=torch.tensor(b["logits"]).double().log_softmax(0)
        divergence.append(float((p.exp()*(p-q)).sum()))
    return {"agreement":sum(a["prediction"]==b["prediction"] for a,b in zip(original,changed))/len(original),
            "mean_choice_kl":sum(divergence)/len(divergence),
            "accuracy_delta":accuracy(changed)["overall"]-accuracy(original)["overall"]}


def dev_pass(original,changed):
    match=comparison(original,changed);a,b=accuracy(original),accuracy(changed)
    return (match["agreement"]>=CONFIG["development_agreement"] and match["mean_choice_kl"]<=CONFIG["development_kl"]
            and b["correct"]>=a["correct"]-1
            and all(b["by_task"][f]["correct"]>=a["by_task"][f]["correct"]-1 for f in DATASETS))


def run_cases(stage,cases,tag):
    records=[]
    with torch.inference_mode():
        for case in cases:
            logits=score(stage,case)
            records.append({"identity":case["identity"],"family":case["family"],"answer":case["answer"],
                            "prediction":int(logits.argmax()),"logits":logits.cpu().tolist()})
            stage.event(tag,completed=len(records),total=len(cases))
    write(stage.path/(tag+".json"),records)
    return records


def model_bytes(model):
    return sum(p.numel()*p.element_size() for p in model.parameters())+sum(b.numel()*b.element_size() for b in model.buffers())


def timing(stage,tag):
    cases=[]
    for family in DATASETS:
        cases += [r for r in stage.tasks["development"] if r["family"]==family][:2]
    samples=[]
    with torch.inference_mode():
        for index,case in enumerate([cases[0],*cases]):
            ids=torch.tensor([case["input_ids"]],device="cuda:0")
            torch.cuda.synchronize();start=time.perf_counter()
            output=stage.model.generate(input_ids=ids,max_new_tokens=CONFIG["generation_tokens"],
                min_new_tokens=CONFIG["generation_tokens"],do_sample=False,use_cache=True,
                pad_token_id=stage.tokenizer.eos_token_id)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if index:
                samples.append({"identity":case["identity"],"prompt_tokens":len(case["input_ids"]),
                                "new_tokens":output.shape[1]-ids.shape[1],"seconds":elapsed})
            del output
    value={"samples":samples,"scope":"batch-one end-to-end generation including prefill; 32 generated tokens; eager attention",
           "tokens_per_second":sum(r["new_tokens"] for r in samples)/sum(r["seconds"] for r in samples),
           "resident_model_tensor_bytes":model_bytes(stage.model)}
    write(stage.path/(tag+"_timing.json"),value)
    return value


def evaluate(root):
    checked(root,"fit");stage=Stage(root,"evaluation")
    fitted=torch.load(root/"fit.pt",map_location="cpu",weights_only=True)
    original_modules={k:stage.model.model.layers[k].mlp for k in CONFIG["layers"]}
    # All original/replacement buffers remain on CPU when inactive.
    def install(count=None,mode="quadratic",compiled=True):
        for layer in CONFIG["layers"]:
            current=stage.model.model.layers[layer].mlp
            current.to("cpu")
            if count is None:
                replacement=original_modules[layer]
            else:
                item=fitted[str(layer)]
                replacement=PolynomialResidualMLP(original_modules[layer],item,item["ranking"][:count],mode,compiled)
            stage.model.model.layers[layer].mlp=replacement.to("cuda:0")
        gc.collect();torch.cuda.empty_cache()

    baseline_dev=run_cases(stage,stage.tasks["development"],"baseline_development")
    development={}
    for count in CONFIG["removed_counts"]:
        install(count)
        rows=run_cases(stage,stage.tasks["development"],f"quadratic_{count}_development")
        development[str(count)]={"accuracy":accuracy(rows),"comparison":comparison(baseline_dev,rows),
                                 "passed":dev_pass(baseline_dev,rows),"model_bytes":model_bytes(stage.model)}
    qualified=[int(n) for n,v in development.items() if v["passed"]]
    selected=max(qualified) if qualified else min(CONFIG["removed_counts"])
    selection={"removed_per_layer":selected,"development_qualified":bool(qualified),"candidates":development,
               "fallback_if_none":"smallest candidate retained for descriptive evaluation, not a gate pass"}
    write(root/"selection.json",selection)
    install()
    baseline=run_cases(stage,stage.tasks["evaluation"],"baseline_evaluation")
    times={"baseline":timing(stage,"baseline_before")}
    results={"baseline":accuracy(baseline)}
    for mode,compiled,tag in (("quadratic",True,"compressed"),("linear",True,"linear"),
                              ("zero",True,"zero"),("quadratic",False,"uncompiled_polynomial")):
        install(selected,mode,compiled)
        rows=run_cases(stage,stage.tasks["evaluation"],tag+"_evaluation")
        results[tag]={"accuracy":accuracy(rows),"comparison":comparison(baseline,rows),
                      "model_bytes":model_bytes(stage.model)}
        if tag in ("compressed","uncompiled_polynomial"):
            times[tag]=timing(stage,tag)
        if tag=="compressed":
            payload={str(k):{name:value.cpu() for name,value in stage.model.model.layers[k].mlp.state_dict().items()}
                     for k in CONFIG["layers"]}
            save(root/"compressed_modules.pt",payload)
            # State reload includes both the residual weights and tensor branch.
            reloaded=torch.load(root/"compressed_modules.pt",map_location="cpu",weights_only=True)
            for k in CONFIG["layers"]:
                stage.model.model.layers[k].mlp.load_state_dict(reloaded[str(k)],strict=True)
            del payload,reloaded
    install()
    times["baseline_after"]=timing(stage,"baseline_after")
    value={"selection":selection,"metrics":results,"timings":times}
    write(root/"evaluation.json",value)
    return stage.finish(["selection.json","evaluation.json","compressed_modules.pt"])


def analyze(root):
    cpu_only();plan=verify(root)
    for name in ("shakedown","capture","fit","evaluation"):checked(root,name)
    result=json.loads((root/"evaluation.json").read_text())
    base=result["metrics"]["baseline"];new=result["metrics"]["compressed"]
    quality=(result["selection"]["development_qualified"] and new["accuracy"]["correct"]>=base["correct"]-1
             and new["comparison"]["agreement"]>=CONFIG["evaluation_agreement"]
             and all(new["accuracy"]["by_task"][f]["correct"]>=base["by_task"][f]["correct"]-1 for f in DATASETS))
    timing=result["timings"]
    original_rate=(timing["baseline"]["tokens_per_second"]+timing["baseline_after"]["tokens_per_second"])/2
    speedup=timing["compressed"]["tokens_per_second"]/original_rate
    reduction=1-new["model_bytes"]/timing["baseline"]["resident_model_tensor_bytes"]
    classification=("quality_and_partial_efficiency_pass" if quality and speedup>=CONFIG["minimum_speedup"] and reduction>0
                    else "quality_preserved_no_speed_gate" if quality else "combined_quality_gate_failed")
    value={"study_id":STUDY_ID,"status":"completed","confirmed":False,"scientific_fingerprint":plan["scientific_fingerprint"],
           "classification":classification,"quality_gate":quality,"speedup":speedup,"model_tensor_byte_reduction":reduction,
           "result":result,"basis_redundancy":{k:v["redundancy"] for k,v in json.loads((root/"fit.json").read_text()).items()},
           "scope":"four length-filtered multiple-choice task families; three MLP blocks; one frozen checkpoint",
           "limitations":["no_semantic_CAV_causality_claim","shared_task_margin_J_space_not_full_vocabulary_J",
                          "Groebner_certificate_covers_graph_surrogate_not_exact_neural_relations",
                          "removed_relation_defects_are_measured_approximations","timing_is_small_paired_pilot_not_confirmatory",
                          "no_general_generation_quality_claim"],
           "artifacts":{name:sha(root/name) for name in ("fit.pt","fit.json","selection.json","evaluation.json","compressed_modules.pt")}}
    write(root/"result.json",value)
    write(root/"campaign_results.json",{**value,"result_sha256":sha(root/"result.json")})
    return {k:v for k,v in value.items() if k not in ("result","artifacts","basis_redundancy")}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","shakedown","capture","fit","evaluate","analyze"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args()
    operations={"prepare":prepare,"shakedown":lambda r:capture(r,True),"capture":capture,
                "fit":fit,"evaluate":evaluate,"analyze":analyze}
    print(json.dumps(operations[args.stage](args.root),allow_nan=False),flush=True)
