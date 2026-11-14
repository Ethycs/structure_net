"""CUDA-graph decoding for the frozen densely stacked polynomial Qwen model.

Engineering extension: the packed layers are ~2.5x faster than the separate
polynomial branch, but reversed-order whole-model timing gains only 0.8%.
Keep weights, eager attention, 73 covered channels and BF16 fixed. Capture one
static-cache decode step including greedy selection and token/position updates.
Use the same eight prompts and 32 tokens, suppressing EOS as min_new_tokens=32
does. Measure HF, uncaptured static decoding and captured decoding. Check graph
versus uncaptured token equality as a systems invariant; HF token agreement is
reported, not a task-accuracy threshold. Re-score all 256 MCQs through static
prefill. Setup/capture time is reported separately from steady-state timing.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
from transformers import StaticCache

from . import jspace_bench_tensor as base
from .jspace_bench_stacked import DEFAULT_ROOT as PACKED, FIT_ROOT, COVERED_ROOT
from .jspace_bench_stacked_timing import DEFAULT_ROOT as TIMING
from .jspace_packed_core import StackedCoreMLP
from .jspace_bench_data import DATASETS, sha, write


DEFAULT_ROOT=base.ROOT/"data/experiments/jspace_bench_tensor/20261113_graph_v1"
CONFIG={"capacity":416,"new_tokens":32,"attention":"eager","precision":"BF16",
        "decode":"one CUDA graph per static shape; token selection and advancement inside graph",
        "arms":["hf","static_uncaptured","cuda_graph"],"reuse_accuracy_pool":True,
        "known_before_freeze":"stacked 209/256; counterbalanced packed 11.83 vs previous 11.73 tok/s"}


class GraphDecoder:
    def __init__(self,model):
        self.model=model
        self.cache=StaticCache(model.config,max_batch_size=1,max_cache_len=CONFIG["capacity"],
                               device="cuda:0",dtype=torch.bfloat16)
        self.token=torch.zeros((1,1),device="cuda:0",dtype=torch.long)
        self.position=torch.zeros(1,device="cuda:0",dtype=torch.long)
        self.step=torch.zeros(1,device="cuda:0",dtype=torch.long)
        self.sequence=torch.zeros(CONFIG["new_tokens"],device="cuda:0",dtype=torch.long)
        self.arange=torch.arange(CONFIG["capacity"],device="cuda:0")
        eos=model.generation_config.eos_token_id
        self.eos=torch.tensor(eos if isinstance(eos,list) else [eos],device="cuda:0",dtype=torch.long)
        self.graph=None

    def choose(self,hidden):
        logits=F.linear(hidden,self.model.lm_head.weight).float()
        logits.index_fill_(-1,self.eos,-torch.inf)
        return logits.argmax(-1)

    def decode_step(self):
        mask=torch.where(self.arange<=self.position,0.,torch.finfo(torch.bfloat16).min).to(torch.bfloat16).view(1,1,1,-1)
        hidden=self.model.model(input_ids=self.token,past_key_values=self.cache,use_cache=True,
            cache_position=self.position,position_ids=self.position.unsqueeze(0),
            attention_mask={"full_attention":mask}).last_hidden_state[:,-1]
        next_token=self.choose(hidden)
        self.sequence.index_copy_(0,self.step,next_token)
        self.token.copy_(next_token.view(1,1));self.position.add_(1);self.step.add_(1)

    @torch.inference_mode()
    def capture(self):
        started=time.perf_counter()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.step.zero_();self.position.zero_();self.decode_step()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.step.zero_();self.position.zero_()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.decode_step()
        torch.cuda.synchronize()
        return time.perf_counter()-started

    @torch.inference_mode()
    def prefill(self,ids):
        self.cache.reset()
        positions=torch.arange(ids.shape[1],device="cuda:0")
        mask=torch.where(self.arange[None,:]<=positions[:,None],0.,torch.finfo(torch.bfloat16).min).to(torch.bfloat16)[None,None]
        hidden=self.model.model(input_ids=ids,past_key_values=self.cache,use_cache=True,
            cache_position=positions,position_ids=positions.unsqueeze(0),
            attention_mask={"full_attention":mask}).last_hidden_state[:,-1]
        return hidden

    @torch.inference_mode()
    def generate(self,ids,captured):
        if ids.shape[1]+CONFIG["new_tokens"]>CONFIG["capacity"]:raise ValueError("Static capacity exceeded")
        first=self.choose(self.prefill(ids))
        self.sequence[0].copy_(first[0]);self.token.copy_(first.view(1,1))
        self.step.fill_(1);self.position.fill_(ids.shape[1])
        for _ in range(CONFIG["new_tokens"]-1):
            if captured:self.graph.replay()
            else:self.decode_step()
        return self.sequence.clone()


def prepare(root):
    base.cpu_only()
    if root.exists():raise FileExistsError(root)
    parent=base.verify(TIMING);base.checked(TIMING,"timing")
    value={**parent,"graph_config":CONFIG,"sources":{**parent["sources"],str(Path(__file__)):sha(__file__)},
           "input_sha256":{**parent["input_sha256"],str(TIMING/"result.json"):sha(TIMING/"result.json")}}
    value.pop("scientific_fingerprint");value["scientific_fingerprint"]=base.digest(value)
    root.mkdir(parents=True)
    for name in value["sources"]:
        path=Path(name);target=root/"producer"/path.relative_to(base.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    shutil.copy2(PACKED/"tasks.json",root/"tasks.json");write(root/"plan.json",value)
    return {"fingerprint":value["scientific_fingerprint"],"graph_config":CONFIG}


def run(root):
    if base.verify(root)["graph_config"]!=CONFIG:raise ValueError("Graph configuration changed")
    stage=base.Stage(root,"graph")
    fit=torch.load(FIT_ROOT/"fit.pt",map_location="cpu",weights_only=True)
    removed={int(k):v["indices"] for k,v in json.loads((COVERED_ROOT/"coverage.json").read_text()).items()}
    for layer,block in enumerate(stage.model.model.layers):
        original=block.mlp.cpu()
        block.mlp=StackedCoreMLP(original,fit[str(layer)] if layer in removed else None,removed.get(layer,[])).cuda()
        del original
    gc.collect();torch.cuda.empty_cache()
    decoder=GraphDecoder(stage.model)
    setup_seconds=decoder.capture()
    cases=[case for family in DATASETS for case in [c for c in stage.tasks["development"] if c["family"]==family][:2]]
    warm=torch.tensor([cases[0]["input_ids"]],device="cuda:0")
    a=decoder.generate(warm,False);b=decoder.generate(warm,True)
    if not torch.equal(a,b):raise ValueError("Captured decode differs from identical uncaptured static algorithm")
    write(stage.path/"graph_shakedown.json",{"static_graph_tokens_equal":True,"tokens":len(a),"capture_seconds":setup_seconds})
    timing={};outputs={}
    for arm in CONFIG["arms"]:
        samples=[];outputs[arm]=[]
        def generate(ids):
            if arm=="hf":
                return stage.model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                    max_new_tokens=32,min_new_tokens=32,pad_token_id=stage.tokenizer.eos_token_id)[0,-32:]
            return decoder.generate(ids,arm=="cuda_graph")
        with torch.inference_mode():
            generate(warm);torch.cuda.synchronize()
            for case in cases:
                stage.guard([0]);ids=torch.tensor([case["input_ids"]],device="cuda:0")
                torch.cuda.synchronize();started=time.perf_counter();tokens=generate(ids);torch.cuda.synchronize()
                samples.append({"identity":case["identity"],"prompt_tokens":ids.shape[1],"new_tokens":len(tokens),"seconds":time.perf_counter()-started})
                outputs[arm].append(tokens.cpu().tolist())
        timing[arm]={"tokens_per_second":sum(v["new_tokens"] for v in samples)/sum(v["seconds"] for v in samples),"samples":samples}
        print(json.dumps({"arm":arm,"tokens_per_second":timing[arm]["tokens_per_second"]}),flush=True)
    if outputs["static_uncaptured"]!=outputs["cuda_graph"]:raise ValueError("Static/captured outputs differ across timing prompts")
    rows=[]
    with torch.inference_mode():
        for case in stage.tasks["evaluation"]:
            ids=torch.tensor([case["input_ids"]],device="cuda:0");hidden=decoder.prefill(ids)
            logits=F.linear(hidden,stage.model.lm_head.weight[stage.label_ids[:len(case["choices"])]]).float()[0]
            if not torch.isfinite(logits).all():raise FloatingPointError("Nonfinite logits")
            rows.append({"identity":case["identity"],"family":case["family"],"answer":case["answer"],
                         "prediction":int(logits.argmax()),"logits":logits.cpu().tolist()})
            stage.event("static_evaluation",completed=len(rows),total=256)
    write(stage.path/"static_evaluation.json",rows)
    reference=json.loads((PACKED/"stacked/stacked_polynomial_core_evaluation.json").read_text())
    matches=sum(x==y for a,b in zip(outputs["hf"],outputs["cuda_graph"]) for x,y in zip(a,b))
    result={"study_id":base.STUDY_ID,"status":"completed","confirmed":False,"scientific_fingerprint":stage.plan["scientific_fingerprint"],
            "classification":"cuda_graph_execution_measurement","accuracy":base.accuracy(rows),
            "comparison_to_packed":base.comparison(reference,rows),"timings":timing,"capture_seconds":setup_seconds,
            "generated_tokens":outputs,"hf_token_matches":matches,"generated_token_comparisons":256,
            "static_graph_exact":True,"graph_speedup":timing["cuda_graph"]["tokens_per_second"]/timing["hf"]["tokens_per_second"],
            "scope":"batch-one, static capacity 416, 32 greedy tokens including prefill; setup amortized, same model and task pool"}
    write(root/"result.json",result);write(root/"campaign_results.json",{**result,"result_sha256":sha(root/"result.json")})
    return stage.finish(["result.json","campaign_results.json"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("prepare","run"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT);args=parser.parse_args()
    print(json.dumps({"prepare":prepare,"run":run}[args.stage](args.root),allow_nan=False),flush=True)
