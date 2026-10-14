"""Numerical controls and CUDA-graph timings for packed Q4 kernels."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from experiments.structure_net.llama_fp16_core_q4_tail import quantize,dequantize,packed_linear,seal
from experiments.structure_net.prepare_llama_cotenancy import write_json

AMENDMENT=Path("docs/07 - Status Reports/2026-10-14_llama32-q4-tail-operator-amendment.md")
DECODE=[(16,bn,64,split,4,2) for bn in (32,64) for split in (1,4,8)]
PREFILL=[(32,64,64,1,4,2),(64,64,64,1,4,3),(64,128,64,1,4,3),
         (128,64,64,1,4,3),(128,128,32,1,8,3)]


def measure(fn):
    for _ in range(3):fn()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        fn()
        with torch.cuda.graph(graph,stream=stream):out=fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    values=[]
    for _ in range(3):
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(20):graph.replay()
        end.record();end.synchronize()
        values.append(begin.elapsed_time(end)/20000)
    return statistics.median(values),values


@torch.inference_mode()
def run(output):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("one selected CUDA GPU required")
    producer=seal(output,"systems_operator_only_not_quality_evidence",
                  {"seed":51014,"decode":DECODE,"prefill":PREFILL,"m":[1,9,512,6300],"repeats":3,"replays":20},
                  [Path(__file__),AMENDMENT])
    torch.manual_seed(51014);rows=[]
    for n,k in ((8192,3072),(3072,8192)):
        w=torch.randn((n,k),device="cuda",dtype=torch.float16)*.03
        p,s=quantize(w);dense=dequantize(p,s)
        for m in (1,9,512,6300):
            x=torch.randn((m,k),device="cuda",dtype=torch.float16)
            expected=torch.nn.functional.linear(x,dense)
            ft,fv=measure(lambda:torch.nn.functional.linear(x,dense))
            for tuning in (DECODE if m<=9 else PREFILL):
                actual=packed_linear(x,p,s,tuning)
                relative=float((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt())
                cosine=float(torch.nn.functional.cosine_similarity(actual.float().flatten(),expected.float().flatten(),dim=0))
                if not torch.isfinite(actual).all() or relative>.002 or cosine<.9999:raise RuntimeError("packed operator numerical gate failed")
                qt,qv=measure(lambda:packed_linear(x,p,s,tuning))
                row={"m":m,"n":n,"k":k,"tuning":tuning,"relative_rms":relative,"cosine":cosine,
                     "q4_seconds":qt,"fp16_seconds":ft,"q4_repeats":qv,"fp16_repeats":fv}
                rows.append(row);write_json(output/"rows.json",{"status":"running","rows":rows})
                print(json.dumps(row),flush=True)
            del expected,actual,x
        del w,p,s,dense
    selected={}
    for label,choices in (("decode",DECODE),("prefill",PREFILL)):
        scores={t:sum(r["q4_seconds"] for r in rows if tuple(r["tuning"])==t and (r["m"]<=9)==(label=="decode")) for t in choices}
        selected[label]=min(choices,key=lambda t:(scores[t],t))
    write_json(output/"rows.json",{"status":"completed","rows":rows})
    write_json(output/"result.json",{"status":"passed","label":"systems_operator_only_not_quality_evidence",
        "selected":selected,"rows":len(rows),"producer_fingerprint":producer["fingerprint"],
        "peak_cuda_bytes":torch.cuda.max_memory_allocated()})
    print(json.dumps(selected),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    run(p.parse_args().output)
