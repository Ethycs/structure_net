"""Archive emitted PTX for selected full/staged precision; no quality evidence."""
import argparse
import json
from pathlib import Path
import re
import shutil
import torch
import triton

from experiments.structure_net.campaign_llama_q4_kernel import PLAN,HYPOTHESIS_ID
from experiments.structure_net.llama_q4_kernel_v3 import core_output_mm
from experiments.structure_net.prepare_llama_cotenancy import sha256,write_json


@torch.inference_mode()
def run(output,operators):
    if output.exists():raise ValueError("fresh diagnostic root required")
    output.mkdir(parents=True)
    selected=json.loads((operators/"result.json").read_text())["selected"]
    torch.manual_seed(51015);m,n,k=512,8192,3072
    x=torch.randn((m,k),device="cuda",dtype=torch.float16)
    q=torch.zeros((k//2,n),device="cuda",dtype=torch.uint8)
    s=torch.ones((k//64,n),device="cuda",dtype=torch.float16)
    c=torch.zeros((m,n),device="cuda",dtype=torch.float16);out=torch.empty_like(c)
    rows=[]
    for name in ("packed","staged"):
        t=selected[name]["prefill"]
        kernel=core_output_mm[(triton.cdiv(m,t["bm"])*triton.cdiv(n,t["bn"]),)](
            x,q,s,c,out,m,n,k,t["bm"],t["bn"],t["bk"],t["group"],t["staged"],num_warps=t["warps"],num_stages=t["stages"])
        ptx=kernel.asm["ptx"];path=output/f"{name}.ptx";path.write_text(ptx)
        mma=sorted(set(re.findall(r"mma\.sync[^\s;]+",ptx)))
        rows.append({"family":name,"tile":t,"n_regs":kernel.n_regs,"shared_bytes":kernel.metadata.shared,
                     "mma_instructions":mma,"ptx_sha256":sha256(path)})
    shutil.copyfile(__file__,output/"producer.py")
    write_json(output/"result.json",{"status":"completed","hypothesis_id":HYPOTHESIS_ID,
        "label":"systems_compiler_inspection_only_not_quality_evidence","source_sha256":sha256(__file__),
        "operator_result_sha256":sha256(operators/"result.json"),"rows":rows})
    print(json.dumps(rows),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);p.add_argument("--operators",type=Path,required=True)
    a=p.parse_args();run(a.output,a.operators)
