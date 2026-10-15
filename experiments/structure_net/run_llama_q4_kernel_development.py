"""One isolated child at a time for the frozen kernel development blocks."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.structure_net.prepare_llama_cotenancy import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--base",type=Path,default=Path("data/experiments/llama_q4_kernel"))
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError("new coordinator root required")
    families=("packed","staged","transient")
    for family in families:
        r=json.loads((a.base/f"20261014_{family}_pilot_v1/result.json").read_text())
        if r["status"]!="passed":raise ValueError("all full-model pilots required")
        if (a.base/f"20261014_{family}_development_v1").exists():raise ValueError("fresh child roots required")
    a.output.mkdir(parents=True);shutil.copyfile(__file__,a.output/"producer.py")
    source_sha=sha256(__file__);completed=[]
    for family in families:
        child=a.base/f"20261014_{family}_development_v1"
        cmd=[sys.executable,"-m","experiments.structure_net.campaign_llama_q4_kernel","run","--family",family,
             "--operators",str(a.base/"20261014_operators_v3"),"--output",str(child)]
        print(json.dumps({"status":"launching","family":family}),flush=True)
        with (a.output/f"{family}.log").open("w") as stream:
            result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(a.output/"execution.json",{"status":"failed","family":family,"returncode":result.returncode,"completed":completed});raise SystemExit(result.returncode)
        terminal=json.loads((child/"result.json").read_text())
        if terminal["status"]!="completed_pending_source_review":raise RuntimeError("child did not complete")
        completed.append({"family":family,"root":str(child),"result_sha256":sha256(child/"result.json")})
        write_json(a.output/"execution.json",{"status":"running","completed":completed})
        print(json.dumps({"status":"completed_pending_source_review","family":family}),flush=True)
    if sha256(__file__)!=source_sha:raise RuntimeError("coordinator changed")
    write_json(a.output/"execution.json",{"status":"completed_pending_source_review","completed":completed,"producer_sha256":source_sha})


if __name__=="__main__":main()
