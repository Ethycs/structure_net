"""Serial lossless follow-up, entered through the authorized study launcher."""
import json
from pathlib import Path
import subprocess
import sys
from experiments.structure_net.prepare_llama_cotenancy import sha256,write_json


def run(root):
    index = {"schema":"nal.llama-lossless-residual-index.v1","status":"running","cells":[],"launcher_sha256":sha256(__file__)}
    (root/"lossless_launcher.py").write_text(Path(__file__).read_text())
    def cell(name,stage,*flags):
        output=root/name
        command=[sys.executable,"-m","experiments.structure_net.run_llama_lossless",stage,"--output",str(output),*map(str,flags)]
        print(json.dumps({"starting":name}),flush=True)
        with (root/(name+".log")).open("w") as log:
            completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        path=output/"result.json"
        if completed.returncode or not path.exists():
            raise RuntimeError("cell failed; preserve log: "+str(output))
        value=json.loads(path.read_text())
        index["cells"].append({"name":name,"result":str(path),"sha256":sha256(path),"status":value["status"]})
        write_json(root/"progress.json",index)
        print(json.dumps({"completed":name,"status":value["status"]}),flush=True)
        return value
    try:
        cell("projection","projection")
        audit=cell("weights","audit_weights")
        cell("original_pilot","measure","--pilot")
        cell("residual_pilot","measure","--pilot","--arm","exact_residual","--checkpoint",audit["checkpoint"],"--reference",root/"original_pilot")
        reference=cell("original","measure","--include-test")
        flags=["--include-test"] if reference["validation_competent"] else []
        cell("exact_residual","measure","--arm","exact_residual","--checkpoint",audit["checkpoint"],"--reference",root/"original",*flags)
        index["status"]="completed"
    except Exception as error:
        index.update(status="stopped_runtime_fault",error=str(error))
        raise
    finally:
        write_json(root/"campaign_index.json",index)
