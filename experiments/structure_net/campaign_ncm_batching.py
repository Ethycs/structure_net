"""Run shared-weight batching after the factorization development producer exits."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.structure_net.prepare_llama_cotenancy import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preceding',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('fresh coordinator root required')
    args.output.mkdir(parents=True);shutil.copyfile(__file__,args.output/'producer.py')
    source=Path('experiments/structure_net/run_ncm_batching.py');expected=sha256(source)
    write_json(args.output/'producer.json',{'command':sys.argv,'worker_sha256':expected,
        'coordinator_sha256':sha256(__file__),'preceding':str(args.preceding),'physical_gpu':0,'concurrency':1})
    while True:
        if args.preceding.exists():
            status=json.loads(args.preceding.read_text())['status']
            if status=='failed':raise RuntimeError('preceding study failed; inspect before launching')
            if status=='development_pending_source_review':break
        time.sleep(5)
    runs=[]
    for stage,batch in (('pilot',9),('run',1),('run',4),('run',9)):
        if sha256(source)!=expected:raise ValueError('batch worker changed')
        root=args.output.parent/(f'20261021_batch{batch}_v1' if stage=='run' else '20261021_pilot_v1')
        command=[sys.executable,'-u','-m','experiments.structure_net.run_ncm_batching',stage,'--batch',str(batch),'--output',str(root)]
        print(json.dumps({'starting_stage':stage,'batch':batch}),flush=True);start=time.perf_counter()
        with (args.output/f'{stage}{batch}.log').open('w') as log:
            child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row={'stage':stage,'batch':batch,'root':str(root),'command':command,'returncode':child.returncode,'seconds':time.perf_counter()-start}
        if not child.returncode:row['result_sha256']=sha256(root/'result.json')
        runs.append(row);write_json(args.output/'execution.json',{'status':'failed' if child.returncode else 'running','runs':runs,
            'preceding_execution_sha256':sha256(args.preceding)})
        if child.returncode:raise RuntimeError('batch worker failed; preserve evidence and inspect log')
    write_json(args.output/'execution.json',{'status':'completed','runs':runs,'preceding_execution_sha256':sha256(args.preceding)})


if __name__=='__main__':main()
