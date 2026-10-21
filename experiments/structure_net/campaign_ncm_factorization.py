"""Run NCM factorization systems/training/development without overlapping timing."""
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
    p.add_argument('--replicas',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('fresh coordinator root required')
    args.output.mkdir(parents=True);shutil.copyfile(__file__,args.output/'producer.py')
    source=Path('experiments/structure_net/run_ncm_factorization.py');expected=sha256(source)
    write_json(args.output/'producer.json',{'command':sys.argv,'worker_sha256':expected,'coordinator_sha256':sha256(__file__),
        'waiting_for':str(args.replicas),'physical_gpu':0,'concurrency':1})
    print(json.dumps({'waiting_for_isolated_replicas':str(args.replicas)}),flush=True)
    while True:
        if args.replicas.exists():
            status=json.loads(args.replicas.read_text())['status']
            if status=='failed':raise RuntimeError('replica campaign failed; inspect before new GPU work')
            if status=='completed':break
        time.sleep(5)
    runs=[]
    for stage in ('pilot','train','development'):
        if sha256(source)!=expected:raise ValueError('worker changed before launch')
        name={'pilot':'pilot','train':'training','development':'development'}[stage]
        root=args.output.parent/f'20261021_{name}_v1'
        command=[sys.executable,'-u','-m','experiments.structure_net.run_ncm_factorization',stage,'--output',str(root)]
        print(json.dumps({'starting_stage':stage}),flush=True);start=time.perf_counter()
        with (args.output/(stage+'.log')).open('w') as log:
            child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row={'stage':stage,'root':str(root),'command':command,'returncode':child.returncode,'seconds':time.perf_counter()-start}
        if not child.returncode:row['result_sha256']=sha256(root/'result.json')
        runs.append(row);write_json(args.output/'execution.json',{'status':'failed' if child.returncode else 'running','runs':runs,
            'preceding_replica_execution_sha256':sha256(args.replicas)})
        if child.returncode:raise RuntimeError('factorization child failed; preserve evidence and inspect log')
        print(json.dumps({'completed_stage':stage}),flush=True)
    write_json(args.output/'execution.json',{'status':'development_pending_source_review','runs':runs,
        'preceding_replica_execution_sha256':sha256(args.replicas)})


if __name__=='__main__':main()
