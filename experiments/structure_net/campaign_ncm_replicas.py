"""Run the fixed NCM replica conditions in isolated CUDA subprocesses."""
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
    p.add_argument('--pilot',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('fresh coordinator root required')
    if json.loads((args.pilot/'result.json').read_text())['status']!='passed':
        raise ValueError('passed concurrency pilot required')
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__,args.output/'producer.py')
    source=Path('experiments/structure_net/run_ncm_replicas.py')
    expected=sha256(source)
    write_json(args.output/'producer.json',{'command':sys.argv,'worker_sha256':expected,
        'coordinator_sha256':sha256(__file__),'pilot_result_sha256':sha256(args.pilot/'result.json'),
        'condition_order':[1,9,2,8,4,7],'physical_gpu':0,'concurrency':1})
    runs=[]
    for copies in (1,9,2,8,4,7):
        if sha256(source)!=expected:raise ValueError('worker changed')
        root=args.output.parent/f'20261021_copies{copies}_v1'
        command=[sys.executable,'-u','-m','experiments.structure_net.run_ncm_replicas','run',
            '--copies',str(copies),'--pilot',str(args.pilot),'--output',str(root)]
        print(json.dumps({'starting_copies':copies}),flush=True)
        start=time.perf_counter()
        with (args.output/f'copies{copies}.log').open('w') as log:
            child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row={'copies':copies,'root':str(root),'command':command,'returncode':child.returncode,
            'seconds':time.perf_counter()-start}
        if not child.returncode:row['result_sha256']=sha256(root/'result.json')
        runs.append(row)
        write_json(args.output/'execution.json',{'status':'failed' if child.returncode else 'running','runs':runs})
        if child.returncode:raise RuntimeError('replica child failed; preserve evidence and inspect log')
        print(json.dumps({'completed_copies':copies}),flush=True)
    write_json(args.output/'execution.json',{'status':'completed','runs':runs})


if __name__=='__main__':main()
