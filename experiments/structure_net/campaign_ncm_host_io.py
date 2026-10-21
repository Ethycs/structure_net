"""Run corrected NCM copies/batching and a fresh original Llama control."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.structure_net.prepare_llama_cotenancy import sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('fresh host campaign required')
    if json.loads((args.pilot/'result.json').read_text())['status']!='passed':raise ValueError('passed pilot required')
    args.output.mkdir(parents=True);shutil.copyfile(__file__,args.output/'producer.py')
    source=Path('experiments/structure_net/run_ncm_host_io.py');expected=sha256(source)
    write_json(args.output/'producer.json',{'command':sys.argv,'worker_sha256':expected,'coordinator_sha256':sha256(__file__),
        'pilot_result_sha256':sha256(args.pilot/'result.json'),'physical_gpu':0,'concurrency':1})
    jobs=[]
    for mode,size in (('replicas',1),('batch',1),('replicas',8),('batch',9),('replicas',9)):
        root=args.output.parent/f'20261021_{mode}{size}_v1'
        command=[sys.executable,'-u','-m','experiments.structure_net.run_ncm_host_io','run','--mode',mode,'--size',str(size),'--output',str(root)]
        jobs.append((f'{mode}{size}',root,command))
    root=args.output.parent/'20261021_teacher_v1'
    jobs.append(('teacher',root,[sys.executable,'-u','-m','experiments.structure_net.run_ncm_llama','evaluate','--condition','teacher','--output',str(root)]))
    runs=[]
    for label,root,command in jobs:
        if sha256(source)!=expected:raise ValueError('host producer changed')
        print(json.dumps({'starting':label}),flush=True);start=time.perf_counter()
        with (args.output/(label+'.log')).open('w') as log:child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row={'label':label,'root':str(root),'command':command,'returncode':child.returncode,'seconds':time.perf_counter()-start}
        if not child.returncode:row['result_sha256']=sha256(root/'result.json')
        runs.append(row);write_json(args.output/'execution.json',{'status':'failed' if child.returncode else 'running','runs':runs})
        if child.returncode:raise RuntimeError('host campaign child failed; preserve evidence')
        print(json.dumps({'completed':label}),flush=True)
    write_json(args.output/'execution.json',{'status':'completed','runs':runs})


if __name__=='__main__':main()
