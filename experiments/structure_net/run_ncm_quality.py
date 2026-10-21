"""Serial coordinator for the prospectively fixed independent quality campaign."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

from experiments.structure_net.ncm_quality import (
    BASE,DATA,PLAN,ORIGINAL_PLAN,CHECKPOINTS,read,write,sha256,check_calibration)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID':
        raise ValueError('select physical GPU0 before coordinator startup')
    check_calibration('primary')
    for model in ('ncm','llama'):
        r=read(BASE/f'20261021_{model}_pilot_v1/result.json')
        if r['status']!='passed' or r['cases']!=2:raise ValueError('both CUDA lifecycle pilots required')
    from experiments.structure_net.ncm_llama import GPT2,INITIAL
    freeze_root=BASE/'20261021_dependency_freeze_v1'
    if freeze_root.exists():raise ValueError('fresh frozen campaign required')
    paths=[PLAN,ORIGINAL_PLAN,DATA/'cases.json',DATA/'calibration.json',INITIAL/'config.yaml',
        *map(Path,CHECKPOINTS),Path('data/models/Llama-3.2-3B-cotenancy/manifest.json')]
    paths += [p for p in GPT2.iterdir() if p.is_file()]
    paths += [Path(__file__),Path('experiments/structure_net/ncm_quality.py'),
        Path('experiments/neural_architecture_lab/finalize_ncm_quality.py'),
        Path('src/neural_architecture_lab/ncm_quality_hypothesis.py'),
        Path('tests/structure_net/test_ncm_quality.py'),Path('tests/neural_architecture_lab/test_ncm_quality_hypothesis.py')]
    manifest={'status':'frozen_before_primary_generation','created_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','flash-attn','numpy','requests')},
        'hashes':{str(p):sha256(p) for p in paths}}
    write(freeze_root/'manifest.json',manifest)
    root=BASE/'20261021_coordinator_v1'
    if root.exists():raise ValueError('fresh coordinator required')
    root.mkdir();(root/'producer.py').write_bytes(Path(__file__).read_bytes())
    jobs=[]
    for model in ('ncm','llama'):
        jobs.append((f'{model}_generation',['generate','--model',model],BASE/f'20261021_{model}_generation_v1'))
    jobs.append(('primary_judge',['judge','--judge','primary'],BASE/'20261021_judge_primary_v1'))
    jobs.append(('secondary_calibration',['calibrate','--judge','secondary'],BASE/'20261021_calibration_secondary_v1'))
    jobs.append(('secondary_judge',['judge','--judge','secondary'],BASE/'20261021_judge_secondary_v1'))
    completed=[]
    for name,args,dst in jobs:
        if name=='secondary_judge' and read(BASE/'20261021_calibration_secondary_v1/result.json')['status']!='passed':
            completed.append({'name':name,'status':'not_run_secondary_calibration_failed'});continue
        for path,h in manifest['hashes'].items():
            if sha256(path)!=h:raise ValueError('frozen campaign source/input changed')
        argv=[sys.executable,'-m','experiments.structure_net.ncm_quality',*args,'--output',str(dst)]
        print(json.dumps({'starting':name,'output':str(dst)}),flush=True)
        stopped_secondary=False
        with (root/(name+'.log')).open('w') as stream:
            process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if name=='secondary_calibration':
                    done=[read(p) for p in (dst/'rows').glob('*.json')]
                    if any(x['status']=='invalid' for x in done):
                        process.terminate();process.wait();stopped_secondary=True
                        producer=read(dst/'producer.json')
                        write(dst/'result.json',{'status':'calibration_failed','reason':'exhausted infrastructure/schema retries; complete gate impossible',
                            'producer_fingerprint':producer['fingerprint'],'requested_pairs':24,'completed_pairs':len(done),
                            'invalid_pairs':sum(x['status']=='invalid' for x in done),'total':2*sum(x['status']=='valid' for x in done),
                            'interrupted':True,'returncode':process.returncode})
                        break
                time.sleep(.5)
            returncode=process.wait()
        row={'name':name,'root':str(dst),'returncode':returncode,'command':argv,
            'status':read(dst/'result.json')['status'] if (dst/'result.json').exists() else 'no_terminal_result'}
        if (dst/'result.json').exists():row['result_sha256']=sha256(dst/'result.json')
        completed.append(row)
        write(root/'execution.json',{'status':'running','runs':completed,'source_sha256':sha256(__file__),
            'dependency_manifest_sha256':sha256(freeze_root/'manifest.json')})
        print(json.dumps(row),flush=True)
        if (returncode and not stopped_secondary) or row['status'] not in ('passed','completed','calibration_failed'):
            raise RuntimeError('campaign stage failed: '+name)
    write(root/'execution.json',{'status':'completed','runs':completed,'source_sha256':sha256(__file__),
        'dependency_manifest_sha256':sha256(freeze_root/'manifest.json')})


if __name__=='__main__':main()
