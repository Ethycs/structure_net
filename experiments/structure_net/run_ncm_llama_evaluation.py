"""Run the four frozen NCM pilot controls sequentially on physical GPU0."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE=Path("data/experiments/ncm_llama")
CONDITIONS=("selected","teacher","fp16","released")


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    root=BASE/"20261021_evaluation_campaign_v1"
    if root.exists() or any((BASE/f"20261021_{a}_heldout_v1").exists() for a in CONDITIONS):
        raise ValueError("fresh append-only evaluation roots required")
    generator=json.loads((BASE/"20261021_generator_v1/result.json").read_text())
    codec=json.loads((BASE/"20261021_codec_v1/result.json").read_text())
    if generator['status']!='generator_adapted' or codec['status'] not in ('codec_adapted','codec_unchanged'):
        raise ValueError("completed fixed training stages required")
    for r in (generator,codec):
        if sha(r['selected_checkpoint'])!=r['model_sha256']:raise ValueError("selected model changed")
    root.mkdir();shutil.copyfile(__file__,root/'producer.py')
    record={'status':'running','source_sha256':sha(__file__),'completed':[],'order':list(CONDITIONS)}
    def save():
        tmp=root/'execution.tmp';tmp.write_text(json.dumps(record,indent=2));tmp.replace(root/'execution.json')
    save()
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',LD_LIBRARY_PATH='')
    for condition in CONDITIONS:
        child=BASE/f'20261021_{condition}_heldout_v1'
        command=[sys.executable,'-m','experiments.structure_net.run_ncm_llama','evaluate','--condition',condition,'--output',str(child)]
        if condition=='selected':command+=['--codec',codec['selected_checkpoint'],'--model',generator['selected_checkpoint']]
        with (root/f'{condition}.log').open('x') as log:
            code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            record.update(status='failed',failed_condition=condition,exit_code=code);save();raise SystemExit(code)
        result=json.loads((child/'result.json').read_text())
        if result['status']!='completed_pending_source_review' or result['rows']!=300:raise ValueError("incomplete evaluation child")
        record['completed'].append({'condition':condition,'root':str(child),'result_sha256':sha(child/'result.json')});save()
        print(json.dumps({'condition':condition,'status':'completed_pending_source_review'}),flush=True)
    if sha(__file__)!=record['source_sha256']:raise ValueError("coordinator source changed")
    record['status']='completed_pending_source_review';save()


if __name__=='__main__':main()
