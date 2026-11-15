"""CPU-only cached-logit loss audit; no model, fitting, GPU or confirmation data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(path):
    stat=Path(path).stat();return [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def summarize(rows):
    result={}
    for split in ('train','development'):
        selected=[r for r in rows if r['split']==split];groups={'all':selected}
        groups.update({family:[r for r in selected if r['family']==family] for family in sorted({r['family'] for r in selected})})
        result[split]={}
        for name,group in groups.items():
            out={'count':len(group)}
            for teacher in ('native','accepted_parent'):
                fields=group[0][teacher].keys()
                out[teacher]={key:sum(r[teacher][key] for r in group)/len(group) for key in fields}
                out[teacher]['choice_correct']=sum(r[teacher]['choice_accuracy'] for r in group)
            for key in ('kl_parent_to_native_t2','kl_native_to_parent_t2'):
                out[key]=sum(r[key] for r in group)/len(group)
            result[split][name]=out
    return result


def run(root,diagnostic_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('This audit requires CUDA hidden')
    os.nice(10);torch.set_num_threads(1)
    root,diagnostic_root=Path(root).resolve(),Path(diagnostic_root).resolve()
    if root.exists():raise FileExistsError(root)
    frozen=read(diagnostic_root/'plan.json');campaign=Path(frozen['campaign']);plan=read(campaign/'plan.json')
    stage_dir=campaign/'stages/block_23';stage=read(stage_dir/'plan.json');capture=read(stage_dir/'capture/manifest.json')
    items_path=Path(plan['items_path']);native_path=Path(plan['native_cache'])/'teacher_logits.npy'
    parent_path=stage_dir/'capture/packed_teacher_logits.npy';items=read(items_path)['items'];labels=plan['label_ids']
    original_rows=plan['original_indices'];inputs={}
    for name,path in (('items',items_path),('native',native_path),('accepted_parent',parent_path)):
        key=str(path)
        if key not in frozen['input_sha256'] or identity(path)!=frozen['input_identity'][key]:
            raise ValueError('Previously authenticated cache identity changed: '+key)
        inputs[name]=dict(path=key,sha256=frozen['input_sha256'][key],identity=identity(path))
    if (capture['status']!='completed' or capture['fingerprint']!=stage['fingerprint']
            or capture['artifacts']['packed_teacher_logits.npy']!=inputs['accepted_parent']['sha256']
            or len(items)!=576 or len({c['identity'] for c in items})!=576
            or sum(c['split']=='train' for c in items)!=512 or sum(c['split']=='development' for c in items)!=64
            or len(original_rows)!=576 or len(set(original_rows))!=576):
        raise ValueError('Frozen complete512TRAIN/64DEV parent cache required')
    for case in items:
        positions=case['loss_positions'];length=len(case['input_ids'])
        expected=[case['input_ids'][p+1] for p in positions[:3]]+[labels[case['answer']]]
        if len(positions)!=4 or positions[-1]!=length-1 or case['targets']!=expected or case['stop']-case['offset']!=length:
            raise ValueError('Supervised token/answer-position mapping differs from frozen capture')
    native=np.load(native_path,mmap_mode='r',allow_pickle=False);parent=np.load(parent_path,mmap_mode='r',allow_pickle=False)
    if (native.dtype!=np.dtype('int16') or parent.dtype!=np.dtype('int16') or not native.dtype.isnative
            or parent.shape!=(576,4,151936) or native.shape[1:]!=(4,151936)
            or any(type(i) is not int or not 0<=i<len(native) for i in original_rows)):
        raise ValueError('Original BF16 four-position logit banks required')
    root.mkdir(parents=True);started=time.monotonic();source=Path(__file__).resolve()
    manifest=dict(schema='jspace.3b-cached-teacher-loss-audit.v1',source=str(source),source_sha256=sha(source),
        inputs=inputs,diagnostic_plan_sha256=sha(diagnostic_root/'plan.json'),diagnostic_fingerprint=frozen['fingerprint'],
        campaign_plan_sha256=sha(campaign/'plan.json'),stage_plan_sha256=sha(stage_dir/'plan.json'),
        capture_manifest_sha256=sha(stage_dir/'capture/manifest.json'),parent_checkpoint_sha256=stage['parent_checkpoint_sha256'],
        original_indices=original_rows,model_forward=False,threads=1,nice_increment=10,temperature=2.,
        scope='All512training and64development cached teacher logits; no student forward, training or held-out data',confirmed=False)
    write(root/'plan.json',manifest);rows=[]
    try:
        with torch.inference_mode():
            for start in range(0,576,4):
                indices=list(range(start,min(start+4,576)));cases=[items[i] for i in indices]
                values={'native':torch.from_numpy(np.array(native[[original_rows[i] for i in indices]],copy=True)).view(torch.bfloat16).float(),
                    'accepted_parent':torch.from_numpy(np.array(parent[indices],copy=True)).view(torch.bfloat16).float()}
                targets=torch.tensor([c['targets'] for c in cases]);measurements={};logs={}
                for name,logits in values.items():
                    if not torch.isfinite(logits).all():raise ValueError('Nonfinite cached logits')
                    logp=logits.log_softmax(-1);ce=-logp.gather(-1,targets[...,None]).squeeze(-1)
                    entropy=-(logp.exp()*logp).sum(-1);top1=(logits.argmax(-1)==targets).float()
                    measurements[name]=[];logs[name]=(logits/2).log_softmax(-1)
                    # Direct check against the exact frozen CE expression.
                    direct=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))
                    if not torch.allclose(ce.mean(),direct,atol=2e-6,rtol=2e-6):raise ValueError('CE reduction mismatch')
                    for j,case in enumerate(cases):
                        choice=logits[j,-1,labels[:len(case['choices'])]];choice_logp=choice.log_softmax(-1)
                        measurements[name].append(dict(ce_four_positions=float(ce[j].mean()),ce_context_positions=float(ce[j,:3].mean()),
                            ce_answer=float(ce[j,-1]),choice_ce=float(-choice_logp[case['answer']]),
                            choice_accuracy=int(choice.argmax())==case['answer'],
                            entropy_four_positions_nats=float(entropy[j].mean()),entropy_answer_nats=float(entropy[j,-1]),
                            entropy_choices_nats=float(-(choice_logp.exp()*choice_logp).sum()),
                            full_vocab_top1_four_positions=float(top1[j].mean()),full_vocab_top1_answer=float(top1[j,-1])))
                    del logp,ce,entropy,top1
                p,n=logs['accepted_parent'],logs['native']
                pn=(p.exp()*(p-n)).sum(-1).mean(-1);np_=(n.exp()*(n-p)).sum(-1).mean(-1)
                for j,(index,case) in enumerate(zip(indices,cases)):
                    rows.append(dict(source_index=index,original_logit_index=original_rows[index],identity=case['identity'],family=case['family'],
                        split=case['split'],loss_positions=case['loss_positions'],targets=case['targets'],
                        native=measurements['native'][j],accepted_parent=measurements['accepted_parent'][j],
                        kl_parent_to_native_t2=float(pn[j]),kl_native_to_parent_t2=float(np_[j])))
                del values,logs,p,n,pn,np_
                if (start+4)%64==0:print(json.dumps(dict(completed=start+4,total=576)),flush=True)
        for entry in inputs.values():
            if identity(entry['path'])!=entry['identity']:raise ValueError('Input changed during audit')
        result=dict(status='completed',summary=summarize(rows),records=rows,elapsed_seconds=time.monotonic()-started,
            ce_reduction='True full-vocabulary CE; equal weight across four original positions and cases',
            entropy='Full-vocabulary or conditional-choice entropy at temperature1; natural logarithms',
            kl='Raw KL at temperature2, averaged over four positions; NOT multiplied by T² or0.7',
            training_objective='0.7*T²*KL(accepted_parent||student)+0.3*CE; native||student KL is diagnostic-only',
            accepted_original_kl_field_direction='KL(native||accepted_parent)',confirmed=False)
        write(root/'result.json',result)
        write(root/'manifest.json',dict(status='completed',plan_sha256=sha(root/'plan.json'),result_sha256=sha(root/'result.json'),source_sha256=sha(source)))
        return dict(root=str(root),summary=result['summary'],elapsed_seconds=result['elapsed_seconds'])
    finally:native._mmap.close();parent._mmap.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--diagnostic-root',required=True,type=Path);args=parser.parse_args()
    print(json.dumps(run(args.root,args.diagnostic_root),allow_nan=False),flush=True)
