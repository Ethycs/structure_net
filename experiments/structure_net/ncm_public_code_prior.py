"""CPU-only empirical code-prior control for the fixed public TRAIN diagnostic."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from experiments.structure_net import ncm_public_pair_fit_probe as probe

data, bundle = probe.data, probe.bundle
ROOTS = {s:data.ROOT/f'20261027_public_code_prior_{s}_v1' for s in ('pilot','run','audit','results')}
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-public-code-prior.md')
TEST = Path('tests/structure_net/test_ncm_public_code_prior.py')
REPORT = Path('docs/08 - Analysis/2026-10-27_ncm-public-code-prior.md')
QUALIFICATION = data.ROOT/'20261027_public_code_prior_CPU_qualification_v1.json'
HYPOTHESIS = 'ncm-public-code-prior-v1'
LEVELS = probe.reference.LEVELS


def add(counts, codes):
    if (len(codes) != counts.shape[0] or any(type(c) is not int or not 0 <= c < counts.shape[1] for c in codes)):
        raise ValueError('complete in-range integer hierarchy required')
    counts[np.arange(len(codes)), codes] += 1


def modes(counts):
    if (counts.ndim != 2 or counts.dtype.kind not in 'iu' or np.any(counts < 0)
            or not np.all(counts.sum(1) == counts.sum(1)[0]) or not counts.sum(1)[0]):
        raise ValueError('nonempty equally sampled positional counts required')
    return counts.argmax(1).tolist()


def compare(predictions, targets, mode):
    a,b,m = np.asarray(predictions),np.asarray(targets),np.asarray(mode)
    if (a.shape != b.shape or a.ndim != 2 or a.shape[1] != len(m) or a.shape[0] == 0):
        raise ValueError('complete paired target and prediction matrices required')
    nonmodal = b != m
    correct = a == b
    return {'correct':int(correct.sum()),'positions':int(b.size),
        'agreement':float(correct.mean()),'nonmodal_correct':int((correct & nonmodal).sum()),
        'nonmodal_positions':int(nonmodal.sum()),
        'nonmodal_agreement':float(correct[nonmodal].mean()) if nonmodal.any() else None}


def entropy(counts):
    p = counts/counts.sum(1,keepdims=True)
    logs = np.zeros_like(p);np.log2(p,out=logs,where=p>0)
    return -(p*logs).sum(1)


def metrics(counts,cases,outputs):
    modal = modes(counts);gold = [r['oracle_codes'] for r in cases]
    predictions = {'positional_mode':[modal]*len(cases)}
    for arm,conditions in outputs.items():
        for mode,rows in conditions.items():
            probe.reference.score_rows(rows,cases,mode=='shuffled')
            predictions[arm+'_'+mode] = [r['codes'] for r in rows]
    scores = {name:compare(rows,gold,modal) for name,rows in predictions.items()}
    levels=[];offset=0;ent=entropy(counts)
    for length in LEVELS:
        c=counts[offset:offset+length];denom=int(c.sum(1)[0])
        levels.append({'length':length,'mean_position_entropy_bits':float(ent[offset:offset+length].mean()),
            'mean_occupied_codes_per_position':float((c>0).sum(1).mean()),
            'mean_mode_fraction':float((c.max(1)/denom).mean()),
            'scores':{name:compare(np.asarray(rows)[:,offset:offset+length],np.asarray(gold)[:,offset:offset+length],modal[offset:offset+length]) for name,rows in predictions.items()}})
        offset+=length
    if offset != counts.shape[0]:raise ValueError('complete twelve-level geometry required')
    return {'scores':scores,'levels':levels,'mode_codes':modal,
        'classification':'source_blind_mode_'+('matches_or_exceeds' if scores['positional_mode']['correct']>=scores['epoch2_matched']['correct'] else 'below')+'_epoch2_code_agreement'}


def ready():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only count analysis required')
    training,environment=probe.broad.runtime.configure()
    qualified=data.read(QUALIFICATION)
    if qualified['actual_tool_exit_code']!=0:raise ValueError('successful CPU contracts required')
    data.verify(qualified['artifacts'])
    probe.reference.receipt(probe.ROOTS['audit']);audit=data.completed(probe.ROOTS['audit'])
    if audit['status']!='public_pair_fit_independently_replayed':raise ValueError('complete fixed diagnostic required')
    probe.reference.receipt(probe.ROOTS['run']);data.completed(probe.ROOTS['run'])
    data.completed(bundle.ROOT)
    manifest=data.read(bundle.ROOT/'manifest.json')
    indexes={manifest[k]:manifest['index_hashes'][manifest[k]] for k in ('pair_index','code_index')}
    data.verify(indexes)
    pair_index=bundle.lines(manifest['pair_index']);code_index=bundle.lines(manifest['code_index'])
    if len(pair_index)!=162353 or len(code_index)!=162353:raise ValueError('complete TRAIN population required')
    for i,(pair,code) in enumerate(zip(pair_index,code_index)):
        if (pair['public_provenance']['split']!='train' or code['catalog_position']!=i
                or any(pair[k]!=code[k] for k in ('id','target_sha256','native_frame_sha256'))):
            raise ValueError('complete original TRAIN target/index bindings required')
    dataset=bundle.target.indexed.IndexedJSONL(manifest['codes'],code_index,manifest['code_payload_sha256'])
    cases=data.read(probe.ROOTS['run']/'cases.json');outputs=data.read(probe.ROOTS['run']/'outputs.json')
    chosen=[pair_index[i]['id'] for i in probe.select(pair_index)]
    if [r['id'] for r in cases]!=chosen:raise ValueError('unchanged fixed public sample required')
    return training,environment,manifest,pair_index,dataset,cases,outputs


def inputs(manifest):
    return [bundle.ROOT/n for n in ('manifest.json','result.json','actual_tool_receipt.json')]+[
        Path(manifest[k]) for k in ('pair_index','code_index','codes')]+[
        probe.ROOTS[s]/n for s in ('run','audit') for n in ('producer.json','result.json','actual_tool_receipt.json')]+[
        probe.ROOTS['run']/n for n in ('cases.json','outputs.json')]+[QUALIFICATION,Path('pixi.lock'),Path('pyproject.toml')]


def freeze(root,stage,training,environment,manifest,extra=()):
    files={Path(__file__).resolve(),PLAN.resolve(),TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            p=Path(name).resolve()
            if p.is_file() and p.suffix=='.py' and p.is_relative_to(Path('experiments').resolve()):files.add(p)
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof={'schema':'nal.ncm-public-code-prior.v1','hypothesis_id':HYPOTHESIS,'stage':stage,
        'environment':environment,'config':{'population':162353,'excluded_diagnostic_cases':32,'positions':641,
            'vocabulary':8192,'tie_rule':'lowest_code_ID','device':'CPU','posthoc_exploratory':True},
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},
        'input_hashes':{str(p):data.sha(p) for p in [*inputs(manifest),*extra]}}
    proof['fingerprint']=training.q.digest(proof);data.write(root/'producer.json',proof);return proof


def finish(root,proof,value,paths):
    data.verify(proof['source_hashes']);data.verify(proof['input_hashes'])
    if torch.cuda.is_initialized():raise ValueError('CPU analysis initialized CUDA')
    result={'schema':proof['schema'],'hypothesis_id':HYPOTHESIS,**value,
        'task_accuracy':None,'model_updates':0,'new_judge_requests':0,'final_test_opened':False,'speed_tested':False,
        'artifacts':{str(p):data.sha(p) for p in [root/'producer.json',*paths]}}
    data.write(root/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('artifacts','measurements')}),flush=True)


def accumulate(dataset,index,cases,limit,independent=False):
    excluded={r['id']:r for r in cases};seen=set();fitted=0;lengths=Counter()
    counts=[Counter() for _ in range(641)] if independent else np.zeros((641,8192),dtype=np.uint32)
    prefix=np.zeros((641,8192),dtype=np.uint32)
    for i in range(limit):
        row=dataset[i];pair=index[i]
        if (any(row[k]!=pair[k] for k in ('id','source_sha256','target_sha256','native_frame_sha256'))
                or row['codec_sha256']!=bundle.CODEC_SHA or len(row['codes'])!=641
                or any(type(c) is not int or not 0<=c<8192 for c in row['codes'])):
            raise ValueError('complete original code-label binding required')
        if row['id'] in excluded:
            if row['codes']!=excluded[row['id']]['oracle_codes']:raise ValueError('diagnostic oracle differs')
            seen.add(row['id']);continue
        if independent:
            for counter,code in zip(counts,row['codes']):counter[code]+=1
        else:add(counts,row['codes'])
        if i<256:add(prefix,row['codes'])
        lengths[pair['native_lexical_tokens']]+=1;fitted+=1
        if (i+1)%20000==0:print(json.dumps({'records_checked':i+1,'fitted':fitted,'independent':independent}),flush=True)
    if independent:
        replay=np.zeros((641,8192),dtype=np.uint32)
        for i,counter in enumerate(counts):
            for code,count in counter.items():replay[i,code]=count
        counts=replay
    if limit==162353 and (len(seen)!=32 or fitted!=162321):raise ValueError('exact diagnostic exclusions required')
    return counts,prefix,{'records_checked':limit,'fitted':fitted,'excluded':sorted(seen),
        'native_lexical_length_histogram':{str(k):v for k,v in sorted(lengths.items())}}


def run(stage):
    root=ROOTS[stage]
    if root.exists():raise ValueError('fresh append-only artifact root required')
    training,environment,manifest,index,dataset,cases,outputs=ready();extra=[]
    if stage=='run':
        probe.reference.receipt(ROOTS['pilot']);pilot=data.completed(ROOTS['pilot'])
        if pilot['status']!='public_code_prior_pilot_passed':raise ValueError('real CPU lifecycle pilot required')
        extra=[ROOTS['pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json','counts.npz')]
    proof=freeze(root,stage,training,environment,manifest,extra)
    counts,prefix,coverage=accumulate(dataset,index,cases,256 if stage=='pilot' else 162353)
    dataset.close()
    if stage=='run':
        with np.load(ROOTS['pilot']/'counts.npz') as saved:
            if not np.array_equal(prefix,saved['counts']):raise ValueError('pilot count replay differs')
    np.savez_compressed(root/'counts.npz',counts=counts)
    data.write(root/'coverage.json',coverage)
    measurements={} if stage=='pilot' else metrics(counts,cases,outputs)
    data.write(root/'measurements.json',measurements)
    finish(root,proof,{'status':'public_code_prior_pilot_passed' if stage=='pilot' else 'public_code_prior_completed',
        'fitted':coverage['fitted'],'systems_lifecycle_only_not_quality_evidence':stage=='pilot',
        'measurements':measurements},[root/n for n in ('counts.npz','coverage.json','measurements.json')])


def audit():
    root=ROOTS['audit']
    if root.exists():raise ValueError('fresh replay root required')
    training,environment,manifest,index,dataset,cases,outputs=ready()
    probe.reference.receipt(ROOTS['run']);data.completed(ROOTS['run'])
    extra=[ROOTS['run']/n for n in ('producer.json','result.json','actual_tool_receipt.json','counts.npz','coverage.json','measurements.json')]
    proof=freeze(root,'audit',training,environment,manifest,extra)
    counts,_,coverage=accumulate(dataset,index,cases,162353,independent=True);dataset.close()
    with np.load(ROOTS['run']/'counts.npz') as saved:
        if not np.array_equal(counts,saved['counts']):raise ValueError('independent positional counts differ')
    measured=metrics(counts,cases,outputs)
    if coverage!=data.read(ROOTS['run']/'coverage.json') or measured!=data.read(ROOTS['run']/'measurements.json'):
        raise ValueError('independent coverage or measurement replay differs')
    data.write(root/'measurements.json',measured)
    finish(root,proof,{'status':'public_code_prior_independently_replayed','fitted':coverage['fitted'],
        'classification':measured['classification'],'measurements':measured,'independent_counter_replay':True},[root/'measurements.json'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('pilot','run','audit'))
    stage=parser.parse_args().stage;root=ROOTS[stage];existed=root.exists()
    try:audit() if stage=='audit' else run(stage)
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
