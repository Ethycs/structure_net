"""Replay complete decoder exposure and compare the fixed held-out reconstructions."""
import argparse
import copy
from dataclasses import asdict
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import torch

from experiments.structure_net import ncm_recovered_decoder_training as study
from experiments.structure_net import ncm_recovered_label_codec_audit as raw_audit

data,runtime = study.data,study.runtime
BASELINE = runtime.ROOT
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-recovered-decoder-evaluation.md')
ROOTS = {s:data.ROOT/f'20261027_recovered_decoder_{s}_v1' for s in ('first_epoch_audit','audit','evaluate','replay')}


def replay_records(records,author,updates):
    orders = study.domain_orders()
    retention = torch.Generator().manual_seed(study.CONFIG.retention_seed)
    augmentation = torch.Generator().manual_seed(study.CONFIG.augmentation_seed)
    visits = torch.zeros(study.CONFIG.frames,dtype=torch.int32);presentations = 0;last = 0
    for last,row in enumerate(records,1):
        if last>updates or row['step'] != last or row['lr'] != study.rate(last): raise ValueError('step or learning rate differs')
        selected = study.positions(last,orders)
        if row['domain_positions'] != selected: raise ValueError('domain positions differ')
        retained = torch.randint(len(author),(study.CONFIG.retention_batch,),generator=retention).tolist()
        augmented = [study.old.augment(author[i],augmentation)[1] for i in retained]
        if row['retention_positions'] != retained or row['augmentation'] != augmented: raise ValueError('retention stream differs')
        if any(not math.isfinite(row[k]) or row[k]<0 for k in ('weighted_domain_CE','weighted_retention_CE','gradient_norm')) or row['gradient_norm']==0:
            raise ValueError('finite nonzero optimization statistics required')
        visits[selected] += 1;presentations += len(selected)
    if last != updates: raise ValueError('incomplete trace')
    epochs = updates//study.CONFIG.steps_per_epoch
    if updates%study.CONFIG.steps_per_epoch==0 and not visits.eq(epochs).all(): raise ValueError('full epoch coverage differs')
    return {'updates':updates,'domain_presentations':presentations,'retention_presentations':updates*study.CONFIG.retention_batch,
        'unique_domain_frames_visited':int(visits.gt(0).sum()),'minimum_domain_visits':int(visits.min()),'maximum_domain_visits':int(visits.max()),
        'all_retention_choices_and_augmentations_replayed':True,'all_domain_order_and_tail_positions_replayed':True}


def freeze(root,stage,training,environment,paths):
    files = {Path(__file__).resolve(),PLAN.resolve(),study.PLAN.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix=='.py' and (p.is_relative_to(Path('experiments').resolve()) or p.is_relative_to(training.native.vendor.VENDOR.resolve())):files.add(p)
    paths += [Path('pixi.lock'),Path('pyproject.toml')]
    proof = {'schema':'nal.ncm-recovered-decoder-evaluation.v1','stage':stage,'environment':environment,
        'training_config':asdict(study.CONFIG),'source_hashes':{str(p):data.sha(p) for p in sorted(files)},
        'input_hashes':{str(p):data.sha(p) for p in paths}}
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    data.write(root/'producer.json',proof);return proof


def audit(first_epoch=False):
    stage = 'first_epoch_audit' if first_epoch else 'audit';root = ROOTS[stage]
    if root.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('fresh CPU-only exposure audit required')
    training,environment = runtime.codec_source.prior.runtime()
    source = study.ROOTS['train'];producer = data.read(source/'producer.json')
    data.verify(producer['source_hashes']);data.verify(producer['input_hashes'])
    updates = study.CONFIG.steps_per_epoch if first_epoch else study.CONFIG.updates
    if not first_epoch:
        completed = study.require(source);study.require(ROOTS['first_epoch_audit'])
        if completed['status']!='full_recovered_decoder_training_completed' or completed['updates']!=updates:raise ValueError('complete fixed final run required')
    checkpoint = study.STORAGE/'train'/f'step{updates}.pt'
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    frozen = data.read(study.ROOTS['pilot']/'result.json')['frozen_state_sha256']
    if (saved['step']!=updates or saved['config']!=asdict(study.CONFIG) or saved['producer_fingerprint']!=producer['fingerprint']
            or saved['frozen_state_sha256']!=frozen):raise ValueError('committed checkpoint metadata differs')
    with (source/'trace.jsonl').open() as stream:
        lines = list(itertools.islice(stream,updates))
        if len(lines)!=updates or any(not line.endswith('\n') for line in lines) or not first_epoch and stream.readline():raise ValueError('exact committed trace extent required')
    paths = [source/'producer.json',checkpoint,study.old.DATA/'frames.pt',study.ROOTS['pilot']/'result.json']
    if not first_epoch:
        paths += [source/n for n in ('result.json','actual_tool_receipt.json','trace.jsonl')]
        paths += [ROOTS['first_epoch_audit']/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
        if data.sha(checkpoint)!=completed['checkpoint_sha256'] or completed['frozen_state_sha256']!=frozen:raise ValueError('final checkpoint identity differs')
    proof = freeze(root,stage,training,environment,paths)
    captured = root/'trace_prefix.jsonl';captured.write_text(''.join(lines))
    author = torch.load(study.old.DATA/'frames.pt',map_location='cpu',weights_only=True,mmap=True)
    records = [json.loads(line) for line in lines];counts = replay_records(iter(records),author,updates)
    if first_epoch:
        rejected = []
        for kind in ('domain','retention'):
            broken = copy.deepcopy(records[0])
            if kind=='domain':broken['domain_positions'][0]=(broken['domain_positions'][0]+1)%study.CONFIG.frames
            else:broken['augmentation'][0]['length']+=1
            try:replay_records(iter([broken]),author,1)
            except ValueError as error:
                expected = 'domain positions differ' if kind=='domain' else 'retention stream differs'
                if str(error)!=expected:raise
                rejected.append(kind)
            else:raise ValueError('tampered control was accepted')
        with (source/'trace.jsonl').open() as stream:
            if ''.join(itertools.islice(stream,updates))!=captured.read_text():raise ValueError('live committed prefix changed')
    else:
        rejected = None
        if any(counts[k]!=completed[k] for k in ('domain_presentations','retention_presentations')):raise ValueError('worker exposure totals differ')
    data.verify(proof['source_hashes']);data.verify(proof['input_hashes'])
    value = {'schema':proof['schema'],'status':'first_epoch_decoder_exposure_replayed' if first_epoch else 'complete_decoder_training_exposure_replayed',
        **counts,'checkpoint_sha256':data.sha(checkpoint),'recorded_frozen_state_sha256':frozen,
        'negative_controls_rejected':rejected,'full_training_completed':not first_epoch,'task_accuracy':None,'final_test_opened':False,
        'artifacts':{str(p):data.sha(p) for p in (root/'producer.json',captured)}}
    data.write(root/'result.json',value);print(json.dumps({k:v for k,v in value.items() if k!='artifacts'}),flush=True)


def comparison(old,new):
    if len(old)!=128 or [r['id'] for r in old]!=[r['id'] for r in new]:raise ValueError('same ordered128 outputs required')
    return {'fixes':sum(not a['exact_rendered'] and b['exact_rendered'] for a,b in zip(old,new)),
        'regressions':sum(a['exact_rendered'] and not b['exact_rendered'] for a,b in zip(old,new)),
        'both_exact':sum(a['exact_rendered'] and b['exact_rendered'] for a,b in zip(old,new)),
        'neither_exact':sum(not a['exact_rendered'] and not b['exact_rendered'] for a,b in zip(old,new))}


def evaluate():
    root = ROOTS['evaluate']
    if root.exists():raise ValueError('fresh fixed reconstruction required')
    audited = study.require(ROOTS['audit']);trained = study.require(study.ROOTS['train']);study.require(BASELINE)
    if audited['status']!='complete_decoder_training_exposure_replayed':raise ValueError('complete independently audited training required')
    training,environment = runtime.codec_source.prior.runtime();runtime.lifecycle.reference.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID'
            or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 2060 SUPER' or torch.cuda.mem_get_info()[0]<runtime.LIMIT):raise ValueError('GPU2 and headroom required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cases,old = [data.read(BASELINE/n) for n in ('cases.json','outputs.json')]
    if len(cases)!=128 or [r['id'] for r in cases]!=[r['id'] for r in old]:raise ValueError('entire fixed128 targets required')
    paths = [BASELINE/n for n in ('producer.json','result.json','actual_tool_receipt.json','cases.json','outputs.json')]
    for source in (study.ROOTS['train'],ROOTS['audit']):paths += [source/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    paths += [Path(trained['checkpoint']),runtime.codec_source.BROAD/'model.pt',training.native.RELEASE/'files/lm1b/tokenizer.pt',training.native.RELEASE/'files/lm1b/small/config.yaml']
    proof = freeze(root,'fixed128_reconstruction',training,environment,paths);data.write(root/'cases.json',cases)
    before_services = runtime.lifecycle.services();torch.cuda.reset_peak_memory_stats()
    codec,frozen = study.load_initial(training)
    saved = torch.load(trained['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    codec.decoder.load_state_dict(saved['decoder'],strict=True);codec.eval().requires_grad_(False);del saved
    if training.native.vendor.module_hash(codec)!=trained['final_codec_sha256'] or study.old.frozen_hash(codec)!=frozen:raise ValueError('exact trained codec with frozen representation required')
    tokenizer = training.native.tokenizer();outputs = []
    with torch.no_grad():
        for row,parent in zip(cases,old):
            gold = torch.tensor([row['native_word_ids']],device='cuda');codes = codec.encode(gold)
            if codes[0].cpu().tolist()!=parent['codes']:raise ValueError('encoded hierarchy changed')
            logits = codec.decode(codes)
            if logits.shape!=(1,128,30528) or not torch.isfinite(logits).all():raise ValueError('finite complete native output required')
            raw = logits.argmax(-1)[0].cpu().tolist();text = training.preparation.rendering.decode(raw,tokenizer)['text']
            outputs.append({'id':row['id'],'stratum':row['stratum'],'target_text':row['target_text'],'native_reference':row['native_reference'],
                'codes':codes[0].cpu().tolist(),'raw_word_ids':raw,'text':text,'invalid_outputs':int(any(v>=len(tokenizer) for v in raw)),
                'tokenizer_digit_sequence_preserved':re.findall(r'\d+',row['target_text'])==re.findall(r'\d+',row['native_reference']),
                **runtime.codec_source.prior.metrics(row['native_word_ids'],raw,row['native_reference'],text)})
            if len(outputs)%16==0:print(json.dumps({'reconstructed':len(outputs),'of':128}),flush=True)
    if training.native.vendor.module_hash(codec)!=trained['final_codec_sha256'] or any(p.grad is not None for p in codec.parameters()):raise ValueError('inference changed codec')
    if torch.cuda.max_memory_allocated()>runtime.LIMIT:raise ValueError('fixed inference memory cap exceeded')
    after_services = runtime.lifecycle.services()
    if any(after_services[u]['ExecMainPID']!=before_services[u]['ExecMainPID'] for u in before_services):raise ValueError('other worker identity changed')
    data.write(root/'outputs.json',outputs);data.write(root/'services.json',{'before':before_services,'after':after_services})
    data.verify(proof['source_hashes']);data.verify(proof['input_hashes'])
    metrics = runtime.aggregate(outputs);whole = metrics['all'];prior = runtime.aggregate(old)['all']
    value = {'schema':proof['schema'],'status':'fixed128_trained_decoder_reconstructions_completed','cases':128,'metrics':metrics,
        'parent_metrics':runtime.aggregate(old),'comparison':comparison(old,outputs),'codec_sha256':trained['final_codec_sha256'],
        'directional_reconstruction_improvement':whole['exact_rendered']>prior['exact_rendered'] and whole['digit_cases_preserved']>=prior['digit_cases_preserved'],
        'strong_reconstruction_gate':whole['exact_rendered']>=122 and whole['digit_cases_preserved']==27,
        'all128_encoded_hierarchies_unchanged':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'task_accuracy':None,'source_only_summary_accuracy':None,'final_test_opened':False,'speed_tested':False,
        'artifacts':{str(p):data.sha(p) for p in root.iterdir() if p.is_file()}}
    data.write(root/'result.json',value);print(json.dumps({k:v for k,v in value.items() if k!='artifacts'}),flush=True)


def replay():
    root = ROOTS['replay'];raw_audit.ROOT = root
    raw_audit.study = SimpleNamespace(ROOT=ROOTS['evaluate'],data=data,codec_source=runtime.codec_source,
        sample=runtime.sample,targets=runtime.targets,aggregate=runtime.aggregate)
    raw_audit.run()
    old = data.read(BASELINE/'outputs.json');new = data.read(ROOTS['evaluate']/'outputs.json')
    paired = comparison(old,new)
    if paired!=data.read(ROOTS['evaluate']/'result.json')['comparison']:raise ValueError('paired outcome counts differ')
    data.write(root/'paired_comparison.json',paired)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=tuple(ROOTS));args=parser.parse_args()
    root=ROOTS[args.stage];existed=root.exists()
    try:
        if args.stage=='first_epoch_audit':audit(True)
        elif args.stage=='audit':audit(False)
        elif args.stage=='evaluate':evaluate()
        else:replay()
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
