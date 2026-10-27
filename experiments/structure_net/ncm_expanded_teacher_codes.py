"""Encode new reviewed targets with the unchanged codec; preserve existing codes."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

from experiments.structure_net import ncm_expanded_teacher_catalog as catalog
from experiments.structure_net import ncm_broad_generator as broad
from experiments.structure_net import ncm_public_code_targets as previous
from experiments.structure_net import ncm_public_batch4_deterministic as deterministic

data, base, codec_source = catalog.data, catalog.base, previous.codec_source
HYPOTHESIS = 'ncm-expanded-reviewed-targets-v1'
SCHEMA = 'nal.ncm-expanded-reviewed-targets.v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-expanded-teacher-codes-plan.md')
TEST = Path('tests/structure_net/test_ncm_expanded_teacher_codes.py')
ROOTS = {s:data.ROOT/f'20261027_expanded_teacher_codes_{s}_v1'
         for s in ('pilot','full','pilot_audit','full_audit')}


def bind(row, codes):
    if (row['split'] != 'TRAIN' or len(codes) != 641
            or any(type(c) is not int or not 0 <= c < 8192 for c in codes)
            or len(row['native_word_ids']) != 128
            or catalog.old.text_hash(row['target_text']) != row['target_sha256']):
        raise ValueError('complete unmodified TRAIN target and integer hierarchy required')
    return {**row,'oracle_codes':codes}


def merge(rows, old_codes, fresh_codes):
    ids = [r['id'] for r in rows]
    if (len(ids) != len(set(ids)) or set(old_codes) & set(fresh_codes)
            or set(old_codes) | set(fresh_codes) != set(ids)):
        raise ValueError('exact disjoint old/fresh code population required')
    out = []
    for row in rows:
        if row['id'] in old_codes:
            value = old_codes[row['id']]
            if value != bind(row,value['oracle_codes']):
                raise ValueError('historical target or code binding changed')
        else:
            value = bind(row,fresh_codes[row['id']])
        out.append(value)
    return out


def same_first_target(before, after):
    # The complete collection file has different row offsets than its pilot.
    # Raw first-response paths, source, target, frame, and codes must all agree.
    return ({k:v for k,v in before.items() if k != 'teacher_first_provenance'}
            == {k:v for k,v in after.items() if k != 'teacher_first_provenance'})


def inputs(stage):
    pilot = stage == 'pilot'
    source = catalog.ROOTS['pilot' if pilot else 'prepare']
    state = base.require(source);base.verify_proof(data.read(source/'producer.json'))
    if pilot:
        if state['status'] != 'expanded_catalog_eight_first_lifecycle_passed':raise ValueError('completed real catalog pilot required')
    else:
        audited = base.require(catalog.ROOTS['audit'])
        base.verify_proof(data.read(catalog.ROOTS['audit']/'producer.json'))
        if audited['status'] != 'expanded_catalog_independently_replayed' or not audited['training_data_admitted']:
            raise ValueError('whole expanded TRAIN catalog must be admitted')
    paths = [source/n for n in ('producer.json','result.json','actual_tool_receipt.json','cases.json','arm_ids.json')]
    if not pilot:paths += [catalog.ROOTS['audit']/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    data.completed(broad.RUNS['prepare'])
    old_rows = data.read(broad.RUNS['prepare']/'cases.json')
    controls = data.read(broad.RUNS['prepare']/'selection.json')['diagnostic_ids']
    if len(old_rows) != 1376 or len(controls) != len(set(controls)) or len(controls) != 16:
        raise ValueError('all1376 original targets and16 unchanged controls required')
    old_codes = {r['id']:r for r in old_rows}
    rows = data.read(source/'cases.json');arms = data.read(source/'arm_ids.json')
    if arms['existing'] != [r['id'] for r in old_rows] or arms['expanded'] != [r['id'] for r in rows]:
        raise ValueError('unchanged existing/expanded arm order required')
    new = set(arms['new_admitted'])
    if not new or new & set(old_codes) or {r['id'] for r in rows} != set(old_codes) | new:
        raise ValueError('all admitted old and new target bindings required')
    if pilot:rows = [r for r in rows if r['id'] in new | set(controls)]
    old_codes = {r['id']:old_codes[r['id']] for r in rows if r['id'] in old_codes}
    paths += [broad.RUNS['prepare']/n for n in ('producer.json','result.json','actual_tool_receipt.json','cases.json','selection.json')]
    paths += [codec_source.BROAD/n for n in ('model.pt','result.json','actual_tool_receipt.json')]
    return rows,old_codes,controls,paths


def freeze(root,stage,environment,training,paths):
    if root.exists():raise ValueError('fresh append-only target stage required')
    files = {Path(__file__).resolve(),PLAN.resolve(),TEST.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                    or p.is_relative_to(training.native.vendor.VENDOR.resolve())):files.add(p)
    paths = list(dict.fromkeys([*paths,training.native.RELEASE/'files/lm1b/tokenizer.pt',
        training.native.RELEASE/'files/lm1b/small/config.yaml',Path('pixi.lock'),Path('pyproject.toml')]))
    proof = {'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,'stage':stage,'environment':environment,
        'config':{'codec_sha256':codec_source.CODEC_SHA,'precision':'FP32','TF32':False,'physical_GPU':0,
            'seed':42,'repeated_encode_every_new_target':True,'existing_control_cases':16,
            'existing_codes_reused_unchanged':True,'model_training':False,'final_test_opened':False},
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},
        'input_hashes':{str(p):data.sha(p) for p in paths}}
    proof['fingerprint'] = base.fingerprint(proof)
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    data.write(root/'producer.json',proof);return proof


def finish(root,proof,value,paths):
    base.verify_proof(proof)
    data.write(root/'result.json',{'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,**value,
        'model_training':False,'new_judge_requests':0,'task_accuracy':None,'final_test_opened':False,
        'artifacts':{str(p):data.sha(p) for p in [root/'producer.json',*paths]}})
    print(json.dumps(value),flush=True)


def encode(stage):
    training,environment = broad.runtime.configure();deterministic.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3060'):
        raise ValueError('physicalGPU0 RTX3060 required; main2060 trainer stays independent')
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    rows,old_codes,controls,paths = inputs(stage)
    pilot = {}
    if stage == 'full':
        checked = base.require(ROOTS['pilot_audit'])
        base.verify_proof(data.read(ROOTS['pilot_audit']/'producer.json'))
        if checked['status'] != 'expanded_target_bindings_replayed' or checked['stage'] != 'pilot':raise ValueError('audited target pilot required')
        pilot = {r['id']:r for r in data.read(ROOTS['pilot']/'cases.json')}
        paths += [ROOTS['pilot_audit']/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
        paths.append(ROOTS['pilot']/'cases.json')
    if data.sha(codec_source.BROAD/'model.pt') != codec_source.CHECKPOINT_SHA:raise ValueError('unchanged repaired-codec checkpoint required')
    root = ROOTS[stage];proof = freeze(root,stage,environment,training,paths)
    codec = training.native.load_codec().float().eval().requires_grad_(False)
    saved = torch.load(codec_source.BROAD/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    codec.load_state_dict(codec_source.codec_state(saved['model'],codec.state_dict()),strict=True);del saved
    if training.native.vendor.module_hash(codec) != codec_source.CODEC_SHA:raise ValueError('repaired codec tensor identity differs')
    torch.cuda.reset_peak_memory_stats();tokenizer = training.native.tokenizer();fresh = {}
    with torch.no_grad():
        for identity in controls:
            row = old_codes[identity]
            codes = codec.encode(torch.tensor([row['native_word_ids']],device='cuda',dtype=torch.long))[0].cpu().tolist()
            if codes != row['oracle_codes']:raise ValueError('old TRAIN code control differs')
        for row in rows:
            if row['id'] in old_codes:continue
            lexical = tokenizer.encode(row['target_text'],add_special_tokens=False)
            if (not 0 < len(lexical) <= 126 or set(lexical)&set(tokenizer.all_special_ids)
                    or row['native_word_ids'] != [101,*lexical,101]+[0]*(126-len(lexical))):
                raise ValueError('complete native target frame differs')
            words = torch.tensor([row['native_word_ids']],device='cuda',dtype=torch.long)
            codes = codec.encode(words)
            if not torch.equal(codes,codec.encode(words)):raise ValueError('new target encode is not exactly repeatable')
            fresh[row['id']] = codes[0].cpu().tolist()
            if len(fresh)%128 == 0:print(json.dumps({'new_targets_encoded':len(fresh)}),flush=True)
    values = merge(rows,old_codes,fresh)
    current = {r['id']:r for r in values}
    if any(k not in current or not same_first_target(v,current[k]) for k,v in pilot.items()):
        raise ValueError('pilot target codes or original first text changed')
    if training.native.vendor.module_hash(codec) != codec_source.CODEC_SHA or any(p.grad is not None for p in codec.parameters()):
        raise ValueError('frozen codec changed or received gradients')
    peak = torch.cuda.max_memory_allocated()
    if peak > 4_900_000_000:raise ValueError('codec GPU allocation exceeded budget')
    data.write(root/'cases.json',values)
    finish(root,proof,{'status':'expanded_targets_encoded','stage':stage,'cases':len(values),
        'new_encoded':len(fresh),'existing_reused':len(old_codes),'existing_controls_exact':16,
        'pilot_cases_reproduced':len(pilot),'all_new_encodes_repeated_exactly':True,'codec_unchanged':True,
        'codec_sha256':codec_source.CODEC_SHA,'peak_allocated_bytes':peak,'training_targets_admitted':False},[root/'cases.json'])


def audit(stage):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():raise ValueError('CPU-only binding replay required')
    training,environment = broad.runtime.configure();source = ROOTS[stage]
    state = base.require(source);base.verify_proof(data.read(source/'producer.json'))
    rows,old_codes,controls,paths = inputs(stage)
    values = data.read(source/'cases.json');fresh = {r['id']:r['oracle_codes'] for r in values if r['id'] not in old_codes}
    if (values != merge(rows,old_codes,fresh) or state['cases'] != len(values)
            or state['new_encoded'] != len(fresh) or state['existing_reused'] != len(old_codes)
            or state['existing_controls_exact'] != len(controls) or not state['all_new_encodes_repeated_exactly']
            or state['codec_sha256'] != codec_source.CODEC_SHA or not state['codec_unchanged']):
        raise ValueError('complete target/code source binding replay differs')
    if stage == 'full':
        current = {r['id']:r for r in values}
        if any(not same_first_target(r,current[r['id']]) for r in data.read(ROOTS['pilot']/'cases.json')):
            raise ValueError('original pilot target changed')
    paths += [source/n for n in ('producer.json','result.json','actual_tool_receipt.json','cases.json')]
    root = ROOTS[stage+'_audit'];proof = freeze(root,'audit_'+stage,environment,training,paths)
    finish(root,proof,{'status':'expanded_target_bindings_replayed','stage':stage,'cases':len(values),
        'new_encoded':len(fresh),'existing_reused':len(old_codes),'independent_GPU_reencoding':False,
        'training_targets_admitted':stage == 'full'},[])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('encode','audit'))
    parser.add_argument('stage',choices=('pilot','full'));args = parser.parse_args()
    root = ROOTS[args.stage+('_audit' if args.action == 'audit' else '')];existed = root.exists()
    try:globals()[args.action](args.stage)
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'error_type':type(error).__name__,'error':str(error)})
        raise
