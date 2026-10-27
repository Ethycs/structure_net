"""Diagnose reserved source-to-code prediction, without semantic success claims."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch

from experiments.structure_net import ncm_public_training_probe as controls
from experiments.structure_net import ncm_codec_roundtrip as codec
from experiments.structure_net import ncm_shared_summary_source_guard as guard
from experiments.structure_net import ncm_shared_summary_inventory as inventory

data, study = controls.data, controls.study
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-reserved-code-probe-plan.md')
TEST = Path('tests/structure_net/test_ncm_reserved_code_probe.py')
HYPOTHESIS = 'ncm-public-epoch2-reserved-code-probe-v1'
SCHEMA = 'nal.ncm-reserved-code-probe.v1'
ROOTS = {s:data.ROOT/f'20261027_reserved_code_probe_{s}_v1' for s in ('pilot','run','audit','results')}
ORACLE = data.ROOT/'20261024_sol64_grounded_oracle_prepare_v1'
PARENT = controls.PARENTS['epoch2']
KEYS = ('id','source_id','group_id','source_component','source_sha256')


def select(cases, oracles):
    rows = [r for r in cases if r['split'] == 'reserved_component']
    indexed = {r['id']:r for r in oracles}
    if len(rows) != 64 or len(indexed) != 64 or len(oracles) != 64 or {r['id'] for r in rows} != set(indexed):
        raise ValueError('all64 original reserved cases and oracle hierarchies required')
    result = []
    for row in rows:
        old = indexed[row['id']]
        if any(row[k] != old[k] for k in (*KEYS,'source','native_word_ids')):
            raise ValueError('original source, target frame or oracle binding changed')
        controls.integers(old['oracle_codes'],641,8192)
        controls.integers(row['native_word_ids'],128,30528)
        result.append({**row,'oracle_codes':old['oracle_codes']})
    return result


def identity_guard(train, held):
    overlaps = {k:len({r[k] for r in train} & {r[k] for r in held}) for k in KEYS}
    if any(overlaps.values()):raise ValueError('TRAIN/reserved source identity overlap')
    return {'train_cases':len(train),'reserved_cases':len(held),'overlaps':overlaps}


def score(outputs, cases, shuffled=False):
    if not cases or len(outputs) != len(cases):raise ValueError('complete ordered diagnostic required')
    levels = [{'positions':0,'teacher_correct':0,'rollout_correct':0} for _ in controls.LEVELS]
    active = matches = 0
    ce = 0.
    for i,(out,row) in enumerate(zip(outputs,cases)):
        source = cases[(i+1)%len(cases)] if shuffled else row
        if (row['split'] != 'reserved_component' or out['id'] != row['id']
                or out['source_id_used'] != source['id'] or out['task_accuracy'] is not None):
            raise ValueError('reserved case/source binding differs')
        for k in ('codes','teacher_codes'):controls.integers(out[k],641,8192)
        controls.integers(out['raw_word_ids'],128,30528)
        controls.integers(row['oracle_codes'],641,8192)
        controls.integers(row['native_word_ids'],128,30528)
        if len(out['levels']) != 12 or out['codes'][0] != out['teacher_codes'][0]:
            raise ValueError('complete levels and identical source-only root required')
        offset = 0
        for j,n in enumerate(controls.LEVELS):
            gold = row['oracle_codes'][offset:offset+n]
            t = sum(a==b for a,b in zip(out['teacher_codes'][offset:offset+n],gold))
            r = sum(a==b for a,b in zip(out['codes'][offset:offset+n],gold))
            m = out['levels'][j]
            if (m['positions'] != n or m['teacher_correct'] != t or m['rollout_correct'] != r
                    or not math.isfinite(m['CE']) or m['CE'] < 0):
                raise ValueError('raw per-level counts or CE differ')
            for key,value in (('positions',n),('teacher_correct',t),('rollout_correct',r)):
                levels[j][key] += value
            ce += m['CE'];offset += n
        count = sum(v != 0 for v in row['native_word_ids'])
        same = sum(a==b and b!=0 for a,b in zip(out['raw_word_ids'],row['native_word_ids']))
        if not count or out['active_words'] != count or out['active_word_matches'] != same:
            raise ValueError('native-word counts differ')
        active += count;matches += same
    positions = len(cases)*641
    teacher = sum(v['teacher_correct'] for v in levels)
    rollout = sum(v['rollout_correct'] for v in levels)
    return {'cases':len(cases),'levels':levels,'teacher_correct_codes':teacher,
        'rollout_correct_codes':rollout,'code_positions':positions,
        'teacher_code_agreement':teacher/positions,'rollout_code_agreement':rollout/positions,
        'teacher_rollout_gap':(teacher-rollout)/positions,
        'root_agreement':levels[0]['teacher_correct']/len(cases),
        'active_word_matches':matches,'active_words':active,'native_word_agreement':matches/active,
        'mean_teacher_CE':ce/(len(cases)*12)}


def classify(matched):
    if matched['teacher_code_agreement'] < .9:return 'reserved_teacher_prediction_also_weak'
    if matched['teacher_rollout_gap'] >= .2:return 'reserved_exposure_error_pattern_supported'
    return 'reserved_exposure_error_pattern_unresolved'


def inputs():
    catalog = study.pipeline.bundle.catalog
    paths = [codec.RUNS['prepare']/n for n in ('producer.json','result.json','actual_tool_receipt.json',
        'cases.json','source_index.json','feature_manifest.json')]
    paths += [ORACLE/n for n in ('producer.json','result.json','worker_terminal_receipt.json','cases.json')]
    paths += [PARENT/n for n in ('producer.json','result.json','actual_tool_receipt.json','model.pt')]
    paths += [catalog.INSPECT/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    paths += [catalog.INSPECTION_AUDIT/n for n in ('result.json','actual_tool_receipt.json','campaign_results.json')]
    paths += [catalog.AUDIT/n for n in ('result.json','actual_tool_receipt.json','campaign_results.json')]
    paths += [catalog.DEST/'training_index.jsonl',catalog.EXISTING/'cases.json',inventory.COMPONENT]
    paths += [data.ROOT/'20261025_codec_roundtrip_results_v1'/n for n in
              ('campaign_results.json','actual_tool_receipt.json','backend_exact_readback.json')]
    return paths


def prepared():
    catalog = study.pipeline.bundle.catalog
    for root in (catalog.INSPECT,catalog.INSPECTION_AUDIT,catalog.AUDIT):data.completed(root)
    controls.receipt(PARENT);data.completed(PARENT)
    # This historical producer predates artifact-bound execution receipts.
    # Retain its real exit receipt and independently verify its original proof,
    # output hashes and the published replay's binding to these exact cases.
    receipt = data.read(codec.RUNS['prepare']/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code'] != 0:raise ValueError('actual original codec preparation completion required')
    codec.verify(data.read(codec.RUNS['prepare']/'producer.json'))
    old_result = data.read(codec.RUNS['prepare']/'result.json')
    data.verify({str(codec.RUNS['prepare']/k):v for k,v in old_result['artifacts'].items()})
    old_replay = data.read(data.ROOT/'20261025_codec_roundtrip_results_v1/campaign_results.json')
    if data.read(data.ROOT/'20261025_codec_roundtrip_results_v1/actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual independent codec replay completion required')
    cases_path = codec.RUNS['prepare']/'cases.json'
    if (old_replay['status'] != 'all400_outputs_independently_replayed'
            or old_replay['artifacts'].get(str(cases_path)) != data.sha(cases_path)):
        raise ValueError('independently replayed original codec cases required')
    rows = select(data.read(codec.RUNS['prepare']/'cases.json'),data.read(ORACLE/'cases.json'))
    original = {r['id']:r for r in data.read(inventory.COMPONENT)}
    if len(original) != 64 or any(any(original[r['id']][k] != r[k] for k in (*KEYS,'source')) for r in rows):
        raise ValueError('exact originally screened component sources required')
    producer = data.read(catalog.INSPECT/'producer.json')
    inspected = data.read(catalog.INSPECT/'result.json')
    if (producer['input_hashes'].get(str(inventory.COMPONENT)) != data.sha(inventory.COMPONENT)
            or not inspected['open_source_and_protected_exact_hash_checks_completed']
            or inspected['counts']['rows'] != 204045):
        raise ValueError('exhaustive public full-text screen must bind these64 sources')
    reviewed = data.read(catalog.EXISTING/'cases.json')
    if len(reviewed) != 1376:raise ValueError('all reviewed TRAIN required')
    proof = {'reviewed':identity_guard(reviewed,rows)}
    text_index = guard.TextIndex([r['source'] for r in rows])
    if any(text_index.matches(r['source']) for r in reviewed):raise ValueError('reviewed TRAIN text overlap')
    with (catalog.DEST/'training_index.jsonl').open() as f:
        public = []
        for entry in map(json.loads,f):
            family = 'xsum_'+catalog.REVISION+'_train_'+entry['public_provenance']['article_id']
            public.append({'id':entry['id'],'source_sha256':entry['source_sha256'],
                'source_id':family,'group_id':family,'source_component':family})
    if len(public) != 162353:raise ValueError('all admitted public TRAIN required')
    proof.update(public=identity_guard(public,rows),reviewed_text_overlaps=0,
        audited_public_text_screen_reused=True,component_source_sha256=data.sha(inventory.COMPONENT),
        sealed_final_opened=False)
    manifest = data.read(codec.RUNS['prepare']/'feature_manifest.json')
    longest = min(rows,key=lambda r:(-manifest[r['id']]['source_tokens'],r['id']))
    pilot_ids = [rows[0]['id'],longest['id']]
    if len(set(pilot_ids)) != 2:raise ValueError('distinct first/longest source pilot required')
    return rows,proof,pilot_ids


def freeze(root, stage, environment, training):
    if root.exists():raise ValueError('fresh append-only stage required')
    files = {Path(__file__).resolve(),PLAN.resolve(),TEST.resolve()}
    files.update(p.resolve() for p in training.native.vendor.VENDOR.rglob('*.py'))
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):files.add(p)
    paths = [*inputs(),Path('pixi.lock'),Path('pyproject.toml')]
    if stage == 'run':paths += [ROOTS['pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json','outputs.json')]
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,'stage':stage,'environment':environment,
        'config':{'cases':64,'seed':42,'physical_GPU':0,'precision':'FP32','TF32':False,
            'cache_bytes':256*1024**2,'peak_limit_bytes':4900000000,'task_accuracy':None,
            'final_test_opened':False,'new_judge_requests':0,'model_updates':0,'speed_tested':False},
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},
        'input_hashes':{str(p):data.sha(p) for p in paths}}
    proof['fingerprint'] = training.q.digest(proof)
    data.write(root/'producer.json',proof);return proof


def verify(proof):
    data.verify(proof['source_hashes']);data.verify(proof['input_hashes'])


def measure(stage):
    root = ROOTS[stage]
    if root.exists():raise ValueError('fresh diagnostic root required')
    training,environment = controls.broad.runtime.configure()
    study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3060'):
        raise ValueError('idle physical GPU0 required')
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    service = controls.live_training()
    rows,split_proof,pilot_ids = prepared()
    if stage == 'run':
        controls.receipt(ROOTS['pilot']);pilot = data.completed(ROOTS['pilot'])
        if pilot['status'] != 'reserved_code_pilot_passed':raise ValueError('completed real CUDA/concurrency pilot required')
        verify(data.read(ROOTS['pilot']/'producer.json'))
    cases = [next(r for r in rows if r['id']==i) for i in pilot_ids] if stage == 'pilot' else rows
    proof = freeze(root,stage,environment,training)
    for name,value in [('cases.json',cases),('split_proof.json',split_proof),('pilot_ids.json',pilot_ids),('training_service_before.json',service)]:data.write(root/name,value)
    index,manifest = [data.read(codec.RUNS['prepare']/n) for n in ('source_index.json','feature_manifest.json')]
    loader = training.preparation.streaming.FeatureCache(index,manifest,max_bytes=256*1024**2)
    model = training.load_initial();state = data.read(PARENT/'result.json')
    if data.sha(PARENT/'model.pt') != state['checkpoint_sha256']:raise ValueError('epoch2 checkpoint bytes differ')
    saved = torch.load(PARENT/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(saved['model'],strict=True);del saved
    model.eval().requires_grad_(False);expected = state['final_model_sha256']
    if training.native.vendor.module_hash(model) != expected:raise ValueError('strict complete model load differs')
    with torch.no_grad():
        for row in cases:
            gold = torch.tensor([row['native_word_ids']],device='cuda')
            if model.codec.encode(gold)[0].cpu().tolist() != row['oracle_codes']:
                raise ValueError('frozen oracle encoding differs')
    outputs,scores = {},{}
    tokenizer = training.native.tokenizer()
    for shuffled in ((False,) if stage=='pilot' else (False,True)):
        name = 'shuffled' if shuffled else 'matched'
        with controls.capture_teacher(model.concepts) as codes:
            out = training.prior.diagnostics(model,cases,loader,tokenizer,shuffle=shuffled)
        if len(codes) != len(out):raise ValueError('complete teacher capture required')
        out = [{**r,'teacher_codes':c} for r,c in zip(out,codes)]
        scores[name] = score(out,cases,shuffled)
        if stage == 'pilot':
            with controls.capture_teacher(model.concepts) as repeated:
                again = training.prior.diagnostics(model,cases,loader,tokenizer)
            if out != [{**r,'teacher_codes':c} for r,c in zip(again,repeated)]:raise ValueError('repeated pilot differs')
        elif not shuffled:
            by_id = {r['id']:r for r in out}
            if any(by_id[r['id']] != r for r in data.read(ROOTS['pilot']/'outputs.json')['matched']):
                raise ValueError('full output differs from pilot')
        if (training.native.vendor.module_hash(model) != expected or any(p.grad is not None for p in model.parameters())
                or torch.cuda.max_memory_allocated() > 4900000000):raise ValueError('model mutation, gradients or GPU memory limit')
        outputs[name] = out;data.write(root/f'outputs_{name}.json',out)
        print(json.dumps({'condition':name,'cases':len(out)}),flush=True)
    data.write(root/'outputs.json',outputs);data.write(root/'training_service_after.json',controls.live_training())
    verify(proof)
    value = {'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,'status':f'reserved_code_{stage}_passed',
        'cases':len(cases),'scores':scores,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'task_accuracy':None,'new_judge_requests':0,'model_updates':0,'final_test_opened':False,'speed_tested':False,
        'artifacts':{str(p):data.sha(p) for p in root.glob('*.json')}}
    data.write(root/'result.json',value);model.close();print(json.dumps({'status':value['status'],'cases':len(cases)}),flush=True)


def audit():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or ROOTS['audit'].exists():raise ValueError('fresh CPU audit required')
    source = ROOTS['run'];controls.receipt(source);state = data.completed(source)
    verify(data.read(source/'producer.json'));cases,_,_ = prepared()
    outputs = data.read(source/'outputs.json')
    if cases != data.read(source/'cases.json') or set(outputs) != {'matched','shuffled'}:
        raise ValueError('complete original64 paired conditions required')
    scores = {name:score(rows,cases,name=='shuffled') for name,rows in outputs.items()}
    if scores != state['scores']:raise ValueError('independent raw replay differs')
    root = ROOTS['audit'];root.mkdir()
    value = {'status':'reserved_code_raw_counts_replayed','classification':classify(scores['matched']),
        'scores':scores,'task_accuracy':None,'final_test_opened':False,'new_judge_requests':0,
        'artifacts':{str(source/n):data.sha(source/n) for n in ('producer.json','result.json','actual_tool_receipt.json','cases.json','outputs.json')}}
    data.write(root/'result.json',value);print(json.dumps({'classification':value['classification'],'scores':scores}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('pilot','run','audit'))
    stage = parser.parse_args().stage;root = ROOTS[stage];existed = root.exists()
    try:audit() if stage == 'audit' else measure(stage)
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
