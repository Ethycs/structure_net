"""Target-aware early-code interventions with unchanged free suffix generation."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

from experiments.structure_net import ncm_reserved_code_probe_v2 as recovery
from experiments.structure_net import ncm_sequential_levels as sequence

prior = recovery.original
data, controls, codec = prior.data, prior.controls, prior.codec
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-oracle-prefix-plan.md')
TEST = Path('tests/structure_net/test_ncm_oracle_prefix.py')
ANALYSIS = Path('experiments/neural_architecture_lab/ncm_oracle_prefix_results.py')
HYPOTHESIS = 'ncm-reserved-oracle-prefix-v1'
SCHEMA = 'nal.ncm-oracle-prefix.v1'
ROOTS = {s:data.ROOT/f'20261027_oracle_prefix_{s}_v1' for s in ('pilot','run','audit','results')}
BASE = data.ROOT/'20261027_reserved_code_probe_run_v2'
BASE_AUDIT = data.ROOT/'20261027_reserved_code_probe_audit_v2'
BASE_RESULTS = data.ROOT/'20261027_reserved_code_probe_results_v2'
ARMS = {'source_only':0,'oracle1':1,'oracle9':9,'oracle25':25,'oracle641':641}


@torch.no_grad()
def generate_with_prefix(model, features, mask, prefix):
    levels = model.concepts.generator.levels
    boundaries = {sum(levels[:i]) for i in range(len(levels)+1)}
    if (model.training or prefix.ndim != 2 or prefix.shape[0] != len(features)
            or prefix.shape[1] not in boundaries or prefix.dtype != torch.long
            or prefix.device != features.device or (prefix < 0).any() or (prefix >= 8192).any()):
        raise ValueError('eval mode and complete in-range prefix levels required')
    cumulative = features.new_zeros(len(features),levels[-1],model.codec.embed_dim)
    previous,output,offset = None,[],0
    for level,length in enumerate(levels):
        if offset < prefix.shape[1]:
            codes = prefix[:,offset:offset+length].detach().clone()
        else:
            codes = sequence.level_logits(model,features,mask,previous,level).argmax(-1)
        output.append(codes)
        cumulative,previous = sequence.advance(model.codec,cumulative,codes,level)
        offset += length
    return torch.cat(output,dim=1)


def suffix_counts(codes,gold,injected):
    controls.integers(codes,641,8192);controls.integers(gold,641,8192)
    if injected not in ARMS.values():raise ValueError('fixed intervention boundary required')
    return sum(a==b for a,b in zip(codes[injected:],gold[injected:])),641-injected


def case_metrics(out,row,arm):
    injected = ARMS[arm]
    if (row['split'] != 'reserved_component' or out['id'] != row['id'] or out['arm'] != arm
            or out['source_id_used'] != row['id'] or out['injected_codes'] != injected
            or out['source_sha256'] != row['source_sha256'] or out['target_sha256'] != row['target_sha256']
            or out['task_accuracy'] is not None):raise ValueError('fixed case/condition/source/target binding required')
    controls.integers(out['raw_word_ids'],128,30528)
    correct,positions = suffix_counts(out['codes'],row['oracle_codes'],injected)
    if out['codes'][:injected] != row['oracle_codes'][:injected]:raise ValueError('injected prefix differs')
    word = codec.metrics(row['native_word_ids'],out['raw_word_ids'],row['native_reference'],out['text'])
    levels,offset = [],0
    for length in controls.LEVELS:
        count = 0 if offset < injected else length
        right = sum(a==b for a,b in zip(out['codes'][offset:offset+count],row['oracle_codes'][offset:offset+count]))
        levels.append({'remaining_positions':count,'correct_remaining_codes':right});offset += length
    return {**word,'remaining_positions':positions,'correct_remaining_codes':correct,'levels':levels}


def aggregate(outputs,cases,arm):
    if len(outputs) != len(cases) or not cases:raise ValueError('complete ordered case population required')
    metrics = [case_metrics(o,r,arm) for o,r in zip(outputs,cases)]
    for out,expected in zip(outputs,metrics):
        if out['metrics'] != expected:raise ValueError('raw case metrics do not replay')
    sums = {k:sum(m[k] for m in metrics) for k in ('remaining_positions','correct_remaining_codes','content_tokens','content_matches')}
    return {**sums,'cases':len(cases),'injected_codes_per_case':ARMS[arm],
        'remaining_code_agreement':sums['correct_remaining_codes']/sums['remaining_positions'] if sums['remaining_positions'] else None,
        'content_agreement':sums['content_matches']/sums['content_tokens'],
        'exact_native_texts':sum(m['exact_rendered'] for m in metrics),
        'digit_cases':sum(m['has_digits'] for m in metrics),
        'digit_sequences_preserved':sum(m['has_digits'] and m['digit_sequence_preserved'] for m in metrics),
        'levels':[{k:sum(m['levels'][i][k] for m in metrics) for k in metrics[0]['levels'][i]} for i in range(12)]}


def effect(measured, baseline_suffix_correct, baseline_content):
    if measured['remaining_positions'] <= 0:raise ValueError('oracle-only control has no generated suffix')
    suffix_change = measured['correct_remaining_codes']-baseline_suffix_correct
    content_change = measured['content_matches']-baseline_content
    suffix_gain = suffix_change/measured['remaining_positions']
    content_gain = content_change/measured['content_tokens']
    return {'suffix_correct_change':suffix_change,'remaining_code_gain':suffix_gain,
        'content_match_change':content_change,'content_gain':content_gain,
        'joint_gate_passed':suffix_gain>=.10 and content_gain>=.20}


def replay(outputs,cases):
    if set(outputs) != set(ARMS):raise ValueError('all five conditions required')
    scores = {arm:aggregate(outputs[arm],cases,arm) for arm in ARMS}
    effects = {}
    for arm,injected in ARMS.items():
        if injected in (0,641):continue
        baseline_correct = sum(suffix_counts(o['codes'],r['oracle_codes'],injected)[0]
                               for o,r in zip(outputs['source_only'],cases))
        effects[arm] = effect(scores[arm],baseline_correct,scores['source_only']['content_matches'])
    passing = [arm for arm in effects if effects[arm]['joint_gate_passed']]
    return {'scores':scores,'effects':effects,'passing_interventions':passing,
        'classification':'early_prefix_rescue_supported' if passing else 'early_prefix_rescue_not_established'}


def compare_controls(outputs,cases,baseline,oracle):
    for row,plain,full in zip(cases,outputs['source_only'],outputs['oracle641']):
        for actual,expected in ((plain,baseline[row['id']]),(full,oracle[row['id']])):
            if any(actual[k] != expected[k] for k in ('codes','raw_word_ids','text')):
                raise ValueError('archived source-only or oracle reconstruction control differs')


def validate_text(outputs,render):
    for rows in outputs.values():
        for out in rows:
            if render(out['raw_word_ids']) != out['text']:raise ValueError('raw native text rendering differs')


def input_paths():
    paths = recovery.inputs()
    paths += [BASE/n for n in ('producer.json','result.json','actual_tool_receipt.json','outputs.json','cases.json','pilot_ids.json','split_proof.json')]
    paths += [BASE_AUDIT/n for n in ('result.json','actual_tool_receipt.json')]
    paths += [BASE_RESULTS/n for n in ('campaign_results.json','actual_tool_receipt.json','backend_exact_readback.json','backend_actual_tool_receipt.json')]
    paths += [codec.RUNS['eval']/n for n in ('producer.json','result.json','outputs.json','actual_tool_receipt.json')]
    paths += [data.ROOT/'20261025_codec_roundtrip_results_v1/campaign_results.json',Path('pixi.lock'),Path('pyproject.toml')]
    if any(not p.is_file() for p in paths):raise ValueError('every fingerprint input must exist before root creation')
    return list(dict.fromkeys(paths))


def prepared():
    recovery.install()
    for root,names in ((BASE,('producer.json','result.json')),(BASE_AUDIT,('result.json',))):
        controls.receipt(root,names);data.completed(root)
    prior.verify(data.read(BASE/'producer.json'))
    campaign = BASE_RESULTS/'campaign_results.json';published = data.read(campaign)
    for filename,target in (('actual_tool_receipt.json',campaign),('backend_actual_tool_receipt.json',BASE_RESULTS/'backend_exact_readback.json')):
        receipt = data.read(BASE_RESULTS/filename)
        if receipt['actual_tool_exit_code'] != 0 or receipt['artifacts'].get(str(target)) != data.sha(target):
            raise ValueError('actual completed report and independent readback required')
        data.verify(receipt['artifacts'])
    backend = data.read(BASE_RESULTS/'backend_exact_readback.json')
    if (backend['status'] != 'independent_registry_readback_passed' or backend['campaign_sha256'] != data.sha(campaign)
            or backend['report_sha256'] != data.sha(published['report'])):raise ValueError('registered baseline differs')
    rows,split,pilot_ids = prior.prepared()
    if rows != data.read(BASE/'cases.json') or pilot_ids != data.read(BASE/'pilot_ids.json'):
        raise ValueError('all original64 cases and pilot identities required')
    baseline = {r['id']:r for r in data.read(BASE/'outputs.json')['matched']}
    old_root = codec.RUNS['eval'];old_path = old_root/'outputs.json'
    old_campaign = data.read(data.ROOT/'20261025_codec_roundtrip_results_v1/campaign_results.json')
    if (data.read(old_root/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or old_campaign['status'] != 'all400_outputs_independently_replayed'
            or old_campaign['artifacts'].get(str(old_path)) != data.sha(old_path)):
        raise ValueError('original independently replayed codec controls required')
    oracle = {r['id']:r for r in data.read(old_path) if r['arm']=='parent_with_source' and r['split']=='reserved_component'}
    if (len(baseline) != 64 or len(oracle) != 64 or set(baseline) != set(oracle)
            or any(oracle[r['id']]['codes'] != r['oracle_codes'] for r in rows)):
        raise ValueError('both archived controls must cover every reserved source')
    return rows,split,pilot_ids,baseline,oracle


def freeze(root,stage,training,environment):
    if root.exists():raise ValueError('fresh append-only intervention root required')
    paths = input_paths()
    if stage=='run':paths += [ROOTS['pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json','outputs.json')]
    files = {Path(__file__).resolve(),PLAN.resolve(),TEST.resolve(),ANALYSIS.resolve(),
             Path('tests/structure_net/test_ncm_sequential_levels.py').resolve()}
    files.update(p.resolve() for p in training.native.vendor.VENDOR.rglob('*.py'))
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix=='.py' and p.is_relative_to(Path('experiments').resolve()):files.add(p)
    input_hashes = {str(p):data.sha(p) for p in paths}
    source_hashes = {str(p):data.sha(p) for p in sorted(files)}
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof = {'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,'stage':stage,'environment':environment,
        'config':{'arms':ARMS,'cases':64,'physical_GPU':0,'precision':'FP32','TF32':False,'seed':42,
            'cache_bytes':256*1024**2,'peak_limit_bytes':4900000000,'target_aware_diagnostic':True,
            'minimum_remaining_code_gain':.10,'minimum_content_gain':.20,
            'new_judge_requests':0,'model_updates':0,'task_accuracy':None,'final_test_opened':False,'speed_tested':False},
        'source_hashes':source_hashes,'input_hashes':input_hashes}
    proof['fingerprint'] = training.q.digest(proof);data.write(root/'producer.json',proof);return proof


@torch.no_grad()
def predict(model,features,mask,row,arm,tokenizer,rendering):
    count = ARMS[arm]
    prefix = torch.tensor([row['oracle_codes'][:count]],dtype=torch.long,device=features.device)
    codes = generate_with_prefix(model,features,mask,prefix)
    logits = model.word_logits(features,mask,codes)
    if logits.shape != (1,128,30528) or not torch.isfinite(logits).all():raise ValueError('finite complete native frame required')
    words = logits.argmax(-1)[0].cpu().tolist()
    out = {'id':row['id'],'arm':arm,'source_id_used':row['id'],'source_sha256':row['source_sha256'],
        'target_sha256':row['target_sha256'],'injected_codes':count,'codes':codes[0].cpu().tolist(),
        'raw_word_ids':words,'text':rendering.decode(words,tokenizer)['text'],'task_accuracy':None}
    out['metrics'] = case_metrics(out,row,arm);return out


def measure(stage):
    root = ROOTS[stage]
    if root.exists():raise ValueError('fresh measurement required')
    training,environment = controls.broad.runtime.configure();prior.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3060'):raise ValueError('idle physical GPU0 required')
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    service = controls.live_training()
    rows,split,pilot_ids,baseline,oracle = prepared()
    if stage=='run':
        controls.receipt(ROOTS['pilot']);state = data.completed(ROOTS['pilot'])
        if state['status'] != 'oracle_prefix_pilot_passed':raise ValueError('completed real CUDA/concurrency pilot required')
        prior.verify(data.read(ROOTS['pilot']/'producer.json'))
    cases = [next(r for r in rows if r['id']==i) for i in pilot_ids] if stage=='pilot' else rows
    proof = freeze(root,stage,training,environment)
    for name,value in [('cases.json',cases),('split_proof.json',split),('pilot_ids.json',pilot_ids),('training_service_before.json',service)]:data.write(root/name,value)
    index,manifest = [data.read(codec.RUNS['prepare']/n) for n in ('source_index.json','feature_manifest.json')]
    loader = training.preparation.streaming.FeatureCache(index,manifest,max_bytes=256*1024**2)
    model = training.load_initial();state = data.read(prior.PARENT/'result.json')
    if data.sha(prior.PARENT/'model.pt') != state['checkpoint_sha256']:raise ValueError('epoch2 checkpoint bytes differ')
    saved = torch.load(prior.PARENT/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(saved['model'],strict=True);del saved
    model.eval().requires_grad_(False);expected = state['final_model_sha256']
    if training.native.vendor.module_hash(model) != expected:raise ValueError('complete strict model load differs')
    tokenizer = training.native.tokenizer();outputs = {arm:[] for arm in ARMS}
    pilot = data.read(ROOTS['pilot']/'outputs.json') if stage=='run' else None
    archived_pilot = {arm:{r['id']:r for r in pilot[arm]} for arm in ARMS} if pilot else None
    for i,row in enumerate(cases):
        features = loader.get(row).unsqueeze(0).cuda();mask = torch.ones(features.shape[:2],device=features.device,dtype=torch.bool)
        with torch.no_grad():
            native = torch.tensor([row['native_word_ids']],device=features.device)
            if model.codec.encode(native)[0].cpu().tolist() != row['oracle_codes']:raise ValueError('frozen target encoding differs')
        for arm in ARMS:
            out = predict(model,features,mask,row,arm,tokenizer,training.preparation.rendering)
            if stage=='pilot' and predict(model,features,mask,row,arm,tokenizer,training.preparation.rendering) != out:
                raise ValueError('repeated intervention differs')
            if archived_pilot and row['id'] in archived_pilot[arm] and out != archived_pilot[arm][row['id']]:
                raise ValueError('full output differs from pilot')
            outputs[arm].append(out)
        if stage=='pilot' or (i+1)%8==0:print(json.dumps({'cases':i+1,'outputs':(i+1)*5}),flush=True)
    compare_controls(outputs,cases,baseline,oracle)
    if (training.native.vendor.module_hash(model) != expected or any(p.grad is not None for p in model.parameters())
            or torch.cuda.max_memory_allocated()>4900000000):raise ValueError('model mutation, gradients or memory limit')
    metrics = replay(outputs,cases);data.write(root/'outputs.json',outputs)
    data.write(root/'training_service_after.json',controls.live_training());prior.verify(proof)
    value = {'schema':SCHEMA,'hypothesis_id':HYPOTHESIS,'status':f'oracle_prefix_{stage}_passed',
        'cases':len(cases),'outputs':len(cases)*5,**metrics,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'archived_source_only_exact':True,'archived_oracle_exact':True,'target_aware_diagnostic':True,
        'task_accuracy':None,'model_updates':0,'new_judge_requests':0,'final_test_opened':False,'speed_tested':False,
        'artifacts':{str(p):data.sha(p) for p in root.glob('*.json')}}
    data.write(root/'result.json',value);model.close()
    print(json.dumps({'status':value['status'],'cases':len(cases),'classification':value['classification']}),flush=True)


def audit():
    root = ROOTS['audit']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or root.exists():raise ValueError('fresh CPU-only audit required')
    training,_ = controls.broad.runtime.configure()
    controls.receipt(ROOTS['run']);state = data.completed(ROOTS['run'])
    prior.verify(data.read(ROOTS['run']/'producer.json'))
    cases,_,_,baseline,oracle = prepared();outputs = data.read(ROOTS['run']/'outputs.json')
    if cases != data.read(ROOTS['run']/'cases.json'):raise ValueError('complete original64 required')
    tokenizer = training.native.tokenizer()
    validate_text(outputs,lambda ids:training.preparation.rendering.decode(ids,tokenizer)['text'])
    compare_controls(outputs,cases,baseline,oracle);metrics = replay(outputs,cases)
    if any(state[k] != v for k,v in metrics.items()):raise ValueError('independent replay differs')
    root.mkdir();value = {'status':'all320_oracle_prefix_outputs_replayed','hypothesis_id':HYPOTHESIS,
        **metrics,'task_accuracy':None,'new_judge_requests':0,'final_test_opened':False,
        'artifacts':{str(ROOTS['run']/n):data.sha(ROOTS['run']/n) for n in
            ('producer.json','result.json','actual_tool_receipt.json','cases.json','outputs.json')}}
    data.write(root/'result.json',value)
    print(json.dumps({'classification':value['classification'],'passing_interventions':value['passing_interventions']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('pilot','run','audit'))
    stage = parser.parse_args().stage;root = ROOTS[stage];existed = root.exists()
    try:audit() if stage=='audit' else measure(stage)
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
