"""Compare fixed domain and expanded decoders on the unchanged64 oracle codes."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_expanded_codec_training as training
from experiments.structure_net.ncm_memory_data import file_hash, write_json

base, native, q, ROOT = training.corpus.component, training.native, training.q, training.ROOT
MODULE = 'experiments.structure_net.ncm_expanded_codec_component'
HYPOTHESIS = 'ncm-expanded-codec-fixed64-oracle-numeric-v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-expanded-codec-component-plan.md')
TEST = Path('tests/structure_net/test_ncm_expanded_codec_component.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_expanded_codec_component.py')
TRAIN_AUDIT = ROOT/'20261024_expanded9936_codec_training_audit_v1'
BASE_AUDIT = ROOT/'20261024_sol64_codec_component_audit_v1'
WORKFLOW, PREP, PILOT, OUTPUTS = (ROOT/('20261024_expanded_codec64_'+n+'_v1') for n in ('workflow','prepare','pilot','outputs'))
ARMS = ('domain','expanded')


def require_training():
    if (q.read(training.WORKFLOW/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0
            or q.read(TRAIN_AUDIT/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0):
        raise ValueError('actual full decoder training and independent CPU replay required')
    native.vendor.terminal(training.TRAIN)
    audit = q.read(TRAIN_AUDIT/'result.json')
    result = q.read(training.TRAIN/'result.json')
    if (audit['status'] != 'expanded9936_FP16_decoder_samples_and_full_state_replayed_semantics_pending'
            or audit['results']['full']['updates'] != 1024
            or result['status'] != 'expanded_codec_FP16_fixed1024_trained_semantics_pending'
            or result['updates'] != 1024 or result['initial_decoder_sha256'] != training.WARM_DECODER_SHA
            or result['checkpoint_sha256'] != audit['results']['full']['checkpoint_sha256']
            or file_hash(training.TRAIN/'model.pt') != result['checkpoint_sha256']):
        raise ValueError('fixed final1024 expanded decoder required, never pilot or source-trained decoder')
    for path,expected in audit['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('audited expanded decoder evidence changed')
    saved = torch.load(training.TRAIN/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    if (saved['step'] != 1024 or saved['config'] != asdict(training.Config())
            or saved['scaler'] != result['scaler'] or saved['scaler']['scale'] != 128.
            or saved['scaler']['_growth_tracker'] != 1024
            or saved['producer_fingerprint'] != q.read(training.TRAIN/'producer.json')['fingerprint']):
        raise ValueError('complete final expanded checkpoint contract differs')
    return result,saved


def cases():
    if q.read(BASE_AUDIT/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual original64 code and rendering audit required')
    original = q.read(BASE_AUDIT/'result.json')
    for path,expected in original['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('unchanged original64 oracle evidence required')
    rows = base.cases()
    by_id = {r['id']:r for r in q.read(base.OUTPUTS/'outputs.json')}
    train = q.read(training.corpus.DATA/'index.json')
    for field in ('id','source_id','group_id','source_sha256','source_component'):
        if {r[field] for r in rows} & {r[field] for r in train}:
            raise ValueError('expanded codec corpus overlaps a component source family')
    return [{**r,'oracle_codes':by_id[r['id']]['codes']} for r in rows]


def load_decoders(device):
    result,saved = require_training()
    codec = training.load_codec('cpu').eval().requires_grad_(False)
    expanded = deepcopy(codec.decoder)
    expanded.load_state_dict(saved['decoder'],strict=True)
    if (native.vendor.module_hash(expanded) != result['final_decoder_sha256']
            or training.lifecycle.frozen_hash(codec) != result['protected_sha256_unchanged']):
        raise ValueError('exact full trained word decoder and original code geometry required')
    return codec.to(device),expanded.to(device).eval().requires_grad_(False),result


def producing(root,stage,inputs=()):
    files = [Path(__file__),PLAN,TEST,AUDITOR,Path(training.__file__),training.PLAN,training.AUDITOR,
        Path(training.corpus.__file__),Path(base.__file__),base.PLAN,Path(base.framed.__file__),
        *sorted(native.vendor.VENDOR.rglob('*.py'))]
    common = [training.TRAIN/n for n in ('model.pt','result.json','producer.json','worker_terminal_receipt.json')]
    common += [training.WORKFLOW/'actual_tool_receipt.json',TRAIN_AUDIT/'result.json',TRAIN_AUDIT/'actual_tool_receipt.json',
        BASE_AUDIT/'result.json',BASE_AUDIT/'actual_tool_receipt.json',base.OUTPUTS/'outputs.json',
        training.corpus.DATA/'index.json',training.corpus.DATA/'split_proof.json']
    p = native.vendor.freeze(root,stage,list(dict.fromkeys([*inputs,*common,*files])),{
        'hypothesis_id':HYPOTHESIS,'cases':64,'arms':ARMS,'same641_oracle_codes':True,
        'baseline_must_match_original_domain_raw_outputs':True,'expanded_checkpoint':'fixed final1024 only',
        'encoder_and_VQ_unchanged':True,'precision':'FP32','TF32':False,'physical_GPU':1,
        'max_peak_bytes':1_500_000_000,'raw_words':128,'full_vocabulary':30528,
        'parent_and_concept_generator_absent':True,'component64_excluded_from_gradients':True,
        'native_CLS_renderer':True,'no_repair':True,'source_prediction':False,'oracle_target_codes':True,
        'student_task_accuracy':None,'component_semantic_accuracy':None,'external_judgments':False,
        'final_test_opened':False,'speed_tested':False})
    p['hypothesis_id'] = HYPOTHESIS
    for i,path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root/'producer_source'/f'expanded_component_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k:v for k,v in p.items() if k!='fingerprint'})
    write_json(root/'producer.json',p)
    return p


def prepare():
    base.configure()
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only metadata preparation required')
    require_training()
    rows = cases()
    longest = min(rows,key=lambda r:(-sum(v!=0 for v in r['native_word_ids']),r['id']))
    ids = [rows[0]['id'],longest['id']]
    if len(set(ids)) != 2:
        raise ValueError('distinct first and longest native frame required')
    p = producing(PREP,'unchanged64_cases_codes_and_train_exclusion')
    write_json(PREP/'cases.json',rows)
    write_json(PREP/'selection.json',{'pilot_ids':ids})
    native.vendor.verify(p)
    write_json(PREP/'result.json',{'status':'unchanged64_expanded_codec_cases_prepared','cases':64})


@torch.no_grad()
def reconstruct(codec,expanded,row,tokenizer):
    codes = torch.tensor([row['oracle_codes']],dtype=torch.long,device='cuda')
    if codes.shape != (1,641) or (codes<0).any() or (codes>=8192).any():
        raise ValueError('all unchanged native codes required')
    latent = codec.vq.idx_to_z_q(codes)
    conditions = {name:base.decode(decoder(latent),tokenizer) for name,decoder in (('domain',codec.decoder),('expanded',expanded))}
    return {'id':row['id'],'source_sha256':row['source_sha256'],'codes':row['oracle_codes'],'conditions':conditions}


def generate(pilot=False):
    base.configure()
    base.require_gpu()
    native.vendor.terminal(PREP)
    rows = q.read(PREP/'cases.json')
    if rows != cases():
        raise ValueError('every original64 case must remain')
    ids = q.read(PREP/'selection.json')['pilot_ids']
    selected = [next(r for r in rows if r['id']==identity) for identity in ids] if pilot else rows
    retained,extra = {},[PREP/n for n in ('cases.json','selection.json','worker_terminal_receipt.json')]
    if not pilot:
        native.vendor.terminal(PILOT)
        retained = {r['id']:r for r in q.read(PILOT/'outputs.json')}
        extra += [PILOT/n for n in ('outputs.json','result.json','worker_terminal_receipt.json')]
    root = PILOT if pilot else OUTPUTS
    p = producing(root,'same_decoder_lifecycle_only' if pilot else 'all64_expanded_decoder_outputs',extra)
    codec,expanded,state = load_decoders('cuda')
    before = {'codec':native.vendor.module_hash(codec),'expanded':native.vendor.module_hash(expanded)}
    original = {r['id']:r for r in q.read(base.OUTPUTS/'outputs.json')}
    tokenizer,outputs = native.tokenizer(),[]
    torch.cuda.reset_peak_memory_stats()
    for row in selected:
        value = reconstruct(codec,expanded,row,tokenizer)
        if value['conditions']['domain'] != original[row['id']]['conditions']['domain']:
            raise ValueError('fixed baseline must reproduce every original domain output exactly')
        if pilot and reconstruct(codec,expanded,row,tokenizer) != value:
            raise ValueError('repeated native inference differs')
        if row['id'] in retained and retained[row['id']] != value:
            raise ValueError('pilot first output changed')
        outputs.append(value)
        write_json(root/'outputs.json',outputs)
    peak = torch.cuda.max_memory_allocated()
    if (peak>1_500_000_000 or before != {'codec':native.vendor.module_hash(codec),'expanded':native.vendor.module_hash(expanded)}
            or any(v.grad is not None for model in (codec,expanded) for v in model.parameters())):
        raise ValueError('frozen decoder state or1.5GB memory budget differs')
    native.vendor.verify(p)
    write_json(root/'result.json',{'status':'expanded_codec_lifecycle_only' if pilot else 'all64_expanded_codec_outputs_semantics_pending',
        'cases':len(outputs),'model_hashes':before,'checkpoint_sha256':state['checkpoint_sha256'],
        'models_unchanged':True,'outputs_sha256':file_hash(root/'outputs.json'),'peak_allocated_bytes':peak,
        'component_semantic_accuracy':None,'student_task_accuracy':None,'source_prediction':False,'final_test_opened':False})
    print(json.dumps(q.read(root/'result.json')),flush=True)


def workflow():
    base.configure()
    require_training()
    if any(root.exists() for root in (WORKFLOW,PREP,PILOT,OUTPUTS)):
        raise ValueError('fresh expanded codec numeric roots required')
    p = producing(WORKFLOW,'supervise_expanded_codec_fixed64_numeric_control')
    receipts = []
    for stage,root,gpu in (('prepare',PREP,''),('pilot',PILOT,'1'),('generate',OUTPUTS,'1')):
        native.vendor.verify(p)
        command = [sys.executable,'-u','-m',MODULE,stage]
        child = subprocess.Popen(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu})
        write_json(WORKFLOW/'progress.json',{'active_stage':stage,'child_pid':child.pid,'stages':receipts})
        code = child.wait()
        receipt = {'stage':stage,'exit_code':code,'child_pid':child.pid,'observer_pid':os.getpid(),'observer':'subprocess.wait','command':command}
        write_json(WORKFLOW/(stage+'_receipt.json'),receipt)
        if root.exists():
            write_json(root/'worker_terminal_receipt.json',receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('expanded codec numeric stage failed: '+stage)
    native.vendor.verify(p)
    write_json(WORKFLOW/'result.json',{'status':'expanded_codec_fixed64_numeric_completed_semantics_pending','stages':receipts})


if __name__=='__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','pilot','generate','workflow'))
    args = parser.parse_args()
    {'prepare':prepare,'pilot':lambda:generate(True),'generate':generate,'workflow':workflow}[args.stage]()
