"""Complete source-only inference for the new checkpoint, without judge calls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_level_conditioning_checkpoint as checkpoint
from experiments.structure_net import ncm_lm1b_native_development as source
from experiments.structure_net.ncm_memory_data import file_hash, write_json

training, native, q, ROOT = checkpoint.training, checkpoint.native, checkpoint.q, checkpoint.ROOT
MODULE = 'experiments.structure_net.ncm_level_conditioning_numeric'
HYPOTHESIS = 'ncm-level-conditioning-fixed128-source-only-inference-v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-level-conditioning-numeric-plan.md')
TEST = Path('tests/structure_net/test_ncm_level_conditioning_numeric.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_level_conditioning_numeric.py')
WORKFLOW, PREP, PILOT, OUTPUTS = (ROOT/('20261024_level_conditioning_numeric_'+n+'_v1') for n in ('workflow','prepare','pilot','outputs'))


def inputs():
    cases,manifest = source.source_inputs()
    source.validate_sources(cases)
    proof = training.split_guard(q.read(training.DATA/'cases.json'),cases)
    return cases,manifest,proof


def producing(root,stage,extra=()):
    files = [Path(__file__),PLAN,TEST,AUDITOR,Path(checkpoint.__file__),Path(source.__file__),
        Path(source.framed.__file__),Path(training.__file__),training.PLAN,training.AUDITOR,
        Path(training.lifecycle.__file__),Path(training.lifecycle.mechanism.__file__),
        Path(training.lifecycle.head.__file__),Path(training.lifecycle.head.concepts.__file__),
        *sorted(native.vendor.VENDOR.rglob('*.py'))]
    common = [source.SOURCE/'cases.json',source.FEATURES/'manifest.json',training.DATA/'cases.json',
        training.TRAIN/'model.pt',training.TRAIN/'result.json',training.TRAIN/'worker_terminal_receipt.json',
        training.WORKFLOW/'result.json',training.WORKFLOW/'tool_terminal_receipt.json',
        checkpoint.TRAIN_AUDIT/'result.json',checkpoint.TRAIN_AUDIT/'actual_tool_terminal_receipt.json']
    p = native.vendor.freeze(root,stage,list(dict.fromkeys([*extra,*common,*files])),{
        'hypothesis_id':HYPOTHESIS,'cases':128,'cohorts':source.COHORTS,'source_only':True,
        'target_fields_in_prediction':False,'oracle_target_codes':False,'checkpoint':'new final1024 FP16native655 only',
        'parent_body_and_MLPs_frozen_and_absent':True,'NCM_generator_and_word_decoder_trained':True,
        'encoder_VQ_frozen':True,'precision':'FP32','TF32':False,'physical_GPU':2,
        'all_source_positions_retained':True,'native_codes':641,'raw_words':128,'word_vocabulary':30528,
        'native_CLS_renderer':True,'repair_or_clipping':False,'max_peak_bytes':3_000_000_000,
        'semantic_cache_read_or_written':False,'external_judgments':False,'model_training':False,
        'source_only_task_accuracy':None,'final_test_opened':False,'speed_tested':False,
        'future_semantic_study_must_reproduce_all128_outputs_exactly':True})
    p['hypothesis_id'] = HYPOTHESIS
    for i,path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root/'producer_source'/f'level_numeric_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k:v for k,v in p.items() if k!='fingerprint'})
    write_json(root/'producer.json',p)
    return p


def prepare():
    native.vendor.corpus.quality.dependencies()
    if torch.cuda.is_initialized() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only source metadata preparation required')
    checkpoint.checkpoint_contract()
    cases,manifest,proof = inputs()
    public = source.reference.verify_public([r for name in source.COHORTS for r in cases[name]])
    p = producing(PREP,'fixed128_target_free_source_metadata',[source.reference.PUBLIC_FILE])
    write_json(PREP/'cases.json',cases)
    write_json(PREP/'split_proof.json',proof)
    write_json(PREP/'public_proof.json',{'public_file_sha256':source.reference.PUBLIC_SHA256,'exact_sources':public})
    write_json(PREP/'selection.json',{'pilot_ids':[r['id'] for r in source.inference_probes(cases,manifest)]})
    native.vendor.verify(p)
    write_json(PREP/'result.json',{'status':'fixed128_target_free_source_metadata_prepared','cases':128,
        'task_accuracy':None,'semantic_cache_untouched':True,'final_test_opened':False})


def generate(pilot=False):
    native.vendor.corpus.quality.dependencies()
    training.require_gpu()
    native.vendor.terminal(PREP)
    cases,manifest,_ = inputs()
    if cases != q.read(PREP/'cases.json'):
        raise ValueError('all original128 source-only cases required')
    rows = source.inference_probes(cases,manifest) if pilot else [r for name in source.COHORTS for r in cases[name]]
    root = PILOT if pilot else OUTPUTS
    extra = [PREP/n for n in ('cases.json','selection.json','split_proof.json','worker_terminal_receipt.json')]
    prior = {}
    if not pilot:
        native.vendor.terminal(PILOT)
        prior = {r['id']:r for r in q.read(PILOT/'outputs.json')}
        extra += [PILOT/n for n in ('outputs.json','result.json','worker_terminal_receipt.json')]
    extra += [Path(manifest[r['id']]['path']) for r in rows]
    p = producing(root,'first_and_longest_frozen_inference_lifecycle' if pilot else 'all128_complete_source_only_predictions',extra)
    model,state = checkpoint.load_selected()
    before = native.vendor.module_hash(model)
    tokenizer,outputs = native.tokenizer(),[]
    torch.cuda.reset_peak_memory_stats()
    try:
        for row in rows:
            value = source.predict(model,row,manifest,tokenizer)
            if pilot and source.predict(model,row,manifest,tokenizer) != value:
                raise ValueError('repeated source-only native output differs')
            if row['id'] in prior and value != prior[row['id']]:
                raise ValueError('fresh full prediction differs from retained pilot first')
            outputs.append(value)
            write_json(root/'outputs.json',outputs)
            if pilot or len(outputs)%16 == 0:
                print(json.dumps({'source_only_generated':len(outputs),'population':len(rows)}),flush=True)
        if native.vendor.module_hash(model) != before or any(v.grad is not None for v in model.parameters()):
            raise ValueError('frozen source-only model changed')
        peak = torch.cuda.max_memory_allocated()
        if peak > 3_000_000_000:
            raise ValueError('source-only inference exceeds3GB allocation budget')
    finally:
        model.close()
    native.vendor.verify(p)
    result = {'status':'new_source_only_inference_lifecycle_passed' if pilot else 'all128_new_source_only_outputs_generated_semantics_pending',
        'cases':len(outputs),'checkpoint_sha256':state['checkpoint_sha256'],'model_sha256':before,
        'model_unchanged':True,'outputs_sha256':file_hash(root/'outputs.json'),'peak_allocated_bytes':peak,
        'source_only':True,'oracle_target_codes':False,'source_only_task_accuracy':None,
        'semantic_cache_untouched':True,'final_test_opened':False,'speed_tested':False}
    write_json(root/'result.json',result)
    print(json.dumps(result),flush=True)


def workflow():
    native.vendor.corpus.quality.dependencies()
    checkpoint.checkpoint_contract()
    if any(root.exists() for root in (WORKFLOW,PREP,PILOT,OUTPUTS)):
        raise ValueError('fresh numeric-only source inference required')
    p = producing(WORKFLOW,'supervise_target_free128_source_inference_without_judging')
    receipts = []
    for stage,root,gpu in (('prepare',PREP,''),('pilot',PILOT,'2'),('generate',OUTPUTS,'2')):
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
            raise RuntimeError('source-only numeric stage failed: '+stage)
    native.vendor.verify(p)
    write_json(WORKFLOW/'result.json',{'status':'all128_numeric_source_inference_completed_semantics_pending','stages':receipts})


if __name__=='__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','pilot','generate','workflow'))
    args = parser.parse_args()
    {'prepare':prepare,'pilot':lambda:generate(True),'generate':generate,'workflow':workflow}[args.stage]()
