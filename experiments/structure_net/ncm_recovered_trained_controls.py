"""Inspect fixed TRAIN controls after the completed recovered-data training."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_recovered_development_outputs as generated

pilot=generated.pilot
verified=generated.verified
study,data=generated.study,generated.data
BASELINE=pilot.ROOT
ROOT=data.ROOT/'20261028_recovered_trained_control_readouts_v1'
CPU=data.ROOT/'20261028_recovered_trained_controls_CPU_v1.json'
PLAN=Path('docs/07 - Status Reports/2026-10-28_ncm-trained-control-readout-diagnostic.md')
TEST=Path('tests/structure_net/test_ncm_recovered_trained_controls.py')
LEVELS=(1,8,16,24,32,40,52,64,76,92,108,128)


def compare_codes(current,baseline,targets):
    """Post-prediction TRAIN diagnostic; exact codes are not semantic accuracy."""
    if (not current or [r['id'] for r in current]!=[r['id'] for r in baseline]
            or len({r['id'] for r in current})!=len(current)):
        raise ValueError('same complete ordered TRAIN controls required')
    rows=[]
    for new,old in zip(current,baseline,strict=True):
        gold=targets[new['id']]
        if any(len(codes)!=sum(LEVELS) or any(type(c) is not int or not 0<=c<8192 for c in codes)
               for codes in (new['codes'],old['codes'],gold)):
            raise ValueError('full native target and predicted code hierarchies required')
        levels=[];offset=0
        for width in LEVELS:
            indices=range(offset,offset+width)
            levels.append({'width':width,
                'initializer_exact_codes':sum(old['codes'][i]==gold[i] for i in indices),
                'trained_exact_codes':sum(new['codes'][i]==gold[i] for i in indices)})
            offset+=width
        rows.append({'id':new['id'],'levels':levels,
            'changed_predicted_codes':sum(a!=b for a,b in zip(new['codes'],old['codes'],strict=True))})
    return {'cases':len(rows),'rows':rows,'TRAIN_only':True,'summary_semantic_accuracy':None,
            'target_codes_used_only_after_prediction':True}


def check_loaded(training,model,trained):
    if (training.native.vendor.module_hash(model)!=trained['final_model_sha256']
            or training.prior.protected(model)!=trained['frozen_hashes']):
        raise ValueError('exact audited final predictor and unchanged protected tensors required')


def run():
    if ROOT.exists():raise ValueError('fresh TRAIN-only diagnostic required')
    checked=data.read(CPU);data.verify(checked['artifacts'])
    if checked['actual_tool_exit_code']!=0 or checked['tests_passed']!=3:
        raise ValueError('trained checkpoint and code comparison contracts required')
    verified.install()
    proofs=verified.ProcessProofs();data.sha=proofs.sha
    trained,audited=study.require(proofs,'train'),study.require(proofs,'audit')
    generated.check_training(trained,audited)
    baseline=proofs.completed(BASELINE)
    if (baseline['status']!='paired_source_predicted_readouts_qualified'
            or not baseline['all_code_pairs_identical'] or not baseline['exact_parent_restore']):
        raise ValueError('completed original same-control paired readout required')
    old_outputs=proofs.read(BASELINE/'outputs.json')
    pilot.paired.paired_identity(old_outputs['parent_with_source'],old_outputs['new_codec_only'])
    checkpoint=study.ROOTS['train']/'model.pt'
    if proofs.bind(checkpoint)!=trained['checkpoint_sha256']:
        raise ValueError('fixed final predictor checkpoint differs')
    old_load,old_require,old_dataset=pilot.task.load_initial,pilot.task.require_parent,pilot.task.dataset
    targets={}
    def load(training):
        model=old_load(training)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(saved['model'],strict=True);del saved
        check_loaded(training,model,trained)
        return model
    def require_parent():
        return [*old_require(),*map(Path,proofs.inputs),CPU,BASELINE/'outputs.json']
    def dataset(arm):
        result=old_dataset(arm)
        targets.update({r['id']:r['oracle_codes'] for r in result[0]})
        return result
    pilot.task.load_initial,pilot.task.require_parent,pilot.task.dataset=load,require_parent,dataset
    pilot.ROOT,pilot.PLAN,pilot.TEST=ROOT,PLAN,TEST
    original_write=data.write
    def write(path,value):
        if Path(path)==ROOT/'producer.json':
            value['config'].update(final_predictor_checkpoint_sha256=trained['checkpoint_sha256'],
                final_predictor_model_sha256=trained['final_model_sha256'],training_updates_completed=6416,
                TRAIN_controls_only=True,gold_codes_for_post_prediction_diagnostics_only=True,
                inherited_application_judgments_changed=False)
            value['fingerprint']=pilot.task.base.fingerprint({k:v for k,v in value.items() if k!='fingerprint'})
        elif Path(path)==ROOT/'result.json':
            outputs=data.read(ROOT/'outputs.json')
            comparison=compare_codes(outputs['parent_with_source'],old_outputs['parent_with_source'],targets)
            original_write(ROOT/'code_comparison.json',comparison)
            value.update(status='trained_paired_readout_TRAIN_controls_completed',
                new_trained_predictor_evaluated=True,training_updates_completed=6416,
                predictor_checkpoint_sha256=trained['checkpoint_sha256'],TRAIN_only=True,
                heldout_cases=0,target_codes_used_only_after_prediction=True)
            value['artifacts'][str(ROOT/'code_comparison.json')]=data.sha(ROOT/'code_comparison.json')
        return original_write(path,value)
    data.write=write
    pilot.run()


if __name__=='__main__':
    existed=ROOT.exists()
    try:run()
    except Exception as error:
        if ROOT.exists() and not existed and not (ROOT/'failure.json').exists():
            data.write(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error),
                'task_accuracy':None,'automatic_retry':False})
        raise
