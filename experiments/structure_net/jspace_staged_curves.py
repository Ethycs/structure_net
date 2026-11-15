"""CPU-only, append-only plots of verified staged recovery transitions.

Every curve distinguishes evaluated candidates from the eligible checkpoint
actually retained. Reused development data are not independent confirmation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil

from . import jspace_3b_staged_campaign as producer

SCHEMA='jspace.staged-development-curves.v1'
NATIVE_CORRECT=48
MINIMUM_CORRECT=45
COUNT=64


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):value.update(chunk)
    return value.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def transition(root,cut):
    """Authenticate a completed selection and fresh replay before reading curves."""
    directory=producer.stage_dir(root,cut)
    for phase in ('train','verify'):
        manifest=directory/phase/'manifest.json'
        if not manifest.is_file():raise ValueError(f'cut{cut} lacks completed fresh verification/training: {manifest}')
    plan,stage=producer.verify_stage(root,cut)
    # checked(verify) recursively authenticates every training artifact and the
    # protected initializer/pilot, then recomputes eligible checkpoint selection.
    fresh=producer.checked(directory,'verify',stage)
    trained=producer.read(directory/'train/result.json')
    if (not fresh['passed'] or trained['completed_epochs']!=12 or trained['updates']!=384
            or trained['initial']['count']!=COUNT or stage['native_correct']!=NATIVE_CORRECT
            or trained['quality']['minimum_correct']!=MINIMUM_CORRECT):
        raise ValueError('Unexpected completed-stage protocol or accuracy reference')
    hashes={str(Path(root)/'plan.json'):sha(Path(root)/'plan.json'),str(directory/'plan.json'):sha(directory/'plan.json')}
    for phase in ('pilot','train','verify'):
        path=directory/phase;manifest=producer.read(path/'manifest.json')
        hashes[str(path/'manifest.json')]=sha(path/'manifest.json')
        for name,digest in manifest['artifacts'].items():hashes[str(path/name)]=digest
    curve=[]
    for index,row in enumerate(trained['history']):
        filename=directory/'train'/f"epoch_{row['epoch']:02d}_development.json"
        report=producer.read(filename)
        if report['metrics']!=row['metrics'] or sha(filename)!=row['development_sha256']:
            raise ValueError('Epoch metrics/reference changed during analysis')
        producer.validate_quality(report)
        selected=producer.training.best_eligible(trained['history'][:index+1],trained['initial'])
        curve.append(dict(epoch=row['epoch'],phase='raw_initializer' if row['epoch']==-1 else
            'feature_fit' if row['epoch']==0 else 'joint_training',
            candidate_correct=row['metrics']['correct'],candidate_accuracy=row['metrics']['correct']/COUNT,
            candidate_metrics=row['metrics'],eligibility=row['eligibility'],
            retained_epoch=selected['epoch'],retained_correct=selected['metrics']['correct'],
            retained_accuracy=selected['metrics']['correct']/COUNT,
            development_path=str(filename),development_sha256=row['development_sha256']))
    final=curve[-1]
    reference=producer.read(directory/'train'/f"epoch_{trained['selected_epoch']:02d}_development.json")
    if (final['retained_epoch']!=trained['selected_epoch'] or final['retained_correct']!=trained['selected']['correct']
            or fresh['metrics']!=trained['selected'] or fresh['selected_checkpoint_sha256']!=trained['selected_checkpoint_sha256']
            or [r['identity'] for r in fresh['records']]!=[r['identity'] for r in reference['records']]
            or [r['prediction'] for r in fresh['records']]!=[r['prediction'] for r in reference['records']]):
        raise ValueError('Fresh replay or final retained checkpoint disagrees with curve')
    return dict(cut=cut,compressed_blocks=36-cut,parent_correct=stage['parent_correct'],
        stage_fingerprint=stage['fingerprint'],root_fingerprint=plan['fingerprint'],native_correct=NATIVE_CORRECT,
        minimum_correct=MINIMUM_CORRECT,count=COUNT,curve=curve,selected_epoch=trained['selected_epoch'],
        selected_correct=trained['selected']['correct'],selected_checkpoint_sha256=trained['selected_checkpoint_sha256'],
        quality_pass=trained['quality_pass'],fresh_verification=fresh['comparison'],storage=fresh['storage']),hashes,plan


def plot_transition(value,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    curve=value['curve'];epochs=[r['epoch'] for r in curve]
    candidates=[r['candidate_correct'] for r in curve];retained=[r['retained_correct'] for r in curve]
    figure,axis=plt.subplots(figsize=(11,6),layout='constrained')
    axis.axvspan(-1.4,.5,color='#edf2f5',alpha=.8)
    axis.plot(epochs,candidates,'o--',color='#637792',linewidth=1.5,markersize=5,label='Evaluated candidate')
    axis.step(epochs,retained,where='post',color='#087f5b',linewidth=2.5,label='Retained eligible checkpoint')
    axis.scatter(epochs,retained,color='#087f5b',s=25,zorder=4)
    rejected=[r for r in curve if not r['eligibility']['eligible']]
    if rejected:
        axis.scatter([r['epoch'] for r in rejected],[r['candidate_correct'] for r in rejected],
            color='#bd4147',marker='x',s=65,linewidths=2,zorder=5,label='Ineligible candidate')
    axis.axhline(NATIVE_CORRECT,color='#353535',linestyle=':',linewidth=1.6,label='Original native: 48/64')
    axis.axhline(MINIMUM_CORRECT,color='#a86b00',linestyle='-.',linewidth=1.4,label='Fixed minimum: 45/64')
    axis.set_xticks(epochs,['Raw\n−1','Feature fit\n0']+[str(i) for i in range(1,13)])
    axis.set_xlim(-1.4,12.4)
    axis.set_ylim(max(0,min(min(candidates),MINIMUM_CORRECT)-2),min(COUNT,max(max(candidates),NATIVE_CORRECT)+3))
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_ylabel('Correct answers out of 64 reused development cases')
    axis.set_xlabel('Raw initializer → feature alignment → joint training epoch')
    axis.grid(axis='y',alpha=.2)
    axis.set_title(f"Cut {value['cut']}: {value['compressed_blocks']-1} → {value['compressed_blocks']} compressed blocks\n"
        f"Selected epoch {value['selected_epoch']}: {value['selected_correct']}/64; fresh verification passed",loc='left',pad=14)
    axis.legend(loc='upper center',ncol=3,fontsize=8.5,frameon=False)
    figure.suptitle('Candidate accuracy and the checkpoint actually retained',fontsize=14,ha='left',x=.02)
    figure.supxlabel('Reused 64 development cases · CE/original-KL eligibility is relative to raw · No held-out or speed measurement',fontsize=8)
    for extension in ('png','svg'):
        path=Path(output)/f"cut_{value['cut']:02d}_development_curve.{extension}"
        if path.exists():raise FileExistsError(path)
        figure.savefig(path,dpi=180)
    plt.close(figure)


def generate(root,cuts,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA for CPU plotting')
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if not cuts or len(set(cuts))!=len(cuts) or any(type(c) is not int or not 0<=c<=26 for c in cuts):
        raise ValueError('Distinct staged cuts in[0,26] required')
    # All requested inputs must qualify before any output directory is created.
    transitions=[];inputs={};fingerprint=None;producer_sources=None
    for cut in sorted(cuts,reverse=True):
        value,hashes,plan=transition(root,cut)
        if fingerprint is not None and fingerprint!=plan['fingerprint']:raise ValueError('Campaign changed during analysis')
        fingerprint=plan['fingerprint'];producer_sources=plan['source_sha256'];transitions.append(value);inputs.update(hashes)
    source=Path(__file__).resolve();source_sha=sha(source)
    summary=dict(schema=SCHEMA,campaign_root=str(root),campaign_fingerprint=fingerprint,transitions=transitions,
        confirmed=False,count=COUNT,reused_development=True,heldout_measured=False,speed_measured=False,
        input_sha256=inputs,producer_source_sha256=producer_sources,plot_source_sha256=source_sha,
        validation='Frozen producer verified source archives, fixed-input identities, complete phase artifacts, eligible selection and fresh replay.',
        created_at=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(),matplotlib=importlib.metadata.version('matplotlib')))
    output.mkdir(parents=True);archive=output/'producer';archive.mkdir();shutil.copy2(source,archive/source.name)
    if sha(archive/source.name)!=source_sha:raise ValueError('Plotting source changed while archiving')
    write(output/'summary.json',summary)
    for value in transitions:plot_transition(value,output)
    artifacts={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write(output/'manifest.json',dict(schema=SCHEMA,status='completed',campaign_fingerprint=fingerprint,
        cuts=[v['cut'] for v in transitions],confirmed=False,artifacts=artifacts))
    return dict(status='completed',output=str(output),cuts=[v['cut'] for v in transitions],
        summary_sha256=sha(output/'summary.json'),manifest_sha256=sha(output/'manifest.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--cuts',required=True,nargs='+',type=int)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    print(json.dumps(generate(args.root,args.cuts,args.output),allow_nan=False))
