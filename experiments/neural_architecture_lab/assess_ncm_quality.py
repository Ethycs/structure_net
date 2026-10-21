"""Append a measurement-qualification assessment; never rewrite the raw campaign."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.structure_net import ncm_quality as q
from experiments.structure_net import ncm_quality_evidence as e
from experiments.neural_architecture_lab.finalize_ncm_quality import verify
from neural_architecture_lab.ncm_quality_assessment import AssessedQualityEvidence,CLASSIFICATION


def assess(args):
    if args.output.exists():raise ValueError('fresh assessment required')
    campaign=e.BASE/'20261021_registered_v1/campaign_results.json';c=q.read(campaign)
    for path,h in c['artifacts'].items():
        if q.sha256(path)!=h:raise ValueError('completed campaign evidence changed')
    ref=e.BASE/'20261021_reference_diagnostic_v1';_,r=verify(ref)
    packets=q.read(ref/'packets.json')['packets'];rows,_=e.audit(ref,packets,'primary')
    if r['status']!='completed' or len(rows)!=20 or r['summary']['n']!=40:raise ValueError('complete reference diagnostic required')
    if c['primary']['ncm']['passes']!=0 or c['primary']['llama']['passes']!=36 or c['primary']['ncm']['fully_supported_summaries']!=0 or r['summary']['passes']!=11:
        raise ValueError('this descriptive assessment is bound to the measured results')
    inputs=[campaign,*ref.rglob('*.json'),e.BASE/'20261021_template_diagnostic_v1/result.json',e.BASE/'20261021_input_audit_v1/result.json']
    result={'schema':'nal.ncm-quality-assessment.v1','hypothesis_id':q.HYPOTHESIS_ID,
        'classification':CLASSIFICATION,'operational_classification':c['classification'],
        'status':'completed_exploratory','confirmed':False,'underpowered':True,
        'strict_primary_passes':{'ncm':0,'llama':36,'n':200},
        'human_reference_strict_passes':11,'human_reference_outputs':40,
        'judge_qualified_for_accuracy':False,'production_accuracy':None,
        'accuracy_status':'unmeasured; strict rubric is not calibrated human-rated accuracy',
        'assessment_boundary':'Post-outcome interpretation after human-reference diagnostic; preserves all original grades and thresholds. No retuning, no model-output regeneration, no new success claim.',
        'artifacts':{str(p):q.sha256(p) for p in dict.fromkeys(inputs)}}
    sources=[Path(__file__),Path('src/neural_architecture_lab/ncm_quality_assessment.py')]
    dst=args.output.parent/'analysis_source';dst.mkdir(parents=True)
    for p in sources:shutil.copyfile(p,dst/p.name)
    result['analysis_source_hashes']={str(p):q.sha256(p) for p in sources};q.write(args.output,result)
    print(json.dumps({k:result[k] for k in ('classification','operational_classification','judge_qualified_for_accuracy','accuracy_status')}))


def register(args):
    a=q.read(args.assessment)
    if a['classification']!=CLASSIFICATION or a['judge_qualified_for_accuracy'] is not False:raise ValueError('qualified interpretation required')
    for path,h in a['artifacts'].items():
        if q.sha256(path)!=h:raise ValueError('assessment evidence changed')
    campaign=e.BASE/'20261021_registered_v1/campaign_results.json'
    record=AssessedQualityEvidence(str(campaign),q.sha256(campaign),str(args.assessment),q.sha256(args.assessment),str(args.report),q.sha256(args.report)).record()
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name='structure_net_completed_hypotheses',queue_dir='data/experiment_queue',
        sent_dir='data/experiment_sent',rejected_dir='data/experiment_rejected',enable_wandb=False,auto_upload=False,
        enable_chromadb=True,chromadb_path='data/chroma_db'))
    logger.log_hypothesis(record);hid=q.HYPOTHESIS_ID;exact=logger.hypotheses_collection.get(ids=[hid],include=['metadatas'])
    if exact['ids']!=[hid] or any(exact['metadatas'][0][k]!=record[k] for k in ('tested','confirmed','confirmation_status','direct_experiment_count')):raise ValueError('exact registry readback')
    semantic=logger.hypotheses_collection.query(query_texts=[record['description']],n_results=10)
    if hid not in semantic['ids'][0]:raise ValueError('semantic registry readback')
    receipt={'status':'registered_and_read_back','record':record,'exact_readback':exact['metadatas'][0],'semantic_ids':semantic['ids'][0]}
    for p in (args.assessment.parent/'registry_readback.json',Path('data/hypothesis_registry')/(hid+'.json')):
        if p.exists():raise ValueError('fresh receipt required')
        q.write(p,receipt)
        if q.read(p)['record']!=record:raise ValueError('full JSON registry readback')
    print(json.dumps(record))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('assess');a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('register');a.add_argument('--assessment',type=Path,required=True);a.add_argument('--report',type=Path,required=True)
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':main()
