"""Audit corrected NCM serving against a fresh unchanged Llama control."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.structure_net.ncm_llama import FIXTURES,digest,decode_ids,sha256,tokenizers,write_json
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.run_ncm_host_io import BASE,HYPOTHESIS_ID
from experiments.structure_net.run_ncm_replicas import REFERENCE,summarize
from experiments.neural_architecture_lab.finalize_ncm_optimizations import verify
from experiments.neural_architecture_lab.finalize_ncm_llama import audit


def read(p):return json.loads(Path(p).read_text())


def finalize(args):
    if args.output.exists():raise ValueError('fresh host-I/O aggregate required')
    pilot=BASE/'20261021_pilot_v1';_,pr=verify(pilot)
    if pr['status']!='passed' or pr['parity_checks']!=35:raise ValueError('full host/batch pilot required')
    if sha256(pilot/'profiles.json')!=pr['profiles_sha256'] or sha256(pilot/'checks.json')!=pr['checks_sha256']:
        raise ValueError('paired profile or batch reset evidence changed')
    if not all(x['parity'] for x in read(pilot/'profiles.json')['rows']+read(pilot/'checks.json')['rows']):
        raise ValueError('pilot output parity failed')
    reference={(r['id'],r['repeat']):r for r in read(REFERENCE)['rows']}
    examples={e['id']:e for e in read(FIXTURES)['examples'] if e['split']=='heldout'}
    old_teacher=Path('data/experiments/ncm_llama/20261021_teacher_heldout_v1/rows.json')
    old={(r['id'],r['repeat']):r for r in read(old_teacher)['rows']}
    teacher_root=BASE/'20261021_teacher_v1';teacher,_,tm,tp=audit(teacher_root,'teacher')
    physical=read(teacher_root/'physical_model_readback.json')
    if physical['status']!='verified' or len(physical['files'])<7:raise ValueError('full original Llama physical readback required')
    if sha256(physical['reference'])!=physical['reference_sha256'] or physical['files']!=read(physical['reference'])['teacher_files']:
        raise ValueError('original Llama file identities differ')
    for x in teacher:
        r=old[x['id'],x['repeat']]
        if x['text']!=r['text'] or x['output_ids']!=r['output_ids'] or x['source_sha256']!=r['source_sha256']:
            raise ValueError('fresh original Llama output differs; source review required')
    teacher_seconds=sum(r['wall_seconds'] for r in teacher);teacher_tokens=sum(r['common_output_tokens'] for r in teacher)
    teacher_summary={'requests':len(teacher),'wall_seconds':teacher_seconds,'common_output_tokens':teacher_tokens,
        'aggregate_tokens_per_second':teacher_tokens/teacher_seconds,'requests_per_second':len(teacher)/teacher_seconds,
        'repetition_tokens_per_second':[sum(x['common_output_tokens'] for x in teacher if x['repeat']==n)/sum(x['wall_seconds'] for x in teacher if x['repeat']==n) for n in range(3)],
        'model_bytes':tm['unique_runtime_weight_buffer_bytes'],'peak_allocated_bytes':tm['peak_allocated_bytes'],
        'peak_reserved_bytes':tm['peak_reserved_bytes'],'exact_parent_output_cells':300}
    gpt,llama=tokenizers();gpt=CachedVocabulary(gpt);conditions=[];paths=[FIXTURES,REFERENCE,old_teacher]
    execution=BASE/'20261021_campaign_v1/execution.json';jobs=read(execution)
    if jobs['status']!='completed' or [r['label'] for r in jobs['runs']]!=['replicas1','batch1','replicas8','batch9','replicas9','teacher']:
        raise ValueError('fixed isolated condition order required')
    for job in jobs['runs']:
        if job['returncode'] or sha256(Path(job['root'])/'result.json')!=job['result_sha256']:raise ValueError('coordinator child changed')
    for mode,size in (('replicas',1),('batch',1),('replicas',8),('batch',9),('replicas',9)):
        root=BASE/f'20261021_{mode}{size}_v1';p,r=verify(root)
        if r['status']!='completed' or p['config']['mode']!=mode or p['config']['size']!=size:raise ValueError('completed host condition differs')
        raw=read(root/'rows.json');rows=raw['rows'];batches=raw['batches']
        if sha256(root/'rows.json')!=r['rows_sha256'] or {(x['id'],x['repeat']) for x in rows}!=set(reference):raise ValueError('host cohort differs')
        for x in rows:
            old=reference[x['id'],x['repeat']]
            if not x['parity'] or x['source_sha256']!=examples[x['id']]['source_sha256'] or x['native_output_ids']!=old['native_output_ids'] or x['text']!=old['text']:
                raise ValueError('host-I/O native output parity failed')
            if x['common_output_tokens']!=len(llama.encode(x['text'],add_special_tokens=False)) or decode_ids(x['native_output_ids'],gpt)!=(x['text'],x['termination']):raise ValueError('host-I/O decoded counts differ')
        summary=summarize(rows,batches,size)
        if summary!=r['summary']:raise ValueError('host aggregate differs from raw timings')
        expected_bytes=713294288*(size if mode=='replicas' else 1)
        if r['model_bytes']!=expected_bytes:raise ValueError('literal/shared model state differs')
        summary.update(mode=mode,size=size,model_bytes=r['model_bytes'],peak_allocated_bytes=r['peak_allocated_bytes'],peak_reserved_bytes=r['peak_reserved_bytes'],
            graph_setup_seconds=r['graph_setup_seconds'],timed_dummy_outputs=r['timed_dummy_outputs'],
            within_original_teacher_model_budget=r['model_bytes']<=6425499904,
            within_original_teacher_allocated_budget=r['peak_allocated_bytes']<=6667318272,
            within_original_teacher_reserved_budget=r['peak_reserved_bytes']<=6794772480,
            token_rate_ratio_vs_fresh_teacher=summary['aggregate_tokens_per_second']/teacher_summary['aggregate_tokens_per_second'],
            request_rate_ratio_vs_fresh_teacher=summary['requests_per_second']/teacher_summary['requests_per_second'])
        conditions.append(summary);paths+=list(root.glob('*.json'))
    eligible=[x for x in conditions if x['within_original_teacher_reserved_budget']]
    passed=any(min(x['token_rate_ratio_vs_fresh_teacher'],x['request_rate_ratio_vs_fresh_teacher'])>=2 for x in eligible)
    legacy_batch=Path('data/experiments/ncm_batching/20261021_batch1_v1')
    partial=read(legacy_batch/'rows.json')
    paths+=list(teacher_root.glob('*.json'))+list(pilot.glob('*.json'))+[execution,execution.parent/'producer.json',execution.parent/'producer.py',
        Path('data/experiments/ncm_batching/20261021_campaign_v1/interruption.json'),
        Path('data/experiments/ncm_replicas/20261021_audit_interruption_v1.json'),
        legacy_batch/'producer.json',legacy_batch/'rows.json',legacy_batch/'failure.json',
        Path('data/experiments/ncm_batching/20261021_pilot_v1/result.json')]
    result={'schema':'nal.ncm-host-io-campaign.v1','hypothesis_id':HYPOTHESIS_ID,'status':'completed_exploratory',
        'classification':'lossless_host_io_throughput_target_met' if passed else 'lossless_host_io_throughput_target_unmet',
        'confirmed':False,'underpowered':True,'direct_timing_cells':1800,'ncm_parity_cells':1500,'fresh_teacher_parent_parity_cells':300,
        'failed_timing_cells':0,'retried_timing_cells':0,'failed_systems_pilots':0,'teacher':teacher_summary,
        'conditions':conditions,'source_fidelity_inherited':{'ncm':100,'teacher':100,'total_cases':100},
        'profile':read(pilot/'profiles.json'),'pilot':pr,'legacy_batching_intentionally_interrupted':True,
        'preserved_legacy_partial_campaign_timed_requests':len(partial['rows']),
        'scientific_boundary':'Used synthetic512/48-token sources, one frozen NCM and technical repeats. Shared batching and literal copies differ. Original Llama batching/optimal equal-memory serving, production accuracy, full application and mobile performance untested.',
        'artifacts':{str(x):sha256(x) for x in dict.fromkeys(paths)}}
    dst=args.output.parent/'analysis_source';dst.mkdir(parents=True)
    for path in (Path(__file__),Path('experiments/structure_net/ncm_vocabulary.py')):shutil.copyfile(path,dst/path.name)
    write_json(args.output,result)
    print(json.dumps({'classification':result['classification'],'teacher':teacher_summary,'conditions':conditions}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args();finalize(args)


if __name__=='__main__':main()
