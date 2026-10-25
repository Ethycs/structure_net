"""Independent replay of decoder adaptation, frozen state and raw diagnostics."""
import hashlib
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_generated_code_decoder as study

DEST=study.ROOT/'20261025_generated_code_decoder_train_audit_v1'


def main():
    training,environment=study.runtime.configure()
    root=study.RUNS['train']
    if DEST.exists():raise ValueError('fresh audit root required')
    training.preparation.completed(root)
    result,proof=[training.q.read(root/n) for n in ('result.json','producer.json')]
    training.native.vendor.verify(proof)
    if proof['environment']!=environment:raise ValueError('runtime mismatch')
    rows=training.q.read(training.base.DATA/'cases.json')
    outputs=training.q.read(root/'outputs.json')
    parent={r['id']:r for r in training.q.read(study.PARENT/'outputs.json')}
    if [r['id'] for r in rows]!=[r['id'] for r in outputs] or len(rows)!=732:
        raise ValueError('complete ordered TRAIN evidence required')
    for row,out in zip(rows,outputs):
        target=row['native_word_ids'];words=out['raw_word_ids']
        if (len(words)!=len(target) or out['codes']!=parent[row['id']]['codes']
                or out['source_sha256']!=row['source_sha256']
                or out['active_words']!=sum(v!=0 for v in target)
                or out['active_word_matches']!=sum(a==b for a,b in zip(words,target) if b!=0)
                or out['rendered_exact']!=all(a==b for a,b in zip(words,target) if b!=0)):
            raise ValueError('raw code/frame/source binding differs')
    if study.decoder.summarize(outputs)!=result['final_TRAIN_diagnostic']:
        raise ValueError('diagnostic arithmetic differs')
    trace=[json.loads(line) for line in (root/'trace.jsonl').read_text().splitlines()]
    if [r['step'] for r in trace]!=list(range(1,1025)):raise ValueError('fixed1024 updates incomplete')
    rng=torch.Generator().manual_seed(study.CONFIG['sampler_seed']);permutation=[]
    exposures={r['id']:0 for r in rows}
    for record in trace:
        if (record['lr']!=study.decoder.rate(record['step'],1024) or len(record['examples'])!=4
                or record['word_CE']!=sum(r['word_CE'] for r in record['examples'])/4):
            raise ValueError('schedule or loss differs')
        for metric in record['examples']:
            if not permutation:permutation=torch.randperm(len(rows),generator=rng).tolist()
            identity=rows[permutation.pop()]['id']
            if metric['id']!=identity:raise ValueError('sample order differs')
            exposures[identity]+=1
    if study.file_hash(root/'model.pt')!=result['checkpoint_sha256']:raise ValueError('checkpoint changed')
    saved=torch.load(root/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    original=torch.load(study.PARENT/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    if set(saved['model'])!=set(original['model']):raise ValueError('architecture changed')
    protected=hashlib.sha256();changed=[]
    for name,value in sorted(saved['model'].items()):
        if name.startswith('concepts.codec.decoder.'):
            if not torch.equal(value,original['model'][name]):changed.append(name)
        else:
            if not torch.equal(value,original['model'][name]):raise ValueError('frozen tensor changed:'+name)
            protected.update(name.encode());protected.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if (not changed or protected.hexdigest()!=result['protected_sha256'] or saved['step']!=1024
            or saved['producer_fingerprint']!=proof['fingerprint'] or saved['sampler_permutation_remaining']!=permutation
            or not torch.equal(saved['sampler_rng'],rng.get_state())
            or any(float(v['step'])!=1024 for v in saved['optimizer']['state'].values())
            or result['peak_allocated_bytes']>study.CONFIG['peak_limit_bytes']
            or result['feature_cache']['peak_resident_bytes']>study.CONFIG['host_feature_cache_bytes']
            or not result['strict_reload_passed']):raise ValueError('state/memory/update integrity failure')
    paths=[root/n for n in ('result.json','producer.json','trace.jsonl','outputs.json','model.pt','actual_tool_receipt.json')]
    value={'status':'generated_code_decoder_training_independently_replayed','updates':1024,
        'cases':732,'presentations':sum(exposures.values()),'exposures_min':min(exposures.values()),
        'exposures_max':max(exposures.values()),'changed_decoder_tensors':changed,
        'frozen_tensors_identical':True,'final_TRAIN_diagnostic':result['final_TRAIN_diagnostic'],
        'first16_mean_CE':sum(r['word_CE'] for r in trace[:16])/16,
        'last16_mean_CE':sum(r['word_CE'] for r in trace[-16:])/16,
        'artifacts':{str(p):study.file_hash(p) for p in paths},'auditor_sha256':study.file_hash(Path(__file__)),
        'task_accuracy':None,'final_test_opened':False,'speed_tested':False}
    DEST.mkdir();study.write_json(DEST/'campaign_results.json',value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('artifacts','changed_decoder_tensors')}))


if __name__=='__main__':main()
