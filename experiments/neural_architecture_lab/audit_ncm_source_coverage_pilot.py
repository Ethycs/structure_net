"""Replay initial parity artifacts; keep the untrained coverage variant unqualified."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_source_coverage_pilot as pilot

data=pilot.data
ROOT=data.ROOT/'20261027_source_coverage_pilot_audit_v1'
REPORT=Path('docs/08 - Analysis/2026-10-27_ncm-source-coverage-pilot.md')


def audit():
    if ROOT.exists() or REPORT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('fresh CPU-only pilot artifact replay required')
    pilot.reference.receipt(pilot.ROOT);value=data.completed(pilot.ROOT)
    proof=data.read(pilot.ROOT/'producer.json');data.verify(proof['source_hashes']);data.verify(proof['input_hashes'])
    if (value['status']!='source_coverage_initial_parity_and_gradients_passed' or value['cases']!=2
            or value['model_updates']!=0 or value['new_judge_requests']!=0 or value['task_accuracy'] is not None
            or value['final_test_opened'] or value['full_training_lifecycle_qualified']
            or not value['systems_lifecycle_only_not_quality_evidence']
            or value['peak_allocated_bytes']>pilot.LIMIT):raise ValueError('complete systems-only pilot required')
    tensors=torch.load(pilot.ROOT/'teacher_logits.pt',map_location='cpu',weights_only=True)
    if set(tensors)!={'before','after'} or any(len(v)!=2 for v in tensors.values()):raise ValueError('all paired logits required')
    for a,b in zip(tensors['before'],tensors['after']):
        if (a.shape!=(1,641,8192) or a.dtype!=torch.float32 or b.dtype!=torch.float32
                or not torch.isfinite(a).all() or not torch.equal(a,b)):
            raise ValueError('raw FP32 teacher-logit parity failed')
    cases=data.read(pilot.ROOT/'cases.json');outputs=data.read(pilot.ROOT/'outputs.json')
    if (outputs['before']!=outputs['after'] or [r['id'] for r in outputs['before']]!=[r['id'] for r in cases]):
        raise ValueError('complete source-only code/word pairing differs')
    for row in outputs['before']:
        pilot.reference.integers(row['codes'],641,8192);pilot.reference.integers(row['raw_word_ids'],128,30528)
    info=data.read(pilot.ROOT/'coverage.json');bindings=data.read(pilot.ROOT/'tensor_bindings.json')
    old,expanded=bindings['original'],bindings['expanded']
    expected=[i for i in range(12) if i not in (8,10,11)]
    if (info['source_blocks']!=list(range(12)) or info['added_source_blocks']!=expected
            or info['new_parameters']!=10616832 or value['trainable_parameters']!=105323136+info['new_parameters']
            or bindings['after_backward']!=expanded or any(expanded.get(k)!=v for k,v in old.items())):
        raise ValueError('original or expanded tensor inventory differs')
    extra={f'concepts.adapters.{i}.{name}.weight' for i in expected for name in ('q','out')}
    if set(expanded)-set(old)!=extra:raise ValueError('unexpected added parameters')
    zero=hashlib.sha256(torch.zeros(768,768,dtype=torch.float32).numpy().tobytes()).hexdigest()
    for i in expected:
        source=min((8,10,11),key=lambda j:(abs(i-j),j))
        if (info['query_templates'][str(i)]!=source
                or expanded[f'concepts.adapters.{i}.q.weight']!=old[f'concepts.adapters.{source}.q.weight']
                or expanded[f'concepts.adapters.{i}.out.weight']!={'dtype':'torch.float32','shape':[768,768],'sha256':zero}):
            raise ValueError('copied query or zero output initialization differs')
    gradients=data.read(pilot.ROOT/'gradients.json')
    if [g['id'] for g in gradients]!=[r['id'] for r in cases]:raise ValueError('all gradient controls required')
    for g in gradients:
        if (not math.isfinite(g['loss']) or g['loss']<0 or set(g['added'])!={str(i) for i in expected}
                or any(not math.isfinite(v['output_L2']) or v['output_L2']<=0 or v['query_nonzero']!=0 for v in g['added'].values())):
            raise ValueError('recorded gradient-routing evidence invalid')
    services=data.read(pilot.ROOT/'services.json')
    if services['before']!=services['after']:raise ValueError('concurrent services changed')
    paths=[Path(__file__),pilot.ROOT/'result.json',pilot.ROOT/'actual_tool_receipt.json',
           *[Path(p) for p in value['artifacts']]]
    ROOT.mkdir()
    campaign={**{k:v for k,v in value.items() if k!='artifacts'},
        'status':'source_coverage_pilot_artifacts_replayed','classification':'source_coverage_systems_pilot_passed',
        'raw_forward_parity_independently_replayed':True,'gradient_statistics_replayed':True,
        'independent_GPU_backward_repeated':False,'artifacts':{str(p):data.sha(p) for p in paths}}
    data.write(ROOT/'campaign_results.json',campaign)
    REPORT.write_text('\n'.join(['# Source attention throughout the NCM generator','',
        'Classification: `source_coverage_systems_pilot_passed`. This is an untrained architecture pilot, not summary accuracy.',
        f"Hypothesis: `{pilot.HYPOTHESIS}`. Aggregate: `{ROOT/'campaign_results.json'}`; SHA256 `{data.sha(ROOT/'campaign_results.json')}`.",'',
        'The original generator reads source memory after blocks 8, 10 and 11. Nine added attention modules now give every existing block source access. Each new query projection copies the nearest original module; every new output projection is zero. The earlier 12-versus-24-block experiment retained only three source-attention modules, so it did not test this intervention.',
        'On two fixed TRAIN controls, the initial expanded model exactly preserved all FP32 teacher logits, source-only generated codes and raw word IDs. The baseline codes and words also reproduced the saved epoch2 checkpoint outputs. Every original tensor stayed unchanged. All nine new output projections received finite, nonzero gradients; their query gradients were zero as expected at zero-output initialization. No optimizer step occurred.',
        'Trainable generator/concept-source parameters increased from 105,323,136 to 115,939,968. Peak own CUDA allocation was 1,949,571,072 bytes on physical GPU2 RTX2060SUPER. The frozen Llama body remained unloaded; original FP32 source features were streamed. Codec and word-source parameters stayed frozen. Both ongoing training service identities were preserved.',
        'CPU replay verified every saved logit, output pair, tensor hash and initialization binding. It validated the recorded gradient statistics without repeating GPU autograd. Three CPU contract tests also passed.',
        'A representative mixed-precision microbatch, fresh optimizer, checkpoint/restart and full concurrency lifecycle still must pass before a fixed-budget training comparison. This pilot establishes no semantic improvement, full-run memory guarantee or speedup. Latest application semantic DEV remains 0/128, and final320 remains sealed.','']) )
    print(json.dumps({'status':campaign['status'],'report':str(REPORT),'full_training_lifecycle_qualified':False}))


def checked():
    receipt=data.read(ROOT/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code']!=0 or any(receipt['artifacts'].get(str(p))!=data.sha(p) for p in (ROOT/'campaign_results.json',REPORT)):
        raise ValueError('actual artifact-bound audit receipt required')
    data.verify(receipt['artifacts']);value=data.read(ROOT/'campaign_results.json');data.verify(value['artifacts'])
    if value['task_accuracy'] is not None or value['full_training_lifecycle_qualified'] or value['model_updates']!=0:
        raise ValueError('untrained systems-only pilot required')
    return {'name':'NCM source attention in all existing generator blocks','classification':value['classification'],
        'task_accuracy_measured':False,'full_training_lifecycle_qualified':False,'model_updates':0,
        'description':'Add nine zero-output source-attention modules to the frozen epoch2 initializer, extending source access from three to twelve generator blocks. Two TRAIN controls preserve every initial FP32 logit, generated code and word. Every added output receives a gradient. CPU raw artifact replay passed. No optimizer update or semantic improvement measured; full training lifecycle remains unqualified.'}


def register():
    record=checked()
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    from experiments.neural_architecture_lab.ncm_result_registry import register_records
    register_records(ROOT/'campaign_results.json',REPORT,ROOT,{pilot.HYPOTHESIS:record})
    print(json.dumps({'status':'registered_and_read_back','hypothesis':pilot.HYPOTHESIS}))


def readback():
    expected=checked()
    import numpy as np
    if not hasattr(np,'NaN'):np.NaN=np.nan
    import chromadb
    h=pilot.HYPOTHESIS;receipt=data.read(ROOT/(h+'_registry_readback.json'));record=receipt['record']
    collection=chromadb.PersistentClient(path='data/chroma_db').get_collection('hypotheses')
    exact=collection.get(ids=[h],include=['metadatas']);semantic=collection.query(query_texts=[record['description']],n_results=10)
    if (exact['ids']!=[h] or any(record.get(k)!=v for k,v in expected.items())
            or any(exact['metadatas'][0].get(k)!=v for k,v in record.items()) or h not in semantic['ids'][0]
            or record['campaign_sha256']!=data.sha(ROOT/'campaign_results.json') or record['report_sha256']!=data.sha(REPORT)
            or data.read(Path('data/hypothesis_registry')/(h+'.json'))!=receipt):raise ValueError('independent registry readback differs')
    value={'status':'independent_registry_readback_passed','hypothesis_id':h,
        'campaign_sha256':data.sha(ROOT/'campaign_results.json'),'report_sha256':data.sha(REPORT)}
    data.write(ROOT/'backend_exact_readback.json',value);print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('audit','register','readback'))
    globals()[parser.parse_args().stage]()
