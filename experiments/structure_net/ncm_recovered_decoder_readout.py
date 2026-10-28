"""Paired new/original decoder readouts with and without frozen source attention."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import torch

from experiments.structure_net import ncm_codec_roundtrip as prior
from experiments.structure_net import ncm_recovered_decoder_evaluation as evaluation
from experiments.structure_net import ncm_generated_code_decoder as parent
from experiments.structure_net import ncm_recovered_paraphrase_catalog as catalog

study, data = evaluation.study, evaluation.data
ROOTS = {s:data.ROOT/f'20261028_recovered_decoder_readout_{s}_v1' for s in ('pilot','eval','replay')}
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-decoder-readout.md')
TEST = Path('tests/structure_net/test_ncm_recovered_decoder_readout.py')
CURRENT = data.ROOT/'20261027_task_supervision_existing_train_v3'
FROZEN = {'codec':'eb962706f10c693871f9c6df007e9fd3e23004c91c6258f435cdb09abf7fb366',
    'word_memory':'0b025b628816d19b2d9edd1e18396e238ec9d7bd45437fe5a4a89cbbbe588596',
    'word_adapters':'1099d764b8c709b2f79b9f1e6abf79df2c1906655d37d2a3abb007675c6e0a39'}
ARMS = ('parent_codec_only','parent_with_source','new_codec_only','new_with_source')
LIMIT = 3_500_000_000


def install_decoder(model, state):
    if set(state) != set(model.codec.decoder.state_dict()):
        raise ValueError('only a complete exact decoder state may be installed')
    model.codec.decoder.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)


def paired(rows, left, right, split):
    a = {r['id']:r for r in rows if r['arm']==left and r['split']==split}
    b = {r['id']:r for r in rows if r['arm']==right and r['split']==split}
    if not a or set(a)!=set(b):raise ValueError('complete paired comparison required')
    return {'cases':len(a),'exact_fixes':sum(not a[k]['exact_rendered'] and b[k]['exact_rendered'] for k in a),
        'exact_regressions':sum(a[k]['exact_rendered'] and not b[k]['exact_rendered'] for k in a),
        'both_exact':sum(a[k]['exact_rendered'] and b[k]['exact_rendered'] for k in a),
        'neither_exact':sum(not a[k]['exact_rendered'] and not b[k]['exact_rendered'] for k in a)}


def summarize(rows):
    metrics=prior.aggregate(rows);comparisons={}
    for split in metrics:
        if set(metrics[split])!=set(ARMS):raise ValueError('all four configurations required')
        comparisons[split]={name:paired(rows,a,b,split) for name,a,b in (
            ('decoder_alone','parent_codec_only','new_codec_only'),
            ('decoder_with_source','parent_with_source','new_with_source'),
            ('new_source_effect','new_codec_only','new_with_source'),
            ('parent_source_effect','parent_codec_only','parent_with_source'))}
    reserved=metrics.get('reserved_component',{})
    gate=None
    if reserved and reserved['new_with_source']['cases']==64:
        a,b=reserved['parent_with_source'],reserved['new_with_source']
        gate=all(b[k]>=a[k] for k in ('exact_rendered','content_agreement','digit_cases_preserved'))
    return {'metrics':metrics,'paired_comparisons':comparisons,'descriptive_integration_nonregression':gate}


def freeze(root,stage,training,environment,paths):
    files={Path(__file__).resolve(),PLAN.resolve(),TEST.resolve()}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            p=Path(filename).resolve()
            if p.is_file() and p.suffix=='.py' and (p.is_relative_to(Path('experiments').resolve()) or p.is_relative_to(training.native.vendor.VENDOR.resolve())):files.add(p)
    paths += [Path('pixi.lock'),Path('pyproject.toml')]
    proof={'schema':'nal.ncm-recovered-decoder-readout.v1','hypothesis_id':'ncm-recovered-decoder-readout-v1',
        'stage':stage,'environment':environment,'config':{'arms':ARMS,'physical_GPU':2,'FP32':True,'TF32':False,
            'batch':1,'peak_limit_bytes':LIMIT,'feature_cache_bytes':256*1024**2,'generator_bypassed':True,
            'target_aware_reconstruction':True,'model_training':False,'new_judge_calls':0,'final_test_opened':False},
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},'input_hashes':{str(p):data.sha(p) for p in dict.fromkeys(paths)}}
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    data.write(root/'producer.json',proof);return proof


def run(stage):
    root=ROOTS[stage]
    if root.exists():raise ValueError('fresh readout diagnostic required')
    training,environment=prior.runtime()
    study.runtime.lifecycle.reference.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID'
            or torch.cuda.get_device_name()!='NVIDIA GeForce RTX 2060 SUPER' or torch.cuda.mem_get_info()[0]<LIMIT):
        raise ValueError('physicalGPU2 with adequate free memory required')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # The previous target-code producer must have released this GPU successfully.
    catalog.receipt_paths(data.ROOT/'20261028_recovered_paraphrase_codes_full_v1')
    trained=study.require(study.ROOTS['train']);study.require(evaluation.ROOTS['audit']);study.require(evaluation.ROOTS['replay'])
    if data.sha(trained['checkpoint'])!=trained['checkpoint_sha256']:raise ValueError('fixed final trained decoder required')
    if data.read(CURRENT/'result.json')['frozen_hashes']!=FROZEN:raise ValueError('current application readout identity differs')
    ready=prior.read(prior.RUNS['prepare']/'result.json')
    for name,sha in ready['artifacts'].items():
        if data.sha(prior.RUNS['prepare']/name)!=sha:raise ValueError('fixed source/target diagnostic changed')
    all_rows=data.read(prior.RUNS['prepare']/'cases.json')
    if len(all_rows)!=80 or sum(r['split']=='reserved_component' for r in all_rows)!=64:raise ValueError('entire64+16 population required')
    pilot_ids=set(data.read(prior.RUNS['prepare']/'selection.json')['pilot_ids'])
    rows=[r for r in all_rows if r['id'] in pilot_ids] if stage=='pilot' else all_rows
    corpus=study.prepared.corpus.ROOT/'frames.sqlite';db=catalog.context.connect(corpus)
    overlap=[]
    for row in all_rows:
        sha=hashlib.sha256(struct.pack('<128H',*row['native_word_ids'])).hexdigest()
        if row['split']=='reserved_component' and db.execute('SELECT 1 FROM frames WHERE sha=?',(sha,)).fetchone():overlap.append(row['id'])
    db.close()
    if overlap:raise ValueError('reserved diagnostic target appears in decoder fitting corpus')
    from experiments.structure_net import ncm_streaming_source_features as streaming
    index,manifest=[data.read(prior.RUNS['prepare']/n) for n in ('source_index.json','feature_manifest.json')]
    selected={r['id'] for r in rows};index=[r for r in index if r['id'] in selected];manifest={i:manifest[i] for i in selected}
    loader=streaming.FeatureCache(index,manifest,max_bytes=256*1024**2)
    paths=[prior.RUNS['prepare']/n for n in ('producer.json','result.json','actual_tool_receipt.json','cases.json','selection.json','source_index.json','feature_manifest.json')]
    paths += [prior.RUNS['eval']/n for n in ('outputs.json','result.json','actual_tool_receipt.json')]
    paths += [Path(trained['checkpoint']),corpus,CURRENT/'result.json',CURRENT/'actual_tool_receipt.json',parent.PARENT/'model.pt']
    for r in (study.ROOTS['train'],evaluation.ROOTS['audit'],evaluation.ROOTS['replay']):paths += [r/n for n in ('result.json','actual_tool_receipt.json')]
    for row in index:paths += [Path(row['path']),Path(manifest[row['id']]['header_path']),Path(manifest[row['id']]['payload_path'])]
    pilot={}
    if stage=='eval':
        study.require(ROOTS['pilot']);pilot={(r['id'],r['arm']):r for r in data.read(ROOTS['pilot']/'outputs.json')}
        paths += [ROOTS['pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json','outputs.json')]
    proof=freeze(root,stage,training,environment,paths);data.write(root/'cases.json',rows)
    torch.cuda.reset_peak_memory_stats();model=parent.load_parent(training);model.eval().requires_grad_(False)
    before={k:training.native.vendor.module_hash(getattr(model,k)) for k in FROZEN}
    if before!=FROZEN:raise ValueError('exact current codec and word-source path required')
    protected=parent.decoder.protected(model);initial=training.native.vendor.module_hash(model)
    original_decoder={k:v.detach().cpu().clone() for k,v in model.codec.decoder.state_dict().items()}
    saved=torch.load(trained['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    if saved['frozen_state_sha256']!=study.old.frozen_hash(model.codec):raise ValueError('frozen representation differs')
    reference={(r['id'],r['arm']):r for r in data.read(prior.RUNS['eval']/'outputs.json')}
    tokenizer=training.native.tokenizer();outputs=[];codes_by_id={};old_matches=0;old_total=0;hashes={}
    with torch.no_grad():
        for version,state in (('parent',original_decoder),('new',saved['decoder'])):
            install_decoder(model,state)
            expected=FROZEN['codec'] if version=='parent' else trained['final_codec_sha256']
            if training.native.vendor.module_hash(model.codec)!=expected or parent.decoder.protected(model)!=protected:
                raise ValueError('decoder-only swap altered protected model state')
            hashes[version]=training.native.vendor.module_hash(model)
            for row in rows:
                codes=model.codec.encode(torch.tensor([row['native_word_ids']],device='cuda',dtype=torch.long))
                values=codes[0].cpu().tolist()
                if codes.shape!=(1,641) or row['id'] in codes_by_id and codes_by_id[row['id']]!=values:raise ValueError('reference hierarchy changed')
                codes_by_id[row['id']]=values
                features=loader.get(row).unsqueeze(0).cuda();mask=torch.ones(features.shape[:2],device='cuda',dtype=torch.bool)
                for mode in ('codec_only','with_source'):
                    def predict():
                        if model._word_bound is not None:raise ValueError('stale source context')
                        logits=model.codec.decode(codes) if mode=='codec_only' else model.word_logits(features,mask,codes)
                        if logits.shape!=(1,128,30528) or not torch.isfinite(logits).all():raise ValueError('complete finite decoder output required')
                        return logits.argmax(-1)[0].cpu().tolist()
                    raw=predict()
                    if stage=='pilot' and raw!=predict():raise ValueError('repeated readout differs')
                    text=training.preparation.rendering.decode(raw,tokenizer)['text']
                    output={'id':row['id'],'split':row['split'],'arm':version+'_'+mode,'target_sha256':row['target_sha256'],
                        'reference':row['native_reference'],'target_text':row['target_text'],'text':text,'codes':values,'raw_word_ids':raw,
                        **prior.metrics(row['native_word_ids'],raw,row['native_reference'],text)}
                    if (row['id'],output['arm']) in pilot and output!=pilot[row['id'],output['arm']]:raise ValueError('pilot/full output differs')
                    if version=='parent':
                        old=reference[row['id'],output['arm']];old_total+=1
                        old_matches+=int(old['raw_word_ids']==raw and old['codes']==values)
                    outputs.append(output)
                if torch.cuda.max_memory_allocated()>LIMIT:raise ValueError('GPU allocation ceiling exceeded')
            if training.native.vendor.module_hash(model)!=hashes[version]:raise ValueError('inference mutated model')
            print(json.dumps({'version':version,'cases':len(rows),'outputs':len(outputs)}),flush=True)
        install_decoder(model,original_decoder)
        if training.native.vendor.module_hash(model)!=initial:raise ValueError('exact original state restore failed')
    data.write(root/'outputs.json',outputs);data.verify(proof['source_hashes']);data.verify(proof['input_hashes'])
    value={'schema':proof['schema'],'status':'recovered_decoder_readout_'+stage+'_completed','cases':len(rows),'outputs':len(outputs),
        **summarize(outputs),'protected_nondecoder_state_unchanged':True,'current_word_source_weights_verified':FROZEN,
        'reserved_frames_in_decoder_fitting':0,'all_reference_codes_unchanged':True,
        'historical_GPU1_parent_outputs_exact':old_matches,'historical_GPU1_parent_outputs_compared':old_total,
        'model_hashes':hashes,'exact_parent_restore':True,'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),
        'task_accuracy':None,'source_only_summary_accuracy':None,'new_judge_calls':0,'final_test_opened':False,'speed_tested':False,
        'artifacts':{str(p):data.sha(p) for p in root.iterdir() if p.is_file()}}
    data.write(root/'result.json',value);print(json.dumps({k:v for k,v in value.items() if k not in ('artifacts','metrics')}),flush=True)
    model.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('pilot','eval'));args=parser.parse_args()
    root=ROOTS[args.stage];existed=root.exists()
    try:run(args.stage)
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
