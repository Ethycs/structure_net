"""Direct target-aware codec diagnostics; never source-only task accuracy."""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('data/experiments/ncm_paper_regime')
RUNS = {s: ROOT / f'20261025_codec_roundtrip_{s}_v1' for s in ('prepare', 'pilot', 'eval')}
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-codec-roundtrip-plan.md')
TEST = Path('tests/structure_net/test_ncm_codec_roundtrip.py')
LABELS = ROOT / '20261024_sol_teacher64_candidates_v2/labels.json'
COMPONENTS = ROOT / '20261024_sol_teacher64_prepare_v2/cases.json'
PARENT = ROOT / '20261025_sequential_full732_pixi_train_v1'
LATEST = ROOT / '20261025_generated_code_decoder_train_v1'
CHECKPOINTS = {
    'parent': (PARENT / 'model.pt', '37d8d70f14d81a63ecd63ca6b9a3f5bf1ac9301d866af1135e24404224e795ad'),
    'adapted': (LATEST / 'model.pt', '5e808e5b204067f839fe627d99f45158d24befe5114d0712e33b00c6a2e0993b'),
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def disjoint(train, held):
    proof = {}
    for key in ('id', 'source_id', 'group_id', 'source_sha256'):
        shared = {r[key] for r in train} & {r[key] for r in held}
        if shared:
            raise ValueError('TRAIN/component overlap: ' + key)
        proof[key] = {'TRAIN_unique': len({r[key] for r in train}),
                      'component_unique': len({r[key] for r in held}), 'overlap': 0}
    return proof


def metrics(gold, raw, reference, text):
    if len(gold) != 128 or len(raw) != 128:
        raise ValueError('complete native128 frames required')
    active = [i for i, v in enumerate(gold) if v != 0]
    content = [i for i in active if gold[i] != 101]
    if not content:
        raise ValueError('nonempty lexical content required')
    digits = re.findall(r'\d+', reference)
    return {'active_matches': sum(raw[i] == gold[i] for i in active), 'active_tokens': len(active),
            'content_matches': sum(raw[i] == gold[i] for i in content), 'content_tokens': len(content),
            'exact_active_sequence': all(raw[i] == gold[i] for i in active),
            'exact_rendered': text == reference, 'has_digits': bool(digits),
            'digit_sequence_preserved': re.findall(r'\d+', text) == digits,
            'empty': not text}


def aggregate(rows):
    result = {}
    for split in sorted({r['split'] for r in rows}):
        result[split] = {}
        for arm in sorted({r['arm'] for r in rows}):
            values = [r for r in rows if r['split'] == split and r['arm'] == arm]
            if not values:
                raise ValueError('all arms must cover every split')
            sums = {k: sum(r[k] for r in values) for k in (
                'active_matches', 'active_tokens', 'content_matches', 'content_tokens',
                'exact_active_sequence', 'exact_rendered', 'has_digits', 'empty')}
            sums.update(cases=len(values), content_agreement=sums['content_matches']/sums['content_tokens'],
                        exact_rendered_fraction=sums['exact_rendered']/len(values),
                        digit_cases_preserved=sum(r['has_digits'] and r['digit_sequence_preserved'] for r in values))
            sums['reconstruction_gate'] = (sums['exact_rendered'] >= 61 and sums['content_agreement'] >= .99
                                            if split == 'reserved_component' and len(values) == 64 else None)
            result[split][arm] = sums
    return result


def runtime():
    from experiments.structure_net import ncm_pixi
    return ncm_pixi.configure()


def freeze(root, stage, inputs, environment):
    if root.exists():
        raise ValueError('fresh append-only root required')
    root.mkdir()
    files = [Path(__file__), PLAN, TEST, Path('pixi.lock'), Path('pyproject.toml')]
    (root / 'producer_source').mkdir()
    for path in files[:3]:
        (root / 'producer_source' / path.name).write_bytes(path.read_bytes())
    proof = {'hypothesis_id': 'ncm-summary-codec-roundtrip-v1', 'stage': stage,
             'config': {'FP32': True, 'TF32': False, 'generator_bypassed': True,
                        'target_aware_diagnostic': True, 'model_training': False,
                        'source_only_task_accuracy': None, 'final_test_opened': False, 'speed_tested': False},
             'source_hashes': {str(p): sha(p) for p in files},
             'input_hashes': {str(p): sha(p) for p in inputs}, 'environment': environment}
    write(root / 'producer.json', proof)
    return proof


def verify(proof):
    for group in ('source_hashes', 'input_hashes'):
        for path, expected in proof[group].items():
            if sha(path) != expected:
                raise ValueError('artifact changed: ' + path)


def prepare():
    training, env = runtime()
    import torch
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU preparation only')
    from experiments.structure_net import ncm_lm1b_native_source_training as previous
    from experiments.structure_net.ncm_additional_training_sources import storage_encoding
    all_train = read(training.base.DATA / 'cases.json')
    selected = read(training.base.DATA / 'selection.json')['diagnostic_ids']
    train_by_id = {r['id']: r for r in all_train}
    held, components = read(LABELS), read(COMPONENTS)
    if len(held) != 64 or len(set(selected)) != 16 or {r['id'] for r in held} != {r['id'] for r in components}:
        raise ValueError('complete fixed16 and reserved64 populations required')
    source_by_id = {r['id']: r for r in components}
    for row in held:
        for key in ('source', 'source_id', 'group_id', 'source_component', 'source_sha256'):
            if row[key] != source_by_id[row['id']][key]:
                raise ValueError('original component source mismatch')
        if not row['teacher_valid'] or not row['capacity_fit']:
            raise ValueError('unaltered complete fitting64 teacher set required')
    old_train = read(previous.DATA / 'cases.json')
    proofs = {'engineering': disjoint(all_train, held), 'earlier_native': disjoint(old_train, held)}
    tokenizer, encoding = training.native.tokenizer(), storage_encoding()
    rows = []
    for split, values in [('TRAIN_control', [train_by_id[i] for i in selected]), ('reserved_component', held)]:
        for row in values:
            lexical = tokenizer.encode(row['target_text'], add_special_tokens=False)
            if not 0 < len(lexical) <= 126 or set(lexical) & set(tokenizer.all_special_ids):
                raise ValueError('unsupported complete frame or unknown/special token; no truncation permitted')
            frame = [101, *lexical, 101] + [0] * (126-len(lexical))
            if frame != row['native_word_ids']:
                raise ValueError('existing native target frame changed')
            keep = {k: row[k] for k in ('id', 'source', 'source_id', 'group_id', 'source_component',
                                      'source_sha256', 'target_text', 'native_word_ids')}
            keep.update(split=split, target_sha256=hashlib.sha256(row['target_text'].encode()).hexdigest(),
                        native_reference=training.preparation.rendering.decode(frame, tokenizer)['text'],
                        native_lexical_tokens=len(lexical), stored_tokens=len(encoding.encode(row['target_text'])))
            rows.append(keep)
    index_paths = [training.base.DATA/'source_index.json', ROOT/'20261024_additional10000_source_metadata_v1/index.json']
    manifest_paths = [training.base.DATA/'feature_manifest.json', ROOT/'20261024_additional10000_source_features_v1/manifest.json']
    ids = {r['id'] for r in rows}
    index = [r for p in index_paths for r in read(p) if r['id'] in ids]
    manifest = {k:v for p in manifest_paths for k,v in read(p).items() if k in ids}
    if len(index) != 80 or len(manifest) != 80:
        raise ValueError('all80 source feature bindings required')
    inputs = [LABELS, COMPONENTS, previous.DATA/'cases.json', training.base.DATA/'cases.json',
              training.base.DATA/'selection.json', *index_paths, *manifest_paths]
    proof = freeze(RUNS['prepare'], 'CPU_prepare', inputs, env)
    for name, value in [('cases.json',rows), ('source_index.json',index), ('feature_manifest.json',manifest),
                        ('split_proof.json',proofs)]:
        write(RUNS['prepare']/name,value)
    pilot = [held[0]['id'], max(held, key=lambda r: (r['native_lexical_tokens'],r['id']))['id']]
    if len(set(pilot)) != 2:
        raise ValueError('distinct first/longest pilot required')
    write(RUNS['prepare']/'selection.json', {'pilot_ids':pilot})
    verify(proof)
    write(RUNS['prepare']/'result.json', {'status':'prepared', 'counts':dict(Counter(r['split'] for r in rows)),
          'native_unknowns':0, 'overflow':0, 'target_truncation':False, 'source_only_task_accuracy':None,
          'artifacts':{p.name:sha(p) for p in RUNS['prepare'].glob('*.json')}})
    print(json.dumps({'prepared':len(rows),'reserved':len(held),'pilot_ids':pilot}),flush=True)


def measure(stage):
    training, env = runtime()
    import torch
    from experiments.structure_net import ncm_generated_code_decoder as study
    from experiments.structure_net import ncm_streaming_source_features as streaming
    training.offload.require_rtx2060()
    for path, expected in CHECKPOINTS.values():
        if sha(path) != expected:
            raise ValueError('checkpoint identity changed')
    ready = read(RUNS['prepare']/'result.json')
    for name, expected in ready['artifacts'].items():
        if sha(RUNS['prepare']/name) != expected:
            raise ValueError('prepared data changed')
    all_rows = read(RUNS['prepare']/'cases.json')
    pilot_ids = read(RUNS['prepare']/'selection.json')['pilot_ids']
    rows = [r for r in all_rows if r['id'] in pilot_ids] if stage == 'pilot' else all_rows
    prior = {}
    if stage == 'eval':
        if read(RUNS['pilot']/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
            raise ValueError('completed pilot required')
        prior = {(r['id'],r['arm']):r for r in read(RUNS['pilot']/'outputs.json')}
    index, manifest = [read(RUNS['prepare']/n) for n in ('source_index.json','feature_manifest.json')]
    index = [r for r in index if r['id'] in {x['id'] for x in rows}]
    manifest = {r['id']:manifest[r['id']] for r in rows}
    loader = streaming.FeatureCache(index,manifest,max_bytes=256*1024**2)
    inputs = [*RUNS['prepare'].glob('*.json'), *[p for p,_ in CHECKPOINTS.values()]]
    for row in index:
        entry = manifest[row['id']]
        inputs.extend([Path(row['path']),Path(entry['header_path']),Path(entry['payload_path'])])
    proof = freeze(RUNS[stage],stage,inputs,env)
    tokenizer = training.native.tokenizer()
    model = study.load_parent(training)
    model.eval().requires_grad_(False)
    initial_hash = training.native.vendor.module_hash(model)
    saved_codec = {k:v.detach().cpu().clone() for k,v in model.codec.state_dict().items()}
    released = torch.load(training.native.RELEASE/'files/lm1b/tokenizer.pt',map_location='cpu',weights_only=True,mmap=True)
    released_path = training.native.RELEASE/'files/lm1b/tokenizer.pt'
    if sha(released_path) != training.native.FILES['lm1b/tokenizer.pt']:
        raise ValueError('official release changed')
    proof['input_hashes'][str(released_path)] = sha(released_path)
    proof['source_hashes'].update({str(p):sha(p) for p in training.native.vendor.VENDOR.rglob('*.py')})
    write(RUNS[stage]/'producer.json',proof)
    outputs, codes_by_id = [], {}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for version in ('released','parent','adapted'):
            if version == 'released':
                model.codec.load_state_dict(released['model'],strict=True)
            elif version == 'parent':
                model.codec.load_state_dict(saved_codec,strict=True)
                if training.native.vendor.module_hash(model) != initial_hash:
                    raise ValueError('parent restore differs')
            else:
                saved = torch.load(CHECKPOINTS['adapted'][0],map_location='cpu',weights_only=True,mmap=True)
                model.load_state_dict(saved['model'],strict=True)
                if training.native.vendor.module_hash(model) != read(LATEST/'result.json')['final_model_sha256']:
                    raise ValueError('latest full state differs')
            model.eval().requires_grad_(False)
            before = training.native.vendor.module_hash(model)
            for row in rows:
                gold = torch.tensor([row['native_word_ids']],device='cuda',dtype=torch.long)
                codes = model.codec.encode(gold)
                as_list = codes[0].cpu().tolist()
                if len(as_list) != 641 or (row['id'] in codes_by_id and codes_by_id[row['id']] != as_list):
                    raise ValueError('frozen encoder/VQ or code geometry differs')
                codes_by_id[row['id']] = as_list
                arms = ('codec_only',) if version == 'released' else ('codec_only','with_source')
                for mode in arms:
                    features = loader.get(row).unsqueeze(0).cuda() if mode == 'with_source' else None
                    valid = torch.ones(features.shape[:2],device='cuda',dtype=torch.bool) if features is not None else None
                    def predict():
                        if model._word_bound is not None:
                            raise ValueError('stale source context')
                        logits = model.codec.decode(codes) if mode == 'codec_only' else model.word_logits(features,valid,codes)
                        if logits.shape != (1,128,30528) or not torch.isfinite(logits).all():
                            raise ValueError('invalid decoder logits')
                        return logits.argmax(-1)[0].cpu().tolist()
                    raw = predict()
                    if stage == 'pilot' and (raw != predict() or not torch.equal(codes,model.codec.encode(gold))):
                        raise ValueError('pilot repeat differs')
                    decoded = training.preparation.rendering.decode(raw,tokenizer)
                    output = {'id':row['id'],'split':row['split'],'arm':version+'_'+mode,
                              'target_sha256':row['target_sha256'],'reference':row['native_reference'],
                              'target_text':row['target_text'],'text':decoded['text'],
                              'raw_word_ids':raw,'codes':as_list,
                              **metrics(row['native_word_ids'],raw,row['native_reference'],decoded['text'])}
                    if (row['id'],output['arm']) in prior and output != prior[(row['id'],output['arm'])]:
                        raise ValueError('full evaluation differs from pilot')
                    outputs.append(output)
                if len(outputs)%32==0:
                    print(json.dumps({'version':version,'outputs':len(outputs)}),flush=True)
            if training.native.vendor.module_hash(model) != before:
                raise ValueError('inference modified parameters')
            write(RUNS[stage]/'outputs.json',outputs)
    result = {'status':'codec_roundtrip_completed', 'cases':len(rows),'outputs':len(outputs),
              'metrics':aggregate(outputs),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'feature_cache':loader.metrics(),'repeated_pilot_exact':stage=='pilot',
              'source_only_task_accuracy':None,'model_training':False,'final_test_opened':False,
              'speed_tested':False,'outputs_sha256':sha(RUNS[stage]/'outputs.json')}
    if result['peak_allocated_bytes'] > 4_900_000_000:
        raise ValueError('diagnostic memory budget exceeded')
    verify(proof)
    write(RUNS[stage]/'result.json',result)
    print(json.dumps(result),flush=True)
    model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=tuple(RUNS))
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else measure(args.stage)
