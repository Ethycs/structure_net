"""Source-only role/date prompts with unchanged raw-token alignment geometry."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

from experiments.structure_net import ncm_recovered_paraphrase_catalog as catalog

pool = catalog.pool
ROOTS = {s: pool.BASE/f'20261028_recovered_paraphrase_metadata_{s}_v1' for s in ('pilot', 'full')}
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-paraphrase-metadata.md')
TEST = Path('tests/structure_net/test_ncm_recovered_paraphrase_metadata.py')
SOURCE_FIELDS = {'id','source_id','group_id','source_component','source_sha256','source',
    'source_cl100k_tokens','source_truncated','split','original_role','original_created_at','spans'}
CONTEXT_RULE = ('\nThe original role and timestamp supplied in source_metadata before the passage '
                'are also source evidence. Use them for attribution without inventing additional context.')


class SourceContextTokenizer:
    def __init__(self, tokenizer, source):
        if set(source) != SOURCE_FIELDS or source['original_role'] not in ('user', 'assistant'):
            raise ValueError('exact source-only fields and recovered role required')
        if source['original_created_at'] is not None and not isinstance(source['original_created_at'], str):
            raise ValueError('literal original timestamp or null required')
        self.tokenizer, self.source = tokenizer, source

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def messages(self, messages):
        if (len(messages) != 2 or messages[0]['role'] != 'system' or messages[1]['role'] != 'user'
                or messages[1]['content'] != '<passage>\n'+self.source['source']+'\n</passage>'):
            raise ValueError('original literal passage prompt required')
        metadata = json.dumps({'role': self.source['original_role'], 'created_at': self.source['original_created_at']},
                              sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        return [{'role': 'system', 'content': messages[0]['content']+CONTEXT_RULE},
                {'role': 'user', 'content': '<source_metadata>'+metadata+'</source_metadata>\n'+messages[1]['content']}]

    def apply_chat_template(self, messages, *args, **kwargs):
        return self.tokenizer.apply_chat_template(self.messages(messages), *args, **kwargs)


def pilot_positions(db, entries):
    lookup = {r['id']: i for i,r in enumerate(entries)}
    selected = [entries[0]['id']]
    rows = []
    for ident, raw in db.execute('SELECT id,record FROM sources'):
        row = json.loads(raw)
        rows.append((ident, row['source_cl100k_tokens'], row['original_role'], row['original_created_at']))
    selected += [min(rows, key=lambda r:(r[1],r[0]))[0], min(rows, key=lambda r:(-r[1],r[0]))[0]]
    for role in sorted({r[2] for r in rows}): selected.append(min(r[0] for r in rows if r[2] == role))
    for has_date in (False, True):
        candidates = [r[0] for r in rows if (r[3] is not None) == has_date]
        if candidates: selected.append(min(candidates))
    duplicate = db.execute('SELECT source_sha FROM sources GROUP BY source_sha HAVING count(DISTINCT created_at)>1 ORDER BY source_sha LIMIT 1').fetchone()
    if duplicate:
        selected += [r[0] for r in db.execute('SELECT min(id) FROM sources WHERE source_sha=? AND created_at IS NOT NULL GROUP BY created_at ORDER BY created_at LIMIT 2', duplicate)]
    return [lookup[i] for i in dict.fromkeys(selected)]


def prepare(source, tokenizers, metadata):
    native, gpt, storage = tokenizers
    wrapped = SourceContextTokenizer(native, source)
    value = metadata.prepare_one(source, wrapped, gpt, storage)
    if (set(value) - SOURCE_FIELDS != {'partition_split','context_ids','source_ids','alignment',
            'alignment_sha256','native_prompt_sha256','source_frame'}
            or value['source_ids'] != gpt(source['source'], add_special_tokens=False)['input_ids']
            or len(value['alignment']) != len(value['source_ids']) or not 1 <= len(value['source_ids']) <= 8192):
        raise ValueError('complete source-only geometry required')
    for row in value['alignment']:
        if not row or any(w <= 0 or not 0 <= i < len(value['context_ids']) for i,w in row) or abs(sum(w for _,w in row)-1) > 1e-6:
            raise ValueError('normalized complete alignment required')
    return value


def run(stage):
    root = ROOTS[stage]
    if root.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only source metadata required')
    from experiments.structure_net import ncm_pixi
    from experiments.structure_net import ncm_additional_source_metadata as metadata
    from experiments.structure_net import ncm_indexed_jsonl as indexed
    _, environment = ncm_pixi.configure()
    inputs = catalog.receipt_paths(catalog.ROOT)
    result = pool.read(catalog.ROOT/'result.json')
    if result['status'] != 'guarded_noisy_paraphrases_indexed_and_read_back': raise ValueError('completed guarded catalog required')
    entries = [json.loads(line) for line in (catalog.ROOT/'sources_index.jsonl').open()]
    source = catalog.ROOT/'sources.jsonl'
    records = indexed.IndexedJSONL(source, entries, result['artifacts'][str(source)])
    db = catalog.context.connect(catalog.ROOT/'catalog.sqlite')
    selected = pilot_positions(db, entries); db.close()
    old = {}
    if stage == 'full':
        inputs += catalog.receipt_paths(ROOTS['pilot'])
        prior = pool.read(ROOTS['pilot']/'result.json')
        if not prior['original_TRAIN_geometry_equal'] or prior['pilot_positions'] != selected:
            raise ValueError('completed same-pilot geometry qualification required')
        old = {r['id']: r for r in map(json.loads, (ROOTS['pilot']/'metadata.jsonl').open())}
    inputs += [catalog.ROOT/n for n in ('producer.json','sources.jsonl','sources_index.jsonl','catalog.sqlite')]
    inputs += [Path('pixi.lock'), Path('pyproject.toml'), metadata.old_training.DATA/'cases.json', metadata.old_training.DATA/'manifest.json']
    for directory in (metadata.base.MODEL_PATH, metadata.base.GPT2):
        inputs += [p for p in directory.iterdir() if p.is_file() and p.name in {'config.json','tokenizer.json',
            'tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt','added_tokens.json'}]
    tokenizers = metadata.tokenizers()
    parity = metadata.parity(*tokenizers)
    files = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve()}
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).resolve().is_relative_to(Path('experiments').resolve()):
            files.add(Path(path).resolve())
    proof = {'schema': 'nal.ncm-recovered-paraphrase-metadata.v1', 'stage': stage, 'environment': environment,
        'config': {'chat_template_date': '23 Oct 2026', 'original_role_date_evidence': True,
            'full_source_retained': True, 'pilot_positions': selected, 'maximum_source_positions': 8192,
            'max_RSS_bytes': catalog.MAX_RSS, 'target_fields_in_prompt': False},
        'source_hashes': {str(p): pool.file_hash(p) for p in files},
        'input_hashes': {str(p): pool.file_hash(p) for p in dict.fromkeys(inputs)}}
    root.mkdir(); (root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)): (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    pool.write(root/'producer.json', proof); pool.write(root/'original_TRAIN_parity.json', parity)
    positions = selected if stage == 'pilot' else range(len(records))
    started = time.monotonic(); count = source_positions = context_positions = max_source = max_context = repeated = 0
    with (root/'metadata.jsonl').open('xb') as output, (root/'index.jsonl').open('x') as idx:
        for position in positions:
            value = prepare(records[position], tokenizers, metadata)
            if value['id'] in old:
                if value != old[value['id']]: raise ValueError('pilot/full metadata differs')
                repeated += 1
            n,c = len(value['source_ids']),len(value['context_ids'])
            indexed.write_record(output, idx, value, catalog_position=position,
                source_sha256=value['source_sha256'], source_component=value['source_component'],
                source_positions=n, native_prompt_positions=c, source_frame=value['source_frame'],
                alignment_sha256=value['alignment_sha256'], native_prompt_sha256=value['native_prompt_sha256'])
            count += 1; source_positions += n; context_positions += c
            max_source = max(max_source,n); max_context = max(max_context,c)
            if stage == 'pilot' or count % 1024 == 0:
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > catalog.MAX_RSS: raise ValueError('host memory limit exceeded')
                print(json.dumps({'stage':stage,'cases':count,'total':len(positions),'elapsed_seconds':time.monotonic()-started}),flush=True)
    records.close()
    catalog.readback(root/'metadata.jsonl', root/'index.jsonl', count)
    if count != len(positions) or stage == 'full' and repeated != len(selected): raise ValueError('complete metadata and pilot repetitions required')
    for collection in (proof['input_hashes'],proof['source_hashes']):
        for p,sha in collection.items():
            if pool.file_hash(p) != sha: raise ValueError('metadata producer changed')
    value = {'schema': proof['schema'], 'status': 'recovered_role_aware_metadata_'+stage+'_completed',
        'cases':count,'pilot_positions':selected,'source_positions':source_positions,'native_prompt_positions':context_positions,
        'max_source_positions':max_source,'max_native_prompt_positions':max_context,
        'estimated_full_FP32_feature_bytes':source_positions*3072*4,'pilot_repetitions_exact':repeated,
        'original_TRAIN_geometry_equal': all(r['all_geometry_exact'] for r in parity),
        'all_byte_offsets_verified':True,'target_fields_in_prompt':False,'training_admitted':False,
        'model_calls':0,'task_accuracy':None,'final_test_opened':False,'elapsed_seconds':time.monotonic()-started,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'artifacts':{str(p):pool.file_hash(p) for p in root.iterdir() if p.is_file()}}
    pool.write(root/'result.json',value);print(json.dumps({k:v for k,v in value.items() if k!='artifacts'}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=tuple(ROOTS));args=parser.parse_args()
    root=ROOTS[args.stage];existed=root.exists()
    try:run(args.stage)
    except Exception as error:
        if root.exists() and not existed:pool.write(root/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
