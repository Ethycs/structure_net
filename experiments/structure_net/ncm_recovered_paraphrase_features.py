"""Bind the qualified packed FP32 feature pipeline to role-aware source inputs."""
import argparse
from pathlib import Path
import sys

from experiments.structure_net import ncm_recovered_paraphrase_metadata as metadata
from experiments.structure_net import ncm_public_feature_store as engine

catalog, pool = metadata.catalog, metadata.pool
data = engine.data
HYPOTHESIS = 'ncm-recovered-role-aware-source-features-v1'
ROOTS = {s:pool.BASE/f'20261028_recovered_paraphrase_features_{s}_v1' for s in ('pilot','full')}
STORAGE = Path('/data2/structure_net/ncm_recovered_paraphrase_20261028_v1/features')
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-paraphrase-features.md')
TEST = Path('tests/structure_net/test_ncm_recovered_paraphrase_features.py')


def select(entries, pilot_ids):
    lookup = {r['id']:i for i,r in enumerate(entries)}
    if len(lookup)!=len(entries) or len(set(pilot_ids))!=len(pilot_ids) or not set(pilot_ids)<=set(lookup):
        raise ValueError('complete unique source population and metadata controls required')
    positions = [lookup[k] for k in pilot_ids]
    for field in ('source_positions','native_prompt_positions'):
        positions += sorted(range(len(entries)),key=lambda i:(-entries[i][field],entries[i]['id']))[:4]
    return list(dict.fromkeys(positions))


def ready():
    from experiments.structure_net import ncm_pixi
    from experiments.structure_net import ncm_indexed_jsonl as indexed
    import json
    _, environment = ncm_pixi.configure()
    root = metadata.ROOTS['full']
    catalog.receipt_paths(root)
    state = pool.read(root/'result.json')
    if (state['status']!='recovered_role_aware_metadata_full_completed' or state['cases']!=40890
            or not state['all_byte_offsets_verified'] or not state['original_TRAIN_geometry_equal']
            or state['pilot_repetitions_exact']!=7 or state['target_fields_in_prompt']):
        raise ValueError('complete role-aware full metadata and exact pilot repetitions required')
    with (root/'index.jsonl').open() as f:entries=list(map(json.loads,f))
    if len(entries)!=state['cases']:raise ValueError('complete source index required')
    path=root/'metadata.jsonl'
    records=indexed.IndexedJSONL(path,entries,state['artifacts'][str(path)])
    pilot_ids=[entries[i]['id'] for i in state['pilot_positions']]
    return environment,records,select(entries,pilot_ids),{}


def freeze(root,stage,environment,selection):
    files={Path(__file__).resolve(),PLAN.resolve(),TEST.resolve(),*map(Path.resolve,engine.TESTS)}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            p=Path(filename).resolve()
            if p.is_file() and p.suffix=='.py' and p.is_relative_to(Path('experiments').resolve()):files.add(p)
    paths=[metadata.ROOTS['full']/n for n in ('producer.json','result.json','actual_tool_receipt.json','metadata.jsonl','index.jsonl')]
    paths += [p for p in engine.original.base.MODEL_PATH.iterdir() if p.is_file()]
    paths += [Path('pixi.lock'),Path('pyproject.toml'),Path(engine.__file__),engine.PLAN,engine.AMENDMENT]
    if stage=='full':paths += [ROOTS['pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json',
        'audit_result.json','audit_actual_tool_receipt.json','raw_hashes.json')]
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    proof={'schema':'nal.ncm-recovered-paraphrase-features.v1','hypothesis_id':HYPOTHESIS,'stage':stage,
        'environment':environment,'selection':selection,'config':{'physical_GPU':0,'body_precision':'BF16',
            'features':'complete_FP32','TF32':False,'encoder_threads':4,'pending_limit':8,
            'shard_bytes':256*1024**2,'minimum_free_disk_bytes':100*1024**3,
            'maximum_GPU_allocated_bytes':8_600_000_000,'source_metadata':str(metadata.ROOTS['full']),
            'storage':str(STORAGE/stage),'target_fields_in_body_input':False,**engine.BOUNDARIES},
        'source_hashes':{str(p):pool.file_hash(p) for p in sorted(files)},
        'input_hashes':{str(p):pool.file_hash(p) for p in dict.fromkeys(paths)}}
    data.write(root/'producer.json',proof);return proof


def run(command,stage):
    engine.ROOTS,engine.STORAGE,engine.HYPOTHESIS=ROOTS,STORAGE,HYPOTHESIS
    engine.ready,engine.freeze=ready,freeze
    (engine.run if command=='extract' else engine.audit)(stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('command',choices=('extract','audit'))
    parser.add_argument('stage',choices=tuple(ROOTS));args=parser.parse_args()
    root=ROOTS[args.stage];existed=root.exists()
    try:run(args.command,args.stage)
    except Exception as error:
        if args.command=='extract' and root.exists() and not existed:
            pool.write(root/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
