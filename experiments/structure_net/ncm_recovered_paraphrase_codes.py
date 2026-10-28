"""Singleton FP32 target codes, with complete pair/frame bindings and disk replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time

from experiments.structure_net import ncm_recovered_paraphrase_catalog as catalog

pool = catalog.pool
ROOTS = {s:pool.BASE/f'20261028_recovered_paraphrase_codes_{s}_v1' for s in ('pilot','full')}
STORAGE = Path('/data2/structure_net/ncm_recovered_paraphrase_20261028_v1/codes')
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-paraphrase-codes.md')
TEST = Path('tests/structure_net/test_ncm_recovered_paraphrase_codes.py')
EXPECTED = '9aace450a49a7dfb44ba1eb00a19974a50d67062ad9f2202a6a2361ea4344e32'
LIMIT = 3_500_000_000


def unique_frames(rows):
    frames = {}; pairs = set()
    for row in rows:
        ids = row['native_word_ids']; n = row['native_lexical_tokens']
        if (len(ids) != 128 or not 1 <= n <= 126 or ids[0] != 101 or ids[n+1] != 101
                or any(ids[n+2:]) or any(type(v) is not int or not 0 <= v < 30522 for v in ids)
                or set(ids[1:n+1]) & {0,100,101,102,103}):
            raise ValueError('complete native128 frame required')
        sha = hashlib.sha256(struct.pack('<128H',*ids)).hexdigest()
        if sha != row['native_frame_sha256'] or row['id'] in pairs:
            raise ValueError('unique pair with exact frame binding required')
        pairs.add(row['id'])
        frame = frames.setdefault(sha, {'frame_sha256':sha,'native_word_ids':ids,
            'native_lexical_tokens':n,'pair_ids':[]})
        if frame['native_word_ids'] != ids: raise ValueError('frame hash collision')
        frame['pair_ids'].append(row['id'])
    values = []
    for key in sorted(frames):
        row = frames[key]; row['pair_ids'].sort(); values.append(row)
    return values, len(pairs)


def controls(frames):
    if len(frames) < 32: raise ValueError('at least32 distinct target frames required')
    candidates = [*range(8),*range(len(frames)-8,len(frames)),
        *sorted(range(len(frames)),key=lambda i:(-frames[i]['native_lexical_tokens'],i))[:8],
        *sorted(range(len(frames)),key=lambda i:(frames[i]['native_lexical_tokens'],i))[:8],*range(len(frames))]
    return list(dict.fromkeys(candidates))[:32]


def run(stage):
    root, storage = ROOTS[stage], STORAGE/stage
    if root.exists() or storage.exists(): raise ValueError('fresh singleton target-code roots required')
    from experiments.structure_net import ncm_recovered_label_codec as runtime
    import torch
    training, environment = runtime.codec_source.prior.runtime()
    runtime.lifecycle.reference.study.pipeline.determinism.configure_runtime()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060 SUPER' or torch.cuda.mem_get_info()[0] < LIMIT):
        raise ValueError('physicalGPU2 and adequate free memory required')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    inputs = catalog.receipt_paths(catalog.ROOT)
    state = pool.read(catalog.ROOT/'result.json')
    if state['status'] != 'guarded_noisy_paraphrases_indexed_and_read_back': raise ValueError('complete catalog required')
    with (catalog.ROOT/'pairs.jsonl').open() as source:
        frames, pair_count = unique_frames(map(json.loads,source))
    if pair_count != state['coverage']['pairs']: raise ValueError('all admitted candidate pairs required')
    selected = controls(frames); pilot_codes = {}
    inputs += [catalog.ROOT/n for n in ('producer.json','pairs.jsonl','pairs_index.jsonl')]
    if stage == 'full':
        inputs += catalog.receipt_paths(ROOTS['pilot'])
        pilot = pool.read(ROOTS['pilot']/'result.json')
        if pilot['pilot_positions'] != selected or pilot['saved_reference_controls_exact'] != 128:
            raise ValueError('completed singleton/reference pilot required')
        pilot_codes = pool.read(ROOTS['pilot']/'pilot_codes.json')
    baseline = runtime.ROOT
    inputs += catalog.receipt_paths(baseline)
    reference_rows, reference_outputs = [pool.read(baseline/n) for n in ('cases.json','outputs.json')]
    if len(reference_rows) != 128 or [r['id'] for r in reference_rows] != [r['id'] for r in reference_outputs]:
        raise ValueError('complete original128 singleton references required')
    inputs += [baseline/n for n in ('cases.json','outputs.json')]
    inputs += [Path('pixi.lock'),Path('pyproject.toml'),training.native.RELEASE/'files/lm1b/tokenizer.pt',
               training.native.RELEASE/'files/lm1b/small/config.yaml']
    files = {Path(__file__).resolve(),PLAN.resolve(),TEST.resolve()}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            p=Path(filename).resolve()
            if p.is_file() and p.suffix=='.py' and (p.is_relative_to(Path('experiments').resolve()) or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                files.add(p)
    proof={'schema':'nal.ncm-recovered-paraphrase-codes.v1','stage':stage,'environment':environment,
        'config':{'pairs':pair_count,'unique_frames':len(frames),'batch':1,'FP32':True,'TF32':False,
            'physical_GPU':2,'codec_sha256':EXPECTED,'maximum_GPU_bytes':LIMIT,'maximum_RSS_bytes':catalog.MAX_RSS,
            'pilot_positions':selected},'source_hashes':{str(p):pool.file_hash(p) for p in sorted(files)},
        'input_hashes':{str(p):pool.file_hash(p) for p in dict.fromkeys(inputs)}}
    root.mkdir();storage.mkdir(parents=True);(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    pool.write(root/'producer.json',proof)
    torch.cuda.reset_peak_memory_stats();codec=training.native.load_codec()
    if training.native.vendor.module_hash(codec)!=EXPECTED:raise ValueError('exact frozen released codec required')
    def encode(ids):
        with torch.no_grad():value=codec.encode(torch.tensor([ids],device='cuda',dtype=torch.long))
        if value.shape!=(1,641) or value.dtype not in (torch.int32,torch.int64) or value.min()<0 or value.max()>=8192:
            raise ValueError('singleton641 bounded integer codes required')
        if torch.cuda.max_memory_allocated()>LIMIT:raise ValueError('GPU memory ceiling exceeded')
        return value[0].cpu().tolist()
    if stage=='pilot':
        for row,reference in zip(reference_rows,reference_outputs,strict=True):
            if encode(row['native_word_ids']) != reference['codes']:raise ValueError('original singleton reference differs')
        for i in selected:
            first=encode(frames[i]['native_word_ids'])
            if first!=encode(frames[i]['native_word_ids']):raise ValueError('repeated singleton encoding differs')
            pilot_codes[frames[i]['frame_sha256']]=first
        pool.write(root/'pilot_codes.json',pilot_codes)
    positions=selected if stage=='pilot' else range(len(frames));started=time.monotonic();count=repeated=0
    binary,index=storage/'codes.uint16.bin',storage/'index.jsonl'
    with binary.open('xb') as out,index.open('x') as idx:
        for i in positions:
            frame=frames[i];codes=encode(frame['native_word_ids']);key=frame['frame_sha256']
            if key in pilot_codes:
                if codes!=pilot_codes[key]:raise ValueError('pilot code changed in sequential pass')
                repeated+=1
            raw=struct.pack('<641H',*codes)
            row={'position':count,'catalog_frame_position':i,'frame_sha256':key,'pair_ids':frame['pair_ids'],
                'offset_bytes':out.tell(),'length_bytes':len(raw),'code_sha256':hashlib.sha256(raw).hexdigest()}
            out.write(raw);idx.write(json.dumps(row)+'\n');count+=1
            if count%512==0:
                out.flush();idx.flush()
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>catalog.MAX_RSS:raise ValueError('host RSS ceiling exceeded')
                print(json.dumps({'encoded':count,'total':len(positions),'elapsed_seconds':time.monotonic()-started}),flush=True)
    checked=0;readback_pairs=0
    with binary.open('rb') as raw_file,index.open() as idx:
        for row in map(json.loads,idx):
            i=positions[checked];frame=frames[i]
            if (row['position']!=checked or row['catalog_frame_position']!=i or row['frame_sha256']!=frame['frame_sha256']
                    or row['pair_ids']!=frame['pair_ids'] or row['offset_bytes']!=raw_file.tell() or row['length_bytes']!=1282):
                raise ValueError('complete ordered pair/frame/code binding differs')
            raw=raw_file.read(1282)
            if hashlib.sha256(raw).hexdigest()!=row['code_sha256'] or max(struct.unpack('<641H',raw))>=8192:
                raise ValueError('saved code bytes or range differ')
            checked+=1;readback_pairs+=len(row['pair_ids'])
        if raw_file.read(1) or checked!=len(positions) or repeated!=32 or stage=='full' and readback_pairs!=pair_count:
            raise ValueError('incomplete code corpus or pilot replay')
    if training.native.vendor.module_hash(codec)!=EXPECTED or any(p.grad is not None for p in codec.parameters()):
        raise ValueError('encoding changed codec')
    for collection in (proof['source_hashes'],proof['input_hashes']):
        for p,sha in collection.items():
            if pool.file_hash(p)!=sha:raise ValueError('producer input changed')
    result={'schema':proof['schema'],'status':'singleton_paraphrase_codes_'+stage+'_completed',
        'frames':count,'pairs_bound':readback_pairs,'all_catalog_pairs':pair_count,'all_catalog_unique_frames':len(frames),
        'batch':1,'code_width':641,'pilot_positions':selected,'pilot_controls_reproduced':repeated,
        'saved_reference_controls_exact':128,'codec_unchanged':True,'codec_sha256':EXPECTED,
        'binary':str(binary),'index':str(index),'all_disk_records_replayed':True,
        'elapsed_seconds':time.monotonic()-started,'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'training_admitted':False,'new_judge_calls':0,'task_accuracy':None,'final_test_opened':False,
        'artifacts':{str(p):pool.file_hash(p) for p in [*filter(Path.is_file,root.iterdir()),binary,index]}}
    if result['peak_RSS_bytes']>catalog.MAX_RSS:raise ValueError('host RSS ceiling exceeded')
    pool.write(root/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=tuple(ROOTS));args=parser.parse_args()
    root=ROOTS[args.stage];existed=root.exists()
    try:run(args.stage)
    except Exception as error:
        if root.exists() and not existed:pool.write(root/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
