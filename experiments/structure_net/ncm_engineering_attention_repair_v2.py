"""Fixed-budget correction on engineering summaries; semantic quality pending."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_attention_repair_data as preparation
from experiments.structure_net import ncm_attention_code_repair_v2 as mechanism
from experiments.structure_net import ncm_attention_repair_objective as objective
from experiments.structure_net import ncm_after_grounded_stop_judgment_lineage as lineage
from experiments.neural_architecture_lab import audit_ncm_engineering_source_features as feature_audit
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

native, q, ROOT = preparation.native, preparation.q, preparation.ROOT
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-engineering-attention-repair-v2-plan.md')
TEST = Path('tests/structure_net/test_ncm_engineering_attention_repair_v2.py')
DATA = ROOT / '20261025_engineering_attention_repair732_data_v1'
RUNS = {mode: ROOT / ('20261025_engineering_attention_repair732_' + mode + '_v2') for mode in ('pilot', 'capacity', 'train')}


@dataclass(frozen=True)
class Config:
    updates: int = 4096
    accumulation: int = 8
    warmup: int = 128
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    sampler_seed: int = 134142
    dropout_seed: int = 135142


def rate(step, updates=4096, warmup=128):
    if not 1 <= step <= updates or not 0 < warmup < updates:
        raise ValueError('step inside fixed warmup/cosine budget required')
    c = Config()
    if step <= warmup: return c.max_lr * step / warmup
    return c.min_lr + .5 * (c.max_lr-c.min_lr) * (1+math.cos(math.pi*(step-warmup)/(updates-warmup)))


def diagnostic_ids(rows, index):
    positions = {r['id']: r['source_positions'] for r in index}
    chosen = []
    for group in sorted({r['source_component'] for r in rows}):
        subset = [r for r in rows if r['source_component'] == group]
        chosen += [subset[0]['id'], min(subset, key=lambda r: (-positions[r['id']], r['id']))['id']]
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) > 16:
        raise ValueError('at most eight independent engineering TRAIN groups expected')
    for row in sorted(rows, key=lambda r: (text_hash('repair732-diagnostic:' + r['id']), r['id'])):
        if len(chosen) == 16: return chosen
        if row['id'] not in chosen: chosen.append(row['id'])
    if len(chosen) != 16: raise ValueError('sixteen fixed TRAIN diagnostics required')
    return chosen


def producing(root, stage, inputs, extra):
    files = [Path(__file__), PLAN, TEST, Path(mechanism.__file__), Path(objective.__file__),
        Path(preparation.__file__), Path(lineage.__file__), Path(preparation.checkpoint.__file__),
        Path(preparation.streaming.__file__), Path(objective.geometry.__file__), Path(feature_audit.__file__)]
    root.mkdir()
    (root / 'producer_source').mkdir()
    for i, path in enumerate(files):
        (root / 'producer_source' / f'{i}_{path.name}').write_bytes(path.read_bytes())
    p = {'schema': 'nal.ncm-engineering-attention-repair.v2', 'stage': stage,
        'source_hashes': {str(p): file_hash(p) for p in files},
        'input_hashes': {str(p): file_hash(p) for p in inputs},
        'source_scope': 'engineering TRAIN summaries only; historical atom labels excluded; original evaluation preserved',
        'config': {**asdict(Config()), **extra, 'corrector': asdict(mechanism.Config()),
            'frozen_NCM_checkpoint_sha256': '3911eab74db6ad13104f87765196e4212893a3e179ba0283e103e65e78273bec',
            'parent_and_MLPs_frozen_and_absent': True, 'native_encoder_VQ_generator_word_decoder_frozen': True,
            'only_corrector_trainable': True, 'classifier_geometry':'bounded_radial_v2_scale4', 'precision': 'FP32', 'TF32': False, 'physical_GPU': 2,
            'objective': 'equal-level code CE +0.05soft-latent MSE +0.5hard-forward word CE',
            'optimizer': {'name': 'AdamW', 'betas': [.9,.999], 'eps': 1e-8, 'weight_decay': .01},
            'gradient_clip': 1., 'host_feature_cache_bytes': 2*1024**3, 'peak_limit_bytes': 4_900_000_000,
            'oracle_targets_in_prediction': False, 'checkpoint_selection': 'fixed final only',
            'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False}}
    write_json(root / 'producer.json', p)
    return p


def prepare():
    native.vendor.corpus.quality.dependencies()
    if DATA.exists(): raise ValueError('fresh engineering correction dataset required')
    if q.read(feature_audit.DEST / 'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual independent full888 feature payload audit required')
    audited = q.read(feature_audit.DEST / 'result.json')
    if (audited['cases'] != 888 or not audited['FP32_bytes_preserved']
            or file_hash(preparation.cache.CACHE / 'manifest.json') != audited['artifacts'][str(preparation.cache.CACHE / 'manifest.json')]):
        raise ValueError('complete independently audited frozen-source feature bindings required')
    source_rows, index, manifest = preparation.corpus('engineering', need_features=True)
    lineage.base_cache()
    code_root = preparation.targets.DEST
    prediction_root = preparation.PREDICTIONS['engineering']
    for root in (code_root, prediction_root, preparation.AUDIT): preparation.completed(root)
    inputs = [*lineage.inputs(), feature_audit.DEST / 'result.json', feature_audit.DEST / 'actual_tool_receipt.json']
    for root in (code_root, prediction_root):
        state = q.read(root / 'result.json')
        if file_hash(root / 'outputs.json') != state['outputs_sha256']:
            raise ValueError('unchanged complete target/prediction outputs required')
        inputs += [root / n for n in ('producer.json','result.json','outputs.json','actual_tool_receipt.json')]
    predicted_state = q.read(prediction_root / 'result.json')
    if (predicted_state['checkpoint_sha256'] != '3911eab74db6ad13104f87765196e4212893a3e179ba0283e103e65e78273bec'
            or predicted_state['model_sha256'] != '2effdc560c0a46bc4e5b1ce419a2f47896230eadff5cbccae70d5e7983561376'):
        raise ValueError('exact original frozen NCM required')
    gold, predictions = q.read(code_root / 'outputs.json'), q.read(prediction_root / 'outputs.json')
    if any([r['id'] for r in values] != [r['id'] for r in source_rows] for values in (gold,predictions)):
        raise ValueError('complete ordered engineering TRAIN bindings required')
    rows = []
    for row,target,actual in zip(source_rows,gold,predictions):
        if (any(target[k] != row[k] for k in ('id','source_sha256','target_sha256','native_word_ids'))
                or actual['source_sha256'] != row['source_sha256']
                or actual['feature_payload_sha256'] != manifest[row['id']]['payload_sha256']
                or len(target['codes']) != 641 or len(actual['codes']) != 641):
            raise ValueError('exact clean target and actual frozen-error bindings required')
        rows.append({**row,'corpus':'engineering','oracle_codes':target['codes'],
            'predicted_codes':actual['codes'],'baseline_raw_word_ids':actual['raw_word_ids']})
    if len(rows)!=732 or len({r['source_sha256'] for r in rows})!=732:
        raise ValueError('all732 distinct engineering TRAIN inputs required')
    inputs += [preparation.AUDIT / n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    inputs += [Path(r['path']) for r in index]
    p = producing(DATA,'engineering732_actual_error_and_oracle_loss_bindings',inputs,
        {'train_cases':732,'historical_core_atom_targets_excluded':True,'student_training_admitted':True,
         'stored_targets_unreviewed':True,'all512_reserved_components_excluded':True})
    write_json(DATA/'cases.json',rows)
    write_json(DATA/'source_index.json',index)
    write_json(DATA/'feature_manifest.json',manifest)
    write_json(DATA/'selection.json',{'diagnostic_ids':diagnostic_ids(rows,index)})
    native.vendor.verify(p)
    write_json(DATA/'result.json',{'status':'complete732_engineering_repair_TRAIN_data_prepared','cases':732,
        'cases_sha256':file_hash(DATA/'cases.json'),'task_accuracy':None,'final_test_opened':False})


def tensors(row, loader):
    features = loader.get(row).unsqueeze(0).cuda()
    mask = torch.ones(features.shape[:2], dtype=torch.bool, device='cuda')
    integers = lambda key: torch.tensor([row[key]], dtype=torch.long, device='cuda')
    return features, mask, integers('predicted_codes'), integers('oracle_codes'), integers('native_word_ids')


def load_renderer():
    """Verify the exact already-audited model without replaying unrelated archives."""
    root=preparation.PREDICTIONS['engineering']
    preparation.completed(root)
    proof=q.read(root/'result.json')
    checkpoint=preparation.checkpoint
    expected='3911eab74db6ad13104f87765196e4212893a3e179ba0283e103e65e78273bec'
    if (proof['checkpoint_sha256']!=expected
            or proof['model_sha256']!='2effdc560c0a46bc4e5b1ce419a2f47896230eadff5cbccae70d5e7983561376'
            or file_hash(checkpoint.training.TRAIN/'model.pt')!=expected):
        raise ValueError('exact complete source-only baseline checkpoint required')
    # Fresh instance routing is unchanged; only this new experiment owns it.
    renderer=checkpoint.training.lifecycle.mechanism.enable(checkpoint.training.lifecycle.head.from_domain(device='cpu'))
    saved=torch.load(checkpoint.training.TRAIN/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    renderer.load_state_dict(saved['model'],strict=True)
    checkpoint.model_contract(renderer,q.read(checkpoint.training.TRAIN/'result.json'))
    if native.vendor.module_hash(renderer)!=proof['model_sha256']:
        raise ValueError('full frozen generator/decoder/attention/encoder/VQ tensors differ')
    return renderer.cuda().eval().requires_grad_(False)


@torch.no_grad()
def diagnostics(corrector, renderer, rows, loader, tokenizer):
    corrector.eval()
    books = [b.weight for b in renderer.codec.vq.embeddings]
    outputs = []
    for row in rows:
        features, mask, predicted, gold, words = tensors(row,loader)
        codes = corrector.repair(predicted,features,mask,books)
        raw = renderer.word_logits(features,mask,codes).argmax(-1).cpu().tolist()[0]
        outputs.append({'id':row['id'],'codes':codes.cpu().tolist()[0],'raw_word_ids':raw,
            **preparation.rendering.decode(raw,tokenizer),
            'changed_codes':int((codes!=predicted).sum()), 'correct_codes':int((codes==gold).sum()),
            'root_correct':bool((codes[:,:1]==gold[:,:1]).all()), 'code_positions':641})
    return outputs


@torch.no_grad()
def oracle_controls(renderer, rows, loader, tokenizer):
    """TRAIN-only representation control, never source-only model prediction."""
    outputs=[]
    for row in rows:
        features,mask,_,gold,words=tensors(row,loader)
        raw=renderer.word_logits(features,mask,gold).argmax(-1)
        active=words!=0
        ids=raw.cpu().tolist()[0]
        outputs.append({'id':row['id'],'oracle_target_codes':True,'TRAIN_diagnostic_only':True,
            'raw_word_ids':ids,**preparation.rendering.decode(ids,tokenizer),
            'active_word_matches':int(((raw==words)&active).sum()),'active_words':int(active.sum()),
            'task_accuracy':None})
    return outputs


def save(root, name, corrector, opt, sampler, step, p):
    path = root / name
    if path.exists(): raise ValueError('checkpoint overwrite forbidden')
    torch.save({'model':{k:v.detach().cpu().clone() for k,v in corrector.state_dict().items()},
        'optimizer':opt.state_dict(),'sampler_rng':sampler.get_state(),'CPU_rng':torch.get_rng_state(),
        'CUDA_rng':torch.cuda.get_rng_state_all(),'step':step,'config':asdict(Config()),
        'producer_fingerprint':q.digest(p)},path)


def train(mode):
    native.vendor.corpus.quality.dependencies()
    preparation.checkpoint.training.require_gpu()
    preparation.completed(DATA)
    if mode not in RUNS: raise ValueError('fixed pilot/capacity/train mode required')
    root = RUNS[mode]
    if root.exists(): raise ValueError('fresh fixed-budget corrector run required')
    if mode != 'pilot':
        preparation.completed(RUNS['pilot'])
        if not q.read(RUNS['pilot']/'result.json')['strict_reload_passed']:
            raise ValueError('real matching objective lifecycle required')
    if mode == 'train': preparation.completed(RUNS['capacity'])
    rows = q.read(DATA/'cases.json')
    if file_hash(DATA/'cases.json') != q.read(DATA/'result.json')['cases_sha256']:
        raise ValueError('unaltered whole732engineering data required')
    index,manifest = q.read(DATA/'source_index.json'),q.read(DATA/'feature_manifest.json')
    ids = q.read(DATA/'selection.json')['diagnostic_ids']
    lookup = {r['id']:r for r in rows}
    probe_rows = [lookup[identity] for identity in ids]
    selected = probe_rows if mode == 'capacity' else rows
    steps, accumulation, warmup = (2,2,1) if mode=='pilot' else (512,4,32) if mode=='capacity' else (4096,8,128)
    inputs = [DATA/n for n in ('producer.json','result.json','cases.json','source_index.json','feature_manifest.json','selection.json','actual_tool_receipt.json')]
    inputs += [preparation.checkpoint.training.TRAIN/'model.pt']
    p = producing(root,'corrector_'+mode,inputs,{'mode':mode,'actual_updates':steps,'actual_accumulation':accumulation,
        'actual_warmup':warmup,'actual_max_lr':1e-4 if mode=='pilot' else Config().max_lr,
        'train_pool_cases':len(selected),'capacity_weights_reused':False,'new_random_residual_initializer_only':True})
    loader = preparation.streaming.FeatureCache(index,manifest,max_bytes=2*1024**3)
    renderer = load_renderer()
    frozen_hash = native.vendor.module_hash(renderer)
    corrector = mechanism.AttentionCodeRepair().cuda()
    initial_hash = native.vendor.module_hash(corrector)
    tokenizer = native.tokenizer()
    initial = diagnostics(corrector,renderer,probe_rows,loader,tokenizer)
    for row,value in zip(probe_rows,initial):
        if value['codes']!=row['predicted_codes'] or value['raw_word_ids']!=row['baseline_raw_word_ids']:
            raise ValueError('identity initialization must preserve complete actual baseline words and codes')
    write_json(root/'initial_outputs.json',initial)
    write_json(root/'TRAIN_oracle_controls.json',oracle_controls(renderer,probe_rows,loader,tokenizer))
    torch.manual_seed(Config().dropout_seed)
    torch.cuda.manual_seed_all(Config().dropout_seed)
    sampler = torch.Generator().manual_seed(Config().sampler_seed)
    opt = torch.optim.AdamW(corrector.parameters(),lr=Config().max_lr,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    books = [b.weight for b in renderer.codec.vq.embeddings]
    longest = min(rows,key=lambda r:(-manifest[r['id']]['source_tokens'],r['id']))
    torch.cuda.reset_peak_memory_stats()
    try:
        with (root/'trace.jsonl').open('x') as stream:
            for step in range(1,steps+1):
                corrector.train()
                lr = 1e-4 if mode=='pilot' else rate(step,steps,warmup)
                for group in opt.param_groups: group['lr']=lr
                opt.zero_grad(set_to_none=True)
                observations = []
                for micro in range(accumulation):
                    position = int(torch.randint(len(selected),(1,),generator=sampler))
                    row = (rows[0] if micro==0 else longest) if mode=='pilot' else selected[position]
                    features,mask,predicted,gold,words = tensors(row,loader)
                    scores = corrector(predicted,features,mask,books)
                    loss,metrics,hard = objective.loss(renderer,scores,gold,words,features,mask)
                    (loss/accumulation).backward()
                    observations.append({'id':row['id'],'source_sha256':row['source_sha256'],
                        'source_positions':features.shape[1],'metrics':metrics,'changed_codes':int((hard!=predicted).sum())})
                    del features,mask,predicted,gold,words,scores,loss,hard
                norm = torch.nn.utils.clip_grad_norm_(corrector.parameters(),1.)
                source_grad = float(corrector.source_kv.weight.grad.abs().sum())
                if (not torch.isfinite(norm) or step==2 and source_grad<=0
                        or any(v.grad is not None for v in renderer.parameters())):
                    raise ValueError('finite corrector/source gradients and protected frozen renderer required')
                opt.step()
                stream.write(json.dumps({'step':step,'lr':lr,'gradient_norm':float(norm),
                    'source_gradient_L1':source_grad,'microbatches':observations},allow_nan=False)+'\n')
                stream.flush()
                if step%16==0 or mode=='pilot':
                    write_json(root/'progress.json',{'step':step,'updates':steps,'mode':mode})
                    print(json.dumps({'repair_step':step,'updates':steps,'mode':mode,'first_losses':observations[0]['metrics']}),flush=True)
                if mode=='train' and step in (1024,2048): save(root,f'step_{step}.pt',corrector,opt,sampler,step,p)
        final = diagnostics(corrector,renderer,probe_rows,loader,tokenizer)
        final_hash = native.vendor.module_hash(corrector)
        save(root,'model.pt',corrector,opt,sampler,steps,p)
        del corrector,opt
        gc.collect()
        torch.cuda.empty_cache()
        saved = torch.load(root/'model.pt',map_location='cpu',weights_only=True)
        corrector = mechanism.AttentionCodeRepair().cuda()
        corrector.load_state_dict(saved['model'],strict=True)
        opt = torch.optim.AdamW(corrector.parameters(),lr=Config().max_lr,betas=(.9,.999),eps=1e-8,weight_decay=.01)
        opt.load_state_dict(saved['optimizer'])
        sampler.set_state(saved['sampler_rng'])
        torch.set_rng_state(saved['CPU_rng'])
        torch.cuda.set_rng_state_all(saved['CUDA_rng'])
        same = preparation.checkpoint.training.lifecycle.same
        if (not same(corrector.state_dict(),saved['model']) or not same(opt.state_dict(),saved['optimizer'])
                or not torch.equal(sampler.get_state(),saved['sampler_rng'])
                or not torch.equal(torch.get_rng_state(),saved['CPU_rng'])
                or not same(torch.cuda.get_rng_state_all(),saved['CUDA_rng'])
                or diagnostics(corrector,renderer,probe_rows,loader,tokenizer)!=final
                or native.vendor.module_hash(corrector)!=final_hash or final_hash==initial_hash
                or native.vendor.module_hash(renderer)!=frozen_hash):
            raise ValueError('strict corrector/optimizer/RNG/hard-output reload or frozen renderer failed')
        peak = torch.cuda.max_memory_allocated()
        if peak>4_900_000_000: raise ValueError('corrector training exceeds4.9GB allocation')
        write_json(root/'outputs.json',final)
        native.vendor.verify(p)
        write_json(root/'result.json',{'status':'corrector_'+mode+'_completed_semantic_quality_pending',
            'updates':steps,'train_pool_cases':len(selected),'strict_reload_passed':True,
            'initial_corrector_sha256':initial_hash,'final_corrector_sha256':final_hash,
            'frozen_renderer_sha256':frozen_hash,'checkpoint_sha256':file_hash(root/'model.pt'),
            'trainable_parameters':sum(v.numel() for v in corrector.parameters()),
            'peak_allocated_bytes':peak,'TRAIN_diagnostic_correct_codes':sum(v['correct_codes'] for v in final),
            'TRAIN_diagnostic_code_positions':16*641,'TRAIN_diagnostic_root_correct':sum(v['root_correct'] for v in final),
            'task_accuracy':None,'final_test_opened':False,'speed_tested':False,'feature_cache':loader.metrics()})
    finally:
        renderer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','pilot','capacity','train'))
    stage=parser.parse_args().stage
    prepare() if stage=='prepare' else train(stage)
