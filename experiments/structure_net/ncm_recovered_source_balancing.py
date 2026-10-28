"""Uniform source epochs, rotating historical labels, and reviewed task anchors."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import resource

from experiments.structure_net import ncm_public_curriculum as shared
from experiments.structure_net import ncm_recovered_paraphrase_catalog as catalog

pool = catalog.pool
ROOT = pool.BASE/'20261028_recovered_source_balancing_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-28_ncm-recovered-source-balancing.md')
TEST = Path('tests/structure_net/test_ncm_recovered_source_balancing.py')
BINDINGS = pool.BASE/'20261027_isolated_task_supervision_preflight_v1/bindings.json'


@dataclass(frozen=True)
class Config:
    recovered_sources: int = 38484
    anchor_cases: int = 2384
    epochs: int = 4
    microbatch: int = 4
    effective_examples: int = 32
    sources_per_anchor: int = 3
    sampler_seed: int = 271142
    variant_seed: int = 272142
    corruption_seed: int = 273142

    @property
    def public_cases(self):return self.recovered_sources
    @property
    def public_per_anchor(self):return self.sources_per_anchor
    @property
    def anchors_per_epoch(self):return math.ceil(self.recovered_sources/self.sources_per_anchor)
    @property
    def presentations_per_epoch(self):return self.recovered_sources+self.anchors_per_epoch
    @property
    def updates_per_epoch(self):return math.ceil(self.presentations_per_epoch/self.effective_examples)
    @property
    def updates(self):return self.epochs*self.updates_per_epoch

    def validate(self):
        for key in ('recovered_sources','anchor_cases','epochs','microbatch','effective_examples','sources_per_anchor'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError('positive integral exposure geometry required')
        if self.effective_examples%self.microbatch:raise ValueError('microbatches must divide the effective update')


CONFIG = Config()


def groups_for(rows, config=CONFIG):
    ids=set();sources=defaultdict(list);components={}
    for i,row in enumerate(rows):
        if row['id'] in ids:raise ValueError('unique target-pair identities required')
        ids.add(row['id']);sha=row['source_sha256'];component=row['source_component']
        if sha in components and components[sha]!=component:raise ValueError('one component per source text required')
        components[sha]=component;sources[sha].append((row['id'],i))
    groups=[]
    for sha in sorted(sources):
        variants=sorted(sources[sha],key=lambda r:(shared.seed(config.variant_seed,sha,r[0]),r[0]))
        if len(variants)>config.epochs:raise ValueError('fixed epochs must cover every target variant')
        groups.append({'source_sha256':sha,'source_component':components[sha],
            'pair_ids':[r[0] for r in variants],'pair_positions':[r[1] for r in variants]})
    if len(groups)!=config.recovered_sources:raise ValueError('complete source population required')
    return groups


class Epoch(shared.Epoch):
    def __init__(self, number, groups, config=CONFIG):
        config.validate()
        if len(groups)!=config.recovered_sources:raise ValueError('complete fixed source groups required')
        positions=[p for g in groups for p in g['pair_positions']]
        if (len(set(positions))!=len(positions) or sorted(positions)!=list(range(len(positions)))
                or any(not g['pair_positions'] or len(g['pair_positions'])>config.epochs for g in groups)):
            raise ValueError('each target belongs to exactly one fully covered source group')
        self.groups=groups
        super().__init__(number,config)

    def example(self, position):
        value=super().example(position)
        if value.population=='public':
            variants=self.groups[value.index]['pair_positions']
            return replace(value,population='recovered',index=variants[(self.number-1)%len(variants)])
        return value


def run():
    if ROOT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('fresh CPU-only scheduling qualification required')
    paths=catalog.receipt_paths(catalog.ROOT)
    bindings=pool.read(BINDINGS);anchor_root=Path(bindings['targets']['full']);anchor_audit=Path(bindings['targets']['full_audit'])
    paths += catalog.receipt_paths(anchor_root)+catalog.receipt_paths(anchor_audit)
    if not pool.read(anchor_audit/'result.json')['training_targets_admitted']:raise ValueError('complete admitted reviewed anchors required')
    anchors=pool.read(anchor_root/'cases.json')
    if len(anchors)!=CONFIG.anchor_cases or len({r['id'] for r in anchors})!=CONFIG.anchor_cases:raise ValueError('all2384 reviewed anchors required')
    rows=[]
    with (catalog.ROOT/'pairs.jsonl').open() as f:
        for row in map(json.loads,f):rows.append({k:row[k] for k in ('id','source_sha256','source_component','source_record_id')})
    if len(rows)!=43894:raise ValueError('all43894 historical pairs required')
    groups=groups_for(rows)
    lookup={r['id']:r for r in rows}
    with (catalog.ROOT/'source_groups.jsonl').open() as f:
        known=list(map(json.loads,f))
    if len(known)!=len(groups):raise ValueError('complete catalog source-group binding required')
    for group,bound in zip(groups,known,strict=True):
        if (group['source_sha256']!=bound['source_sha256'] or group['source_component']!=bound['source_component']
                or sorted(group['pair_ids'])!=sorted(v['pair_id'] for v in bound['variants'])
                or any(v['source_record_id']!=lookup[v['pair_id']]['source_record_id'] for v in bound['variants'])):
            raise ValueError('original source/target/context groups changed')
    paths += [BINDINGS,anchor_root/'cases.json',catalog.ROOT/'pairs.jsonl',catalog.ROOT/'source_groups.jsonl',Path('pixi.lock'),Path('pyproject.toml')]
    proof={'schema':'nal.ncm-recovered-source-balancing.v1','config':asdict(CONFIG),
        'source_hashes':{str(p):pool.file_hash(p) for p in (Path(__file__),Path(shared.__file__),PLAN,TEST)},
        'input_hashes':{str(p):pool.file_hash(p) for p in dict.fromkeys(paths)},
        'model_training':False,'training_admitted':False,'accuracy_measured':False,'final_test_opened':False}
    ROOT.mkdir();(ROOT/'producer_source').mkdir()
    for i,p in enumerate(map(Path,proof['source_hashes'])):(ROOT/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    pool.write(ROOT/'producer.json',proof);pool.write(ROOT/'groups.json',groups)
    counts=Counter();sources=Counter();targets=Counter();anchor_counts=Counter();sizes=Counter()
    with (ROOT/'planned_examples.jsonl').open('x') as out:
        for number in range(1,CONFIG.epochs+1):
            epoch=Epoch(number,groups)
            for update,size,examples in epoch.batches():
                for example in examples:
                    row=rows[example.index] if example.population=='recovered' else anchors[example.index]
                    out.write(json.dumps({**asdict(example),'id':row['id'],'source_sha256':row['source_sha256'],
                        'update_in_epoch':update,'update_denominator':size})+'\n')
            sizes[min(CONFIG.effective_examples,CONFIG.presentations_per_epoch-(CONFIG.updates_per_epoch-1)*CONFIG.effective_examples)]+=1
    with (ROOT/'planned_examples.jsonl').open() as f:
        for number in range(1,CONFIG.epochs+1):
            epoch=Epoch(number,groups)
            for update,size,examples in epoch.batches():
                for example in examples:
                    saved=json.loads(next(f));row=rows[example.index] if example.population=='recovered' else anchors[example.index]
                    expected={**asdict(example),'id':row['id'],'source_sha256':row['source_sha256'],
                        'update_in_epoch':update,'update_denominator':size}
                    if saved!=expected:raise ValueError('planned stream identity/order/noise/denominator differs')
                    counts[example.population]+=1;counts['noisy']+=example.noisy
                    if example.population=='recovered':sources[row['source_sha256']]+=1;targets[row['id']]+=1
                    else:anchor_counts[row['id']]+=1
            cut=CONFIG.updates_per_epoch//2
            resumed=list(epoch.batches(start_update=cut));fresh=list(Epoch(number,groups).batches(start_update=cut))
            if resumed!=fresh:raise ValueError('position-only resume suffix differs')
        if f.read(1):raise ValueError('unaccounted planned presentations')
    if (set(sources.values())!={4} or len(targets)!=43894 or len(anchor_counts)!=2384
            or counts!={'recovered':153936,'anchor':51312,'noisy':102624}):raise ValueError('fixed source-balanced exposure differs')
    for group in groups:
        visits=[targets[k] for k in group['pair_ids']]
        if min(visits)<1 or max(visits)-min(visits)>1:raise ValueError('target rotation failed')
    overlap=set(sources)&{r['source_sha256'] for r in anchors}
    for collection in (proof['input_hashes'],proof['source_hashes']):
        for p,sha in collection.items():
            if pool.file_hash(p)!=sha:raise ValueError('scheduling input changed')
    result={'schema':proof['schema'],'status':'all_source_balanced_presentations_replayed',
        'epochs':4,'updates':CONFIG.updates,'presentations':CONFIG.epochs*CONFIG.presentations_per_epoch,
        'counts':dict(counts),'recovered_sources':len(sources),'source_visits_each':4,'recovered_targets_visited':len(targets),
        'target_visit_histogram':dict(Counter(targets.values())),'anchor_cases_visited':len(anchor_counts),
        'anchor_minimum_visits':min(anchor_counts.values()),'anchor_maximum_visits':max(anchor_counts.values()),
        'exact_source_overlap_with_anchors':len(overlap),'novel_recovered_source_texts_relative_to_anchors':len(sources)-len(overlap),
        'final_update_denominators':dict(sizes),'all_variants_covered':True,'all_resume_suffixes_identical':True,
        'model_training':False,'training_admitted':False,'task_accuracy':None,'new_judge_calls':0,'final_test_opened':False,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'artifacts':{str(p):pool.file_hash(p) for p in ROOT.iterdir() if p.is_file()}}
    if result['peak_RSS_bytes']>catalog.MAX_RSS:raise ValueError('4GB host memory ceiling exceeded')
    pool.write(ROOT/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(__doc__).parse_args();existed=ROOT.exists()
    try:run()
    except Exception as error:
        if ROOT.exists() and not existed:pool.write(ROOT/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
