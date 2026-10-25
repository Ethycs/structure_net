"""Independent arithmetic audit and explicit TRAIN text-review record."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path('data/experiments/ncm_paper_regime')
RUN=ROOT/'20261024_sequential_engineering16_capacity_v1'
DEST=ROOT/'20261024_sequential_engineering16_review_v1'
DATA=ROOT/'20261025_engineering_attention_repair732_data_v1'


def read(path): return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    if DEST.exists(): raise ValueError('fresh immutable review root required')
    result=read(RUN/'result.json')
    receipt=read(RUN/'actual_tool_receipt.json')
    if receipt['actual_tool_exit_code']!=0 or not receipt['handle_closed']:
        raise ValueError('observed completed worker required')
    if sha(RUN/'model.pt')!=result['checkpoint_sha256']:
        raise ValueError('checkpoint hash differs')
    cases={r['id']:r for r in read(DATA/'cases.json')}
    ids=read(DATA/'selection.json')['diagnostic_ids']
    output,oracle=read(RUN/'outputs.json'),read(RUN/'TRAIN_oracle_controls.json')
    if [r['id'] for r in output]!=ids or [r['id'] for r in oracle]!=ids:
        raise ValueError('unchanged16 TRAIN selection required')
    trace=[json.loads(s) for s in (RUN/'trace.jsonl').read_text().splitlines()]
    counts=Counter(m['id'] for row in trace for m in row['metrics'])
    if ([r['step'] for r in trace]!=list(range(1,513)) or set(counts)!=set(ids)
            or set(counts.values())!={64}):
        raise ValueError('exact512 updates,1024 examples,64 exposures per TRAIN case required')
    matches=sum(a==b for r in output for a,b in zip(r['codes'],cases[r['id']]['oracle_codes']))
    word_matches=sum(a==b for r in output for a,b in zip(r['raw_word_ids'],cases[r['id']]['native_word_ids']) if b!=0)
    if matches!=result['final']['rollout_correct_codes'] or word_matches!=result['final']['active_word_matches']:
        raise ValueError('reported final code/word counts differ from saved outputs')
    comments=[
        'Hierarchical pipeline phrase is damaged; domain specialization becomes domain settings.',
        'LLM approaches becomes rem approaches; implementations omitted.',
        'Reference content preserved in the decoded text.',
        'Archived qualifier omitted; principal proposal preserved.',
        'Reference content preserved in the decoded text.',
        'Empirical and entropy qualifiers lost; numerical targets retained.',
        'Trainers omitted from the interface list.',
        'Reference content preserved in the decoded text.',
        'Reference content preserved in the decoded text.',
        'Reference content preserved in the decoded text.',
        'Similarity thresholds becomes threshold thresholds.',
        'Tensor products becomes aggregation products.',
        'Dependency-aware becomes perspectives-aware; manual pinning becomes manual dashboard.',
        'Semantic qualifier omitted; dependency becomes perspectives.',
        'Commits omitted after five unmerged;394 tests and seven MCP tools retained.',
        'Converting omitted; listed mechanisms and decay endpoints retained.',
    ]
    pairs=[{'id':r['id'],'reference':cases[r['id']]['target_text'],'generated':r['text'],
        'reference_code_rendering':o['text'],'matches_reference_code_text':r['text']==o['text'],
        'review_note':note,'review_scope':'assistant inspection against supplied TRAIN reference; not independent semantic grading'}
        for r,o,note in zip(output,oracle,comments)]
    numeric=result['numeric_learnability_pass']
    empty=sum(not r['text'].strip() for r in output)
    material=[ids[i] for i in (1,5,10,11,12,13,14)]
    artifacts=[RUN/n for n in ('producer.json','result.json','actual_tool_receipt.json','outputs.json',
        'initial_outputs.json','shuffled_source_outputs.json','TRAIN_oracle_controls.json','trace.jsonl','parity.json')]
    artifacts += [DATA/n for n in ('cases.json','selection.json')]+[Path(__file__)]
    DEST.mkdir()
    write(DEST/'reviewed_pairs.json',pairs)
    summary={'schema':'nal.ncm-sequential-engineering-review.v1','hypothesis_id':result['hypothesis_id'],
        'conformance':'EXPLORATORY','classification':'native_generator_fits_TRAIN_codes_frozen_rendering_quality_failed',
        'confirmed':False,'completed_training_runs':1,'completed_pilots':1,'updates':512,'train_cases':16,
        'numeric_learnability_pass':numeric,'empty_outputs':empty,'decoded_text_reviewed':True,
        'decoded_text_adequate_to_scale':False,'material_reference_deviation_examples':material,
        'full732_training_admitted':False,'code_matches':matches,'code_positions':10256,
        'word_matches':word_matches,'active_word_positions':672,
        'all_decoded_texts_equal_reference_code_rendering':all(r['matches_reference_code_text'] for r in pairs),
        'source_only_reference_code_agreement':matches/10256,
        'source_only_TRAIN_word_agreement':word_matches/672,
        'shuffled_source_TRAIN_word_agreement':result['shuffled_source']['active_word_agreement'],
        'initial_teacher_CE':result['initial']['mean_teacher_CE'],'final_teacher_CE':result['final']['mean_teacher_CE'],
        'all_roots_correct':all(r['levels'][0]['rollout_correct']==1 for r in output),
        'all_errors_in_last_level':sum(r['levels'][-1]['positions']-r['levels'][-1]['rollout_correct'] for r in output)==10256-matches,
        'trainable_parameters':result['trainable_parameters'],'peak_allocated_bytes':result['peak_allocated_bytes'],
        'optimizer_loop_seconds':trace[-1]['elapsed_seconds'],
        'training_and_postflight_seconds':result['training_seconds'],
        'checkpoint_sha256':result['checkpoint_sha256'],'strict_reload_passed':result['strict_reload_passed'],
        'task_accuracy':None,'heldout_quality_measured':False,'speed_tested':False,'final_test_opened':False,
        'comparison_boundary':'Versus prior repair, native capacity, objective, optimizer and exposures change; cannot attribute gain solely to sequential execution.',
        'noise_counter_boundary':'Trace history_corruptions includes final-level replacements that have no subsequent prediction; actual conditioning uses only earlier levels.',
        'artifacts':{str(p):sha(p) for p in artifacts},'reviewed_pairs_sha256':sha(DEST/'reviewed_pairs.json')}
    write(DEST/'campaign_results.json',summary)
    print(json.dumps(summary,allow_nan=False))


if __name__=='__main__': main()
