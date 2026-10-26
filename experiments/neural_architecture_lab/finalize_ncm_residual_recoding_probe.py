"""Independently replay the frozen-codebook probe's bindings and readout counts."""
import json
from pathlib import Path

from experiments.structure_net import ncm_residual_recoding_probe as probe

DEST = probe.study.runtime.ROOT / '20261025_residual_recoding_probe_results_v1'


def main():
    if DEST.exists():
        raise ValueError('fresh append-only diagnostic aggregate required')
    training, _ = probe.study.runtime.configure()
    q = training.q
    for root in probe.ROOTS.values():
        training.preparation.completed(root)
    full = q.read(probe.ROOTS['full'] / 'outputs.json')
    pilot = q.read(probe.ROOTS['pilot'] / 'outputs.json')
    ids = q.read(probe.study.RUNS['prepare'] / 'selection.json')['diagnostic_ids']
    cases = {r['id']: r for r in q.read(probe.study.RUNS['prepare'] / 'cases.json')}
    predictions = {r['id']: r for r in q.read(probe.study.RUNS['train'] / 'outputs.json')}
    ordered = [(i, k, a) for i in ids for k in probe.PREFIX_LEVELS
               for a in ('original_suffix', 'adaptive_suffix')]
    actual = [(r['id'], r['prefix_levels'], r['arm']) for r in full]
    if len(ids) != 16 or actual != ordered:
        raise ValueError('complete ordered16 by8 by2 comparisons required')
    lookup = {k: r for k, r in zip(actual, full)}
    if any(lookup[r['id'], r['prefix_levels'], r['arm']] != r for r in pilot):
        raise ValueError('pilot/full identity differs')
    tokenizer = training.native.tokenizer()
    levels = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)
    for r in full:
        case, prediction = cases[r['id']], predictions[r['id']]
        if (case['split'] != 'TRAIN' or r['target_sha256'] != case['target_sha256']
                or r['source_sha256'] != case['source_sha256']
                or prediction['source_id_used'] != r['id']):
            raise ValueError('unchanged TRAIN target/source binding required')
        end = sum(levels[:r['prefix_levels']])
        codes, gold = r['codes'], case['oracle_codes']
        if (len(codes) != 641 or any(type(x) is not int or not 0 <= x < 8192 for x in codes)
                or codes[:end] != prediction['codes'][:end]
                or r['arm'] == 'original_suffix' and codes[end:] != gold[end:]
                or r['suffix_changes_from_original'] != sum(a != b for a,b in zip(codes[end:],gold[end:]))):
            raise ValueError('forced-prefix or fixed-suffix code contract differs')
        if r['prefix_levels'] in (0, 12):
            other = lookup[r['id'], r['prefix_levels'], 'original_suffix']
            if any(r[k] != other[k] for k in ('codes', 'codec_only', 'with_source', 'latent_MSE')):
                raise ValueError('endpoint equality control differs')
        for mode in ('codec_only', 'with_source'):
            value, target = r[mode], case['native_word_ids']
            raw = value['raw_word_ids']
            if len(raw) != 128 or any(type(x) is not int or not 0 <= x < 30528 for x in raw):
                raise ValueError('complete native raw readout required')
            positions = [i for i,t in enumerate(target) if t != 0]
            expected = {'active_words':len(positions),
                'active_word_matches':sum(raw[i] == target[i] for i in positions),
                'exact_active_frame':all(raw[i] == target[i] for i in positions),
                **training.preparation.rendering.decode(raw,tokenizer)}
            if any(value[k] != v for k,v in expected.items()):
                raise ValueError('raw token metric or rendering replay differs')
            if mode == 'with_source' and r['prefix_levels'] == 12 and raw != prediction['raw_word_ids']:
                raise ValueError('source-only baseline readout differs')
    metrics = []
    for count in probe.PREFIX_LEVELS:
        for arm in ('original_suffix', 'adaptive_suffix'):
            rows = [lookup[i,count,arm] for i in ids]
            item = {'prefix_levels':count,'arm':arm,'cases':16,
                'mean_latent_MSE':sum(r['latent_MSE'] for r in rows)/16}
            for mode in ('codec_only', 'with_source'):
                total = sum(r[mode]['active_words'] for r in rows)
                correct = sum(r[mode]['active_word_matches'] for r in rows)
                item[mode] = {'active_words':total,'active_word_matches':correct,
                    'active_word_agreement':correct/total,
                    'exact_active_frames':sum(r[mode]['exact_active_frame'] for r in rows)}
            metrics.append(item)
    reported = q.read(probe.ROOTS['full'] / 'result.json')
    if (reported['metrics'] != metrics or not reported['model_unchanged']
            or reported['outputs_sha256'] != probe.study.file_hash(probe.ROOTS['full']/'outputs.json')
            or reported['task_accuracy'] is not None or reported['training_updates'] != 0
            or reported['final_test_opened'] or reported['source_only_inference']):
        raise ValueError('diagnostic aggregate or scope differs')
    deltas = []
    for count in probe.PREFIX_LEVELS:
        changes = [lookup[i,count,'adaptive_suffix']['with_source']['active_word_matches']
                   - lookup[i,count,'original_suffix']['with_source']['active_word_matches'] for i in ids]
        deltas.append({'prefix_levels':count,'improved_cases':sum(d>0 for d in changes),
            'worsened_cases':sum(d<0 for d in changes),'unchanged_cases':sum(d==0 for d in changes),
            'net_active_token_matches':sum(changes)})
    paths = [Path(__file__), *[p for root in probe.ROOTS.values()
        for p in root.glob('*.json')]]
    result = {'hypothesis_id':'ncm-residual-recoding-probe-v1',
        'classification':'oracle_residual_recoding_capacity_measured',
        'cases':16,'comparisons':256,'readouts':512,'metrics':metrics,'paired_changes':deltas,
        'oracle_target_access':True,'source_only_inference':False,'task_accuracy':None,
        'training_updates':0,'new_judge_requests':0,'final_test_opened':False,
        'final_task_goal_achieved':False,'confirmed':False,'conformance':'EXPLORATORY_UNDERPOWERED',
        'independently_replayed':'all bindings, prefix identities, raw token metrics, renderings and aggregates',
        'latent_MSE':'producer measurement aggregated; no independent encoder rerun',
        'artifacts':{str(p):probe.study.file_hash(p) for p in paths}}
    DEST.mkdir()
    probe.study.write_json(DEST/'campaign_results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','artifacts')}),flush=True)


if __name__ == '__main__':
    main()
