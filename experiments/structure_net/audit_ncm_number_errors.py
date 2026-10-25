"""Exploratory number-token audit of saved open TRAIN/DEV outputs; no training."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re


ROOT = Path('data/experiments/ncm_paper_regime')
DEST = ROOT / '20261025_ncm_number_error_audit_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_kind(piece):
    if any(c.isdigit() for c in piece):
        return 'digit_bearing'
    if any(c.isalpha() for c in piece):
        return 'other_alphabetic'
    return 'punctuation'


def rates(counts):
    return {name: {'correct': value[0], 'total': value[1],
                   'fraction': value[0] / value[1] if value[1] else None}
            for name, value in sorted(counts.items())}


def compare(rows, outputs, tokenizer, subset=None):
    counts = {name: [0, 0] for name in ('digit_bearing', 'other_alphabetic', 'punctuation')}
    examples = []
    for row in rows:
        if subset is not None and row['id'] not in subset:
            continue
        expected, actual = row['native_word_ids'], outputs[row['id']]['raw_word_ids']
        if len(expected) != 128 or len(actual) != 128:
            raise ValueError('Complete native frames required')
        groups = {name: [0, 0] for name in counts}
        for gold, prediction in zip(expected, actual):
            if gold in tokenizer.all_special_ids:
                continue
            name = token_kind(tokenizer.convert_ids_to_tokens(gold))
            groups[name][0] += int(gold == prediction)
            groups[name][1] += 1
        for name, value in groups.items():
            counts[name][0] += value[0]
            counts[name][1] += value[1]
        examples.append({'id': row['id'], 'source_component': row['source_component'],
            'target': row['target_text'], 'output': outputs[row['id']]['text'],
            'position_matches': rates(groups)})
    return rates(counts), examples


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or DEST.exists():
        raise ValueError('CPU-only execution and fresh artifact root required')
    from experiments.structure_net.ncm_pixi import configure
    training, runtime = configure()
    tokenizer = training.native.tokenizer()
    paths = {
        'train': ROOT / '20261025_engineering_attention_repair732_data_v1/cases.json',
        'generated': ROOT / '20261025_sequential_full732_pixi_train_v1/outputs.json',
        'oracle': ROOT / '20261025_sequential_decoder_train_v1/oracle_outputs.json',
        'review': ROOT / '20261025_pixi_data_review_v1/records.jsonl',
        'train_result': ROOT / '20261025_sequential_full732_pixi_train_v1/result.json',
        'decoder_result': ROOT / '20261025_sequential_decoder_train_v1/result.json',
        'tokenizer': training.native.RELEASE / 'files/lm1b/tokenizer.pt',
    }
    hashes = {k: digest(v) for k, v in paths.items()}
    rows = json.loads(paths['train'].read_text())
    generated = {r['id']: r for r in json.loads(paths['generated'].read_text())}
    oracle = {r['id']: r for r in json.loads(paths['oracle'].read_text())}
    if len(rows) != 732 or {r['id'] for r in rows} != set(generated) or set(oracle) != set(generated):
        raise ValueError('All 732 cases must be paired')
    for row in rows:
        for field in ('source', 'target'):
            text = row['source'] if field == 'source' else row['target_text']
            if hashlib.sha256(text.encode()).hexdigest() != row[field + '_sha256']:
                raise ValueError('Original text hash mismatch')
        ids = tokenizer.encode(row['target_text'], add_special_tokens=False)
        if row['native_word_ids'] != [101, *ids, 101] + [0] * (126 - len(ids)):
            raise ValueError('Original complete target frame mismatch')
        if oracle[row['id']]['source_only'] or oracle[row['id']]['codes'] != row['oracle_codes']:
            raise ValueError('Oracle control must use the same correct codes')
    numeric_ids = {r['id'] for r in rows if any(c.isdigit() for c in r['target_text'])}
    generated_rates, details = compare(rows, generated, tokenizer)
    oracle_rates, _ = compare(rows, oracle, tokenizer)
    numeric_row_rates, _ = compare(rows, generated, tokenizer, numeric_ids)
    vocab = tokenizer.get_vocab()
    numeric_vocab = {p: i for p, i in vocab.items() if token_kind(p) == 'digit_bearing'}
    numeric_vocab_ids = set(numeric_vocab.values())
    trained_counts = Counter(i for r in rows for i in r['native_word_ids'] if i in numeric_vocab_ids)
    probes = []
    for text in ('95%', '94%', '1,024', '1024', '253,366', '13,843', '0.000123', '7.25 GB', '-3.14', '2026-10-25'):
        ids = tokenizer.encode(text, add_special_tokens=False)
        probes.append({'text': text, 'ids': ids, 'pieces': tokenizer.convert_ids_to_tokens(ids),
                       'rendered': tokenizer.decode(ids), 'has_UNK': tokenizer.unk_token_id in ids})
    review = [json.loads(line) for line in paths['review'].read_text().splitlines()]
    dev = [r for r in review if r['split'].startswith('DEV')]
    if len(dev) != 128:
        raise ValueError('All open DEV cases required')
    no_digit_outputs = [r for r in dev if not re.search(r'\d', r['model_summary'])]
    result = {
        'schema': 'ncm.number-error-exploratory-audit.v1',
        'status': 'saved_output_diagnostic_complete',
        'classification': 'number_tokens_underperform_other_alphabetic_tokens_on_TRAIN',
        'scope': 'Post-hoc audit; no new training, model forward, judge calls or sealed test access',
        'measurement': 'Exact native token identity at the original target position, excluding special tokens',
        'limitations': [
            'Position agreement is not numerical fact accuracy or semantic summary quality.',
            'Digit-bearing tokens include identifiers, versions and dates; spelled-out numbers are excluded.',
            'Commas, decimal points, signs and units are not included in the digit-bearing token score.',
            'The oracle control is TRAIN reconstruction; unseen-number generalization remains untested.',
            'Training references are stored summaries, not independently verified gold.',
            'Class comparisons are descriptive and not controlled for rarity or position.',
        ],
        'TRAIN_cases': len(rows), 'TRAIN_cases_with_digits': len(numeric_ids),
        'generated_TRAIN_token_position_agreement': generated_rates,
        'oracle_TRAIN_token_position_agreement': oracle_rates,
        'generated_same_number_containing_cases': numeric_row_rates,
        'tokenizer': type(tokenizer).__name__, 'vocabulary_size': len(vocab),
        'digit_bearing_vocabulary_entries': len(numeric_vocab),
        'distinct_digit_bearing_tokens_in_TRAIN_targets': len(trained_counts),
        'number_encoding_probes': probes,
        'DEV_outputs_without_digits': len(no_digit_outputs),
        'DEV_outputs_without_digits_semantic_passes': sum(r['judge']['passed'] for r in no_digit_outputs),
        'DEV_cases': len(dev),
        'input_hashes': {str(paths[k]): v for k, v in hashes.items()},
        'producer_sha256': digest(Path(__file__)), 'runtime_executable': runtime['executable'],
        'task_accuracy_reestimated': False,
    }
    if generated_rates['digit_bearing']['fraction'] >= generated_rates['other_alphabetic']['fraction']:
        result['classification'] = 'number_specific_deficit_not_observed'
    if any(digest(paths[k]) != value for k, value in hashes.items()):
        raise ValueError('Inputs changed during audit')
    DEST.mkdir()
    (DEST / 'producer.py').write_bytes(Path(__file__).read_bytes())
    (DEST / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (DEST / 'cases.json').write_text(json.dumps(details, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('input_hashes', 'number_encoding_probes')}, indent=2))


if __name__ == '__main__':
    main()
