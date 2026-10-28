"""Render the four preselected before/after controls from recovered-data training."""
import hashlib
import json
from pathlib import Path

from experiments.structure_net import ncm_lm1b_native_development as rendering

BASE = Path('data/experiments/ncm_paper_regime')
TRAIN = BASE/'20261028_recovered_paraphrase_training_train_v2'
AUDIT = BASE/'20261028_recovered_paraphrase_training_audit_v2'
CATALOG = BASE/'20261028_recovered_paraphrase_catalog_v1'
ROOT = BASE/'20261028_recovered_saved_examples_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def result(root):
    receipt = read(root/'actual_tool_receipt.json')
    if (receipt['actual_tool_exit_code'] != 0
            or receipt['artifacts'].get(str(root/'result.json')) != sha(root/'result.json')):
        raise ValueError('completed artifact-bound result required')
    return read(root/'result.json')


def indexed_rows(stem, keys, catalog):
    """Read only requested payloads; verify each against the bound byte index."""
    index = CATALOG/(stem+'_index.jsonl')
    if catalog['artifacts'][str(index)] != sha(index):
        raise ValueError('catalog index changed')
    rows = {}
    with index.open() as entries, (CATALOG/(stem+'.jsonl')).open('rb') as stream:
        for entry in map(json.loads, entries):
            if entry['id'] not in keys:
                continue
            stream.seek(entry['offset_bytes'])
            payload = stream.read(entry['length_bytes'])
            if hashlib.sha256(payload).hexdigest() != entry['record_sha256']:
                raise ValueError('selected record bytes changed')
            value = json.loads(payload)
            if value['id'] != entry['id'] or value['id'] in rows:
                raise ValueError('selected record identity differs')
            rows[value['id']] = value
    if set(rows) != keys:
        raise ValueError('all preselected controls required')
    return rows


def run():
    if ROOT.exists():
        raise ValueError('fresh descriptive export root required')
    trained, audited, catalog = result(TRAIN), result(AUDIT), result(CATALOG)
    if (trained['updates'] != 6416
            or audited['status'] != 'recovered_full_training_replayed'
            or audited['source_checkpoint_sha256'] != trained['checkpoint_sha256']):
        raise ValueError('latest completed and independently audited training required')
    names = ('controls.json', 'initial_outputs.json', 'final_outputs.json')
    for name in names:
        if trained['artifacts'][str(TRAIN/name)] != sha(TRAIN/name):
            raise ValueError('saved training control changed')
    controls, initial, final = (read(TRAIN/name) for name in names)
    if len(controls) != 4 or any(r['population'] != 'recovered' for r in controls):
        raise ValueError('all four fixed recovered-data controls required')
    pairs = indexed_rows('pairs', {r['id'] for r in controls}, catalog)
    sources = indexed_rows('sources', {r['source_record_id'] for r in pairs.values()}, catalog)
    tokenizer = rendering.native.tokenizer()
    rows = []
    for control, before, after in zip(controls, initial, final, strict=True):
        pair = pairs[control['id']]
        source = sources[pair['source_record_id']]
        if (hashlib.sha256(source['source'].encode()).hexdigest() != pair['source_sha256']
                or hashlib.sha256(pair['target_text'].encode()).hexdigest() != pair['target_sha256']):
            raise ValueError('original source/target binding differs')
        rows.append({**control, 'source': source['source'],
            'original_role': source['original_role'], 'original_created_at': source['original_created_at'],
            'reference': pair['target_text'], 'reference_kind': pair['reference_kind'],
            'before': rendering.decode(before['raw_word_ids'], tokenizer),
            'after': rendering.decode(after['raw_word_ids'], tokenizer),
            'changed_code_positions': sum(a != b for a, b in zip(before['codes'], after['codes'], strict=True)),
            'changed_raw_word_positions': sum(a != b for a, b in zip(before['raw_word_ids'], after['raw_word_ids'], strict=True))})
    paths = [Path(__file__).resolve(), Path(rendering.__file__).resolve(),
             Path(rendering.framed.__file__).resolve(), Path(rendering.native.__file__).resolve(),
             *[p for p in rendering.native.INVENTORY.joinpath('bert-tokenizer').iterdir() if p.is_file()],
             *[TRAIN/n for n in names], CATALOG/'pairs_index.jsonl', CATALOG/'sources_index.jsonl',
             *[root/n for root in (TRAIN, AUDIT, CATALOG) for n in ('result.json', 'actual_tool_receipt.json')]]
    evidence = {'status': 'saved_preselected_TRAIN_before_after_examples_rendered',
        'updates': 6416, 'checkpoint_sha256': trained['checkpoint_sha256'],
        'selection': 'all four controls selected before training: first, shortest source, longest source, longest target',
        'TRAIN_only': True, 'representative_random_sample': False,
        'new_model_calls': 0, 'new_judge_calls': 0, 'task_accuracy': None, 'final_test_opened': False,
        'rows': rows, 'artifacts': {str(p): sha(p) for p in paths}}
    ROOT.mkdir()
    write(ROOT/'examples.json', evidence)
    blocks = ['Latest recovered-data run: 6,416 updates, 43,894 training pairs.\n'
              'All four controls were selected before training. These are TRAIN examples, not held-out accuracy.\n'
              'The reference is a historical paraphrase, not independently verified factual gold.\n'
              'Before/after texts use the unchanged native renderer; no output repair or model calls.']
    for i, row in enumerate(rows, 1):
        blocks.append(f"EXAMPLE {i} — {row['id']}\n\nSOURCE\n{row['source']}\n\n"
                      f"REFERENCE\n{row['reference']}\n\nBEFORE TRAINING\n{row['before']['text']}\n\n"
                      f"AFTER TRAINING\n{row['after']['text']}")
    with (ROOT/'examples.txt').open('x') as stream:
        stream.write('\n\n' + ('\n\n' + '='*72 + '\n\n').join(blocks) + '\n')
    print(json.dumps({k: v for k, v in evidence.items() if k != 'artifacts'}, ensure_ascii=False))


if __name__ == '__main__':
    run()
