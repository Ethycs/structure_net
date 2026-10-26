"""Preserve and account for incomplete XSum rows after v1's fail-fast finding.

The pinned official TRAIN shard has an empty document at row 7686, article
30280969. Version 1 stopped there before any model training. This data-only
amendment exports empty strings unchanged and counts them; the unchanged token
screen excludes them. It does not lower training or evaluation requirements.
Run with the same Pixi environments as v1, replacing the module name with v2.
"""
import argparse
from collections import Counter
from pathlib import Path

from experiments.structure_net import ncm_public_xsum_data as original

FAILED = original.EXPORT
EXPORT = original.ROOT/'20261026_public_xsum_train_export_v2'
INSPECT = original.ROOT/'20261026_public_xsum_train_inspection_v2'
TEST = Path('tests/structure_net/test_ncm_public_xsum_data_v2.py')


def validate_shape(row):
    if set(row) != {'document', 'summary', 'id'}:
        raise ValueError('unexpected public dataset fields')
    if any(not isinstance(value, str) for value in row.values()):
        raise ValueError('non-string field: preserve failure before choosing a null policy')
    if not row['id'].isascii() or not row['id'].isdigit():
        raise ValueError('expected BBC numeric article identifier')


def configure():
    receipt = original.read(FAILED/'actual_tool_receipt.json')
    diagnostic = original.read(FAILED/'first_invalid_row_diagnostic.json')
    if (receipt['actual_tool_exit_code'] != 1 or receipt['tool_chunk'] != 'b63338'
            or diagnostic['row_number'] != 7686 or diagnostic['id'] != '30280969'
            or diagnostic['fields']['document']['characters'] != 0):
        raise ValueError('preserved original empty-document failure required')
    original.verify(original.read(FAILED/'producer.json')['input_hashes'])
    old_start, old_finish = original.start, original.finish
    counts = Counter()

    def validate(row):
        validate_shape(row)
        counts['complete_rows' if row['document'].strip() and row['summary'].strip()
               else 'incomplete_rows'] += 1
        if not row['document'].strip():
            counts['empty_document_rows'] += 1
        if not row['summary'].strip():
            counts['empty_summary_rows'] += 1

    def start(root, stage, inputs, environment, config):
        return old_start(root, stage, [*inputs, Path(__file__), TEST,
            *[FAILED/name for name in ('producer.json', 'failure.json', 'actual_tool_receipt.json',
                                      'first_invalid_row_diagnostic.json')]], environment,
            {**config, 'data_preparation_amendment': 'preserve_and_exclude_empty_strings_v2',
             'first_known_empty_document_row': 7686, 'original_failure_preserved': True,
             'training_thresholds_changed': False})

    def finish(root, hashes, result):
        return old_finish(root, hashes, {**result, 'row_completeness': dict(counts),
            'empty_strings_preserved_and_ineligible': True})

    original.EXPORT, original.INSPECT = EXPORT, INSPECT
    original.validate_row, original.start, original.finish = validate, start, finish


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('export', 'inspect'))
    stage = parser.parse_args().stage
    configure()
    target = EXPORT if stage == 'export' else INSPECT
    existed = target.exists()
    try:
        original.export() if stage == 'export' else original.inspect()
    except Exception as error:
        if not existed and target.exists():
            original.write(target/'failure.json', {'error_type': type(error).__name__,
                'error': str(error), **original.BOUNDARIES})
        raise
