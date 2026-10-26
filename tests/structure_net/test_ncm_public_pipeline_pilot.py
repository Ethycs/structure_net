from dataclasses import replace

import torch

from experiments.structure_net import ncm_public_pipeline_pilot as pilot


def test_actual_byte_comparison_covers_both_populations_and_tail():
    config = replace(pilot.curriculum.CONFIG, public_cases=11, anchor_cases=3,
                     epochs=2, effective_examples=8)
    epoch = pilot.curriculum.Epoch(1, config)
    class Sources:
        def fetch(self, refs):
            return [{'id': r.population+str(r.index), 'source_sha256': str(r.index),
                     'native_word_ids': [r.index], 'oracle_codes': [r.index],
                     'presentation': pilot.asdict(r), 'features': torch.full((r.index+1, 3), float(r.index))}
                    for r in refs]
    value = pilot.pipeline_identity(Sources(), epoch)
    assert value['all_records_equal']
    assert [r['denominator'] for r in value['batches']] == [8, 8, 7, 7]
    assert sum(len(r['positions']) for r in value['batches']) == 15
