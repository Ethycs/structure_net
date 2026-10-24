"""Native raw-text framing retains lexical positions and excludes whole families."""
from copy import deepcopy

import pytest

from experiments.structure_net import ncm_expanded_codec_corpus as run


def test_literal_markers_are_lexical_and_all_positions_survive():
    run.component.configure()
    tokenizer = run.native.tokenizer()
    text = 'Explain [CLS], [PAD], and [SEP] in the tokenizer. ' * 30
    ids, frames = run.frame_text(text, tokenizer)
    assert not set(ids) & {0, 101}
    assert len(frames) > 1
    restored = []
    for frame in frames:
        assert len(frame) == 128 and frame[0] == 101
        end = frame.index(101, 1)
        restored.extend(frame[1:end])
        assert all(v == 0 for v in frame[end+1:])
    assert restored == ids


def test_actual9936_source_split_keeps_every_component_family_out():
    run.component.configure()
    selected, held, proof = run.sources()
    assert len(selected) == 9936 and len(held) == 64
    assert all(v['overlap'] == 0 for v in proof.values())
    rows = run.q.read(run.teacher.data.DEST / 'cases.json')
    bad = deepcopy(rows)
    target = next(r for r in bad if r['id'] not in {r['id'] for r in held})
    target['spans'][0]['turn_text_sha256'] = held[0]['spans'][0]['turn_text_sha256']
    with pytest.raises(ValueError, match='overlap'):
        run.split(bad, held)
