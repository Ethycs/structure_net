import json

import pytest

from experiments.structure_net import ncm_source_hierarchy_training as study


def test_actual_qualified_tree_updates_replay():
    # Use the saved real CUDA training records, not a reconstructed synthetic trace.
    rows=study.data.read(study.broad.RUNS['prepare']/'cases.json')
    counts=study.replay_trace(study.qualified_tree.ROOT/'trace.jsonl',rows,2)
    assert counts['presentations']==64 and counts['noisy_presentations']==32
    assert counts['distinct_sources_visited']==64


def test_modified_real_corruption_count_is_rejected(tmp_path):
    rows=study.data.read(study.broad.RUNS['prepare']/'cases.json')
    records=[json.loads(line) for line in (study.qualified_tree.ROOT/'trace.jsonl').read_text().splitlines()]
    records[0]['microbatches'][0]['history_changes'][0]+=1
    path=tmp_path/'tampered.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError,match='code corruption differs'):study.replay_trace(path,rows,2)
