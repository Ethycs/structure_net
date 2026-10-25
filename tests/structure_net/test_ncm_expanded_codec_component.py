"""All original code bindings and whole-source-family exclusions remain."""
from experiments.structure_net import ncm_expanded_codec_component as run


def test_fixed64_codes_and_whole_families_are_held_out():
    run.base.configure()
    rows=run.cases()
    assert len(rows)==64 and run.ARMS==('domain','expanded')
    assert all(len(r['oracle_codes'])==641 and all(0<=c<8192 for c in r['oracle_codes']) for r in rows)
    corpus=run.q.read(run.training.corpus.DATA/'index.json')
    assert not {r['source_component'] for r in rows}&{r['source_component'] for r in corpus}
