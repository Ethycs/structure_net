import re
import pytest
from experiments.neural_architecture_lab.run_word_games_packed_experts import GROUPS,packed_command,packed_verify


def test_packed_layout_covers_all_overridden_banks_once():
    assert sorted(i for ids in GROUPS.values() for i in ids)==list(range(26))
    assert len(GROUPS['CPU'])==2
    args=packed_command(10,18088,'test')
    assert args[args.index('--tensor-split')+1]=='1,0,0'
    rules=args[args.index('--override-tensor')+1].split(',')
    for dev,ids in GROUPS.items():
        for i in ids:
            found=[d for r in rules for pattern,d in [r.split('=')] if re.search(pattern,f'blk.{i}.ffn_down_exps.weight')]
            assert found==[dev]


def test_packed_verifier_requires_unchanged_attention_and_complete_projections():
    log='\n'.join(f'layer {i} assigned to device CUDA0, detail' for i in range(42))
    log+='\n'+'\n'.join(f'tensor blk.{i}.ffn_{part}_exps.weight (144 MiB q4_K) buffer type overridden to {dev}'
                        for dev,ids in GROUPS.items() for i in ids for part in ('up','gate','down'))
    assert packed_verify(log,10)['gpu_banks']==38
    with pytest.raises(ValueError):packed_verify(log.replace('blk.23.ffn_up_exps','blk.24.ffn_up_exps'),10)
