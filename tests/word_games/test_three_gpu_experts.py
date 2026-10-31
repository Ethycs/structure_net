import re
import pytest
from experiments.word_games.three_gpu_experts import command, verify, SMALL


def test_two_2060_banks_preserve_attention_and_weight_precision():
    args=command(10,18088,'test')
    assert SMALL in args[args.index('--gpus')+1]
    for flag,value in {'--device':'CUDA0,CUDA1,CUDA2','--tensor-split':'1,0,0',
                       '--n-cpu-moe':'0','--cache-type-k':'q8_0','--cache-type-v':'q8_0'}.items():
        assert args[args.index(flag)+1]==value
    overrides=args[args.index('--override-tensor')+1].split(',')
    for i in range(40):
        matches=[dev for rule in overrides for pattern,dev in [rule.split('=')]
                 if re.search(pattern,f'blk.{i}.ffn_up_exps.weight')]
        expected=['CUDA1'] if i<8 else ['CUDA2'] if i<18 else ['CPU'] if i<26 else []
        assert matches==expected


def test_placement_verifier_requires_every_projection_on_correct_gpu():
    log='\n'.join(f'layer {i} assigned to device CUDA0, detail' for i in range(42))
    log+='\n'+'\n'.join(f'tensor blk.{i}.ffn_{part}_exps.weight (144 MiB q4_K) buffer type overridden to '+
                        ('CUDA1' if i<8 else 'CUDA2' if i<18 else 'CUDA_Host')
                        for i in range(26) for part in ('up','gate','down'))
    assert verify(log,10)['cpu_layers']==list(range(18,26))
    with pytest.raises(ValueError):verify(log.replace('CUDA2','CUDA1'),10)
    with pytest.raises(ValueError):command(12,1,'test')


def test_unload_waits_for_two_actual_free_memory_samples(monkeypatch,tmp_path):
    from experiments.word_games import three_gpu_experts as m
    values=iter((93,5507,93,5507,5507));sleeps=[]
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**kw:f'{SMALL}, {next(values)}\n')
    monkeypatch.setattr(m.time,'sleep',lambda n:sleeps.append(n))
    m.wait_for_release(tmp_path)
    result=m.read(tmp_path/'release_check.json')
    assert result['released'] and len(result['samples'])==5 and len(sleeps)==4
