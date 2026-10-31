import pytest
from experiments.word_games.moe_placement import CONFIGS, Config, command, quality_compare, verify_placement


def test_resident_gpu_split_preserves_precision_and_protected_gpu():
    args=command(CONFIGS['gpu8'],18088,'test')
    for k,v in {'--device':'CUDA1,CUDA0','--n-cpu-moe':'0','--split-mode':'layer',
                '--threads':'12','--flash-attn':'on','--cache-type-k':'q8_0','--cache-type-v':'q8_0'}.items():
        assert args[args.index(k)+1]==v
    assert 'GPU-8cf440df' not in str(args)
    assert args[args.index('--tensor-split')+1]=='7.5,34.5'
    override=args[args.index('--override-tensor')+1]
    import re
    regex=override.split('=')[0]
    assert all(bool(re.search(regex,f'blk.{i}.ffn_gate_exps.weight'))==(8<=i<26) for i in range(42))


def test_placement_rejects_missing_or_wrong_device():
    config=CONFIGS['gpu8']
    log='\n'.join(f'layer {i} assigned to device CUDA{int(i<8)}, is_swa = 0' for i in range(42))
    log+='\n'+'\n'.join(f'tensor blk.{i}.ffn_{p}_exps.weight (144 MiB q4_K) buffer type overridden to CUDA_Host'
                        for i in range(8,26) for p in ('up','gate','down'))
    assert verify_placement(log,config)['cpu_layers']==list(range(8,26))
    for wrong in (log.replace('layer 0 assigned to device CUDA1','layer 0 assigned to device CUDA0'),
                  log.replace('blk.8.ffn_up_exps','blk.7.ffn_up_exps'),log.replace('CUDA_Host','CUDA0')):
        with pytest.raises(ValueError):verify_placement(wrong,config)


def test_quality_is_parsed_equality_not_just_valid_json():
    ref={str(i):{'parsed':{'value':i},'response':{'content':str(i),'stop_type':'eos'}} for i in range(36)}
    assert quality_compare(ref,ref)['quality_parity']
    changed={**ref,'0':{'parsed':{'value':123},'response':{'content':'123','stop_type':'eos'}}}
    assert not quality_compare(ref,changed)['quality_parity']
    assert quality_compare(ref,changed)['changed']==['0']
    with pytest.raises(ValueError):quality_compare(ref,{})


def test_configuration_bounds():
    for args in ({'threads':48},{'super_layers':20},{'pin':True,'threads':24},{'cache_mib':9000}):
        with pytest.raises(ValueError):Config('bad',**args)


def test_registry_cannot_upgrade_speed_with_changed_answers():
    import runpy
    from pathlib import Path
    cls=runpy.run_path(str(Path(__file__).resolve().parents[2]/
        'src/neural_architecture_lab/word_games_moe_placement_hypothesis.py'))['MoEPlacementEvidence']
    args=dict(classification='speedup_with_replay_parity',best_condition='gpu8',speedup=1.2,
              parsed_equal=36,full_passes=5,campaign='a.json',campaign_sha256='a'*64,
              report='r.md',report_sha256='b'*64)
    record=cls(**args).record()
    assert record['confirmed'] is False and record['quality_measure']=='finite_frozen_replay_parity'
    for change in ({'parsed_equal':35},{'speedup':1.01},{'speedup':float('inf')},{'full_passes':1}):
        with pytest.raises(ValueError):cls(**{**args,**change})


def test_cache_guard_uses_positive_native_allocation():
    log='\n'.join(f'layer {i} assigned to device CUDA0, is_swa = 0' for i in range(42))
    log+='\n'+'\n'.join(f'tensor blk.{i}.ffn_{p}_exps.weight (144 MiB q4_K) buffer type overridden to CUDA_Host'
                        for i in range(26) for p in ('up','gate','down'))
    native='\n0.09 I impl: CUDA0 MoE cache size = 2046.77 MiB for 12024.00 MiB of host experts'
    assert verify_placement(log+native,CONFIGS['cache2048'])['verified']
    for text in (log,log+native.replace('2046.77','0.00')):
        with pytest.raises(ValueError):verify_placement(text,CONFIGS['cache2048'])


def test_structural_diagnostic_is_separate_from_prose_parity():
    from experiments.neural_architecture_lab.analyze_word_games_moe_placement import structural_diagnostic
    metaphor=dict(name='library',entity='book',location='shelf',slot='position',place_verb='place',
                  return_verb='return',approval='permission',deadline='date')
    ref={'metaphor':{'parsed':metaphor},'cases/x/receiver':{'parsed':{'decision':'applied','occupied_slots':[],
         'summary':'Supported statement.'}},'cases/x/sender/turn_01':{'parsed':{'kind':'final','proposal':{},'target_text':'old'}}}
    changed={**ref,'cases/x/receiver':{'parsed':{**ref['cases/x/receiver']['parsed'],'summary':'Invented approval.'}}}
    diagnostic=structural_diagnostic(ref,changed)
    assert diagnostic['receiver_state_equal']==1 and diagnostic['selector_valid']
    assert diagnostic['post_hoc'] is True and diagnostic['live_workflows'] is False


def test_experts_only_keeps_layer_and_kv_assignment_on_primary_gpu():
    import re
    config=CONFIGS['experts8'];args=command(config,18088,'test')
    assert args[args.index('--device')+1]=='CUDA0,CUDA1'
    assert args[args.index('--tensor-split')+1]=='1,0'
    host,gpu=args[args.index('--override-tensor')+1].split(',')
    assert all(bool(re.search(gpu.split('=')[0],f'blk.{i}.ffn_gate_exps.weight'))==(i<8) for i in range(42))
    log='\n'.join(f'layer {i} assigned to device CUDA0, is_swa = 0' for i in range(42))
    log+='\n'+'\n'.join(f'tensor blk.{i}.ffn_{p}_exps.weight (144 MiB q4_K) buffer type overridden to '+('CUDA1' if i<8 else 'CUDA_Host')
                       for i in range(26) for p in ('up','gate','down'))
    log+='\nload_tensors: CUDA1 model buffer size = 3852.00 MiB'
    assert verify_placement(log,config)['gpu_expert_overrides']==list(range(8))
    for bad in (log.replace('CUDA1','CUDA0'),log.replace('3852.00','0.00'),log.replace('blk.7.ffn_up_exps','blk.8.ffn_up_exps')):
        with pytest.raises(ValueError):verify_placement(bad,config)
