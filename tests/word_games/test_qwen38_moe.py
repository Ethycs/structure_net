import pytest
from pathlib import Path
import runpy

from experiments.word_games.qwen38_moe_runtime import command, validate_metadata, verify_expert_placement


def test_user_requested_placement_and_precision_are_explicit():
    args=command(18087,'test')
    for flag,value in {'--n-cpu-moe':'26','--flash-attn':'on','--cache-type-k':'q8_0',
                       '--cache-type-v':'q8_0','--spec-type':'none','--fit':'off'}.items():
        assert args[args.index(flag)+1]==value
    assert '--spec-draft-model' not in args


def test_metadata_rejects_wrong_quant_or_backbone():
    data={'general.architecture':'qwen35moe','general.file_type':15,'qwen35moe.block_count':41,
          'qwen35moe.nextn_predict_layers':1,'qwen35moe.expert_count':256,'qwen35moe.expert_used_count':8}
    validate_metadata(data)
    for key,value in [('general.file_type',10),('general.architecture','qwen35'),('qwen35moe.expert_used_count',4)]:
        with pytest.raises(ValueError): validate_metadata({**data,key:value})


def test_placement_gate_rejects_missing_or_extra_cpu_expert_blocks():
    def log(n):
        return '\n'.join(f'tensor blk.{i}.ffn_{p}_exps.weight (144 MiB q4_K) buffer type overridden to CPU_REPACK'
                         for i in range(n) for p in ('gate','up','down'))
    assert len(verify_expert_placement(log(26)))==26
    assert len(verify_expert_placement(log(26).replace('CPU_REPACK','CUDA_Host')))==26
    for invalid in (log(25),log(27),log(26).replace('CPU_REPACK','CUDA0'),log(26).replace('ffn_gate_exps','missing')):
        with pytest.raises(ValueError): verify_expert_placement(invalid)


def test_registry_preserves_manual_semantic_limitations_and_user_configuration():
    cls=runpy.run_path(str(Path(__file__).resolve().parents[2]/
        'src/neural_architecture_lab/word_games_qwen38_moe_hypothesis.py'))['Qwen38Evidence']
    args=dict(classification='higher_joint_task_success',successes=12,cases=12,rate=15,
              campaign='c.json',campaign_sha256='a'*64,report='r.md',report_sha256='b'*64,semantic_supported=10)
    record=cls(**args).record()
    assert record['confirmed'] is False and record['semantic_supported_summaries']==10
    assert record['cpu_expert_layers']==26 and record['weight_quant']=='Q4_K_M'
    stopped=cls(**{**args,'classification':'metaphor_gate_failed','cases':0,'successes':0,'semantic_supported':0}).record()
    assert stopped['cases_completed']==0 and stopped['semantic_review_method']=='not_reached'
    for change in ({'semantic_supported':13},{'rate':float('nan')},{'cases':0},{'successes':5}):
        with pytest.raises(ValueError): cls(**{**args,**change})
