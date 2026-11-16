from copy import deepcopy
import numpy as np
import pytest
import torch
from transformers import Phi3Config, Phi3ForCausalLM
from experiments.structure_net import jspace_phi_native as native


def cases():
    items=[dict(identity=f'case{i}',split='train' if i==0 else 'development',
                input_ids=ids) for i,ids in enumerate(([1,4,7],[1,3,5,6]))]
    responses=[dict(example_id=c['identity'],split=c['split'],prompt_ids=c['input_ids'],
                    response_ids=[8,2],eos_token_id=2,finish_reason='eos') for c in items]
    return items,responses


def test_native_packing_keeps_prompt_end_separate_from_gold_suffix():
    items,responses=cases(); rows=native.full_items(items,responses)
    assert [(r['offset'],r['stop'],r['full_stop']) for r in rows]==[(0,3,5),(5,9,11)]
    assert rows[0]['full_input_ids']==[1,4,7,8,2]
    bad=deepcopy(responses);bad[0]['response_ids']=[8,3]
    with pytest.raises(ValueError):native.full_items(items,bad)
    bad=deepcopy(responses);bad[0]['prompt_ids']=[9]
    with pytest.raises(ValueError):native.full_items(items,bad)


def test_streamed_native_matches_model_and_future_response_cannot_leak(tmp_path):
    torch.manual_seed(3851)
    config=Phi3Config(hidden_size=16,intermediate_size=24,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,vocab_size=32,
        max_position_embeddings=32,original_max_position_embeddings=32,
        tie_word_embeddings=False,bos_token_id=1,eos_token_id=2,pad_token_id=0,
        resid_pdrop=0.,embd_pdrop=0.,attention_dropout=0.)
    config._attn_implementation='eager'
    model=Phi3ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    items=native.full_items(*cases())
    bank=np.lib.format.open_memmap(tmp_path/'scratch_a.npy',mode='w+',dtype=np.int16,
                                   shape=(items[-1]['full_stop'],16))
    for c in items:
        native.put(bank,c['offset'],model.model.embed_tokens(torch.tensor([c['full_input_ids']])))
    bank.flush();calls=[]
    result=native.stream_blocks(tmp_path,items,16,3,bank,lambda i:model.model.layers[i],
                               model.model.rotary_emb,'cpu',lambda:calls.append(1))
    prefix=np.load(tmp_path/'prefix.npy',mmap_mode='r')
    for c in items:
        full=torch.tensor([c['full_input_ids']]);prompt=torch.tensor([c['input_ids']])
        actual=native.get(result,c,'cpu')
        expected=model.model(input_ids=full,use_cache=False,output_hidden_states=True)
        assert torch.equal(model.model.norm(actual),expected.last_hidden_state)
        assert torch.equal(native.get(prefix,c,'cpu'),expected.hidden_states[2])
        prompt_result=model.model(input_ids=prompt,use_cache=False).last_hidden_state
        assert torch.equal(model.model.norm(actual[:,:len(c['input_ids'])]),prompt_result)
    assert len(calls)==3*(1+len(items))
    result._mmap.close();prefix._mmap.close()


def test_nonfinite_bank_rejected(tmp_path):
    bank=np.zeros((2,4),dtype=np.int16)
    with pytest.raises(ValueError):native.put(bank,0,torch.full((1,2,4),float('nan'),dtype=torch.bfloat16))
    with pytest.raises(ValueError):native.put(bank,0,torch.ones(1,2,4))


def test_prompt_and_teacher_forced_banks_split_without_crossing_rows(tmp_path):
    full,prompt,width=7,5,4
    expected=np.arange((full+prompt)*width,dtype=np.int16).reshape(full+prompt,width)
    for name in ('prefix.npy','teacher_hidden.npy'):np.save(tmp_path/name,expected,allow_pickle=False)
    native.separate_banks(tmp_path,full,prompt,width)
    for name in ('prefix.npy','teacher_hidden.npy'):
        assert np.array_equal(np.load(tmp_path/name),expected[:full])
        assert np.array_equal(np.load(tmp_path/('prompt_'+name)),expected[full:])
        assert not (tmp_path/('combined_'+name)).exists()
