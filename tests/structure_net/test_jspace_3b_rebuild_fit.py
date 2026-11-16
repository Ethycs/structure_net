"""Native-first full-response fitting, real tiny Qwen, and adjacent inheritance."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_rebuild_fit as study
from experiments.structure_net.jspace_3b_response_train import module

data = module("distillation_data")


class Dataset:
    def __init__(self, native, cut):
        self.records=[];self.boundaries=[];self.next=[]
        provenance=dict(source_uri="fixture",source_revision="v1",tokenizer_sha256="fixture",
            chat_template_sha256="fixture",prompt_provenance="fixture",response_provenance="reference",
            assistant_boundary_verified=True)
        for i in range(12):
            record=data.ResponseRecord(example_id=f"case{i}",group_id=f"group{i}",split="train",
                task="a" if i%2==0 else "b",prompt_ids=(1,2+i),response_ids=(15,)*((i%3)+1)+(31,),
                provenance=provenance,finish_reason="eos",eos_token_id=31)
            self.records.append(record)
            def first(_m,args):self.boundaries.append(args[0].detach().clone())
            def last(_m,args):self.next.append(args[0].detach().clone())
            handles=[native.model.layers[cut].register_forward_pre_hook(first)]
            following=native.model.norm if cut+1==len(native.model.layers) else native.model.layers[cut+1]
            handles.append(following.register_forward_pre_hook(last))
            with torch.no_grad():native(input_ids=torch.tensor([record.input_ids]),use_cache=False)
            for handle in handles:handle.remove()

    def batch(self,indices,device="cpu",dtype=torch.bfloat16):
        i,=indices
        return dict(boundary_hidden=self.boundaries[i].to(device),next_boundary_hidden=self.next[i].to(device))


def fixture(tmp_path):
    torch.manual_seed(29834)
    cfg=Qwen2Config(hidden_size=8,intermediate_size=16,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,vocab_size=32,max_position_embeddings=32,
        tie_word_embeddings=True,attention_dropout=0.,eos_token_id=31,pad_token_id=0)
    cfg._attn_implementation="eager"
    native=Qwen2ForCausalLM(cfg).bfloat16().eval().requires_grad_(False)
    path=tmp_path/"native";native.save_pretrained(path)
    config=study.FitConfig(rank=4,head_dim=2,anchors=2,remove_count=4,svd_rank=4,oversample=2,
                          probes_per_group=2,position_capacity=12,chunk_rows=32)
    return native,path,config


def test_response_calibration_uses_every_response_and_eos_and_rejects_dev(tmp_path):
    native,_,_=fixture(tmp_path);dataset=Dataset(native,1)
    cases=study.calibration_cases(dataset.records,range(8),range(8,12),production=False)
    assert [len(c["loss_positions"]) for c in cases]==[len(r.response_ids) for r in dataset.records]
    assert all(c["targets"][-1]==31 for c in cases)
    assert all(c["loss_positions"][0]==1 for c in cases)
    with pytest.raises(ValueError,match="disjoint"):
        study.calibration_cases(dataset.records,[0,1],[1,2],production=False)
    dataset.records[8]=replace(dataset.records[8],split="development")
    with pytest.raises(ValueError,match="TRAIN"):
        study.calibration_cases(dataset.records,range(8),range(8,12),production=False)


def test_bootstrap_full_response_fit_and_adjacent_parent_inheritance(tmp_path,monkeypatch):
    native,path,config=fixture(tmp_path)
    monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:pytest.fail("CPU fixture initialized CUDA"))
    fits={};previous=None;old=None
    for cut in (1,0):
        dataset=Dataset(native,cut)
        receipt=study.fit_stage(tmp_path/f"stage{cut}",dataset,model_path=path,cut=cut,
            calibration_indices=list(range(8)),diagnostic_indices=list(range(8,12)),teacher_fingerprint="native-fixture",
            parent_fits=fits,parent_checkpoint=previous,config=config,production=False,progress=lambda _:None)
        assert receipt["status"]=="completed" and receipt["native_seam_checks"]==12
        assert receipt["response_positions"]==sum(len(r.response_ids) for r in dataset.records[:8])
        assert receipt["order"][2]=="capture compiled SwiGLU"
        directory=receipt["fit_directory"];fits[cut]=directory
        interface=json.loads((tmp_path/f"stage{cut}"/"fits"/f"block_{cut:02d}"/"interface.json").read_text())
        assert interface["native_calibration"].endswith("compiled_train.npy")
        activations=np.load(interface["native_calibration"])
        assert activations.shape==(receipt["response_positions"],16)
        assert activations.dtype==np.int16
        capture=json.loads((tmp_path/f"stage{cut}"/"capture"/"metadata.json").read_text())
        assert len(capture["normalization"]["per_record"])==8
        jac=np.load(capture["blocks"][str(cut)]["mlp_jacobian_bf16"],mmap_mode="r")
        assert jac.shape==(sum(len(r.input_ids) for r in dataset.records[:8])*4,16)
        jac._mmap.close()
        stack,norm,E,rotary,inheritance=study.builder.build_stack(path,fits,cut,previous,
            device="cpu",trainable=False,rank=4,checkpoint_blocks=False)
        if old is not None:
            assert inheritance["unchanged_layers"]==[1] and inheritance["norm_exact"]
            for name,value in old["stack"].items():
                if name.startswith("blocks.0."):
                    assert torch.equal(value,stack.state_dict()[name.replace("blocks.0.","blocks.1.")])
        assert all(block.has_polynomial for block in stack.blocks)
        old={"stack":stack.state_dict(),"norm":norm.state_dict()}
        previous=tmp_path/f"parent{cut}.pt";torch.save(old,previous)


def test_native_boundary_mismatch_is_a_hard_failure(tmp_path):
    native,path,config=fixture(tmp_path);dataset=Dataset(native,1)
    dataset.next[0]=dataset.next[0]+1
    with pytest.raises(ValueError,match="next-boundary"):
        study.fit_stage(tmp_path/"bad",dataset,model_path=path,cut=1,calibration_indices=range(8),
            diagnostic_indices=range(8,12),teacher_fingerprint="fixture",config=config,production=False,progress=lambda _:None)


def test_means_use_all_train_native_bank_and_global_fill_without_dev_or_forward():
    values=torch.tensor([[1.,1.],[1.,1.],[3.,3.],[3.,3.],[3.,3.],[100.,100.]]).bfloat16()
    dataset=SimpleNamespace(records=[SimpleNamespace(split="train",input_ids=[1,2]),
        SimpleNamespace(split="train",input_ids=[3,4,5]),SimpleNamespace(split="development",input_ids=[6])],
        boundary=values.view(torch.int16).numpy(),rows=[dict(offset=0,stop=2),dict(offset=2,stop=5),dict(offset=5,stop=6)],
        batch=lambda *_a,**_k:pytest.fail("Means loaded teacher targets or ran a model"))
    result=study.training_position_means(dataset,width=2,capacity=4,progress=lambda _:None)
    assert result["training_records"]==2 and result["training_tokens"]==5
    assert torch.equal(result["counts"],torch.tensor([2.,2.,1.,0.]))
    torch.testing.assert_close(result["means"],torch.tensor([[2.,2.],[2.,2.],[3.,3.],[2.2,2.2]]))
    assert result["unobserved_positions"]==[3]
