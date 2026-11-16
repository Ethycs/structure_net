"""Option text, gold remapping and original-replay gating are independent."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from experiments.structure_net import jspace_3b_option_order as study


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):
        assert messages[0]==dict(role="system",content=study.sanity.data_source.SYSTEM)
        assert kwargs==dict(tokenize=True,add_generation_prompt=True)
        return [1,*hashlib.sha256(messages[1]["content"].encode()).digest()[:12],2]


def cases():
    tokenizer=Tokenizer();result=[]
    for i in range(3):
        question=f"Sentence {i} about the _ with a blank."
        choices=[f"first text {i}",f"second text {i}"]
        prompt=question+"\n\nA. "+choices[0]+"\nB. "+choices[1]
        ids=tokenizer.apply_chat_template([dict(role="system",content=study.sanity.data_source.SYSTEM),
            dict(role="user",content=prompt)],tokenize=True,add_generation_prompt=True)
        result.append(dict(identity=study.sanity.data_source.identity(question,choices),family="winogrande",
            prompt=prompt,choices=choices,answer=i%2,input_ids=ids,split="sanity"))
    return result


def record(case,prediction):
    return dict(identity=case["identity"],family=case["family"],answer=case["answer"],prediction=prediction,
        logits=[3.,0.] if prediction==0 else [0.,3.])


def test_swapping_changes_only_option_order_and_remaps_gold_without_losing_identity():
    original=cases();items=study.make_pairs(original,Tokenizer())
    assert len(items)==6 and len({c["identity"] for c in items})==6
    for i in range(3):
        a,b=items[i],items[i+3]
        assert a["input_ids"]==original[i]["input_ids"] and a["prompt"]==original[i]["prompt"]
        assert a["identity"]==original[i]["identity"]==b["original_identity"]
        assert b["choices"]==list(reversed(a["choices"])) and b["answer"]==1-a["answer"]
        assert b["prompt"].split("\n\n")[0]==a["prompt"].split("\n\n")[0]
        assert b["input_ids"]!=a["input_ids"]
    assert items[0]["offset"]==0 and all(a["stop"]==b["offset"] for a,b in zip(items,items[1:]))


@pytest.mark.parametrize("defect",["tokens","identity","prompt","duplicate_options","length"])
def test_changed_original_template_or_ambiguous_options_are_rejected(defect):
    original=cases();tokenizer=Tokenizer()
    if defect=="tokens":original[0]["input_ids"][0]=99
    elif defect=="identity":original[0]["identity"]="wrong"
    elif defect=="prompt":original[0]["prompt"]+="\nextra"
    elif defect=="duplicate_options":original[0]["choices"][1]=original[0]["choices"][0]
    else:
        tokenizer.apply_chat_template=lambda *_a,**_kw:list(range(385))
        original[0]["input_ids"]=list(range(385))
    with pytest.raises(ValueError):study.make_pairs(original,tokenizer)


def test_semantic_consistency_differs_from_always_A_letter_bias():
    items=study.make_pairs(cases(),Tokenizer());n=3
    semantic=[record(c,c["answer"]) for c in items]
    result=study.order_metrics(semantic,items)
    assert result["original_correct"]==result["swapped_correct"]==result["both_correct"]==n
    assert result["remapped_choice_consistency"]==1.
    always_A=[record(c,0) for c in items];result=study.order_metrics(always_A,items)
    assert result["original_A_predictions"]==result["swapped_A_predictions"]==3
    assert result["original_text_option0_chosen_before"]==3 and result["original_text_option0_chosen_after"]==0
    assert result["remapped_choice_consistency"]==0. and result["mean_variant_accuracy"]==.5
    assert result["both_correct"]==0 and result["either_correct"]==3


def test_original_logit_mismatch_aborts_before_any_swapped_forward():
    items=study.make_pairs(cases(),Tokenizer());reference=[record(c,c["answer"]) for c in items[:3]];calls=[]
    def scorer(case):
        calls.append(case["variant"]);value=record(case,case["answer"])
        if len(calls)==1:value["logits"][case["answer"]]=3.25  # Same prediction is insufficient.
        return value
    with pytest.raises(ValueError,match="exactly"):study.score_variants(items,reference,scorer)
    assert calls==["original"]*3


def test_exact_original_replay_allows_swapped_phase_without_altering_records():
    items=study.make_pairs(cases(),Tokenizer());reference=[record(c,c["answer"]) for c in items[:3]];events=[]
    records,proof=study.score_variants(items,reference,lambda c:record(c,c["answer"]),lambda i,c:events.append((i,c["variant"])))
    assert proof==dict(passed=True,exact_records=True,exact_logits=True,count=3)
    assert records[:3]==reference and [x[1] for x in events]==["original"]*3+["swapped"]*3


def test_semantically_wrong_gold_or_swapped_correspondence_rejected():
    items=study.make_pairs(cases(),Tokenizer());items[-1]["original_identity"]="other"
    with pytest.raises(ValueError,match="correspondence"):
        study.order_metrics([record(c,c["answer"]) for c in items],items)


def test_denied_reservation_precedes_model_and_cuda(tmp_path,monkeypatch):
    monkeypatch.setattr(study,"verify",lambda *_:{"fingerprint":"p"})
    def deny(*_):raise RuntimeError("reservation denied")
    monkeypatch.setattr(study.sanity,"wait_ready",deny)
    monkeypatch.setattr(torch.cuda,"init",lambda:pytest.fail("CUDA before lease"))
    monkeypatch.setattr(study.sanity.execution,"load_native_model",lambda *_a,**_k:pytest.fail("model before lease"))
    with pytest.raises(RuntimeError,match="reservation denied"):study.run(tmp_path,"native")


def test_compact_requires_completed_native_replay_before_reservation(tmp_path,monkeypatch):
    monkeypatch.setattr(study,"verify",lambda *_:{"fingerprint":"p"})
    def deny(*_):raise ValueError("native replay incomplete")
    monkeypatch.setattr(study,"checked",deny)
    monkeypatch.setattr(study.sanity,"wait_ready",lambda *_:pytest.fail("reservation before native proof"))
    with pytest.raises(ValueError,match="incomplete"):study.run(tmp_path,"compact")


def test_verification_rejects_retokens_even_when_saved_reference_is_unchanged(tmp_path):
    items=study.make_pairs(cases(),Tokenizer());references={"native":[],"compact":[]}
    plan=dict(schema=study.SCHEMA,policy=study.POLICY,checkpoint_sha256=study.sanity.CHECKPOINT_SHA,
        allocation_mib=11264,allocator_mib=11008,input_identity={},source_sha256={},
        items_digest=study.base.digest(items),references_digest=study.base.digest(references))
    plan["fingerprint"]=study.base.digest(plan)
    study.base.write(tmp_path/"plan.json",plan);study.base.write(tmp_path/"items.json",dict(items=items));study.base.write(tmp_path/"references.json",references)
    assert study.verify(tmp_path)["fingerprint"]==plan["fingerprint"]
    items[0]["input_ids"][1]+=1
    (tmp_path/"items.json").write_text(json.dumps(dict(items=items)))
    with pytest.raises(ValueError,match="cases or references"):study.verify(tmp_path)
