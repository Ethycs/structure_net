"""Meaningful-position, permutation, identity and sealed-audit data contracts."""
import copy
from pathlib import Path

import pytest

from experiments.structure_net import jspace_3b_broad_recovery_data as data


class CharacterTokenizer:
    def apply_chat_template(self,messages,*,tokenize,add_generation_prompt):
        assert add_generation_prompt
        rendered="<system>"+messages[0]["content"]+"</system><user>"+messages[1]["content"]+"</user><assistant>\n"
        return [ord(c) for c in rendered] if tokenize else rendered

    def __call__(self,text,*,add_special_tokens,return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        return dict(input_ids=[ord(c) for c in text],offset_mapping=[(i,i+1) for i in range(len(text))])


def case(family,index,answer=0,choices=None):
    choices=choices or ["first word","second word"]
    question=f"Select the item numbered {index}."
    layout=data.ContentLayout(CharacterTokenizer(),[65,66,67,68])
    return dict(layout.encode(question,choices,answer),identity=data.identity(question,choices),family=family,
        choices=choices,answer=answer,split="train",source_split="train",source_row=index)


def test_content_targets_are_inside_user_text_and_exclude_fixed_chat_and_labels():
    tokenizer=CharacterTokenizer();layout=data.ContentLayout(tokenizer,[65,66,67,68])
    row=layout.encode("Which animal can swim?",["A fish","A stone"],0)
    begin,end=row["user_content_token_span"]
    for p in row["content_loss_positions"][:3]:
        assert begin<=p<p+1<end
        assert chr(row["input_ids"][p+1]).isalnum()
    assert row["content_targets"][-1]==65
    assert row["legacy_loss_positions"]==[0,(len(row["input_ids"])-2)//2,len(row["input_ids"])-2,len(row["input_ids"])-1]
    assert row["content_loss_positions"][-1]==len(row["input_ids"])-1


def test_train_permutation_balances_positions_remaps_answers_preserves_every_source():
    source=[case(family,i,answer=0) for family in data.DATASETS for i in range(7)]
    # Ensure task identities are unique across families, matching the real frozen population.
    for i,row in enumerate(source):
        replacement=case(row["family"],i,answer=0);row.clear();row.update(replacement)
    layout=data.ContentLayout(CharacterTokenizer(),[65,66,67,68])
    transformed,summary=data.permute_training(source,layout,seed=19)
    again,_=data.permute_training(source,layout,seed=19)
    assert transformed==again and len(transformed)==len(source)
    assert {c["source_identity"] for c in transformed}=={c["identity"] for c in source}
    assert [c["family"] for c in transformed[:8]]==list(data.DATASETS)*2
    for c in transformed:
        original=source[c["source_case_index"]]
        assert c["choices"][c["answer"]]==original["choices"][original["answer"]]
        assert c["permutation"][c["answer"]]==c["source_answer"]
        assert c["semantic_identity"]==data.semantic_identity(data.question_from_case(original),original["choices"])
    assert all(max(d["answer_counts"].values())-min(d["answer_counts"].values())<=1 for d in summary.values())


def test_position_dependent_choices_keep_original_order():
    source=[case(family,i,choices=["first","all of the above"]) for i,family in enumerate(data.DATASETS)]
    transformed,summary=data.permute_training(source,data.ContentLayout(CharacterTokenizer(),[65,66,67,68]))
    assert all(c["permutation"]==[0,1] and c["permutation_preserved_reason"] for c in transformed)
    assert all(len(d["preserved_flags"])==1 for d in summary.values())


def test_answer_balance_is_separate_for_each_choice_count():
    source=[case("arc",i,choices=["one","two","three","four"]) for i in range(12)]
    source.append(case("arc",12,choices=["one","two","three"]))
    source.extend(case(family,20+i) for i,family in enumerate(list(data.DATASETS)[1:]))
    _,summary=data.permute_training(source,data.ContentLayout(CharacterTokenizer(),[65,66,67,68]))
    assert summary["arc"]["answer_counts_by_choice_count"]["4"]==dict.fromkeys(["0","1","2","3"],3)
    assert sum(summary["arc"]["answer_counts_by_choice_count"]["3"].values())==1


def test_semantic_identity_is_option_order_invariant_and_blocks_fresh_reordered_case():
    rows=[dict(sentence="The _ is here.",option1="first",option2="second")]
    question,choices=data.sanity.prompt_parts("winogrande",rows[0])
    semantic=data.semantic_identity(question,list(reversed(choices)))
    assert semantic==data.semantic_identity(question,choices)
    with pytest.raises(ValueError,match="Insufficient fresh"):
        data.select_fresh("winogrande",rows,set(),{semantic},CharacterTokenizer(),seed=4,count=1)


def test_dev_and_audit_selection_are_sequential_disjoint_and_do_not_read_answers():
    rows=[dict(sentence=f"The _ is here at {i}.",option1="first",option2="second") for i in range(12)]
    dev,_=data.select_fresh("winogrande",rows,set(),set(),CharacterTokenizer(),seed=4,count=4)
    audit,_=data.select_fresh("winogrande",rows,{c["identity"] for c in dev},
        {c["semantic_identity"] for c in dev},CharacterTokenizer(),seed=5,count=4)
    assert not {c["identity"] for c in dev}&{c["identity"] for c in audit}
    assert all("answer" not in c for c in dev+audit)


def test_verifier_does_not_decode_audit_tasks(tmp_path,monkeypatch):
    # A minimal authenticated fixture reaches population validation without opening audit content.
    root=tmp_path
    items=dict(items=[]);tasks=dict(splits={});audit=dict(identities=[f"{i:064x}" for i in range(128)],semantic_identities=[])
    exclusions=dict(training_source_identities=[],identities=[])
    values={"items.json":items,"tasks.json":tasks,"audit_identities.json":audit,"exclusions.json":exclusions}
    for name,value in values.items():data.write(root/name,value)
    (root/"audit_tasks.json").write_text("SEALED: NOT JSON")
    plan=dict(config=data.CONFIG,input_sha256={},source_sha256={},training_order=list(range(4608)),
        items_digest=data.digest(items["items"]),tasks_digest=data.digest(tasks),audit_identities_digest=data.digest(audit),
        exclusions_digest=data.digest(exclusions),counts={"train":4608,"development":256},token_counts={},option_balance={},payload_estimates={})
    plan["fingerprint"]=data.digest(plan);data.write(root/"plan.json",plan)
    data.write(root/"manifest.json",dict(status="frozen_unscored",fingerprint=plan["fingerprint"],
        artifacts={name:data.sha(root/name) for name in [*values,"audit_tasks.json","plan.json"]}))
    original_read=data.read
    def guarded_read(path):
        assert Path(path).name!="audit_tasks.json"
        return original_read(path)
    monkeypatch.setattr(data,"read",guarded_read);monkeypatch.setattr(data,"validate_items",lambda *a:None)
    assert data.verify(root)["counts"]["train"]==4608


def test_cuda_is_hidden_before_any_data_access(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(ValueError,match="CUDA hidden"):
        data.freeze(tmp_path/"new")
    assert not (tmp_path/"new").exists()
