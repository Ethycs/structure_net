import hashlib
import json

import pytest

from experiments.structure_net import jspace_recovery_expansion as data


def test_balanced_quotas_exhaust_small_family_without_duplicates():
    assert data.balanced_quotas(dict(arc=973,boolq=9283,hellaswag=39761,winogrande=9102),4096) == dict(
        arc=973,boolq=1041,hellaswag=1041,winogrande=1041)
    assert data.balanced_quotas(dict(a=2,b=100,c=100),11) == dict(a=2,b=5,c=4)
    with pytest.raises(ValueError): data.balanced_quotas(dict(a=2,b=3),6)


def test_confirmation_reader_extracts_only_identity_lines_without_json_decoder(tmp_path,monkeypatch):
    ids = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(3)]
    path = tmp_path/"tasks.json"
    path.write_text(json.dumps({"cases":[dict(identity=i,prompt='do not inspect "identity": "fake"',
                                             answer=77,choices=["secret"]) for i in ids]},indent=2))
    monkeypatch.setattr(json,"loads",lambda *_a,**_k: pytest.fail("Confirmation content was parsed"))
    assert data.confirmation_identities_only(path,expected_count=3) == set(ids)
    with pytest.raises(ValueError): data.confirmation_identities_only(path,expected_count=4)


class Tokenizer:
    def apply_chat_template(self,batch,*,tokenize,add_generation_prompt):
        assert tokenize and add_generation_prompt
        assert all(c[0] == {"role":"system","content":data.SYSTEM} for c in batch)
        return [[1]* (500 if "TOO_LONG" in c[1]["content"] else 10) for c in batch]


def test_scan_counts_full_population_but_retains_bounded_deterministic_training_prefix():
    rows=[dict(passage=f"passage{i}",question="question",answer=bool(i%2)) for i in range(7)]
    rows.append(dict(passage="TOO_LONG",question="question",answer=True))
    rows.append(rows[0])
    question,choices,_=data.normalize("boolq",rows[1])
    excluded={hashlib.sha256(json.dumps([question,choices]).encode()).hexdigest()}
    first,inventory,seen=data.scan_training_family("boolq",rows,excluded,Tokenizer(),seed=42,keep_limit=3,batch_size=2)
    again,same,_=data.scan_training_family("boolq",rows,excluded,Tokenizer(),seed=42,keep_limit=3,batch_size=3)
    assert first == again and inventory == same
    assert inventory["eligible_total"] == 6 and inventory["outside_token_limit"] == 1
    assert inventory["excluded_or_duplicate"] == 2 and len(first) == 3 and len(seen) == 7
    assert all(c["source_split"] == "train" and c["identity"] not in excluded for c in first)


def original_items():
    return [dict(identity=f"old{i}",split="train" if i<512 else "development",stop=(i+1)*10,
                 input_ids=list(range(10)),answer=i%2,choices=["n","y"])
            for i in range(576)]


def test_expansion_preserves_every_old_row_and_dev_and_four_loss_positions():
    old=original_items()
    new=dict(identity="new",family="boolq",source_split="train",source_row=42,
             input_ids=list(range(11)),answer=1,choices=["n","y"])
    items=data.assemble_items(old,[new],[32,33,34,35])
    assert items[:576] == old
    assert [c for c in items if c["split"] == "development"] == old[512:]
    assert items[-1]["loss_positions"] == [0,4,9,10]
    assert items[-1]["targets"] == [1,5,10,33]
    assert items[-1]["offset"] == 5760 and items[-1]["stop"] == 5771
    for changed in (dict(new,identity="old0"),dict(new,source_split="validation")):
        with pytest.raises(ValueError): data.assemble_items(old,[changed],[32,33,34,35])
