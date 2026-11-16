"""Focused sanity selection and sealed-confirmation identity contracts."""
import json
from pathlib import Path

import pytest

from experiments.structure_net import jspace_3b_sanity_data as data


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert messages[0] == dict(role="system", content=data.SYSTEM)
        assert kwargs == dict(tokenize=True, add_generation_prompt=True)
        return list(range(385 if "LONG" in messages[1]["content"] else 12))


def row(i):
    return dict(sentence=f"The _ case {i}.", option1="first", option2="second")


def test_identity_scan_handles_nested_and_compact_history_without_json_decode(tmp_path):
    a, b, c = "a"*64, "b"*64, "c"*64
    path = tmp_path/"tasks.json"
    path.write_text(json.dumps(dict(splits={"train":[dict(identity=a)], "development":[dict(identity=b)]}, previous_identities=[c])))
    assert data.identity_candidates(path) == {a,b,c}


def test_unlabeled_selection_disjoint_deterministic_and_length_limited():
    rows = [row(i) for i in range(12)] + [row(1)] + [row("LONG")]
    excluded = {data.identity(*data.prompt_parts("winogrande", rows[0]))}
    selected, _ = data.select_unlabeled("winogrande", rows, excluded, Tokenizer(), seed=17, count=11)
    assert len(selected) == len({c["identity"] for c in selected}) == 11
    assert not excluded & {c["identity"] for c in selected}
    assert all("answer" not in c and len(c["input_ids"]) == 12 for c in selected)
    assert selected == data.select_unlabeled("winogrande", rows, excluded, Tokenizer(), seed=17, count=11)[0]


def test_cannot_fill_sanity_by_reusing_reserved_or_long_examples():
    rows = [row(1), row("LONG")]
    reserved = {data.identity(*data.prompt_parts("winogrande", rows[0]))}
    with pytest.raises(ValueError, match="Only 0"):
        data.select_unlabeled("winogrande", rows, reserved, Tokenizer(), seed=1, count=1)


def test_balance_and_all_population_disjointness_gate():
    cases = [dict(identity=str(i)*64, family=f, input_ids=[1,2,3], choices=["a","b"], answer=0, split="sanity")
             for i,f in enumerate(data.DATASETS)]
    data.validate_cases(cases, set(), per_family=1)
    with pytest.raises(ValueError, match="disjointness"):
        data.validate_cases(cases, {cases[2]["identity"]}, per_family=1)
    with pytest.raises(ValueError, match="balance"):
        data.validate_cases(cases[:-1], set(), per_family=1)


def test_reserved_identity_receipt_never_decodes_confirmation_tasks(tmp_path, monkeypatch):
    confirmation, expansion = tmp_path/"confirmation", tmp_path/"expansion"
    confirmation.mkdir();expansion.mkdir()
    cp=dict(config={});cp["fingerprint"]=data.digest(cp)
    ep=dict(confirmation_fingerprint=cp["fingerprint"]);ep["fingerprint"]=data.digest(ep)
    data.write(confirmation/"plan.json",cp);data.write(expansion/"plan.json",ep)
    (confirmation/"tasks.json").write_text("SEALED CONTENT MUST NOT BE JSON-DECODED")
    data.write(confirmation/"exclusions.json",dict(identities=[]))
    ids=[f"{i:064x}" for i in range(512)]
    data.write(expansion/"confirmation_identities.json",dict(identities=ids,source_fingerprint=cp["fingerprint"]))
    data.write(confirmation/"manifest.json",dict(status="frozen_unscored",fingerprint=cp["fingerprint"],
        artifacts={n:data.sha(confirmation/n) for n in ("plan.json","tasks.json","exclusions.json")}))
    data.write(expansion/"manifest.json",dict(status="frozen_untrained",fingerprint=ep["fingerprint"],
        artifacts={"confirmation_identities.json":data.sha(expansion/"confirmation_identities.json")}))
    real_read=data.read
    def guarded_read(path):
        assert Path(path)!=confirmation/"tasks.json"
        return real_read(path)
    monkeypatch.setattr(data,"read",guarded_read)
    found,_,_=data.reserved_identities(confirmation,expansion)
    assert found==set(ids)
    (expansion/"confirmation_identities.json").write_text("{}")
    with pytest.raises(ValueError,match="identity-only"):
        data.reserved_identities(confirmation,expansion)


def test_cpu_only_freeze_guard_precedes_source_reads(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(ValueError,match="CUDA hidden"):
        data.freeze(tmp_path/"output")
    assert not (tmp_path/"output").exists()
