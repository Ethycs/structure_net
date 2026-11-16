from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from experiments.neural_architecture_lab import distillation_recovery as cli
data = cli.module("distillation_data")


def record(name="a", *, prompt=(1, 2), response=(3, 4, 9), split="train", **kwargs):
    provenance = dict(source_uri="fixture://document/"+name, source_revision="revision1",
        tokenizer_sha256="tokenizer", chat_template_sha256="template", prompt_provenance="source",
        response_provenance="reference", assistant_boundary_verified=True)
    provenance.update(kwargs.pop("provenance", {}))
    return data.ResponseRecord(name, kwargs.pop("group_id", name), split, "summary", prompt, response,
        provenance, kwargs.pop("finish_reason", "eos"), kwargs.pop("eos_token_id", 9), **kwargs)


def test_every_response_token_and_real_eos_have_exact_causal_targets_even_when_eos_is_padding():
    records = [record(), record("b", prompt=(5,), response=(6, 9))]
    batch = data.response_batch(records, pad_token_id=9, max_length=5)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 4, 9], [5, 6, 9, 9, 9]]
    assert batch["response_mask"].tolist() == [[False, False, True, True, True], [False, True, True, False, False]]
    assert batch["loss_mask"].tolist() == [[False, True, True, True, False], [True, True, False, False, False]]
    assert batch["targets"][batch["loss_mask"]].tolist() == [3, 4, 9, 6, 9]
    assert int(batch["response_mask"].sum()) == int(batch["loss_mask"].sum()) == 5
    assert batch["counts"] == dict(input_tokens=8, prompt_tokens=3, response_tokens=5, supervised_tokens=5)
    # Real model-shaped logits: the unshifted contract gives the known five targets.
    logits = torch.zeros(2, 5, 10, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits[batch["loss_mask"]], batch["targets"][batch["loss_mask"]])
    loss.backward()
    assert torch.equal(logits.grad.abs().sum(-1)>0, batch["loss_mask"])


def test_cap_is_not_eos_and_no_silent_truncation():
    row = record(response=(3, 4), finish_reason="length")
    batch = data.response_batch([row], pad_token_id=0, max_length=4)
    assert batch["targets"][batch["loss_mask"]].tolist() == [3, 4]
    with pytest.raises(ValueError, match="observed terminal"): record(response=(3, 4))
    with pytest.raises(ValueError, match="Token-cap"): record(finish_reason="length")
    with pytest.raises(ValueError, match="after an observed"): record(response=(9, 4, 9))
    with pytest.raises(ValueError, match="truncation"): data.response_batch([row], pad_token_id=0, max_length=3)


def test_boundary_verified_against_actual_pinned_template_prefix():
    row = record()
    fields = {k:v for k,v in asdict(row).items() if k not in {"prompt_ids", "response_ids", "provenance"}}
    good = data.record_from_tokenized(full_ids=row.input_ids, assistant_boundary=2, expected_prompt_ids=(1,2),
        provenance=row.provenance, **fields)
    assert good.input_ids == row.input_ids
    assert good.provenance["tokenized_prefix_sha256"] == data.digest((1,2))
    with pytest.raises(ValueError, match="assistant boundary"):
        data.record_from_tokenized(full_ids=row.input_ids, assistant_boundary=2, expected_prompt_ids=(1,7),
            provenance=row.provenance, **fields)
    with pytest.raises(ValueError, match="actual prompt"):
        replace(good, prompt_ids=(1,7))


def test_group_derivatives_and_exact_duplicate_inputs_cannot_cross_splits():
    train = record()
    dev = record("dev", prompt=(7,8), split="development", group_id=train.group_id)
    with pytest.raises(ValueError, match="source group"): data.dataset_fingerprint([train,dev])
    dev = record("dev", split="development", response=(6,9))
    with pytest.raises(ValueError, match="exact prompt"): data.dataset_fingerprint([train,dev])
    dev = record("dev", prompt=(7,8), split="development")
    assert data.dataset_fingerprint([train,dev]) == data.dataset_fingerprint(iter([train,dev]))
    with pytest.raises(ValueError, match="Repeated example"): data.dataset_fingerprint([train,train])
    changed = replace(dev, provenance={**dev.provenance, "tokenizer_sha256":"different"})
    with pytest.raises(ValueError, match="tokenizer"): data.dataset_fingerprint([train,changed])


def test_streaming_buckets_preserve_all_records_deterministically_without_packing(tmp_path):
    rows = [record(str(i), prompt=(1,)*(1+i%5)) for i in range(19)]
    path = tmp_path/"train.jsonl"
    path.write_text("".join(json.dumps(asdict(r))+"\n" for r in rows))
    def collect():
        return [[r.example_id for r in group] for group in data.length_buckets(
            data.stream_records(path), batch_size=3, buffer_size=7, max_length=8, seed=11)]
    first = collect()
    assert first == collect() and sorted(x for group in first for x in group) == sorted(r.example_id for r in rows)
    assert all(len(g)<=3 for g in first)
    with pytest.raises(ValueError, match="truncation"):
        list(data.length_buckets(rows, batch_size=2, buffer_size=4, max_length=6))


def test_rollout_identity_binds_student_round_seed_teacher_and_exact_prefix():
    provenance = dict(response_provenance="student_rollout", generator_model="student@sha",
        student_checkpoint="sha1", teacher_id="teacher@sha", round_id="round0", decoding={"do_sample":True}, seed=3)
    row = record(provenance=provenance)
    changes = [dict(student_checkpoint="sha2"), dict(round_id="round1"), dict(seed=4), dict(teacher_id="other")]
    assert all(data.record_fingerprint(replace(row, provenance={**row.provenance,**p})) != data.record_fingerprint(row) for p in changes)
    with pytest.raises(ValueError, match="TRAIN"): replace(row, split="development")
    with pytest.raises(ValueError, match="seed"): record(provenance={**provenance,"seed":None})
    with pytest.raises(ValueError, match="round teacher"):
        data.target_contract(row,teacher_id="different",temperature=2.,vocabulary_size=10)


def test_streamed_full_cache_returns_exact_requested_rows_and_hashes_only_at_admission(tmp_path, monkeypatch):
    row = record(); logits = torch.arange(30).reshape(3,10).bfloat16()
    directory=tmp_path/"cache"
    data.write_target_cache(directory,row,teacher_id="teacher",temperature=2.,vocabulary_size=10,logits=logits)
    assert data.target_contract(row,teacher_id="teacher",temperature=2.,vocabulary_size=10)["coverage"]=="complete_response"
    assert data.target_contract(row,teacher_id="teacher",temperature=2.,vocabulary_size=10,positions=[1])["coverage"]=="declared_subset"
    calls=[]; original=data._sha
    monkeypatch.setattr(data,"_sha",lambda path:(calls.append(str(path)),original(path))[1])
    store=data.TeacherTargetStore({data.record_fingerprint(row):directory},teacher_id="teacher",temperature=2.,vocabulary_size=10)
    batch=data.response_batch([row],pad_token_id=0,max_length=5)
    positions=torch.tensor([[0,3],[0,1]])
    assert torch.equal(store(batch,positions),logits.float()[[2,0]])
    assert torch.equal(store(batch,positions),logits.float()[[2,0]])
    assert len(calls)==1
    batch["input_ids"][0,0]=7
    with pytest.raises(ValueError,match="context tokens"):store(batch,positions)
    batch=data.response_batch([row],pad_token_id=0,max_length=5)
    batch["attention_mask"][0,0]=False
    with pytest.raises(ValueError,match="attention context"):store(batch,positions)
    batch=data.response_batch([row],pad_token_id=0,max_length=5)
    with (directory/"logits.npy").open("ab") as stream:stream.write(b"tamper")
    with pytest.raises(ValueError,match="admitted teacher cache changed"):store(batch,positions)


@pytest.mark.parametrize("teacher,temperature",[("different",2.),("teacher",1.)])
def test_wrong_teacher_or_temperature_cache_rejected(tmp_path,teacher,temperature):
    row=record();directory=tmp_path/"cache"
    data.write_target_cache(directory,row,teacher_id="teacher",temperature=2.,vocabulary_size=10,logits=torch.zeros(3,10))
    store=data.TeacherTargetStore({data.record_fingerprint(row):directory},teacher_id=teacher,temperature=temperature,vocabulary_size=10)
    with pytest.raises(ValueError,match="provenance mismatch"):
        store(data.response_batch([row],pad_token_id=0,max_length=5),[[0,1]])


def test_topk_tail_is_normalized_labelled_approximate_and_cannot_masquerade_as_exact(tmp_path):
    row=record();directory=tmp_path/"cache"
    manifest=data.write_target_cache(directory,row,teacher_id="teacher",temperature=2.,vocabulary_size=10,
        topk_ids=[[2,3]]*3,probabilities=[[.3,.2]]*3,tail_mass=[.5]*3)
    assert manifest["format"]=="topk_plus_tail_approximation"
    assert manifest["contract"]["positions"]==[1,2,3]
    store=data.TeacherTargetStore({data.record_fingerprint(row):directory},teacher_id="teacher",temperature=2.,vocabulary_size=10)
    with pytest.raises(ValueError,match="separate approximation"):
        store(data.response_batch([row],pad_token_id=0,max_length=5),[[0,1]])
    for ids,tail in [([[2,2]]*3,[.5]*3),([[2,3]]*3,[.1]*3),([[2.2,3.]]*3,[.5]*3)]:
        with pytest.raises(ValueError):data.write_target_cache(tmp_path/"bad",row,teacher_id="teacher",temperature=2.,
            vocabulary_size=10,topk_ids=ids,probabilities=[[.3,.2]]*3,tail_mass=tail)


def test_pointer_only_corpus_index_deduplicates_without_copying_or_admitting(tmp_path):
    from experiments.structure_net import distillation_corpus as corpus
    rows=[record(),record("duplicate"),record("dev",prompt=(7,8),split="development")]
    path=tmp_path/"responses.jsonl"
    path.write_text("".join(json.dumps(asdict(r))+"\n" for r in rows))
    receipt=corpus.prepare(path,tmp_path/"prepared",max_length=5)
    assert receipt["counts"]["input_records"]==3 and receipt["counts"]["active_records"]==2
    assert receipt["counts"]["exact_duplicate_records"]==1
    assert receipt["counts"]["response_tokens"]==receipt["counts"]["supervised_tokens"]==6
    assert receipt["new_study_admitted"] is False
    pointers=[json.loads(line) for line in (tmp_path/"prepared/index.jsonl").read_text().splitlines()]
    assert pointers[1]["duplicate_of"]=="a"
    assert all("prompt_ids" not in p and "response_ids" not in p for p in pointers)
    with path.open("rb") as stream:
        for pointer in pointers:
            stream.seek(pointer["offset_bytes"])
            value=stream.read(pointer["length_bytes"])
            assert data.hashlib.sha256(value).hexdigest()==pointer["record_sha256"]
    with pytest.raises(FileExistsError):corpus.prepare(path,tmp_path/"prepared",max_length=5)
    with pytest.raises(ValueError,match="admission limit"):corpus.prepare(path,tmp_path/"overlimit",max_records=2)
    assert not (tmp_path/"overlimit").exists()
